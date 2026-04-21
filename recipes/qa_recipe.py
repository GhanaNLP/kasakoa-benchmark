"""
qa_recipe.py
Handles:
  1. LLM inference for QA (statement + question → model answer)
  2. Similarity scoring via NVIDIA baai/bge-m3 embeddings
"""

import os
import re
import time
import threading
import concurrent.futures
from datetime import datetime

import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# ─── Rate limiter ─────────────────────────────────────────────────────────────

class RateLimiter:
    """Token bucket rate limiter. Thread-safe."""
    def __init__(self, max_requests_per_minute):
        self.interval = 60.0 / max_requests_per_minute
        self.last_request_time = 0.0
        self.lock = threading.Lock()

    def acquire(self):
        with self.lock:
            now = time.time()
            wait = self.interval - (now - self.last_request_time)
            if wait > 0:
                time.sleep(wait)
                now = time.time()
            self.last_request_time = now


# Global limiter: 35 requests per minute
_RATE_LIMITER = RateLimiter(max_requests_per_minute=35)

# ─── Client factory ───────────────────────────────────────────────────────────

def _get_client(provider):
    if provider == "nvidia":
        return OpenAI(
            base_url="https://integrate.api.nvidia.com/v1",
            api_key=os.getenv("NVIDIA_BUILD_API_KEY"),
        )
    elif provider == "openai":
        return OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    elif provider == "mistral":
        return OpenAI(
            base_url="https://api.mistral.ai/v1",
            api_key=os.getenv("MISTRAL_API_KEY"),
        )
    elif provider == "anthropic":
        import anthropic
        return anthropic.Anthropic(api_key=os.getenv("CLAUDE_API_KEY"))
    elif provider == "gemini":
        return None  # handled separately
    return None


# ─── Prompt builder ───────────────────────────────────────────────────────────

def _build_prompt(statement, question, lang_name):
    return (
        f"You are evaluating comprehension of a {lang_name} statement.\n\n"
        f"Statement: {statement}\n"
        f"Question: {question}\n\n"
        f"Give a short, direct answer in no more than 5 words. "
        f"Just the answer, inside square brackets.\n"
        f"Answer:"
    )


def _extract_bracketed(text):
    match = re.search(r'\[(.*?)\]', text, flags=re.S)
    if match:
        return match.group(1).strip()
    return text.strip()


# ─── Single call with retries ─────────────────────────────────────────────────

def _call_model(client, statement, question, lang_name, model_id, provider, max_retries=5):
    prompt = _build_prompt(statement, question, lang_name)

    for attempt in range(max_retries):
        # Rate limit check BEFORE each attempt
        _RATE_LIMITER.acquire()
        try:
            if provider == "anthropic":
                response = client.messages.create(
                    model=model_id,
                    max_tokens=512,
                    temperature=0.3,
                    messages=[{"role": "user", "content": prompt}],
                )
                return _extract_bracketed(response.content[0].text)

            elif provider == "gemini":
                import google.generativeai as genai
                genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
                model = genai.GenerativeModel(model_id)
                response = model.generate_content(
                    prompt, generation_config={"temperature": 0.3}
                )
                return _extract_bracketed(response.text)

            else:
                kwargs = {
                    "model": model_id,
                    "messages": [{"role": "user", "content": prompt}],
                    "stream": False,
                }
                if provider == "openai":
                    kwargs["max_completion_tokens"] = 512
                    if not any(r in model_id.lower() for r in ["o1", "o3"]):
                        kwargs["temperature"] = 0.3
                        kwargs["top_p"] = 0.95
                else:
                    kwargs["max_tokens"] = 512
                    kwargs["temperature"] = 0.3
                    kwargs["top_p"] = 0.95

                # Disable reasoning chain for known NVIDIA reasoning models
                if provider == "nvidia":
                    thinking_keywords = ["deepseek", "kimi", "nemotron", "r1", "think"]
                    if any(k in model_id.lower() for k in thinking_keywords):
                        kwargs["extra_body"] = {
                            "chat_template_kwargs": {"thinking": False}
                        }

                completion = client.chat.completions.create(**kwargs)
                return _extract_bracketed(completion.choices[0].message.content)

        except Exception as e:
            ts = datetime.now().strftime('%H:%M:%S')
            print(f"  [{ts}] Attempt {attempt+1} failed: {e}")
            if attempt < max_retries - 1:
                # Shorter backoff: 1, 2, 4, 8, 16 instead of 2, 3, 5, 9, 17
                time.sleep(2 ** attempt)
            else:
                return ""

    return ""


# ─── Batch inference ──────────────────────────────────────────────────────────

# Removed BATCH_SIZE, BATCH_PAUSE, STAGGER_DELAY.
# Use enough workers to keep the pipeline full while rate limiter throttles.
MAX_WORKERS = 4


def run_qa(df, lang_code, lang_name, model_id, provider):
    """
    Takes a DataFrame with columns: statement, question, correct_answer_text
    Returns the same DataFrame with an added 'model_answer' column.
    """
    client = _get_client(provider) if provider != "gemini" else None

    result_df = df.copy()
    result_df['model_answer'] = ""
    total = len(result_df)
    completed = 0
    lock = threading.Lock()

    print(f"    Rows: {total}  |  Workers: {MAX_WORKERS}  |  Rate limit: 35 req/min")

    def worker(row_idx):
        nonlocal completed
        row = result_df.iloc[row_idx]
        try:
            answer = _call_model(
                client,
                row['statement'],
                row['question'],
                lang_name,
                model_id,
                provider,
            )
        except Exception as e:
            ts = datetime.now().strftime('%H:%M:%S')
            print(f"    [{ts}] Row {row_idx+1} error: {e}")
            answer = ""

        with lock:
            result_df.at[row_idx, 'model_answer'] = answer
            completed += 1
            current = completed

        ts = datetime.now().strftime('%H:%M:%S')
        snippet = answer[:60] if answer else "[empty]"
        print(f"    [{ts}] Row {row_idx+1}/{total} ({current} done): {snippet}")

    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [executor.submit(worker, i) for i in range(total)]
        concurrent.futures.wait(futures)

    return result_df


# ─── Similarity scoring via NVIDIA BGE-M3 ────────────────────────────────────

_EMB_CLIENT = None


def _get_emb_client():
    global _EMB_CLIENT
    if _EMB_CLIENT is None:
        _EMB_CLIENT = OpenAI(
            base_url="https://integrate.api.nvidia.com/v1",
            api_key=os.getenv("NVIDIA_BUILD_API_KEY"),
        )
    return _EMB_CLIENT


def _cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(x * x for x in b) ** 0.5
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def _embed_batch(texts, max_retries=5):
    """Embed a list of texts using NVIDIA BGE-M3. Returns list of vectors."""
    client = _get_emb_client()
    for attempt in range(max_retries):
        try:
            response = client.embeddings.create(
                input=texts,
                model="baai/bge-m3",
                encoding_format="float",
                extra_body={"truncate": "NONE"},
            )
            return [item.embedding for item in response.data]
        except Exception as e:
            ts = datetime.now().strftime('%H:%M:%S')
            print(f"  [{ts}] Embedding attempt {attempt+1} failed: {e}")
            if attempt < max_retries - 1:
                time.sleep((2 ** attempt) + 1)
    return [[0.0]] * len(texts)


EMB_BATCH_SIZE = 32


def compute_similarity(df):
    """
    Adds 'similarity_score' column to df.
    Requires 'model_answer' and 'correct_answer_text' columns.
    """
    result_df = df.copy()
    answers = result_df['model_answer'].fillna('').tolist()
    refs = result_df['correct_answer_text'].fillna('').tolist()
    scores = []

    total = len(answers)
    for i in range(0, total, EMB_BATCH_SIZE):
        batch_ans = answers[i:i + EMB_BATCH_SIZE]
        batch_ref = refs[i:i + EMB_BATCH_SIZE]
        # Embed both in one call to reduce API round-trips
        combined = batch_ans + batch_ref
        vecs = _embed_batch(combined)
        ans_vecs = vecs[:len(batch_ans)]
        ref_vecs = vecs[len(batch_ans):]
        for a, r in zip(ans_vecs, ref_vecs):
            scores.append(_cosine(a, r))
        print(f"    Scored rows {i+1}–{min(i+EMB_BATCH_SIZE, total)}/{total}")

    result_df['similarity_score'] = scores
    return result_df
