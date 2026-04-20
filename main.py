import os
import pandas as pd
import sys
import json
import time
from datetime import timedelta

sys.path.append(os.path.join(os.path.dirname(__file__), 'recipes'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'utils'))

import qa_recipe
import reporting
from dotenv import load_dotenv

load_dotenv()


# ─── State helpers ────────────────────────────────────────────────────────────

def load_state(state_file="processing_state.json"):
    if os.path.exists(state_file):
        try:
            with open(state_file, 'r') as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def save_state(state, state_file="processing_state.json"):
    try:
        with open(state_file, 'w') as f:
            json.dump(state, f, indent=2)
    except Exception as e:
        print(f"[WARN] Could not save state: {e}")


# ─── Config loaders ───────────────────────────────────────────────────────────

def load_models(csv_path="recipes/models.csv"):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"{csv_path} not found.")
    df = pd.read_csv(csv_path)
    return df[df['tested'].str.lower() == 'yes'].to_dict('records')


def load_language_mapping(csv_path="utils/language_mapping.csv"):
    if not os.path.exists(csv_path):
        return {}
    df = pd.read_csv(csv_path)
    return dict(zip(df['language_code'].str.lower(), df['language_name']))


def setup_env():
    if not os.path.exists('.env'):
        print("No .env found. Creating template.")
        with open('.env', 'w') as f:
            f.write(
                "NVIDIA_BUILD_API_KEY=\n"
                "OPENAI_API_KEY=\n"
                "CLAUDE_API_KEY=\n"
                "GEMINI_API_KEY=\n"
                "MISTRAL_API_KEY=\n"
            )


# ─── Language selection ───────────────────────────────────────────────────────

def pick_language(input_dir, lang_map):
    csv_files = [f for f in os.listdir(input_dir) if f.endswith('.csv')]
    if not csv_files:
        print(f"No CSV files found in '{input_dir}/'.")
        sys.exit(1)

    # Extract language codes from filenames (e.g. twi_reasoning_dataset.csv -> twi)
    available = {}
    for f in csv_files:
        # Try to detect a language code as the first underscore-separated token
        code = f.split('_')[0].lower()
        name = lang_map.get(code, code.upper())
        available[code] = (name, f)

    print("\nAvailable languages in input/:")
    options = sorted(available.items())
    for i, (code, (name, fname)) in enumerate(options, 1):
        print(f"  [{i}] {name} ({code})  —  {fname}")

    while True:
        choice = input("\nEnter number or language code: ").strip()
        if choice.isdigit():
            idx = int(choice) - 1
            if 0 <= idx < len(options):
                code, (name, fname) = options[idx]
                return code, name, fname
        else:
            choice = choice.lower()
            if choice in available:
                code = choice
                name, fname = available[code]
                return code, name, fname
        print("  Invalid selection, try again.")


# ─── Stage 1: LLM inference ───────────────────────────────────────────────────

def run_inference(lang_code, lang_name, input_file, input_dir, output_dir, models, state):
    print(f"\n{'='*60}")
    print(f"STAGE 1 — LLM Inference  [{lang_name} / {lang_code}]")
    print(f"{'='*60}")

    df = pd.read_csv(os.path.join(input_dir, input_file))
    lang_out = os.path.join(output_dir, lang_code)
    os.makedirs(lang_out, exist_ok=True)

    total = sum(
        1 for m in models
        if not state.get(f"inference/{lang_code}/{m['model_id'].replace('/', '_')}", {}).get('done')
    )
    done = 0
    start = time.time()

    for model in models:
        safe_id = model['model_id'].replace('/', '_')
        key = f"inference/{lang_code}/{safe_id}"
        out_path = os.path.join(lang_out, f"{lang_code}_{safe_id}.csv")

        if state.get(key, {}).get('done'):
            print(f"  [SKIP] {model['model_id']} — already done")
            continue

        print(f"\n  → {model['model_id']} ({model['provider']})")
        try:
            result_df = qa_recipe.run_qa(
                df=df,
                lang_code=lang_code,
                lang_name=lang_name,
                model_id=model['model_id'],
                provider=model['provider'],
            )
            result_df.to_csv(out_path, index=False)

            state[key] = {
                'done': True,
                'rows': len(result_df),
                'timestamp': pd.Timestamp.now().isoformat(),
            }
            save_state(state)
            done += 1

            elapsed = time.time() - start
            eta = str(timedelta(seconds=int(elapsed / done * (total - done)))) if done < total else "0:00:00"
            print(f"  Progress: {done}/{total} — ETA {eta}")
        except Exception as e:
            print(f"  [ERROR] {model['model_id']}: {e}")

    print(f"\nInference complete in {timedelta(seconds=int(time.time()-start))}.")


# ─── Stage 2: Similarity scoring ─────────────────────────────────────────────

def run_similarity(lang_code, output_dir, state):
    print(f"\n{'='*60}")
    print(f"STAGE 2 — Similarity Scoring  [{lang_code}]")
    print(f"{'='*60}")

    lang_out = os.path.join(output_dir, lang_code)
    csv_files = [f for f in os.listdir(lang_out) if f.endswith('.csv')]

    for fname in csv_files:
        key = f"similarity/{lang_code}/{fname}"
        if state.get(key, {}).get('done'):
            print(f"  [SKIP] {fname}")
            continue

        fpath = os.path.join(lang_out, fname)
        df = pd.read_csv(fpath)

        if 'model_answer' not in df.columns or 'correct_answer_text' not in df.columns:
            print(f"  [SKIP] {fname} — missing required columns")
            continue

        print(f"  Scoring {fname} ...")
        df = qa_recipe.compute_similarity(df)
        df.to_csv(fpath, index=False)

        state[key] = {'done': True, 'timestamp': pd.Timestamp.now().isoformat()}
        save_state(state)
        print(f"    Done — avg score: {df['similarity_score'].mean():.4f}")

    print("Similarity scoring complete.")


# ─── Stage 3: Reports ─────────────────────────────────────────────────────────

def run_reports(lang_code, lang_name, output_dir, reports_dir, state):
    print(f"\n{'='*60}")
    print(f"STAGE 3 — Report Generation  [{lang_name} / {lang_code}]")
    print(f"{'='*60}")

    key = f"reports/{lang_code}"
    if state.get(key, {}).get('done'):
        print("  Reports already generated. Re-generating to include any new data...")

    lang_out = os.path.join(output_dir, lang_code)
    lang_rep = os.path.join(reports_dir, lang_code)
    os.makedirs(lang_rep, exist_ok=True)

    reporting.generate_reports(
        lang_code=lang_code,
        lang_name=lang_name,
        output_dir=lang_out,
        reports_dir=lang_rep,
    )

    state[key] = {'done': True, 'timestamp': pd.Timestamp.now().isoformat()}
    save_state(state)
    print(f"  Reports saved to reports/{lang_code}/")


# ─── Entry point ──────────────────────────────────────────────────────────────

def main():
    setup_env()

    input_dir = "input"
    output_dir = "output"
    reports_dir = "reports"
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(reports_dir, exist_ok=True)

    models = load_models("recipes/models.csv")
    lang_map = load_language_mapping("utils/language_mapping.csv")
    state = load_state()

    lang_code, lang_name, input_file = pick_language(input_dir, lang_map)
    print(f"\nSelected: {lang_name} ({lang_code})  —  {input_file}")

    run_inference(lang_code, lang_name, input_file, input_dir, output_dir, models, state)
    run_similarity(lang_code, output_dir, state)
    run_reports(lang_code, lang_name, output_dir, reports_dir, state)

    print(f"\n{'='*60}")
    print("All stages complete.")
    print(f"  Results:  output/{lang_code}/")
    print(f"  Reports:  reports/{lang_code}/")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
