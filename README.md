# Kasakoa Benchmark

A GhanaNLP project to evaluate the reasoning and comprehension ability of Large Language Models on idioms used in Ghanaian languages.

## Project Overview

**Kasakoa** (meaning *"indirect speech"* in Twi) benchmarks LLMs on their ability to correctly answer comprehension questions given statements based on idioms in Ghanaian languages. Unlike translation benchmarks, this tests deeper understanding — whether a model can correctly interpret meaning beyond the direct meaning of these idioms and understnd the complex logic behind the idioms that are used in Ghanaian languages.

Similarity between model answers and reference answers is computed using **[BAAI/BGE-M3](https://huggingface.co/BAAI/bge-m3)** embeddings via NVIDIA Build, giving a language-agnostic semantic similarity score.

## Task Format

Each input CSV contains rows with three columns:

| Column | Description |
|---|---|
| `statement` | A sentence or passage in the target Ghanaian language |
| `question` | A comprehension question about the statement |
| `correct_answer_text` | The reference/gold answer (in English) |

The model is prompted to answer the question based on the statement and its answer is scored against the reference using cosine similarity of BGE-M3 embeddings.

## Results (Overview)

> Results will appear here after running the benchmark across multiple languages.

<div style="display:flex; flex-wrap:wrap; gap:10px;">

<a href="reports/model_performance.png">
  <img src="reports/model_performance.png" width="45%" alt="Overall Model Performance">
</a>

<a href="reports/language_performance.png">
  <img src="reports/language_performance.png" width="45%" alt="Language Performance">
</a>

</div>

## Language-Specific Results

<details>
<summary>Click to expand language results</summary>

Results per language are stored in `reports/<lang_code>/` and include:
- `performance_comparison.png` — ranked bar chart of model scores
- `performance_vs_consistency_quadrant.png` — performance vs consistency scatter
- `summary.csv` — raw numbers

</details>

## Evaluated Models

Models are configured in `recipes/models.csv`. Set `tested = yes` to include a model in runs.

Current active models:

| Model | Provider |
|---|---|
| deepseek-ai/deepseek-v3.2 | nvidia |
| google/gemma-3-1b-it | nvidia |
| google/gemma-3-27b-it | nvidia |
| openai/gpt-oss-120b | nvidia |
| moonshotai/kimi-k2-instruct-0905 | nvidia |
| meta/llama-3.1-405b-instruct | nvidia |
| meta/llama-4-maverick-17b-128e-instruct | nvidia |
| mistralai/mistral-large-3-675b-instruct-2512 | nvidia |
| qwen/qwen3.5-397b-a17b | nvidia |
| gemini-3-flash-preview | gemini |
| gpt-5.1 | openai |

## Setup

```bash
git clone https://github.com/GhanaNLP/kasakoa-benchmark
cd kasakoa-benchmark
pip install -r requirements.txt
cp .env.example .env
# Fill in your API keys in .env
```

## Running the Benchmark

```bash
python main.py
```

You will be prompted to select which language to benchmark:

```
Available languages in input/:
  [1] Twi (twi)  —  twi_reasoning_dataset.csv
  [2] Ewe (ewe)  —  ewe_reasoning_dataset.csv

Enter number or language code: 1
```

The pipeline then runs three stages automatically:

1. **Inference** — queries each active model with all QA pairs
2. **Similarity** — scores model answers against reference answers using BGE-M3
3. **Reports** — generates charts and summary CSVs

All stages support **resume**: if the process is interrupted, re-running `main.py` and selecting the same language will skip already-completed work.

## Input File Naming

Input CSVs should be named with the language code as a prefix:

```
input/twi_reasoning_dataset.csv
input/ewe_reasoning_dataset.csv
input/dag_reasoning_dataset.csv
```

Language codes follow ISO 639-3 conventions (same as `utils/language_mapping.csv`).

## Repo Structure

```
kasakoa-benchmark/
├── main.py                      # Entry point — runs full pipeline
├── requirements.txt
├── .env.example
├── processing_state.json        # Auto-generated resume state (gitignored)
│
├── recipes/
│   ├── models.csv               # Model registry — set tested=yes to enable
│   └── qa_recipe.py             # LLM inference + BGE-M3 similarity scoring
│
├── utils/
│   ├── language_mapping.csv     # ISO codes → language names
│   └── reporting.py             # Chart and summary generation
│
├── input/                       # Put your input CSVs here
│   └── twi_reasoning_dataset.csv
│
├── output/                      # Auto-generated per-model results (gitignored)
│   └── twi/
│       └── twi_deepseek-ai_deepseek-v3.2.csv
│
└── reports/                     # Charts and summaries
    ├── model_performance.png    # Cross-language overall model ranking
    ├── language_performance.png # Cross-language language difficulty
    └── twi/
        ├── performance_comparison.png
        ├── performance_vs_consistency_quadrant.png
        └── summary.csv
```

## Similarity Metric

Answers are scored using cosine similarity between BGE-M3 embeddings of the model answer and the reference answer. BGE-M3 is a multilingual embedding model that handles both English and Ghanaian language text, making it appropriate for cross-lingual answer comparison.

Scores range from 0–1, where 1 = semantically identical.

## Contributing

1. Add your language CSV to `input/` following the naming convention
2. Run the benchmark
3. Submit results or open a PR with your `reports/<lang_code>/` output

For questions: [info@ghananlp.org](mailto:info@ghananlp.org)

## License

Open community project. Contributions welcome.
