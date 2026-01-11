# GPQA experiment README (single-file handoff)

## Overview
This repo supports running a **GPQA debate → judge → accuracy** pipeline where:
- **Debater 1** and **Debater 2** defend two *different* multiple-choice options (A/B/C/D).
- Each debater is **bound to the original model’s reasoning trace + model** (label-bound, not “correctness-bound”).
- The **judge directly outputs the correct option letter**: `Answer: <A|B|C|D>`.
- You can run **swap=False and swap=True** and average accuracy to mitigate prefix bias.

**What “swap” means (GPQA):** we run the *same question pair twice* but **flip which side is “Debater 1” vs “Debater 2”** (presentation order). This tests/averages out any judge bias from seeing one side first. It does **not** change the ground-truth answer and is **not** the same as the multiple-choice option letters A/B/C/D.

## Prereqs
- **Python**: 3.11 recommended.
- **Secrets**: create `SECRETS` file with `OPENROUTER_API_KEY` if you’ll use OpenRouter.
- **GPQA trace files** (not committed; they’re ignored):
  - `data/gpqa/GPQA_Reasoning_Traces_openai_gpt-4o_main_all_448.json`
  - `data/gpqa/GPQA_Reasoning_Traces_anthropic_claude-sonnet-4_main_all_448.json`

If your trace files live elsewhere, update the constants in `core/load/gpqa_loader.py` (`MODEL_A_FILE`, `MODEL_B_FILE`).

## Small end-to-end smoke test (2 rows)
Start here to verify your environment + keys + prompts work end-to-end quickly.

```bash
python -m core.load.gpqa_loader ./data/gpqa_runs_test/debate_sim/data0.csv --limit 2

python -m core.debate \
  +experiment=gpqa_debate \
  exp_dir=./data/gpqa_runs_test \
  dataset_type=gpqa \
  llm_provider=openrouter \
  limit=2

python -m core.judge \
  +experiment=gpqa_debate \
  exp_dir=./data/gpqa_runs_test \
  dataset_type=gpqa \
  llm_provider=openrouter \
  limit=2 \
  ++judge.language_model.model=gpt-4o-mini \
  ++judge_name=gpt-4o-mini

python -m core.scoring.gpqa_accuracy score_both \
  --filename=./data/gpqa_runs_test/debate_sim/data0.csv \
  --judge_name=gpt-4o-mini \
  --verbose=True
```

**Why `limit=2` is good for testing:** it makes debate/judge finish quickly and cheaply while exercising the full pipeline. Remove `limit` for full runs.

## 1) Generate the dataset CSV (from the two trace JSONs)
This produces a CSV of pairs where the two models disagree and exactly one is correct. If you omit `--limit`, it writes the full filtered set.

```bash
python -m core.load.gpqa_loader ./data/gpqa_runs/debate_sim/data0.csv
```

## 2) Run debates (writes both swap and non-swap)
This will create:
- `./data/gpqa_runs/debate_sim/data0.csv`
- `./data/gpqa_runs/debate_sim/data0_swap.csv`

```bash
python -m core.debate \
  +experiment=gpqa_debate \
  exp_dir=./data/gpqa_runs \
  dataset_type=gpqa \
  llm_provider=openrouter
```

## 3) Run judge (writes both swap and non-swap judgements)
This will create (under a judge subdir):
- `.../gpt-4o-mini/data0_judgement.csv`
- `.../gpt-4o-mini/data0_swap_judgement.csv`

```bash
python -m core.judge \
  +experiment=gpqa_debate \
  exp_dir=./data/gpqa_runs \
  dataset_type=gpqa \
  llm_provider=openrouter \
  ++judge.language_model.model=gpt-4o-mini \
  ++judge_name=gpt-4o-mini
```

## 4) Score accuracy (average across swap)
```bash
python -m core.scoring.gpqa_accuracy score_both \
  --filename=./data/gpqa_runs/debate_sim/data0.csv \
  --judge_name=gpt-4o-mini \
  --verbose=True
```

## Notes (important behaviors)
- **Judge prompt**: `core/config/experiment/judge/gpqa/default.yaml`
- **GPQA experiment config**: `core/config/experiment/gpqa_debate.yaml`
- **Debater naming**: user-facing names are **Debater 1 / Debater 2** (to avoid confusion with option letters A/B/C/D).
- **Caching**: per-file caches live in `.../debate_sim/cache_data0/`. Delete to force a rerun.
