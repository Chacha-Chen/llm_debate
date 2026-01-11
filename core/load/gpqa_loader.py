import csv
import json
from pathlib import Path
from typing import Dict, List, Optional

import fire

from core.load.gpqa import (
    GPQAEntry,
    build_row,
    filter_disagreements,
    load_gpqa_file,
    match_pairs,
)
from core.utils import typer_async

MODEL_A_NAME = "GPT-4o"
MODEL_B_NAME = "Claude Sonnet 4"

MODEL_A_FILE = Path("data/gpqa/GPQA_Reasoning_Traces_openai_gpt-4o_main_all_448.json")
MODEL_B_FILE = Path(
    "data/gpqa/GPQA_Reasoning_Traces_anthropic_claude-sonnet-4_main_all_448.json"
)


def _load_entries() -> Dict[str, Dict[int, GPQAEntry]]:
    model_a_entries = load_gpqa_file(MODEL_A_FILE)
    model_b_entries = load_gpqa_file(MODEL_B_FILE)
    return {"A": model_a_entries, "B": model_b_entries}


def _write_csv(rows: List[Dict], filepath: Path | str):
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "id",
                "question",
                "choices",
                "correct answer",
                "correct answer letter",
                "negative answer",
                "model_a_name",
                "model_b_name",
                "model_a_reasoning",
                "model_b_reasoning",
                "model_a_answer",
                "model_b_answer",
                "correct_model",
                "complete",
                "transcript",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def build_rows(limit: Optional[int] = None) -> List[Dict]:
    entries = _load_entries()
    pairs = match_pairs(entries["A"], entries["B"])
    filtered_pairs = filter_disagreements(pairs)
    if limit is not None:
        filtered_pairs = filtered_pairs[: int(limit)]

    rows = []
    for i, (entry_a, entry_b) in enumerate(filtered_pairs):
        rows.append(build_row(i, entry_a, entry_b, MODEL_A_NAME, MODEL_B_NAME))
    return rows


@typer_async
async def main(
    filepath: Path | str,
    split: Optional[List[str]] = None,
    max_tokens: Optional[int] = None,
    limit: Optional[int] = None,
    take_from_end: bool = False,
    write_to_file: bool = True,
    sources: Optional[List[str]] = None,
    difficulty: Optional[int] = None,
    ignore_nyu: bool = True,
    minimize_story_duplication: Optional[bool] = None,
    max_answerability: Optional[float] = None,
    min_untimed_accuracy: Optional[float] = None,
    max_speed_accuracy: Optional[float] = None,
    min_context_required: Optional[float] = None,
    skip_conflicting_labels: Optional[bool] = None,
    max_num_from_same_story: Optional[int] = None,
    human_experiments: Optional[List[str]] = None,
):
    """
    GPQA loader to align with the Quality loader signature.
    Extra parameters are accepted for compatibility but ignored.
    """
    _ = (
        split,
        max_tokens,
        take_from_end,
        sources,
        difficulty,
        ignore_nyu,
        minimize_story_duplication,
        max_answerability,
        min_untimed_accuracy,
        max_speed_accuracy,
        min_context_required,
        skip_conflicting_labels,
        max_num_from_same_story,
        human_experiments,
    )  # unused

    filepath = Path(filepath)
    rows = build_rows(limit=limit)
    if write_to_file:
        _write_csv(rows, filepath)
    return rows


if __name__ == "__main__":
    fire.Fire(main)
