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

# Default model configuration (GPQA)
DEFAULT_MODEL_A_NAME = "GPT-4o"
DEFAULT_MODEL_B_NAME = "Claude Sonnet 4"

DEFAULT_MODEL_A_FILE = Path("data/gpqa/GPQA_Reasoning_Traces_openai_gpt-4o_main_all_448.json")
DEFAULT_MODEL_B_FILE = Path(
    "data/gpqa/GPQA_Reasoning_Traces_anthropic_claude-sonnet-4_main_all_448.json"
)


def _load_entries(
    model_a_file: Optional[Path] = None,
    model_b_file: Optional[Path] = None,
) -> Dict[str, Dict[int, GPQAEntry]]:
    model_a_file = model_a_file or DEFAULT_MODEL_A_FILE
    model_b_file = model_b_file or DEFAULT_MODEL_B_FILE
    
    model_a_entries = load_gpqa_file(model_a_file)
    model_b_entries = load_gpqa_file(model_b_file)
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


def build_rows(
    limit: Optional[int] = None,
    model_a_file: Optional[Path] = None,
    model_b_file: Optional[Path] = None,
    model_a_name: Optional[str] = None,
    model_b_name: Optional[str] = None,
) -> List[Dict]:
    model_a_name = model_a_name or DEFAULT_MODEL_A_NAME
    model_b_name = model_b_name or DEFAULT_MODEL_B_NAME
    
    entries = _load_entries(model_a_file, model_b_file)
    pairs = match_pairs(entries["A"], entries["B"])
    filtered_pairs = filter_disagreements(pairs)
    if limit is not None:
        filtered_pairs = filtered_pairs[: int(limit)]

    rows = []
    for i, (entry_a, entry_b) in enumerate(filtered_pairs):
        rows.append(build_row(i, entry_a, entry_b, model_a_name, model_b_name))
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
    model_a_file: Optional[str] = None,
    model_b_file: Optional[str] = None,
    model_a_name: Optional[str] = None,
    model_b_name: Optional[str] = None,
):
    """
    GPQA loader to align with the Quality loader signature.
    Extra parameters are accepted for compatibility but ignored.
    
    Args:
        filepath: Output CSV filepath
        limit: Limit number of questions to load
        model_a_file: Path to model A reasoning traces JSON (optional, defaults to GPQA GPT-4o)
        model_b_file: Path to model B reasoning traces JSON (optional, defaults to GPQA Claude Sonnet 4)
        model_a_name: Display name for model A (optional, defaults to "GPT-4o")
        model_b_name: Display name for model B (optional, defaults to "Claude Sonnet 4")
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
    
    # Convert string paths to Path objects if provided
    model_a_path = Path(model_a_file) if model_a_file else None
    model_b_path = Path(model_b_file) if model_b_file else None
    
    rows = build_rows(
        limit=limit,
        model_a_file=model_a_path,
        model_b_file=model_b_path,
        model_a_name=model_a_name,
        model_b_name=model_b_name,
    )
    if write_to_file:
        _write_csv(rows, filepath)
    return rows


if __name__ == "__main__":
    fire.Fire(main)
