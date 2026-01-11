import json
import re
from pathlib import Path
from typing import Dict, List, Tuple

from pydantic import BaseModel


class GPQAEntry(BaseModel):
    index: int
    choices: str
    correct_choice: str
    model_answer: str
    model_reasoning: List[str]
    model_correct: bool
    Question: str

    class Config:
        extra = "allow"


def parse_choices(choices: str) -> Dict[str, str]:
    """
    Parse a choices string like "A. opt1, B. opt2, C. opt3, D. opt4" into a map.
    Handles occasional newlines or trailing commas.
    """
    # Normalize whitespace/newlines
    normalized = " ".join(choices.replace("\n", " ").split())
    pattern = re.compile(r"([A-D])\.\s*([^,]+)")
    matches = pattern.findall(normalized)
    return {letter.strip().upper(): text.strip() for letter, text in matches}


def load_gpqa_file(path: Path) -> Dict[int, GPQAEntry]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    entries = {}
    for raw in data:
        entry = GPQAEntry(**raw)
        entries[entry.index] = entry
    return entries


def match_pairs(
    model_a_entries: Dict[int, GPQAEntry],
    model_b_entries: Dict[int, GPQAEntry],
) -> List[Tuple[GPQAEntry, GPQAEntry]]:
    shared_indices = set(model_a_entries.keys()) & set(model_b_entries.keys())
    pairs = []
    for idx in sorted(shared_indices):
        pairs.append((model_a_entries[idx], model_b_entries[idx]))
    return pairs


def filter_disagreements(
    pairs: List[Tuple[GPQAEntry, GPQAEntry]]
) -> List[Tuple[GPQAEntry, GPQAEntry]]:
    """Keep only pairs where models disagree and exactly one is correct."""
    filtered = []
    for a, b in pairs:
        answers_differ = (
            a.model_answer.strip().upper() != b.model_answer.strip().upper()
        )
        exactly_one_correct = (a.model_correct ^ b.model_correct) is True
        if answers_differ and exactly_one_correct:
            filtered.append((a, b))
    return filtered


def build_row(
    idx: int,
    entry_a: GPQAEntry,
    entry_b: GPQAEntry,
    model_a_name: str,
    model_b_name: str,
) -> Dict:
    choices_map = parse_choices(entry_a.choices)
    correct_letter = entry_a.correct_choice.strip().upper()

    # Determine which model is correct in this pair
    correct_model = "A" if entry_a.model_correct else "B"
    incorrect_entry = entry_b if entry_a.model_correct else entry_a

    correct_answer_text = choices_map.get(correct_letter, "")
    negative_answer_text = choices_map.get(
        incorrect_entry.model_answer.strip().upper(), ""
    )

    return {
        "id": idx,
        "question": entry_a.Question.strip(),
        "choices": entry_a.choices,
        "correct answer": correct_answer_text,
        "correct answer letter": correct_letter,  # Store the ground truth letter directly
        "negative answer": negative_answer_text,
        "model_a_name": model_a_name,
        "model_b_name": model_b_name,
        "model_a_reasoning": json.dumps(entry_a.model_reasoning),
        "model_b_reasoning": json.dumps(entry_b.model_reasoning),
        "model_a_answer": entry_a.model_answer.strip().upper(),
        "model_b_answer": entry_b.model_answer.strip().upper(),
        "correct_model": correct_model,
        "complete": False,
        "transcript": "",
    }
