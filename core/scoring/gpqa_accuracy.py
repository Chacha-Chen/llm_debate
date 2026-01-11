import logging
import re
from pathlib import Path
from typing import Optional

import fire
import pandas as pd

from core.load.gpqa import parse_choices
from core.scoring.accuracy import find_answer, find_inconclusive, get_complete_rows

LOGGER = logging.getLogger(__name__)


def func_correct_gpqa(
    judgement,
    correct_answer_letter,
    choices_str=None,
    correct_answer_text=None,
    debug=False,
):
    """
    Determine if judge's answer is correct for GPQA.

    Args:
        judgement: The judge's answer text
        correct_answer_letter: The ground truth answer letter (A, B, C, or D) - preferred
        choices_str: The choices string (e.g., "A. option1, B. option2, ...") - fallback if letter not available
        correct_answer_text: The text of the correct answer - fallback if letter not available
        debug: Whether to print debug info

    Returns:
        True if judge picked the correct answer letter, False if incorrect, "Unknown" if unclear
    """
    judgement = judgement.strip()
    if len(judgement) <= 3:
        return "Unknown"

    # Get the ground truth correct answer letter
    if correct_answer_letter:
        # Direct: use the stored letter
        correct_letter = str(correct_answer_letter).strip().upper()
    elif choices_str and correct_answer_text:
        # Fallback: reverse-lookup from text (for backward compatibility with old CSVs)
        choices_map = parse_choices(choices_str)
        correct_letter = None
        for letter, text in choices_map.items():
            if text.strip() == correct_answer_text.strip():
                correct_letter = letter
                break
        if correct_letter is None:
            if debug:
                LOGGER.warning(
                    f"Could not find correct answer letter for text: {correct_answer_text}"
                )
            return "Unknown"
    else:
        if debug:
            LOGGER.warning(
                "Missing both correct_answer_letter and (choices_str + correct_answer_text)"
            )
        return "Unknown"

    # Extract judge's answer letter (A, B, C, or D)
    judge_picked_letter = None
    for letter in ["A", "B", "C", "D"]:
        if find_answer(judgement, letter):
            judge_picked_letter = letter
            break

    # If no letter found, check for inconclusive
    if judge_picked_letter is None:
        if find_inconclusive(judgement):
            return False
        if debug:
            print(judgement)
            print("======================\n\n")
            user_input = input("Do you want to see more? (y/n)")
            if user_input == "n":
                raise StopIteration("User stopped")
        return "Unknown"

    # Direct comparison: judge's answer letter vs ground truth answer letter
    return judge_picked_letter == correct_letter


def get_gpqa_accuracy(
    df,
    swap: bool = False,  # Kept for backward compatibility, no longer used for correctness checking
    debug: bool = False,
    n_votes: int = 1,
    verbose: bool = False,
):
    if not isinstance(df, pd.DataFrame):
        df = pd.read_csv(df, encoding="utf-8")

    full = len(df)
    correct_columns = []

    for n_vote in range(n_votes):
        suffix = n_vote if n_vote > 0 else ""
        complete_column = f"complete_judge{suffix}"
        judge_column = f"answer_judge{suffix}"
        correct_column = f"correct{suffix}"
        correct_columns.append(correct_column)

        # Check if judge columns exist
        if complete_column not in df.columns or judge_column not in df.columns:
            raise ValueError(
                f"Missing judge columns: {complete_column} and/or {judge_column}. "
                f"Please run the judge first using: python -m core.judge +experiment=gpqa_debate exp_dir=<your_exp_dir> dataset_type=gpqa"
            )

        df = get_complete_rows(df, complete_column, ensure_complete=True)

        # Check if required columns exist
        # Prefer "correct answer letter" if available (direct comparison), otherwise fall back to text lookup
        if "correct answer letter" in df.columns:
            # Direct comparison: use the stored ground truth letter
            df[correct_column] = df.apply(
                lambda row: func_correct_gpqa(
                    row[judge_column],
                    correct_answer_letter=row.get("correct answer letter"),
                    choices_str=row.get("choices"),
                    correct_answer_text=row.get("correct answer"),
                    debug=debug,
                ),
                axis=1,
            )
        elif "correct answer" in df.columns and "choices" in df.columns:
            # Fallback: reverse-lookup from text (for backward compatibility)
            df[correct_column] = df.apply(
                lambda row: func_correct_gpqa(
                    row[judge_column],
                    correct_answer_letter=None,
                    choices_str=row["choices"],
                    correct_answer_text=row["correct answer"],
                    debug=debug,
                ),
                axis=1,
            )
        else:
            missing_columns = []
            if (
                "correct answer letter" not in df.columns
                and "correct answer" not in df.columns
            ):
                missing_columns.append("correct answer letter (or correct answer)")
            if (
                "choices" not in df.columns
                and "correct answer letter" not in df.columns
            ):
                missing_columns.append("choices")
            raise ValueError(
                f"Missing required columns: {missing_columns}. "
                f"Need either 'correct answer letter' (preferred) or ('correct answer' + 'choices') for GPQA accuracy calculation."
            )

        if verbose:
            accuracy = (df[correct_column] == True).sum() / full
            LOGGER.info(f"Accuracy {suffix}: {accuracy}")

    df_tmp = df.copy()
    df_tmp["correct_true_count"] = df_tmp[correct_columns].apply(
        lambda row: (row == True).sum(), axis=1
    )
    df_tmp["correct_false_count"] = df_tmp[correct_columns].apply(
        lambda row: (row == False).sum(), axis=1
    )
    df_tmp["correct_voted"] = (
        df_tmp["correct_true_count"] > df_tmp["correct_false_count"]
    )
    count_unknown = (df_tmp[correct_columns] == "Unknown").sum().sum()

    accuracy = (df_tmp["correct_voted"] == True).sum() / full

    return accuracy, count_unknown, full, df


def score_file(
    filename: str,
    swap: bool = False,
    method: Optional[str] = None,
    model: Optional[str] = None,
    dataset: Optional[str] = None,
    results_file: Optional[Path] = None,
    resave_df: bool = False,
    verbose: bool = False,
    debug: bool = False,
    n_votes: int = 1,
    judge_name: Optional[str] = None,
):
    filename_path = Path(filename)

    # If judge columns don't exist in main file, try to find judgement file
    df_test = pd.read_csv(filename_path, encoding="utf-8", nrows=1)
    if "complete_judge" not in df_test.columns or "answer_judge" not in df_test.columns:
        # Look for judgement file in subdirectories
        parent_dir = filename_path.parent
        file_stem = filename_path.stem  # e.g., "data0"
        file_suffix = "_swap_judgement" if swap else "_judgement"

        # Try to find judge subdirectory (e.g., gpt-4, gpt-4-1106-preview, etc.)
        judge_dir = None
        judge_file = None

        # First try the specified judge_name if provided
        if judge_name:
            potential_dir = parent_dir / judge_name
            potential_file = potential_dir / f"{file_stem}{file_suffix}.csv"
            if potential_file.exists():
                judge_dir = potential_dir
                judge_file = potential_file
                if verbose:
                    LOGGER.info(
                        f"Found judgement file with specified judge_name: {judge_file}"
                    )

        # If not found, auto-detect: look for subdirectories with judgement files
        if judge_file is None:
            judge_dirs = [d for d in parent_dir.iterdir() if d.is_dir()]
            for d in judge_dirs:
                potential_file = d / f"{file_stem}{file_suffix}.csv"
                if potential_file.exists():
                    judge_dir = d
                    judge_file = potential_file
                    if verbose:
                        LOGGER.info(f"Auto-detected judgement file: {judge_file}")
                    break

        if judge_file and judge_file.exists():
            filename_path = judge_file
            if verbose:
                LOGGER.info(f"Using judgement file: {judge_file}")
        else:
            raise ValueError(
                f"Missing judge columns in {filename_path}. "
                f"Could not find judgement file: {file_stem}{file_suffix}.csv in subdirectories of {parent_dir}. "
                f"Please run the judge first using: python -m core.judge +experiment=gpqa_debate exp_dir=<your_exp_dir> dataset_type=gpqa"
            )

    accuracy, count_unknown, total, df = get_gpqa_accuracy(
        filename_path, swap=swap, debug=debug, n_votes=n_votes, verbose=verbose
    )

    results = pd.DataFrame(
        {
            "method": [method],
            "accuracy": [accuracy],
            "dataset": [dataset],
            "model": [model],
            "swap": [swap],
            "n_votes": [n_votes],
            "unknown_proportion": [100 * count_unknown / total / n_votes],
            "num_matches": [total],
        }
    )
    if verbose:
        print(results.round(3).to_markdown(index=False))
    if results_file is not None:
        if results_file.exists():
            print(f"Appending to {results_file}")
            results.to_csv(results_file, mode="a", header=False, index=False)
        else:
            results.to_csv(results_file, mode="w", header=True, index=False)
    if resave_df:
        # Save to the file that was actually loaded (filename_path), not the original input
        df.to_csv(filename_path, index=False)
    return results


def score_both_swap_conditions(
    filename: str,
    method: Optional[str] = None,
    model: Optional[str] = None,
    dataset: Optional[str] = None,
    results_file: Optional[Path] = None,
    verbose: bool = False,
    debug: bool = False,
    n_votes: int = 1,
    judge_name: Optional[str] = None,
):
    """
    Calculate accuracy by averaging across both swap=False and swap=True conditions.
    This mitigates prefix bias by averaging results from both label orderings.
    """
    filename_path = Path(filename)
    parent_dir = filename_path.parent
    file_stem = filename_path.stem  # e.g., "data0"

    # Process both swap conditions
    results_list = []
    for swap in [False, True]:
        file_suffix = "_swap_judgement" if swap else "_judgement"

        # Find the judgement file
        judge_file = None
        if judge_name:
            potential_dir = parent_dir / judge_name
            potential_file = potential_dir / f"{file_stem}{file_suffix}.csv"
            if potential_file.exists():
                judge_file = potential_file

        if judge_file is None:
            judge_dirs = [d for d in parent_dir.iterdir() if d.is_dir()]
            for d in judge_dirs:
                potential_file = d / f"{file_stem}{file_suffix}.csv"
                if potential_file.exists():
                    judge_file = potential_file
                    break

        if judge_file and judge_file.exists():
            if verbose:
                LOGGER.info(f"Processing {judge_file} (swap={swap})")
            result = score_file(
                str(judge_file),
                swap=swap,
                method=method,
                model=model,
                dataset=dataset,
                results_file=None,  # Don't save individual results
                verbose=verbose,
                debug=debug,
                n_votes=n_votes,
                judge_name=judge_name,
            )
            results_list.append(result)
        else:
            if verbose:
                LOGGER.warning(
                    f"Could not find {file_stem}{file_suffix}.csv, skipping swap={swap}"
                )

    if not results_list:
        raise ValueError(
            f"Could not find any judgement files for {file_stem} in subdirectories of {parent_dir}"
        )

    # Average the accuracies
    combined_results = pd.concat(results_list, ignore_index=True)
    avg_accuracy = combined_results["accuracy"].mean()
    total_matches = combined_results["num_matches"].sum()
    avg_unknown_proportion = combined_results["unknown_proportion"].mean()

    # Create averaged result
    averaged_result = pd.DataFrame(
        {
            "method": [method],
            "accuracy": [avg_accuracy],
            "dataset": [dataset],
            "model": [model],
            "swap": ["averaged"],
            "n_votes": [n_votes],
            "unknown_proportion": [avg_unknown_proportion],
            "num_matches": [total_matches],
        }
    )

    if verbose:
        print("\nIndividual swap condition results:")
        print(combined_results.round(3).to_markdown(index=False))
        print("\nAveraged result:")
        print(averaged_result.round(3).to_markdown(index=False))

    if results_file is not None:
        if results_file.exists():
            print(f"Appending to {results_file}")
            averaged_result.to_csv(results_file, mode="a", header=False, index=False)
        else:
            averaged_result.to_csv(results_file, mode="w", header=True, index=False)

    return averaged_result


if __name__ == "__main__":
    # Allow calling either score_file directly or score_both_swap_conditions
    # Usage:
    #   python -m core.scoring.gpqa_accuracy score_both --filename=... --judge_name=...
    #   python -m core.scoring.gpqa_accuracy --filename=... --swap=False ...
    import sys

    # Check if first argument is "score_both"
    if len(sys.argv) > 1 and sys.argv[1] == "score_both":
        # Remove "score_both" from argv so Fire can parse the rest
        # Also convert flag=value to --flag=value for Fire compatibility
        new_argv = [sys.argv[0]]
        for arg in sys.argv[2:]:
            if "=" in arg and not arg.startswith("--"):
                # Convert filename=value to --filename=value
                new_argv.append("--" + arg)
            else:
                new_argv.append(arg)
        sys.argv = new_argv
        fire.Fire(score_both_swap_conditions)
    else:
        fire.Fire(score_file)
