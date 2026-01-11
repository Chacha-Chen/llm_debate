import asyncio
import json
import logging
import time
import traceback
from typing import Optional

import pandas as pd

from core.file_handler import Method
from core.load.gpqa import parse_choices
from core.rollouts.quality_sim import QualitySimRollout
from core.rollouts.utils import (
    Answers,
    CacheManager,
    DebaterNames,
    Round,
    StubCacheManager,
    TranscriptConfig,
)

LOGGER = logging.getLogger(__name__)


class GPQASimRollout(QualitySimRollout):
    def _names_for_row(self, row: pd.Series, swap: bool) -> DebaterNames:
        # Anonymize debater display names to avoid model-identification bias.
        # Use "Debater 1" and "Debater 2" instead of "A" and "B" to avoid confusion with multiple choice options.
        correct_name = "Debater 1"
        incorrect_name = "Debater 2"
        if swap:
            correct_name, incorrect_name = incorrect_name, correct_name
        return DebaterNames(correct=correct_name, incorrect=incorrect_name, judge=None)

    def _extra_for_row(self, row: pd.Series) -> dict:
        def _maybe_load(value):
            if isinstance(value, str):
                try:
                    return json.loads(value)
                except json.JSONDecodeError:
                    return value
            return value

        return {
            "model_a_reasoning": _maybe_load(row["model_a_reasoning"]),
            "model_b_reasoning": _maybe_load(row["model_b_reasoning"]),
            # Keep real model names in extra for logging/analysis, but prompts use anonymized names.
            "model_a_name": row["model_a_name"],
            "model_b_name": row["model_b_name"],
            "correct_model": row["correct_model"],
            "choices_map": parse_choices(row["choices"]),
            "model_a_answer_letter": row.get("model_a_answer"),
            "model_b_answer_letter": row.get("model_b_answer"),
        }

    async def run(self, index: int, row: pd.Series, swap: bool = False):
        names = self._names_for_row(row, swap)

        transcript = TranscriptConfig(
            index=index,
            story=None,
            story_title=None,
            question=row["question"],
            question_set_id=None,
            answers=Answers(
                correct=row["correct answer"],
                incorrect=row["negative answer"],
            ),
            names=names,
            swap=swap,
            rollout_type=self.config.rollout_type,
            extra=self._extra_for_row(row),
        )

        cache_manager = CacheManager(self.cache_dir, transcript.index)
        current_step, transcript_cache, _ = cache_manager.unpack_results()
        if transcript_cache is not None:
            transcript = TranscriptConfig(**json.loads(transcript_cache))
        transcript_string = transcript.json()

        duration = 0
        while current_step < int(self.config.num_steps):
            if self.correct_debater or self.incorrect_debater or self.cross_examiner:
                try:
                    start_time = time.time()
                    transcript = await self.debate_turn(
                        transcript, current_step, cache_manager
                    )
                    duration = time.time() - start_time
                    LOGGER.info(
                        f"Step {current_step} completed in {duration:.3f} (index {index})"
                    )
                    current_step += 1
                    transcript_string = transcript.json()
                except (RuntimeError, IndexError, ValueError) as e:
                    transcript_string = (
                        f"Error occurred on debate {transcript.index}, step {current_step}. "
                        f"Error message: {e}."
                    )
                    current_step = -1
                    LOGGER.info(transcript_string)
                    LOGGER.info(traceback.format_exc())
                    break
            else:
                current_step += 1
        complete = current_step >= int(self.config.num_steps)
        if complete:
            if self.correct_debater or self.incorrect_debater:
                debater = (
                    self.correct_debater
                    if self.correct_debater
                    else self.incorrect_debater
                )
                LOGGER.info(f"Total cost: {debater.api_handler.running_cost:.3f}")
            LOGGER.info(f"Completed: {transcript.index}")

        return {
            "transcript": transcript_string,
            "complete": complete,
        }
