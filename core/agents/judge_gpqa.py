from core.agents.judge_quality import JudgeQuality
from core.rollouts.utils import TranscriptConfig


class JudgeGPQA(JudgeQuality):
    """
    GPQA judge: evaluates logical soundness, no quote/story context.
    """

    @staticmethod
    def _reasoning_text(extra: dict, key: str) -> str:
        """Return a newline-joined reasoning trace if present."""
        if not extra:
            return ""
        val = extra.get(key, "")
        if isinstance(val, list):
            return "\n".join(val)
        return str(val)

    @staticmethod
    def _choices_text(extra: dict) -> str:
        choices_map = extra.get("choices_map") if extra else {}
        if not isinstance(choices_map, dict) or not choices_map:
            return ""
        parts = []
        for letter in sorted(choices_map.keys()):
            parts.append(f"{letter}. {choices_map[letter]}")
        return "\n".join(parts)

    def get_transcript(self, transcript: TranscriptConfig) -> str:
        """Override to skip story/quote verification for GPQA (no story field)."""
        transcript_string = ""
        if not transcript.swap:
            positions = ["judge", "cross_examiner", "correct", "incorrect"]
        else:
            positions = ["judge", "cross_examiner", "incorrect", "correct"]

        for i, round in enumerate(transcript.rounds):
            transcript_string += f"Round {i + 1}:\n\n"
            for position in positions:
                message = getattr(round, position, None)
                if message:
                    if position == "judge":
                        name = "Judge"
                    else:
                        name = getattr(transcript.names, position, position)
                    assert name is not None
                    transcript_string += f'{name}: """{message}"""\n\n'
        return transcript_string.strip()

    def fill_in_content(self, content: str, transcript: TranscriptConfig):
        extra = transcript.extra or {}
        model_a_reasoning = self._reasoning_text(extra, "model_a_reasoning")
        model_b_reasoning = self._reasoning_text(extra, "model_b_reasoning")
        choices_text = self._choices_text(extra)
        placeholders = {
            "QUESTION": lambda: transcript.question,
            "ANSWER_A": lambda: transcript.answers.correct
            if not transcript.swap
            else transcript.answers.incorrect,
            "ANSWER_B": lambda: transcript.answers.incorrect
            if not transcript.swap
            else transcript.answers.correct,
            "NAME_A": lambda: transcript.names.correct or "Debater 1",
            "NAME_B": lambda: transcript.names.incorrect or "Debater 2",
            "TRANSCRIPT": lambda: self.get_transcript(transcript),
            # Anonymize model names to avoid bias - use debater labels for consistency
            "MODEL_A_NAME": lambda: "Debater 1",
            "MODEL_B_NAME": lambda: "Debater 2",
            "MODEL_A_ANSWER_LETTER": lambda: extra.get("model_a_answer_letter", ""),
            "MODEL_B_ANSWER_LETTER": lambda: extra.get("model_b_answer_letter", ""),
            "MODEL_A_REASONING": lambda: model_a_reasoning,
            "MODEL_B_REASONING": lambda: model_b_reasoning,
            "CHOICES": lambda: choices_text,
            "WORD_LIMIT": lambda: str(self.config.prompts.word_limit),
        }
        for placeholder, placeholder_filler in placeholders.items():
            if f"<{placeholder}>" in content:
                content = content.replace(f"<{placeholder}>", placeholder_filler())

        return content
