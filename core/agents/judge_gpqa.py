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
        """Override to skip story/quote verification for GPQA (no story field).

        MODEL_A is always "Debater 1", MODEL_B is always "Debater 2" in the transcript.
        """
        extra = transcript.extra or {}
        correct_model = str(extra.get("correct_model", "A")).upper()

        # MODEL_A is always "Debater 1", MODEL_B is always "Debater 2"
        model_a_name = "Debater 1"
        model_b_name = "Debater 2"

        # Map correct/incorrect positions to MODEL_A/B names
        if correct_model == "A":
            # MODEL_A is correct, MODEL_B is incorrect
            correct_debater_name = model_a_name
            incorrect_debater_name = model_b_name
        else:
            # MODEL_B is correct, MODEL_A is incorrect
            correct_debater_name = model_b_name
            incorrect_debater_name = model_a_name

        # Determine order based on swap
        if not transcript.swap:
            positions = ["judge", "cross_examiner", "correct", "incorrect"]
        else:
            positions = ["judge", "cross_examiner", "incorrect", "correct"]

        transcript_string = ""
        for i, round in enumerate(transcript.rounds):
            transcript_string += f"Round {i + 1}:\n\n"
            for position in positions:
                message = getattr(round, position, None)
                if message:
                    if position == "judge":
                        name = "Judge"
                    elif position == "cross_examiner":
                        name = transcript.names.cross_examiner or "Judge"
                    elif position == "correct":
                        name = correct_debater_name
                    elif position == "incorrect":
                        name = incorrect_debater_name
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
        choices_map = extra.get("choices_map", {}) if extra else {}

        # Get model answer letters
        model_a_letter = str(extra.get("model_a_answer_letter", "")).strip().upper()
        model_b_letter = str(extra.get("model_b_answer_letter", "")).strip().upper()

        # Bind ANSWER_A to MODEL_A's prediction and ANSWER_B to MODEL_B's prediction
        answer_a_text = choices_map.get(model_a_letter, "")
        answer_b_text = choices_map.get(model_b_letter, "")

        # MODEL_A is always "Debater 1", MODEL_B is always "Debater 2"
        placeholders = {
            "QUESTION": lambda: transcript.question,
            "ANSWER_A": lambda: answer_a_text,
            "ANSWER_B": lambda: answer_b_text,
            "NAME_A": lambda: "Debater 1",  # MODEL_A is always Debater 1
            "NAME_B": lambda: "Debater 2",  # MODEL_B is always Debater 2
            "TRANSCRIPT": lambda: self.get_transcript(transcript),
            # Anonymize model names to avoid bias - use debater labels for consistency
            "MODEL_A_NAME": lambda: "Debater 1",
            "MODEL_B_NAME": lambda: "Debater 2",
            "MODEL_A_ANSWER_LETTER": lambda: model_a_letter,
            "MODEL_B_ANSWER_LETTER": lambda: model_b_letter,
            "MODEL_A_REASONING": lambda: model_a_reasoning,
            "MODEL_B_REASONING": lambda: model_b_reasoning,
            "CHOICES": lambda: choices_text,
            "WORD_LIMIT": lambda: str(self.config.prompts.word_limit),
        }
        for placeholder, placeholder_filler in placeholders.items():
            if f"<{placeholder}>" in content:
                content = content.replace(f"<{placeholder}>", placeholder_filler())

        return content
