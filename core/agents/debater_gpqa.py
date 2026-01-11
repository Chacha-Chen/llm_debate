from typing import Optional

from core.agents.debater_quality import DebaterQuality
from core.file_handler import Method
from core.rollouts.utils import TranscriptConfig
from core.utils import SEPARATOR


class DebaterGPQA(DebaterQuality):
    """
    GPQA debater: reasoning-focused, no quotes/story context.
    """

    MODEL_NAME_MAP = {
        # normalize to lowercase/no-spaces before lookup
        "gpt4o": "openai/gpt-4o",
        "gpt-4o": "openai/gpt-4o",
        "gpt-4-1106-preview": "openai/gpt-4o",
        "claudesonnet4": "anthropic/claude-sonnet-4",
        "claude-sonnet-4": "anthropic/claude-sonnet-4",
    }

    def _internal_model_key(self, transcript: TranscriptConfig) -> str:
        """Return 'A' or 'B' to identify which of the two models/debaters this is.

        This is NOT about multiple choice answer letters (A/B/C/D). This is an internal
        identifier to access the correct data source:
        - Returns "A" → access model_a_* data (model_a_reasoning, model_a_answer_letter, etc.)
        - Returns "B" → access model_b_* data (model_b_reasoning, model_b_answer_letter, etc.)

        Example: If model_a predicted answer "C" and model_b predicted answer "B":
        - Debater representing model_a gets key "A" → accesses model_a_answer_letter="C"
        - Debater representing model_b gets key "B" → accesses model_b_answer_letter="B"

        Display names shown to debaters and judge are "Debater 1" and "Debater 2" to avoid
        confusion with multiple choice options (A, B, C, D).
        """
        if self.correct:
            return "A" if transcript.names.correct == "Debater 1" else "B"
        else:
            return "A" if transcript.names.incorrect == "Debater 1" else "B"

    def _model_id_for_label(self, transcript: TranscriptConfig) -> str:
        """Map model_a/model_b name to provider ID; fall back to config default."""
        model_key = self._internal_model_key(transcript)
        extra = transcript.extra or {}
        name_key = "model_a_name" if model_key == "A" else "model_b_name"
        raw_name = extra.get(name_key, "") or ""
        normalized = raw_name.lower().replace(" ", "").replace("_", "")
        return self.MODEL_NAME_MAP.get(normalized, self.config.language_model.model)

    def _reasoning_text_for_label(self, transcript: TranscriptConfig) -> str:
        """Pick the matching reasoning trace based on internal model key, not correctness.

        Uses internal model key (A/B) to access model_a_reasoning or model_b_reasoning.
        Display names are "Debater 1" and "Debater 2", but internally we use A/B to map to model_a/model_b.
        """
        model_key = self._internal_model_key(transcript)
        extra = transcript.extra or {}
        reasoning_key = "model_a_reasoning" if model_key == "A" else "model_b_reasoning"
        reasoning = extra.get(reasoning_key, [])
        return "\n".join(reasoning) if reasoning else ""

    def _choices_text(self, transcript: TranscriptConfig) -> str:
        """Render full choice list for display to debaters/judge."""
        extra = transcript.extra or {}
        choices_map = extra.get("choices_map") or {}
        if not isinstance(choices_map, dict) or not choices_map:
            return ""
        # Stable order by letter
        parts = []
        for letter in sorted(choices_map.keys()):
            parts.append(f"{letter}. {choices_map[letter]}")
        return "\n".join(parts)

    def _model_answer_letter_for_label(self, transcript: TranscriptConfig) -> str:
        """Return the source model's answer letter (A-D) for this debater's side.

        This returns the multiple choice answer letter (A, B, C, or D) that the original
        model predicted, NOT the internal model key (A/B).
        """
        model_key = self._internal_model_key(transcript)
        extra = transcript.extra or {}
        key = "model_a_answer_letter" if model_key == "A" else "model_b_answer_letter"
        letter = extra.get(key, "")
        return str(letter).strip().upper()

    def get_transcript_string(self, transcript: TranscriptConfig):
        """Override to skip story verification for GPQA (no story field)."""
        # Skip TranscriptParser.verify_strict() since GPQA has no story/quotes to verify
        our_name, opponent_name = self.names_from_transcript(transcript)
        transcript_string = ""
        for i, round in enumerate(transcript.rounds):
            our_arg, opponent_arg = self.args_from_round(round)
            transcript_string += f"Round {i + 1}:\n\n"
            if round.judge is not None and len(round.judge) > 0:
                judge_name = "Judge"
                transcript_string += f'{judge_name}: """{round.judge}"""\n\n'
            if round.cross_examiner is not None and len(round.cross_examiner) > 0:
                cross_examiner_name = transcript.names.cross_examiner or "Judge"
                transcript_string += (
                    f'{cross_examiner_name}: """{round.cross_examiner}"""\n\n'
                )
            transcript_string += (
                f'{our_name}: """{our_arg}"""\n\n' if our_arg is not None else ""
            )
            if self.method == Method.debate:
                transcript_string += (
                    f'{opponent_name}: """{opponent_arg}"""\n\n'
                    if opponent_arg is not None
                    else ""
                )
            if i + 1 < len(transcript.rounds):
                transcript_string += f"{SEPARATOR}\n\n"
        return transcript_string.strip()

    def fill_in_content(self, content: str, transcript: TranscriptConfig):
        reasoning_text = self._reasoning_text_for_label(transcript)
        answer_defending, answer_opposing = self.answers_from_transcript(transcript)
        letter_defending, letter_opposing = self.answer_letters_from_transcript(
            transcript
        )
        name, opponent_name = self.names_from_transcript(transcript)
        choices_text = self._choices_text(transcript)
        model_answer_letter = self._model_answer_letter_for_label(transcript)

        placeholders = {
            "QUESTION": lambda: transcript.question,
            "NAME": lambda: name,
            "OPPONENT_NAME": lambda: opponent_name,
            "ANSWER_DEFENDING": lambda: answer_defending,
            "ANSWER_DEFENDING_LETTER": lambda: letter_defending,
            "ANSWER_OPPOSING_LETTER": lambda: letter_opposing,
            "ANSWER_OPPOSING": lambda: answer_opposing,
            "CHOICES": lambda: choices_text,
            "MODEL_ANSWER_LETTER": lambda: model_answer_letter,
            "TRANSCRIPT_MESSAGE": lambda: self.create_transcript_message(transcript),
            "TRANSCRIPT": lambda: self.get_transcript_string(transcript),
            "NEW_ARGUMENT_REQUEST": lambda: self.get_new_argument_request(transcript),
            "THINKING_ADVICE": lambda: self.get_thinking_advice(transcript),
            "WORD_LIMIT": lambda: str(self.config.prompts.word_limit),
            "MODEL_REASONING": lambda: reasoning_text,
        }

        for placeholder, placeholder_filler in placeholders.items():
            if f"<{placeholder}>" in content:
                content = content.replace(f"<{placeholder}>", placeholder_filler())

        return content

    async def get_completion(self, transcript: TranscriptConfig):
        """Override to select model by debater label (internal A/B mapping) instead of correctness.

        Note: Display names are "Debater 1" and "Debater 2" to avoid confusion with multiple choice options.
        """
        prompt = self.construct_messages(transcript)
        if (
            "ft:" in self.config.language_model.model
            or "gpt-3.5-turbo-16k" == self.config.language_model.model
        ):
            content = "\n".join([x["content"] for x in prompt])
            num_tokens = len(self.tokenizer.encode(content))
            if num_tokens > 3750:
                arguments = [
                    f"<argument>{TOKEN_LIMIT_ARGUMENT}</argument>"
                ] * self.config.BoN
                LOGGER.warning(
                    f"Prompt doesn't fit in fine-tuned model with {num_tokens} tokens. Using round end argument {arguments[0]}"
                )
                return arguments

        model_id = self._model_id_for_label(transcript)
        responses = await self.api_handler(
            model_ids=model_id,
            prompt=prompt,
            temperature=self.config.language_model.temperature,
            top_p=self.config.language_model.top_p,
            max_tokens=self.config.language_model.max_tokens,
            n=self.config.BoN,
            num_candidates_per_completion=self.config.language_model.num_candidates_per_completion,
            is_valid=self.is_valid,
            insufficient_valids_behaviour="pad_invalids",
        )
        responses = [x.completion.strip() for x in responses]
        assert len(responses) == self.config.BoN
        return responses

    def is_valid(self, completion: str):
        """GPQA: only require argument tags and word limits; no quote requirements."""
        if "<argument>" not in completion:
            return False
        try:
            argument = self.extract_argument(completion)
        except ValueError:
            return False
        word_count = len(argument.split(" "))
        return (
            word_count >= self.config.language_model.min_words
            and word_count <= self.config.language_model.max_words
        )
