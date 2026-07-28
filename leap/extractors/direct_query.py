"""LEAP's existing Query(T, Q) answer generation as an extractor."""

from __future__ import annotations

import ast

from vllm import SamplingParams

from leap.core.types import ExtractorResult
from leap.generation.prompt_builder import PromptBuilder

from .base import ExtractionContext


class DirectQueryExtractor:
    name = "direct_query"

    def __init__(self, prompt_builder: PromptBuilder) -> None:
        self.prompt_builder = prompt_builder

    async def extract(self, context: ExtractionContext) -> ExtractorResult:
        prompt = self.prompt_builder.build_query_prompt(
            question=context.question,
            table=context.table,
            action_history=list(context.action_history),
            worker=context.worker,
        )
        params = SamplingParams(
            temperature=0.0,
            max_tokens=200,
            stop_token_ids=[context.worker.tokenizer.eos_token_id],
            stop=["\n", "\n\n"],
        )
        response = await context.worker.generate_text(prompt, f"{context.request_id}_extract_direct_query", params)
        if not response or not response.strip():
            return ExtractorResult(method=self.name, answers=[], accuracy=0.0, error="Empty model response")
        answers = parse_direct_answers(response)
        if not answers:
            return ExtractorResult(
                method=self.name,
                answers=[],
                accuracy=0.0,
                error="Could not parse a non-empty answer",
                metadata={"response": response},
            )
        return ExtractorResult(method=self.name, answers=answers, accuracy=0.0, metadata={"response": response})


def parse_direct_answers(response: str) -> list[str]:
    response = response.strip()
    try:
        parsed = ast.literal_eval(response)
        if isinstance(parsed, list):
            return [str(item).strip() for item in parsed if str(item).strip()]
    except (ValueError, SyntaxError):
        pass
    return [response] if response else []
