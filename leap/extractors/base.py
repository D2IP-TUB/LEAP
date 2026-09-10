"""Common contracts for answer extractors."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, Protocol

from leap.core.table import Table
from leap.core.types import ExtractorResult


@dataclass(frozen=True)
class ExtractionContext:
    question: str
    table: Table
    request_id: str
    action_history: tuple[str, ...]
    worker: Any


class AnswerExtractor(Protocol):
    name: str

    async def extract(self, context: ExtractionContext) -> ExtractorResult: ...


async def run_extractors(extractors: tuple[AnswerExtractor, ...], context: ExtractionContext) -> list[ExtractorResult]:
    """Run independent extractors concurrently without allowing one failure to abort the rest."""

    outcomes = await asyncio.gather(*(extractor.extract(context) for extractor in extractors), return_exceptions=True)
    results: list[ExtractorResult] = []
    for extractor, outcome in zip(extractors, outcomes):
        if isinstance(outcome, BaseException):
            results.append(ExtractorResult(method=extractor.name, answers=[], accuracy=0.0, error=str(outcome)))
        else:
            results.append(outcome)
    return results
