"""Extractor registry and construction."""

from __future__ import annotations

from leap.generation.prompt_builder import PromptBuilder

from .autoprep_extractors import CoTEnd2EnderExtractor, End2EnderExtractor, NL2CodeExtractor
from .direct_query import DirectQueryExtractor
from .nl2sql import NL2SQLExtractor

EXTRACTOR_TYPES = {
    DirectQueryExtractor.name: DirectQueryExtractor,
    NL2SQLExtractor.name: NL2SQLExtractor,
    NL2CodeExtractor.name: NL2CodeExtractor,
    End2EnderExtractor.name: End2EnderExtractor,
    CoTEnd2EnderExtractor.name: CoTEnd2EnderExtractor,
}


def build_extractors(names: tuple[str, ...], prompt_builder: PromptBuilder):
    extractors = []
    for name in names:
        extractor_type = EXTRACTOR_TYPES[name]
        extractors.append(extractor_type(prompt_builder) if name == "direct_query" else extractor_type())
    return tuple(extractors)
