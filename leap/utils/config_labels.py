"""Utilities for concise experiment/configuration display labels.

Labels include only dimensions that are meaningful for the displayed run/configuration:
backend and global constraints are shown only for constrained action-generation runs,
and direct-query runs omit action-generation details entirely.
"""

from __future__ import annotations

from typing import Any


def _get(source: Any, key: str, default: Any = None) -> Any:
    if isinstance(source, dict):
        return source.get(key, default)
    return getattr(source, key, default)


def build_generation_config_label(
    *,
    model_id: str,
    generation: Any,
    strategy: str | None = None,
    use_constraints: bool | None = None,
    use_global_constraints: bool | None = None,
    constraint_backend: str | None = None,
    output_format: str | None = None,
    force_zero_temperature: bool | None = None,
) -> str:
    """Build a compact label containing only meaningful generation dimensions.

    Separators are inserted only between fields that are actually present in the
    final label, so impossible combinations such as ``unconstrained | xgrammar``
    are not displayed.
    """
    strategy = strategy or _get(generation, "strategy", None)
    use_constraints = bool(_get(generation, "use_constraints", use_constraints))
    use_global_constraints = bool(_get(generation, "use_global_constraints", use_global_constraints))
    constraint_backend = _get(generation, "constraint_backend", constraint_backend)
    output_format = _get(generation, "output_format", output_format)
    force_zero_temperature = bool(_get(generation, "force_zero_temperature", force_zero_temperature))

    parts = [model_id]
    if strategy:
        parts.append(str(strategy))

    # direct_query does not generate actions, so action constraints/backend/format are not meaningful.
    if strategy != "direct_query":
        parts.append("constrained" if use_constraints else "unconstrained")
        if use_constraints and constraint_backend:
            parts.append(str(constraint_backend))
        if use_constraints and use_global_constraints:
            parts.append("global")
        if output_format:
            parts.append(str(output_format))

    if force_zero_temperature:
        parts.append("zero_temp")

    return " | ".join(part for part in parts if part)
