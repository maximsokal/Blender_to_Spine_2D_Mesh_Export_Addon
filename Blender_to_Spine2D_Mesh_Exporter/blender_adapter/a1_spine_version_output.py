"""Resolve one immutable Spine JSON target across A1 multi-object sources."""

from __future__ import annotations

from typing import Tuple

from ..domain.spine.version_target import SpineJsonTarget
from .a1_multi_object_contracts import A1MultiObjectSource


def resolve_a1_sources_spine_target(
    *source_groups: Tuple[A1MultiObjectSource, ...],
) -> SpineJsonTarget:
    """Return the common target used by every source in one output transaction.

    Multi and mixed output must never infer a target from a composed document or accept
    different per-object targets. The immutable source settings are the authoritative
    request boundary, so mismatches are rejected before texture staging and composition.
    """

    if not source_groups:
        raise ValueError("at least one source group is required")

    sources: list[A1MultiObjectSource] = []
    for group_index, group in enumerate(source_groups):
        if not isinstance(group, tuple):
            raise TypeError(f"source_groups[{group_index}] must be tuple")
        for source_index, source in enumerate(group):
            if not isinstance(source, A1MultiObjectSource):
                raise TypeError(
                    f"source_groups[{group_index}][{source_index}] must be "
                    "A1MultiObjectSource"
                )
            sources.append(source)

    if not sources:
        raise ValueError("source groups cannot all be empty")

    expected = sources[0].settings.export.spine_target
    mismatches = tuple(
        source
        for source in sources[1:]
        if source.settings.export.spine_target is not expected
    )
    if mismatches:
        assignments = ", ".join(
            f"{source.component_id}={source.settings.export.spine_target.exact_version}"
            for source in sources
        )
        raise ValueError(
            "All A1 sources in one output transaction must use the same Spine JSON "
            f"target; assignments: {assignments}"
        )
    return expected


__all__ = ["resolve_a1_sources_spine_target"]
