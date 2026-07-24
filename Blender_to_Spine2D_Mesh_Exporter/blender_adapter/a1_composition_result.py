"""Immutable document replacement for A1 composition result variants."""

from __future__ import annotations

from dataclasses import replace

from ..domain.spine import (
    ConnectedGroupBuildResult,
    SpineDocument,
    SpineDocumentCompositionResult,
)


A1CompositionResult = SpineDocumentCompositionResult | ConnectedGroupBuildResult


def replace_a1_composition_document(
    result: A1CompositionResult,
    document: SpineDocument,
) -> A1CompositionResult:
    """Return one result whose every document owner references ``document``."""

    if not isinstance(document, SpineDocument):
        raise TypeError("document must be SpineDocument")
    if isinstance(result, ConnectedGroupBuildResult):
        nested = replace(result.composition, document=document)
        return replace(
            result,
            document=document,
            composition=nested,
        )
    if isinstance(result, SpineDocumentCompositionResult):
        return replace(result, document=document)
    raise TypeError(
        "result must be SpineDocumentCompositionResult or ConnectedGroupBuildResult"
    )


__all__ = [
    "A1CompositionResult",
    "replace_a1_composition_document",
]
