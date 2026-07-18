"""Compose finalized prepared A1 objects into one typed Spine document.

This module owns document composition only. It performs no Blender reads, rendering,
serialization, or file writes. Callers must finalize every render-derived attachment before
calling :func:`compose_a1_multi_object_document`.
"""

from __future__ import annotations

from typing import Tuple

from ..application import A1MultiObjectExportSettings, A1MultiObjectMode
from ..domain.spine import (
    ConnectedGroupBuildResult,
    ConnectedGroupSettings,
    ConnectedObjectDocument,
    ConstraintOrderPolicy,
    SpineCompositionSettings,
    SpineDocumentComponent,
    SpineDocumentCompositionResult,
    build_connected_group_document,
    compose_spine_documents,
)
from .a1_multi_object_contracts import A1MultiObjectSource
from .a1_object_preparation import PreparedA1Object


def _validate_composition_inputs(
    sources: Tuple[A1MultiObjectSource, ...],
    prepared: Tuple[PreparedA1Object, ...],
    settings: A1MultiObjectExportSettings,
) -> None:
    if not isinstance(settings, A1MultiObjectExportSettings):
        raise TypeError("settings must be A1MultiObjectExportSettings")
    if settings.mode not in {
        A1MultiObjectMode.STANDALONE,
        A1MultiObjectMode.CONNECTED,
    }:
        raise ValueError(
            "multi-object composition accepts STANDALONE or CONNECTED mode only"
        )
    if not isinstance(sources, tuple) or not sources:
        raise ValueError("sources must be a non-empty tuple")
    if not all(isinstance(item, A1MultiObjectSource) for item in sources):
        raise TypeError("sources must contain A1MultiObjectSource values")
    if (
        not isinstance(prepared, tuple)
        or len(prepared) != len(sources)
        or not prepared
    ):
        raise ValueError("prepared objects must correspond one-to-one with sources")
    if not all(isinstance(item, PreparedA1Object) for item in prepared):
        raise TypeError("prepared must contain PreparedA1Object values")

    component_ids = tuple(source.component_id for source in sources)
    if len(component_ids) != len(set(component_ids)):
        raise ValueError("component_id values must be unique during composition")
    if settings.mode is A1MultiObjectMode.CONNECTED:
        if len(sources) < 2:
            raise ValueError("CONNECTED composition requires at least two objects")
        if settings.anchor_component_id is not None and (
            settings.anchor_component_id not in set(component_ids)
        ):
            raise ValueError("anchor_component_id is not present in composition sources")


def compose_a1_multi_object_document(
    sources: Tuple[A1MultiObjectSource, ...],
    prepared: Tuple[PreparedA1Object, ...],
    settings: A1MultiObjectExportSettings,
) -> SpineDocumentCompositionResult | ConnectedGroupBuildResult:
    """Compose already-finalized object documents exactly once."""

    _validate_composition_inputs(sources, prepared, settings)

    if settings.mode is A1MultiObjectMode.STANDALONE:
        components = tuple(
            SpineDocumentComponent(
                component_id=source.component_id,
                document=item.document,
                animation_namespace=(
                    source.animation_namespace or source.component_id
                ),
            )
            for source, item in zip(sources, prepared, strict=True)
        )
        return compose_spine_documents(
            components,
            SpineCompositionSettings(
                shared_bone_names=("root",),
                constraint_order_policy=ConstraintOrderPolicy.REBASE_CONTIGUOUS,
                namespace_animations=settings.namespace_animations,
                animation_separator=settings.animation_separator,
            ),
        )

    dimensions = {
        (
            item.settings.export.texture_width,
            item.settings.export.texture_height,
        )
        for item in prepared
    }
    if len(dimensions) != 1:
        raise ValueError(
            "CONNECTED mode requires identical texture dimensions for every object"
        )
    texture_width, texture_height = next(iter(dimensions))
    connected_objects = tuple(
        ConnectedObjectDocument(
            component_id=source.component_id,
            prefix=item.prefix,
            document=item.document,
            world_position=item.world_position,
            animation_namespace=(
                source.animation_namespace or source.component_id
            ),
        )
        for source, item in zip(sources, prepared, strict=True)
    )
    return build_connected_group_document(
        connected_objects,
        ConnectedGroupSettings(
            texture_width=texture_width,
            texture_height=texture_height,
            group_prefix=settings.connected_group_prefix,
            anchor_component_id=settings.anchor_component_id,
            z_tolerance=settings.z_tolerance,
            scale_mode=settings.connected_scale_mode,
            animation_separator=settings.animation_separator,
        ),
    )


__all__ = ["compose_a1_multi_object_document"]
