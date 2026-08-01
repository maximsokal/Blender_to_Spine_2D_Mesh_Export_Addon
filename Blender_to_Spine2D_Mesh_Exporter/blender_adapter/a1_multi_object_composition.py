"""Compose finalized prepared A1 objects into one typed Spine document.

This module owns document composition only. It performs no Blender reads, rendering,
serialization, or file writes. Callers must finalize every render-derived attachment before
calling :func:`compose_a1_multi_object_document`.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Tuple

from ..application import (
    A1MultiObjectExportSettings,
    A1MultiObjectMode,
    A1SingleObjectExportSettings,
    resolve_a1_multi_object_preparation_settings,
)
from ..domain.baking import CameraProjectionPlan
from ..domain.geometry import calculate_a1_projected_snapshot_depth_range
from ..domain.projection import A1ProjectionDirection
from ..domain.spine import (
    ConnectedPlacementSpace,
    ConstraintOrderPolicy,
    SpineCompositionSettings,
    SpineDocumentComponent,
    SpineDocumentCompositionResult,
    compose_spine_documents,
    resolve_a1_rig_profile,
)
from ..domain.spine.connected_group_assembly import build_connected_group_document
from ..domain.spine.connected_group_contracts import (
    ConnectedGroupBuildResult,
    ConnectedGroupSettings,
    ConnectedObjectDocument,
)
from ..domain.spine.object_block_draw_order import (
    SpineObjectBlockDepth,
    apply_object_block_setup_draw_order,
)
from .a1_multi_object_contracts import A1MultiObjectSource
from .a1_object_preparation import PreparedA1Object


def _expected_prepared_settings(
    source: A1MultiObjectSource,
    mode: A1MultiObjectMode,
) -> A1SingleObjectExportSettings:
    """Resolve expected settings through the shared preparation policy owner."""

    if not isinstance(source, A1MultiObjectSource):
        raise TypeError("source must be A1MultiObjectSource")
    return resolve_a1_multi_object_preparation_settings(source.settings, mode)


def _validate_source_prepared_pair(
    source: A1MultiObjectSource,
    prepared: PreparedA1Object,
    mode: A1MultiObjectMode,
    *,
    pair_index: int,
) -> None:
    """Reject tuple reordering and settings drift before component IDs are assigned."""

    if not isinstance(pair_index, int) or isinstance(pair_index, bool) or pair_index < 0:
        raise ValueError("pair_index must be a non-negative integer")
    if prepared.source_object is not source.source_object:
        raise ValueError(
            f"sources[{pair_index}] component '{source.component_id}' does not match "
            "the prepared object's live source_object"
        )
    expected_settings = _expected_prepared_settings(source, mode)
    if prepared.settings != expected_settings:
        raise ValueError(
            f"sources[{pair_index}] component '{source.component_id}' settings do not "
            f"match the prepared {mode.value} object settings"
        )


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

    for pair_index, (source, item) in enumerate(
        zip(sources, prepared, strict=True)
    ):
        _validate_source_prepared_pair(
            source,
            item,
            settings.mode,
            pair_index=pair_index,
        )

    profiles = tuple(
        resolve_a1_rig_profile(item.rig.profile.profile_id) for item in prepared
    )
    if len(set(profiles)) != 1:
        raise ValueError(
            "All objects in one multi-object document must use the same rig profile"
        )

    targets = tuple(item.settings.export.spine_target for item in prepared)
    if len(set(targets)) != 1:
        raise ValueError(
            "All prepared objects in one multi-object document must use the same "
            "Spine JSON target"
        )


def _connected_placement_space(
    prepared: PreparedA1Object,
) -> ConnectedPlacementSpace:
    """Resolve where one finalized attachment already stores its visible XY position."""

    if not isinstance(prepared, PreparedA1Object):
        raise TypeError("prepared must be PreparedA1Object")
    if isinstance(prepared.bake_plan, CameraProjectionPlan):
        return ConnectedPlacementSpace.PRESERVE_DOCUMENT
    return ConnectedPlacementSpace.ANCHOR_RELATIVE_WORLD


def _standalone_axis_projection_direction(
    prepared: Tuple[PreparedA1Object, ...],
) -> A1ProjectionDirection | None:
    """Resolve one shared object-bake projection or preserve Camera Projection order."""

    if not isinstance(prepared, tuple) or not prepared:
        raise ValueError("prepared must be a non-empty tuple")
    camera_flags = tuple(
        isinstance(item.bake_plan, CameraProjectionPlan) for item in prepared
    )
    if any(camera_flags):
        # Existing rendered Camera Projection setup order remains independent from the
        # Normal / UV Segments Active Camera layout introduced by Slice 4.
        return None

    directions = tuple(item.settings.projection_direction for item in prepared)
    if not all(isinstance(item, A1ProjectionDirection) for item in directions):
        raise TypeError("prepared projection directions must be A1ProjectionDirection")
    unique_directions = set(directions)
    if len(unique_directions) != 1:
        raise ValueError(
            "Standalone Normal / UV Segments objects must use one shared "
            "projection_direction"
        )
    return directions[0]


def _standalone_object_block_depths(
    sources: Tuple[A1MultiObjectSource, ...],
    prepared: Tuple[PreparedA1Object, ...],
) -> Tuple[SpineObjectBlockDepth, ...]:
    """Build world-depth records from already projected immutable snapshots."""

    if len(sources) != len(prepared):
        raise ValueError("sources and prepared must correspond one-to-one")

    result: list[SpineObjectBlockDepth] = []
    for source_input_index, (source, item) in enumerate(
        zip(sources, prepared, strict=True)
    ):
        depth_range = calculate_a1_projected_snapshot_depth_range(
            item.source_snapshot
        )
        result.append(
            SpineObjectBlockDepth(
                component_id=source.component_id,
                source_input_index=source_input_index,
                nearest_vertex_index=depth_range.nearest_vertex_id.index,
                nearest_vertex_depth=depth_range.nearest_vertex_depth,
                farthest_vertex_index=depth_range.farthest_vertex_id.index,
                farthest_vertex_depth=depth_range.farthest_vertex_depth,
            )
        )
    return tuple(result)


def _compose_standalone_document(
    sources: Tuple[A1MultiObjectSource, ...],
    prepared: Tuple[PreparedA1Object, ...],
    settings: A1MultiObjectExportSettings,
) -> SpineDocumentCompositionResult:
    """Compose source-order internals, then move complete setup slot blocks only."""

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
    composition = compose_spine_documents(
        components,
        SpineCompositionSettings(
            shared_bone_names=("root",),
            constraint_order_policy=ConstraintOrderPolicy.REBASE_CONTIGUOUS,
            namespace_animations=settings.namespace_animations,
            animation_separator=settings.animation_separator,
        ),
    )

    projection_direction = _standalone_axis_projection_direction(prepared)
    if projection_direction is None:
        return composition

    depth_entries = _standalone_object_block_depths(sources, prepared)
    reordered_document = apply_object_block_setup_draw_order(
        composition.document,
        components,
        depth_entries,
        depth_tolerance=settings.z_tolerance,
    )
    if reordered_document is composition.document:
        return composition
    return replace(composition, document=reordered_document)


def compose_a1_multi_object_document(
    sources: Tuple[A1MultiObjectSource, ...],
    prepared: Tuple[PreparedA1Object, ...],
    settings: A1MultiObjectExportSettings,
) -> SpineDocumentCompositionResult | ConnectedGroupBuildResult:
    """Compose already-finalized object documents exactly once."""

    _validate_composition_inputs(sources, prepared, settings)

    if settings.mode is A1MultiObjectMode.STANDALONE:
        return _compose_standalone_document(sources, prepared, settings)

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
            placement_space=_connected_placement_space(item),
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
            namespace_animations=settings.namespace_animations,
            animation_separator=settings.animation_separator,
        ),
        profile=prepared[0].rig.profile,
        spine_target=prepared[0].settings.export.spine_target,
    )


__all__ = ["compose_a1_multi_object_document"]
