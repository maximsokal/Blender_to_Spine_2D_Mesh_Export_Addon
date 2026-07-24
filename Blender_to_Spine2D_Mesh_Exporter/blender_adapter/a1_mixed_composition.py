"""Compose finalized connected and standalone subgroups into one mixed document."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Tuple

from ..application import (
    A1MultiObjectExportSettings,
    A1MultiObjectMode,
    GroupedCameraOverlayResult,
)
from ..domain.spine import (
    ConnectedGroupBuildResult,
    ConstraintOrderPolicy,
    SpineCompositionSettings,
    SpineDocument,
    SpineDocumentComponent,
    SpineDocumentCompositionResult,
    compose_spine_documents,
)
from .a1_grouped_output import apply_staged_grouped_camera_overlay
from .a1_mixed_settings import (
    build_connected_subgroup_settings,
    build_standalone_subgroup_settings,
)
from .a1_multi_object_composition import compose_a1_multi_object_document
from .a1_multi_object_contracts import A1MultiObjectSource
from .a1_object_preparation import PreparedA1Object
from .grouped_camera_projection_executor import GroupedCameraProjectionStageResult
from .grouped_camera_projection_policy import GroupedCameraProjectionRequest


@dataclass(frozen=True, slots=True)
class A1MixedObjectPartition:
    connected: Tuple[PreparedA1Object, ...]
    standalone: Tuple[PreparedA1Object, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.connected, tuple) or len(self.connected) < 2:
            raise ValueError("connected must contain at least two prepared objects")
        if not isinstance(self.standalone, tuple) or not self.standalone:
            raise ValueError("standalone must contain at least one prepared object")
        if not all(
            isinstance(item, PreparedA1Object)
            for item in self.connected + self.standalone
        ):
            raise TypeError("partition values must contain PreparedA1Object instances")


@dataclass(frozen=True, slots=True)
class A1MixedCompositionResult:
    document: SpineDocument
    connected_composition: ConnectedGroupBuildResult
    standalone_composition: SpineDocumentCompositionResult
    outer_composition: SpineDocumentCompositionResult
    overlay: GroupedCameraOverlayResult | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.document, SpineDocument):
            raise TypeError("document must be SpineDocument")
        if not isinstance(self.connected_composition, ConnectedGroupBuildResult):
            raise TypeError("connected_composition must be ConnectedGroupBuildResult")
        if not isinstance(self.standalone_composition, SpineDocumentCompositionResult):
            raise TypeError(
                "standalone_composition must be SpineDocumentCompositionResult"
            )
        if not isinstance(self.outer_composition, SpineDocumentCompositionResult):
            raise TypeError("outer_composition must be SpineDocumentCompositionResult")
        if self.outer_composition.document is not self.document:
            raise ValueError("outer composition must own the returned document")
        if (
            self.connected_composition.composition.document
            is not self.connected_composition.document
        ):
            raise ValueError(
                "connected composition metadata must own the connected final document"
            )
        if self.overlay is not None and not isinstance(
            self.overlay,
            GroupedCameraOverlayResult,
        ):
            raise TypeError("overlay must be GroupedCameraOverlayResult or None")


def partition_mixed_prepared_objects(
    finalized_objects: Tuple[PreparedA1Object, ...],
    connected_sources: Tuple[A1MultiObjectSource, ...],
    standalone_sources: Tuple[A1MultiObjectSource, ...],
) -> A1MixedObjectPartition:
    """Partition finalized objects using the validated source-group cardinalities."""

    if not isinstance(finalized_objects, tuple):
        raise TypeError("finalized_objects must be tuple")
    if not isinstance(connected_sources, tuple) or len(connected_sources) < 2:
        raise ValueError("connected_sources must contain at least two objects")
    if not isinstance(standalone_sources, tuple) or not standalone_sources:
        raise ValueError("standalone_sources must contain at least one object")
    if not all(
        isinstance(item, A1MultiObjectSource)
        for item in connected_sources + standalone_sources
    ):
        raise TypeError("sources must contain A1MultiObjectSource values")
    expected_count = len(connected_sources) + len(standalone_sources)
    if len(finalized_objects) != expected_count:
        raise ValueError(
            "finalized object count does not match connected and standalone sources"
        )
    split = len(connected_sources)
    return A1MixedObjectPartition(
        connected=finalized_objects[:split],
        standalone=finalized_objects[split:],
    )


def _compose_outer_document(
    connected_document: SpineDocument,
    standalone_document: SpineDocument,
    settings: A1MultiObjectExportSettings,
) -> SpineDocumentCompositionResult:
    """Compose connected first and standalone second as two explicit visual blocks.

    Spine draws later slots on top. Mixed mode currently has no cross-group Z contract,
    so the standalone subgroup intentionally remains above the connected subgroup while
    each subgroup preserves its own deterministic internal ordering.
    """

    return compose_spine_documents(
        (
            SpineDocumentComponent(
                component_id="connected_group",
                document=connected_document,
            ),
            SpineDocumentComponent(
                component_id="standalone_group",
                document=standalone_document,
            ),
        ),
        SpineCompositionSettings(
            shared_bone_names=("root",),
            constraint_order_policy=ConstraintOrderPolicy.REBASE_CONTIGUOUS,
            namespace_animations=False,
            animation_separator=settings.animation_separator,
        ),
    )


def compose_a1_mixed_document(
    connected_sources: Tuple[A1MultiObjectSource, ...],
    standalone_sources: Tuple[A1MultiObjectSource, ...],
    partition: A1MixedObjectPartition,
    settings: A1MultiObjectExportSettings,
    *,
    grouped_request: GroupedCameraProjectionRequest | None = None,
    grouped_stage: GroupedCameraProjectionStageResult | None = None,
) -> A1MixedCompositionResult:
    """Compose both subgroups exactly once and optionally apply static flattening."""

    if not isinstance(settings, A1MultiObjectExportSettings):
        raise TypeError("settings must be A1MultiObjectExportSettings")
    if settings.mode is not A1MultiObjectMode.MIXED:
        raise ValueError("mixed composition requires A1MultiObjectMode.MIXED")
    if not isinstance(partition, A1MixedObjectPartition):
        raise TypeError("partition must be A1MixedObjectPartition")
    if len(connected_sources) != len(partition.connected):
        raise ValueError("connected sources do not match connected prepared objects")
    if len(standalone_sources) != len(partition.standalone):
        raise ValueError("standalone sources do not match standalone prepared objects")
    if (grouped_request is None) != (grouped_stage is None):
        raise ValueError("grouped request and stage must either both be present or both absent")

    anchor = settings.anchor_component_id or connected_sources[0].component_id
    connected_settings = build_connected_subgroup_settings(settings, anchor)
    connected = compose_a1_multi_object_document(
        connected_sources,
        partition.connected,
        connected_settings,
    )
    if not isinstance(connected, ConnectedGroupBuildResult):
        raise TypeError("connected subgroup composition returned an unexpected result type")

    overlay = None
    if grouped_request is not None and grouped_stage is not None:
        overlay = apply_staged_grouped_camera_overlay(
            connected.document,
            grouped_request,
            grouped_stage,
        )
        updated_connected_composition = replace(
            connected.composition,
            document=overlay.document,
        )
        connected = replace(
            connected,
            document=overlay.document,
            composition=updated_connected_composition,
        )

    standalone = compose_a1_multi_object_document(
        standalone_sources,
        partition.standalone,
        build_standalone_subgroup_settings(settings),
    )
    if not isinstance(standalone, SpineDocumentCompositionResult):
        raise TypeError("standalone subgroup composition returned an unexpected result type")

    outer = _compose_outer_document(
        connected.document,
        standalone.document,
        settings,
    )
    return A1MixedCompositionResult(
        document=outer.document,
        connected_composition=connected,
        standalone_composition=standalone,
        outer_composition=outer,
        overlay=overlay,
    )


__all__ = [
    "A1MixedCompositionResult",
    "A1MixedObjectPartition",
    "compose_a1_mixed_document",
    "partition_mixed_prepared_objects",
]
