"""Compose finalized connected and standalone subgroups into one mixed document."""

from __future__ import annotations

from dataclasses import dataclass, replace
from types import MappingProxyType
from typing import Mapping, Tuple

from ..application import (
    A1MultiObjectExportSettings,
    A1MultiObjectMode,
    GroupedCameraOverlayResult,
)
from ..domain.spine import (
    ConnectedGroupBuildResult,
    ConstraintOrderPolicy,
    SpineCompositionError,
    SpineCompositionSettings,
    SpineDocument,
    SpineDocumentComponent,
    SpineDocumentCompositionResult,
    compose_spine_documents,
)
from ..domain.spine.connected_group_serialization_validator import (
    ConnectedGroupSerializationValidator,
)
from ..domain.spine.validator import SpineValidator
from .a1_composition_result import replace_a1_composition_document
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


_CONNECTED_COMPONENT_ID = "connected_group"
_STANDALONE_COMPONENT_ID = "standalone_group"


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


@dataclass(frozen=True, slots=True)
class _ConnectedOuterCompositionInput:
    """Strict outer-composition view of one validated connected subgroup.

    Spine 4.2 connected parity intentionally permits order ties for independent
    same-layer constraints. The generic composition service validates strict component
    documents before it rebases their orders, so MIXED composition needs a temporary
    unique-order view. ``original_order_by_name`` keeps the connected schedule provenance
    intact in the returned outer composition metadata.
    """

    document: SpineDocument
    original_order_by_name: Mapping[str, int]

    def __post_init__(self) -> None:
        if not isinstance(self.document, SpineDocument):
            raise TypeError("document must be SpineDocument")
        if not isinstance(self.original_order_by_name, Mapping):
            raise TypeError("original_order_by_name must be a mapping")
        constraint_count = len(self.document.ik) + len(self.document.transform)
        if len(self.original_order_by_name) != constraint_count:
            raise ValueError(
                "original order metadata must cover every connected constraint"
            )
        for name, order in self.original_order_by_name.items():
            if not isinstance(name, str) or not name.strip():
                raise ValueError("constraint metadata names must be non-empty strings")
            if isinstance(order, bool) or not isinstance(order, int) or order < 0:
                raise ValueError("constraint metadata orders must be non-negative ints")


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


def _connected_constraint_records(
    document: SpineDocument,
) -> tuple[tuple[int, int, int, str, object], ...]:
    """Return the exact stable order used by generic Spine composition.

    Records are ordered by the current runtime order, then IK before transform, then
    their original collection index. This mirrors the generic composer's deterministic
    tie-break contract without modifying the connected rig schedule itself.
    """

    if not isinstance(document, SpineDocument):
        raise TypeError("document must be SpineDocument")
    records: list[tuple[int, int, int, str, object]] = []
    for index, constraint in enumerate(document.ik):
        records.append((constraint.order, 0, index, "ik", constraint))
    for index, constraint in enumerate(document.transform):
        records.append((constraint.order, 1, index, "transform", constraint))
    records.sort(key=lambda item: (item[0], item[1], item[2]))
    return tuple(records)


def _prepare_connected_outer_component(
    document: SpineDocument,
) -> _ConnectedOuterCompositionInput:
    """Validate connected semantics and create a strict temporary order view.

    Only ``DUPLICATE_CONSTRAINT_ORDER`` is accepted by the connected validator. Every
    other structural, reference, skin, attachment, and weighted-mesh issue remains a hard
    error. The returned document is then checked by the normal strict validator so the
    outer composer never receives an invalid component.
    """

    if not isinstance(document, SpineDocument):
        raise TypeError("document must be SpineDocument")
    try:
        ConnectedGroupSerializationValidator().validate_or_raise(document)
    except Exception as exc:
        raise SpineCompositionError(
            f"Component '{_CONNECTED_COMPONENT_ID}' is not a valid connected "
            f"Spine document: {exc}"
        ) from exc

    order_by_identity: dict[tuple[str, int], int] = {}
    original_order_by_name: dict[str, int] = {}
    for global_order, record in enumerate(_connected_constraint_records(document)):
        _order, _kind_rank, local_index, kind, constraint = record
        name = constraint.name
        if name in original_order_by_name:
            raise SpineCompositionError(
                f"Connected constraint name is duplicated: {name!r}"
            )
        original_order_by_name[name] = constraint.order
        order_by_identity[(kind, local_index)] = global_order

    rebased_document = replace(
        document,
        ik=tuple(
            replace(
                constraint,
                order=order_by_identity[("ik", local_index)],
            )
            for local_index, constraint in enumerate(document.ik)
        ),
        transform=tuple(
            replace(
                constraint,
                order=order_by_identity[("transform", local_index)],
            )
            for local_index, constraint in enumerate(document.transform)
        ),
    )
    try:
        SpineValidator().validate_or_raise(rebased_document)
    except Exception as exc:
        raise SpineCompositionError(
            f"Component '{_CONNECTED_COMPONENT_ID}' failed strict validation "
            f"after temporary order rebasing: {exc}"
        ) from exc

    return _ConnectedOuterCompositionInput(
        document=rebased_document,
        original_order_by_name=MappingProxyType(dict(original_order_by_name)),
    )


def _restore_connected_order_provenance(
    composition: SpineDocumentCompositionResult,
    connected_document: SpineDocument,
    original_order_by_name: Mapping[str, int],
) -> SpineDocumentCompositionResult:
    """Restore the original connected component and its historical order metadata."""

    if not isinstance(composition, SpineDocumentCompositionResult):
        raise TypeError("composition must be SpineDocumentCompositionResult")
    if not isinstance(connected_document, SpineDocument):
        raise TypeError("connected_document must be SpineDocument")
    if not isinstance(original_order_by_name, Mapping):
        raise TypeError("original_order_by_name must be a mapping")

    connected_component_count = 0
    components = []
    for component in composition.components:
        if component.component_id != _CONNECTED_COMPONENT_ID:
            components.append(component)
            continue
        connected_component_count += 1
        components.append(replace(component, document=connected_document))
    if connected_component_count != 1:
        raise SpineCompositionError(
            "Outer composition must contain exactly one connected component; "
            f"found={connected_component_count}"
        )

    found_names: set[str] = set()
    assignments = []
    for assignment in composition.constraint_orders:
        if assignment.component_id != _CONNECTED_COMPONENT_ID:
            assignments.append(assignment)
            continue
        if assignment.constraint_name not in original_order_by_name:
            raise SpineCompositionError(
                "Outer composition produced unknown connected constraint metadata: "
                f"{assignment.constraint_name!r}"
            )
        found_names.add(assignment.constraint_name)
        assignments.append(
            replace(
                assignment,
                original_order=original_order_by_name[assignment.constraint_name],
            )
        )

    missing = set(original_order_by_name) - found_names
    if missing:
        raise SpineCompositionError(
            "Outer composition lost connected constraint metadata for: "
            f"{tuple(sorted(missing))}"
        )
    return replace(
        composition,
        components=tuple(components),
        constraint_orders=tuple(assignments),
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

    Spine 4.2 connected output may contain intentional same-layer order ties. The generic
    composer will make all final MIXED orders unique, but its component validator runs
    first. A strict temporary view bridges those contracts without changing the original
    connected result or its diagnostic order metadata.
    """

    connected_input = _prepare_connected_outer_component(connected_document)
    outer = compose_spine_documents(
        (
            SpineDocumentComponent(
                component_id=_CONNECTED_COMPONENT_ID,
                document=connected_input.document,
            ),
            SpineDocumentComponent(
                component_id=_STANDALONE_COMPONENT_ID,
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
    return _restore_connected_order_provenance(
        outer,
        connected_document,
        connected_input.original_order_by_name,
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
        connected = replace_a1_composition_document(
            connected,
            overlay.document,
        )
        if not isinstance(connected, ConnectedGroupBuildResult):
            raise TypeError("connected composition replacement changed result type")

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
