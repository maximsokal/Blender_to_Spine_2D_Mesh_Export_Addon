"""Create stable A1 Z-group assignments from the original mesh snapshot.

Z values are inspected exactly once on the source snapshot. Every later segment or
triangulated copy resolves its parent group through ``SourceVertexId`` rather than
comparing transformed floating-point coordinates.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

from ..domain.geometry import MeshSnapshot, MeshSnapshotValidator, SourceVertexId
from ..domain.spine.legacy_rig_contracts import LegacyZGroup
from .a1_attachment_projection import A1VertexZBinding
from .a1_numeric_contracts import (
    require_finite_number,
    require_identity,
    require_integer,
)


class A1ZGroupAssignmentError(ValueError):
    """Raised when source lineage cannot provide an unambiguous Z parent."""


@dataclass(frozen=True, slots=True)
class A1ZGroupHeightOverride:
    z_value: float
    height_real_pixels: float

    def __post_init__(self) -> None:
        require_finite_number(self.z_value, "z_value")
        require_finite_number(self.height_real_pixels, "height_real_pixels")


@dataclass(frozen=True, slots=True)
class A1SourceVertexZBinding:
    source_vertex_id: SourceVertexId
    z_group_index: int

    def __post_init__(self) -> None:
        if not isinstance(self.source_vertex_id, SourceVertexId):
            raise TypeError("source_vertex_id must be SourceVertexId")
        require_integer(self.z_group_index, "z_group_index", minimum=0)


@dataclass(frozen=True, slots=True)
class A1ZGroupAssignmentPlan:
    source_snapshot_id: str
    z_index_base: int
    groups: Tuple[LegacyZGroup, ...]
    source_bindings: Tuple[A1SourceVertexZBinding, ...]

    def __post_init__(self) -> None:
        require_identity(self.source_snapshot_id, "source_snapshot_id")
        require_integer(self.z_index_base, "z_index_base", minimum=0)
        if not isinstance(self.groups, tuple) or not self.groups:
            raise ValueError("groups must be a non-empty tuple")
        if not all(isinstance(group, LegacyZGroup) for group in self.groups):
            raise TypeError("groups must contain LegacyZGroup values")
        if not isinstance(self.source_bindings, tuple) or not self.source_bindings:
            raise ValueError("source_bindings must be a non-empty tuple")
        if not all(
            isinstance(binding, A1SourceVertexZBinding)
            for binding in self.source_bindings
        ):
            raise TypeError(
                "source_bindings must contain A1SourceVertexZBinding values"
            )
        source_ids = tuple(binding.source_vertex_id for binding in self.source_bindings)
        if len(source_ids) != len(set(source_ids)):
            raise ValueError("source_bindings contain duplicate SourceVertexId values")
        valid_indices = set(range(self.z_index_base, self.z_index_base + len(self.groups)))
        invalid = tuple(
            sorted(
                {
                    binding.z_group_index
                    for binding in self.source_bindings
                    if binding.z_group_index not in valid_indices
                }
            )
        )
        if invalid:
            raise ValueError(f"source_bindings reference invalid group indices: {invalid}")

    def group_index_for_source(self, source_vertex_id: SourceVertexId) -> int:
        if not isinstance(source_vertex_id, SourceVertexId):
            raise TypeError("source_vertex_id must be SourceVertexId")
        mapping = {
            binding.source_vertex_id: binding.z_group_index
            for binding in self.source_bindings
        }
        try:
            return mapping[source_vertex_id]
        except KeyError as exc:
            raise A1ZGroupAssignmentError(
                f"No Z-group assignment for source vertex {source_vertex_id}"
            ) from exc

    def projection_bindings(
        self,
        derived_snapshot: MeshSnapshot,
    ) -> Tuple[A1VertexZBinding, ...]:
        """Resolve local derived vertices exclusively through source lineage."""

        if not isinstance(derived_snapshot, MeshSnapshot):
            raise TypeError("derived_snapshot must be MeshSnapshot")
        MeshSnapshotValidator().validate_or_raise(derived_snapshot)
        source_mapping = {
            binding.source_vertex_id: binding.z_group_index
            for binding in self.source_bindings
        }
        missing = tuple(
            sorted(
                {
                    vertex.source_id
                    for vertex in derived_snapshot.vertices
                    if vertex.source_id not in source_mapping
                },
                key=lambda item: (item.object_id, item.vertex_index),
            )
        )
        if missing:
            raise A1ZGroupAssignmentError(
                "Derived snapshot contains vertices without source Z assignment: "
                + str(missing)
            )
        return tuple(
            A1VertexZBinding(
                vertex_id=vertex.id,
                z_group_index=source_mapping[vertex.source_id],
            )
            for vertex in sorted(
                derived_snapshot.vertices,
                key=lambda item: item.id.index,
            )
        )


def build_a1_z_group_assignment(
    source_snapshot: MeshSnapshot,
    *,
    height_overrides: Tuple[A1ZGroupHeightOverride, ...] = (),
    z_index_base: int = 1,
) -> A1ZGroupAssignmentPlan:
    """Build sorted legacy Z groups and source-vertex bindings exactly once."""

    if not isinstance(source_snapshot, MeshSnapshot):
        raise TypeError("source_snapshot must be MeshSnapshot")
    MeshSnapshotValidator().validate_or_raise(source_snapshot)
    if not isinstance(height_overrides, tuple):
        raise TypeError("height_overrides must be tuple")
    if not all(isinstance(item, A1ZGroupHeightOverride) for item in height_overrides):
        raise TypeError("height_overrides must contain A1ZGroupHeightOverride values")
    require_integer(z_index_base, "z_index_base", minimum=0)

    override_values = tuple(float(item.z_value) for item in height_overrides)
    if len(override_values) != len(set(override_values)):
        raise ValueError("height_overrides contain duplicate z_value entries")
    override_by_z = {
        float(item.z_value): float(item.height_real_pixels)
        for item in height_overrides
    }

    z_values_by_source: dict[SourceVertexId, float] = {}
    for vertex in source_snapshot.vertices:
        z_value = float(vertex.position[2])
        existing = z_values_by_source.get(vertex.source_id)
        if existing is not None and existing != z_value:
            raise A1ZGroupAssignmentError(
                f"Source vertex {vertex.source_id} appears with conflicting Z values "
                f"{existing} and {z_value}"
            )
        z_values_by_source[vertex.source_id] = z_value

    ordered_z_values = tuple(sorted(set(z_values_by_source.values())))
    unknown_overrides = tuple(sorted(set(override_by_z) - set(ordered_z_values)))
    if unknown_overrides:
        raise A1ZGroupAssignmentError(
            f"height_overrides reference Z values absent from source snapshot: "
            f"{unknown_overrides}"
        )
    groups = tuple(
        LegacyZGroup(
            z_value=z_value,
            height_real_pixels=override_by_z.get(z_value),
        )
        for z_value in ordered_z_values
    )
    group_index_by_z = {
        z_value: z_index_base + offset
        for offset, z_value in enumerate(ordered_z_values)
    }
    bindings = tuple(
        A1SourceVertexZBinding(
            source_vertex_id=source_vertex_id,
            z_group_index=group_index_by_z[z_value],
        )
        for source_vertex_id, z_value in sorted(
            z_values_by_source.items(),
            key=lambda item: (item[0].object_id, item[0].vertex_index),
        )
    )
    return A1ZGroupAssignmentPlan(
        source_snapshot_id=source_snapshot.snapshot_id,
        z_index_base=z_index_base,
        groups=groups,
        source_bindings=bindings,
    )
