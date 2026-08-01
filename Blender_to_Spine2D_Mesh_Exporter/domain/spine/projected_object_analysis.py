"""Immutable per-object projection analysis shared by all composition modes."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Tuple

from ..geometry import MeshSnapshot, calculate_a1_projected_snapshot_depth_range
from ..projection import A1ProjectionDirection
from .object_block_draw_order import SpineObjectBlockDepth


Vector3 = Tuple[float, float, float]


def _finite_number(value: object, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{field_name} must be a finite number")
    resolved = float(value)
    if not isfinite(resolved):
        raise ValueError(f"{field_name} must be finite")
    return 0.0 if resolved == 0.0 else resolved


def _canonical_name(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")
    if value != value.strip():
        raise ValueError(f"{field_name} cannot contain boundary whitespace")
    return value


def _projected_origin(snapshot: MeshSnapshot) -> Vector3:
    matrix = snapshot.world_matrix
    if not isinstance(matrix, tuple) or len(matrix) != 16:
        raise ValueError("snapshot.world_matrix must contain sixteen values")
    values = tuple(
        _finite_number(value, f"snapshot.world_matrix[{index}]")
        for index, value in enumerate(matrix)
    )
    tolerance = 1.0e-10
    expected_linear = (
        1.0, 0.0, 0.0,
        0.0, 1.0, 0.0,
        0.0, 0.0, 1.0,
    )
    actual_linear = (
        values[0], values[1], values[2],
        values[4], values[5], values[6],
        values[8], values[9], values[10],
    )
    if any(
        abs(actual - expected) > tolerance
        for actual, expected in zip(actual_linear, expected_linear, strict=True)
    ) or any(abs(values[index]) > tolerance for index in (12, 13, 14)) or abs(
        values[15] - 1.0
    ) > tolerance:
        raise ValueError(
            "projected object analysis requires a translation-only canonical snapshot"
        )
    return values[3], values[7], values[11]


def _projected_world_position(
    origin: Vector3,
    local_position: Vector3,
    projection_direction: A1ProjectionDirection,
) -> Vector3:
    """Recover true projected U/V/depth from one internal canonical vertex.

    Active Camera stores local Mesh Y reflected so the established attachment projector's
    later Y inversion produces screen-up Spine coordinates. Diagnostics and bounds must
    undo that storage reflection. Signed-axis snapshots store canonical V directly.
    """

    local_u, local_v, local_depth = tuple(
        _finite_number(value, f"local_position[{index}]")
        for index, value in enumerate(local_position)
    )
    world_v = (
        origin[1] - local_v
        if projection_direction is A1ProjectionDirection.ACTIVE_CAMERA
        else origin[1] + local_v
    )
    return (
        origin[0] + local_u,
        world_v,
        origin[2] + local_depth,
    )


@dataclass(frozen=True, slots=True)
class A1ProjectedBounds:
    minimum_u: float
    maximum_u: float
    minimum_v: float
    maximum_v: float
    minimum_depth: float
    maximum_depth: float

    def __post_init__(self) -> None:
        for field_name in (
            "minimum_u",
            "maximum_u",
            "minimum_v",
            "maximum_v",
            "minimum_depth",
            "maximum_depth",
        ):
            object.__setattr__(
                self,
                field_name,
                _finite_number(getattr(self, field_name), field_name),
            )
        if self.minimum_u > self.maximum_u:
            raise ValueError("minimum_u cannot exceed maximum_u")
        if self.minimum_v > self.maximum_v:
            raise ValueError("minimum_v cannot exceed maximum_v")
        if self.minimum_depth > self.maximum_depth:
            raise ValueError("minimum_depth cannot exceed maximum_depth")


@dataclass(frozen=True, slots=True)
class A1ProjectedObjectAnalysis:
    component_id: str
    prefix: str
    source_input_index: int
    projection_direction: A1ProjectionDirection
    projected_origin_u: float
    projected_origin_v: float
    projected_origin_depth: float
    nearest_vertex_index: int
    nearest_vertex_world_position: Vector3
    nearest_vertex_depth: float
    farthest_vertex_index: int
    farthest_vertex_depth: float
    projected_bounds: A1ProjectedBounds
    owned_slot_names: Tuple[str, ...]

    def __post_init__(self) -> None:
        _canonical_name(self.component_id, "component_id")
        _canonical_name(self.prefix, "prefix")
        if (
            isinstance(self.source_input_index, bool)
            or not isinstance(self.source_input_index, int)
            or self.source_input_index < 0
        ):
            raise ValueError("source_input_index must be a non-negative integer")
        if type(self.projection_direction) is not A1ProjectionDirection:
            raise TypeError("projection_direction must be A1ProjectionDirection")
        for field_name in (
            "projected_origin_u",
            "projected_origin_v",
            "projected_origin_depth",
            "nearest_vertex_depth",
            "farthest_vertex_depth",
        ):
            object.__setattr__(
                self,
                field_name,
                _finite_number(getattr(self, field_name), field_name),
            )
        for field_name in ("nearest_vertex_index", "farthest_vertex_index"):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{field_name} must be a non-negative integer")
        if (
            not isinstance(self.nearest_vertex_world_position, tuple)
            or len(self.nearest_vertex_world_position) != 3
        ):
            raise TypeError("nearest_vertex_world_position must contain three values")
        object.__setattr__(
            self,
            "nearest_vertex_world_position",
            tuple(
                _finite_number(value, f"nearest_vertex_world_position[{index}]")
                for index, value in enumerate(self.nearest_vertex_world_position)
            ),
        )
        if self.farthest_vertex_depth > self.nearest_vertex_depth:
            raise ValueError(
                "farthest_vertex_depth cannot exceed nearest_vertex_depth"
            )
        if not isinstance(self.projected_bounds, A1ProjectedBounds):
            raise TypeError("projected_bounds must be A1ProjectedBounds")
        if not isinstance(self.owned_slot_names, tuple) or not self.owned_slot_names:
            raise ValueError("owned_slot_names must be a non-empty tuple")
        for index, name in enumerate(self.owned_slot_names):
            _canonical_name(name, f"owned_slot_names[{index}]")
        if len(self.owned_slot_names) != len(set(self.owned_slot_names)):
            raise ValueError("owned_slot_names cannot contain duplicates")

    @property
    def block_depth(self) -> SpineObjectBlockDepth:
        return SpineObjectBlockDepth(
            component_id=self.component_id,
            source_input_index=self.source_input_index,
            nearest_vertex_index=self.nearest_vertex_index,
            nearest_vertex_depth=self.nearest_vertex_depth,
            farthest_vertex_index=self.farthest_vertex_index,
            farthest_vertex_depth=self.farthest_vertex_depth,
        )


def analyse_projected_object(
    *,
    component_id: str,
    prefix: str,
    source_input_index: int,
    projection_direction: A1ProjectionDirection,
    snapshot: MeshSnapshot,
    owned_slot_names: Tuple[str, ...],
) -> A1ProjectedObjectAnalysis:
    """Build one complete projected placement and draw-order diagnostic record."""

    if not isinstance(snapshot, MeshSnapshot):
        raise TypeError("snapshot must be MeshSnapshot")
    if type(projection_direction) is not A1ProjectionDirection:
        raise TypeError("projection_direction must be A1ProjectionDirection")
    origin = _projected_origin(snapshot)
    depth_range = calculate_a1_projected_snapshot_depth_range(snapshot)
    vertex_by_index = {vertex.id.index: vertex for vertex in snapshot.vertices}
    if len(vertex_by_index) != len(snapshot.vertices):
        raise ValueError("snapshot contains duplicate VertexId indices")
    try:
        nearest_vertex = vertex_by_index[depth_range.nearest_vertex_id.index]
    except KeyError as exc:
        raise ValueError("nearest projected vertex is absent from snapshot") from exc

    world_positions = tuple(
        _projected_world_position(
            origin,
            vertex.position,
            projection_direction,
        )
        for vertex in snapshot.vertices
    )
    if not world_positions:
        raise ValueError("snapshot contains no vertices")
    nearest_world = _projected_world_position(
        origin,
        nearest_vertex.position,
        projection_direction,
    )
    return A1ProjectedObjectAnalysis(
        component_id=component_id,
        prefix=prefix,
        source_input_index=source_input_index,
        projection_direction=projection_direction,
        projected_origin_u=origin[0],
        projected_origin_v=origin[1],
        projected_origin_depth=origin[2],
        nearest_vertex_index=depth_range.nearest_vertex_id.index,
        nearest_vertex_world_position=nearest_world,
        nearest_vertex_depth=depth_range.nearest_vertex_depth,
        farthest_vertex_index=depth_range.farthest_vertex_id.index,
        farthest_vertex_depth=depth_range.farthest_vertex_depth,
        projected_bounds=A1ProjectedBounds(
            minimum_u=min(position[0] for position in world_positions),
            maximum_u=max(position[0] for position in world_positions),
            minimum_v=min(position[1] for position in world_positions),
            maximum_v=max(position[1] for position in world_positions),
            minimum_depth=min(position[2] for position in world_positions),
            maximum_depth=max(position[2] for position in world_positions),
        ),
        owned_slot_names=owned_slot_names,
    )


__all__ = [
    "A1ProjectedBounds",
    "A1ProjectedObjectAnalysis",
    "analyse_projected_object",
]
