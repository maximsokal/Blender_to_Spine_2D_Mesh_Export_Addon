"""Resolve one immutable world-space pivot for a selected multi-object export.

The pivot is the center of the aggregate world-space AABB of the exact Mesh snapshots
owned by the export route. It is deliberately computed from exported vertices rather
than Blender Object origins or ``Object.bound_box`` so arbitrary origins, rotation,
scale, shear, and ignored/evaluated modifier policy cannot skew the assembly pivot.

No Blender data is mutated and no operator or BMesh state is used.
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
from math import isfinite
from typing import Any, Tuple

from ..application.a1_shared_pivot import (
    A1SharedPivotWorld,
    supports_a1_shared_pivot,
    validate_a1_shared_pivot_world,
)
from ..domain.geometry import normalize_mesh_snapshot_world_transform
from ..domain.spine import calculate_uniform_scale
from .a1_multi_object_contracts import A1MultiObjectSource
from .a1_source_geometry_preparation import _read_source_snapshot, object_name


logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class A1SharedPivotResolution:
    """Resolved aggregate bounds and their world-space midpoint."""

    pivot_world: A1SharedPivotWorld
    minimum_world: A1SharedPivotWorld
    maximum_world: A1SharedPivotWorld
    vertex_count: int
    object_count: int

    def __post_init__(self) -> None:
        pivot = validate_a1_shared_pivot_world(self.pivot_world)
        minimum = validate_a1_shared_pivot_world(self.minimum_world)
        maximum = validate_a1_shared_pivot_world(self.maximum_world)
        object.__setattr__(self, "pivot_world", pivot)
        object.__setattr__(self, "minimum_world", minimum)
        object.__setattr__(self, "maximum_world", maximum)
        if isinstance(self.vertex_count, bool) or not isinstance(self.vertex_count, int):
            raise TypeError("vertex_count must be int")
        if isinstance(self.object_count, bool) or not isinstance(self.object_count, int):
            raise TypeError("object_count must be int")
        if self.vertex_count <= 0:
            raise ValueError("vertex_count must be positive")
        if self.object_count < 2:
            raise ValueError("object_count must be at least two")
        if any(minimum[index] > maximum[index] for index in range(3)):
            raise ValueError("minimum_world cannot exceed maximum_world")
        expected = tuple(
            (minimum[index] + maximum[index]) / 2.0 for index in range(3)
        )
        if pivot != expected:
            raise ValueError(
                "pivot_world must equal the aggregate world-space bounds midpoint"
            )


def _validate_shared_source_contract(
    sources: Tuple[A1MultiObjectSource, ...],
) -> None:
    """Require one coherent signed-axis/pixel coordinate system for the assembly."""

    if not isinstance(sources, tuple) or len(sources) < 2:
        raise ValueError("sources must contain at least two A1MultiObjectSource values")
    if not all(isinstance(source, A1MultiObjectSource) for source in sources):
        raise TypeError("sources must contain A1MultiObjectSource values")

    first_settings = sources[0].settings
    first_direction = first_settings.projection_direction
    first_scale = calculate_uniform_scale(
        first_settings.export.texture_width,
        first_settings.export.texture_height,
        first_settings.rig_scale_mode,
    )

    for index, source in enumerate(sources):
        settings = source.settings
        if settings.shared_pivot_world is not None:
            raise ValueError(
                "Shared pivot override is owned by the multi-object transaction; "
                f"component {source.component_id!r} already carries shared_pivot_world"
            )
        if not settings.use_world_location_for_main_bone:
            raise ValueError(
                "Shared selection pivot requires world-location main-bone placement; "
                f"component={source.component_id!r}"
            )
        if not supports_a1_shared_pivot(
            settings.bake_execution.texture_export_mode,
            settings.projection_direction,
            len(sources),
        ):
            raise ValueError(
                "Shared selection pivot requires Normal / UV Segments with one of "
                "+X, -X, +Y, -Y, +Z, -Z for every source; "
                f"component={source.component_id!r}, "
                f"mode={settings.bake_execution.texture_export_mode.value!r}, "
                f"projection={settings.projection_direction.value!r}"
            )
        if settings.projection_direction is not first_direction:
            raise ValueError(
                "Shared selection pivot requires one projection direction across the "
                f"multi-object transaction; first={first_direction.value!r}, "
                f"component[{index}]={settings.projection_direction.value!r}"
            )
        scale = calculate_uniform_scale(
            settings.export.texture_width,
            settings.export.texture_height,
            settings.rig_scale_mode,
        )
        if scale != first_scale:
            raise ValueError(
                "Shared selection pivot requires one Spine pixel scale across the "
                f"multi-object transaction; first={first_scale}, component[{index}]={scale}"
            )


def _world_vertices_from_export_snapshot(
    source: A1MultiObjectSource,
    *,
    scene: Any | None,
) -> tuple[tuple[float, float, float], ...]:
    """Return exact pre-projection world vertices for one immutable export snapshot."""

    object_id = object_name(source.source_object)
    snapshot, _modifier_count, _warnings, _uv_report = _read_source_snapshot(
        source.source_object,
        object_id,
        source.settings,
        scene=scene,
    )
    normalized = normalize_mesh_snapshot_world_transform(snapshot)
    normalized_snapshot = normalized.snapshot
    matrix = normalized_snapshot.world_matrix
    if not isinstance(matrix, tuple) or len(matrix) != 16:
        raise ValueError("normalized snapshot world_matrix must contain 16 values")
    translation = (float(matrix[3]), float(matrix[7]), float(matrix[11]))
    if not all(isfinite(value) for value in translation):
        raise ValueError(
            f"Normalized world translation is non-finite for {object_id!r}"
        )

    vertices = tuple(
        (
            translation[0] + float(vertex.position[0]),
            translation[1] + float(vertex.position[1]),
            translation[2] + float(vertex.position[2]),
        )
        for vertex in normalized_snapshot.vertices
    )
    if not vertices:
        raise ValueError(
            f"Shared selection pivot cannot use empty Mesh object {object_id!r}"
        )
    if not all(isfinite(value) for vertex in vertices for value in vertex):
        raise ValueError(
            f"World geometry contains non-finite coordinates for {object_id!r}"
        )
    return vertices


def resolve_a1_shared_pivot_world(
    sources: Tuple[A1MultiObjectSource, ...],
    *,
    scene: Any | None = None,
) -> A1SharedPivotResolution:
    """Resolve the aggregate exported-geometry AABB midpoint exactly once."""

    _validate_shared_source_contract(sources)

    minimum = [float("inf"), float("inf"), float("inf")]
    maximum = [float("-inf"), float("-inf"), float("-inf")]
    vertex_count = 0

    for source in sources:
        vertices = _world_vertices_from_export_snapshot(source, scene=scene)
        vertex_count += len(vertices)
        for vertex in vertices:
            for axis in range(3):
                value = float(vertex[axis])
                if value < minimum[axis]:
                    minimum[axis] = value
                if value > maximum[axis]:
                    maximum[axis] = value

    minimum_world = validate_a1_shared_pivot_world(tuple(minimum))
    maximum_world = validate_a1_shared_pivot_world(tuple(maximum))
    pivot_world = validate_a1_shared_pivot_world(
        tuple(
            (minimum_world[axis] + maximum_world[axis]) / 2.0
            for axis in range(3)
        )
    )
    result = A1SharedPivotResolution(
        pivot_world=pivot_world,
        minimum_world=minimum_world,
        maximum_world=maximum_world,
        vertex_count=vertex_count,
        object_count=len(sources),
    )
    logger.info(
        "Resolved shared selection pivot: objects=%d vertices=%d pivot=%s bounds=(%s, %s)",
        result.object_count,
        result.vertex_count,
        result.pivot_world,
        result.minimum_world,
        result.maximum_world,
    )
    return result


__all__ = [
    "A1SharedPivotResolution",
    "resolve_a1_shared_pivot_world",
]
