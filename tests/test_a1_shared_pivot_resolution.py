from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from Blender_to_Spine2D_Mesh_Exporter.application import (
    A1SingleObjectExportSettings,
    ExportSettings,
)
from Blender_to_Spine2D_Mesh_Exporter.blender_adapter.a1_multi_object_contracts import (
    A1MultiObjectSource,
)
from Blender_to_Spine2D_Mesh_Exporter.blender_adapter.a1_shared_pivot import (
    resolve_a1_shared_pivot_world,
)
from Blender_to_Spine2D_Mesh_Exporter.domain.geometry.ids import (
    SourceVertexId,
    VertexId,
)
from Blender_to_Spine2D_Mesh_Exporter.domain.geometry.model import MeshSnapshot, MeshVertex
from Blender_to_Spine2D_Mesh_Exporter.domain.projection import A1ProjectionDirection


def _matrix(
    sx: float,
    sy: float,
    sz: float,
    tx: float,
    ty: float,
    tz: float,
) -> tuple[float, ...]:
    return (
        sx,
        0.0,
        0.0,
        tx,
        0.0,
        sy,
        0.0,
        ty,
        0.0,
        0.0,
        sz,
        tz,
        0.0,
        0.0,
        0.0,
        1.0,
    )


def _snapshot(
    object_id: str,
    positions: tuple[tuple[float, float, float], ...],
    world_matrix: tuple[float, ...],
) -> MeshSnapshot:
    return MeshSnapshot(
        snapshot_id=f"{object_id}-snapshot",
        source_object_id=object_id,
        object_name=object_id,
        vertices=tuple(
            MeshVertex(
                id=VertexId(index),
                source_id=SourceVertexId(object_id, index),
                position=position,
                normal=(0.0, 0.0, 1.0),
            )
            for index, position in enumerate(positions)
        ),
        edges=(),
        loops=(),
        faces=(),
        world_matrix=world_matrix,
    )


def _settings(
    *,
    direction: A1ProjectionDirection = A1ProjectionDirection.POSITIVE_Z,
    use_world_location_for_main_bone: bool = True,
) -> A1SingleObjectExportSettings:
    return A1SingleObjectExportSettings(
        export=ExportSettings(
            texture_width=128,
            texture_height=128,
            output_directory=Path("shared-pivot-resolution-output"),
        ),
        projection_direction=direction,
        use_world_location_for_main_bone=use_world_location_for_main_bone,
    )


def _source(object_id: str, settings: A1SingleObjectExportSettings):
    obj = SimpleNamespace(type="MESH", name=object_id, name_full=object_id, data=object())
    return A1MultiObjectSource(
        source_object=obj,
        component_id=object_id,
        settings=settings,
    )


def test_shared_pivot_is_center_of_aggregate_exported_world_geometry() -> None:
    source_a = _source("A", _settings())
    source_b = _source("B", _settings())
    snapshots = {
        "A": _snapshot(
            "A",
            ((-1.0, -2.0, -1.0), (2.0, 1.0, 3.0)),
            _matrix(2.0, 1.0, 0.5, 10.0, -1.0, 4.0),
        ),
        "B": _snapshot(
            "B",
            ((-3.0, 2.0, 1.0), (1.0, 5.0, 4.0)),
            _matrix(1.0, 2.0, 1.0, -5.0, 3.0, -2.0),
        ),
    }

    def read_snapshot(_obj, object_id, _settings_value, **_kwargs):
        return snapshots[object_id], 0, (), object()

    with patch(
        "Blender_to_Spine2D_Mesh_Exporter.blender_adapter.a1_shared_pivot._read_source_snapshot",
        side_effect=read_snapshot,
    ):
        resolved = resolve_a1_shared_pivot_world((source_a, source_b))
        reversed_resolved = resolve_a1_shared_pivot_world((source_b, source_a))

    # A world vertices: (8,-3,3.5), (14,0,5.5)
    # B world vertices: (-8,7,-1), (-4,13,2)
    assert resolved.minimum_world == pytest.approx((-8.0, -3.0, -1.0))
    assert resolved.maximum_world == pytest.approx((14.0, 13.0, 5.5))
    assert resolved.pivot_world == pytest.approx((3.0, 5.0, 2.25))
    assert resolved.vertex_count == 4
    assert resolved.object_count == 2
    assert reversed_resolved == resolved


def test_shared_pivot_resolution_rejects_different_projection_axes() -> None:
    sources = (
        _source("A", _settings(direction=A1ProjectionDirection.POSITIVE_Z)),
        _source("B", _settings(direction=A1ProjectionDirection.POSITIVE_X)),
    )

    with pytest.raises(ValueError, match="one projection direction"):
        resolve_a1_shared_pivot_world(sources)


def test_shared_pivot_resolution_requires_world_location_main_bones() -> None:
    sources = (
        _source("A", _settings()),
        _source("B", _settings(use_world_location_for_main_bone=False)),
    )

    with pytest.raises(ValueError, match="world-location main-bone"):
        resolve_a1_shared_pivot_world(sources)


def test_shared_pivot_resolution_rejects_preexisting_per_object_override() -> None:
    first = _settings()
    second = A1SingleObjectExportSettings(
        export=first.export,
        projection_direction=first.projection_direction,
        shared_pivot_world=(0.0, 0.0, 0.0),
    )
    sources = (_source("A", first), _source("B", second))

    with pytest.raises(ValueError, match="owned by the multi-object transaction"):
        resolve_a1_shared_pivot_world(sources)
