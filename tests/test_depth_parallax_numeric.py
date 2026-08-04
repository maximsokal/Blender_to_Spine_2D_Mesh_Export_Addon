"""Numerical regressions for accumulated parallax horizon angles."""

from __future__ import annotations

from math import pi

import pytest

from Blender_to_Spine2D_Mesh_Exporter.domain.geometry import SourceVertexId
from Blender_to_Spine2D_Mesh_Exporter.domain.geometry.depth_parallax import (
    _FaceGeometry,
    _dihedral_angle,
)


_OBJECT_ID = "ParallaxNumeric"


def _face(
    face_index: int,
    normal: tuple[float, float, float],
) -> _FaceGeometry:
    return _FaceGeometry(
        face_index=face_index,
        source_face_index=face_index,
        source_vertex_ids=(
            SourceVertexId(_OBJECT_ID, face_index * 3),
            SourceVertexId(_OBJECT_ID, face_index * 3 + 1),
            SourceVertexId(_OBJECT_ID, face_index * 3 + 2),
        ),
        world_points=(
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
        ),
        normal_world=normal,
        centroid_world=(1.0 / 3.0, 1.0 / 3.0, 0.0),
    )


def test_proportional_coplanar_normals_have_exactly_zero_cost() -> None:
    first = _face(0, (0.7071067811865475, 0.0, -0.7071067811865475))
    second = _face(1, (0.35355339059327373, 0.0, -0.35355339059327373))

    assert _dihedral_angle(first, second) == 0.0


def test_dihedral_angle_is_independent_of_normal_length() -> None:
    front = _face(0, (0.0, 0.0, 4.0))
    diagonal = _face(1, (8.0, 0.0, -8.0))

    assert _dihedral_angle(front, diagonal) == pytest.approx(pi / 4.0, abs=1.0e-15)
