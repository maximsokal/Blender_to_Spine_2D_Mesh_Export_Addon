from dataclasses import replace
from math import cos, radians, sin

import pytest

from Blender_to_Spine2D_Mesh_Exporter.application import (
    A1GeometryPreparationSettings,
    prepare_a1_geometry_regions,
)
from Blender_to_Spine2D_Mesh_Exporter.domain.geometry import (
    A1AngularMode,
    SegmentationSettings,
)

from test_a1_segmentation_decomposition import build_three_quad_strip


def _normal_at_degrees(angle: float):
    value = radians(angle)
    return (-sin(value), 0.0, cos(value))


def _fold_snapshot():
    return build_three_quad_strip()


def _segment_indices(result):
    return tuple(
        tuple(face_id.index for face_id in segment.face_ids)
        for segment in result.segmentation.segments
    )


def test_geometry_preparation_defaults_to_seed_cone():
    settings = A1GeometryPreparationSettings()

    assert settings.angular_mode is A1AngularMode.SEED_CONE
    assert settings.local_angle_limit_degrees is None


def test_geometry_preparation_forwards_hybrid_dihedral_contract():
    result = prepare_a1_geometry_regions(
        _fold_snapshot(),
        A1GeometryPreparationSettings(
            segmentation=SegmentationSettings(
                angle_limit_degrees=30.0,
                split_uv_boundaries=False,
            ),
            angular_mode=A1AngularMode.SEED_CONE_AND_LOCAL_DIHEDRAL,
            local_angle_limit_degrees=30.0,
        ),
    )

    assert _segment_indices(result) == ((0, 1), (2,))
    assert result.settings.angular_mode is A1AngularMode.SEED_CONE_AND_LOCAL_DIHEDRAL
    assert result.settings.local_angle_limit_degrees == 30.0


@pytest.mark.parametrize("value", (-1.0, 181.0, float("inf"), float("nan")))
def test_geometry_preparation_rejects_invalid_local_angle_limit(value):
    with pytest.raises(ValueError):
        A1GeometryPreparationSettings(
            angular_mode=A1AngularMode.SEED_CONE_AND_LOCAL_DIHEDRAL,
            local_angle_limit_degrees=value,
        )


def test_geometry_preparation_requires_typed_angular_mode():
    with pytest.raises(TypeError, match="A1AngularMode"):
        A1GeometryPreparationSettings(
            angular_mode="SEED_CONE_AND_LOCAL_DIHEDRAL",
        )


def test_removed_legacy_seed_cone_value_is_rejected():
    with pytest.raises(ValueError, match="Unsupported A1 angular mode"):
        from Blender_to_Spine2D_Mesh_Exporter.domain.geometry.a1_segmentation import (
            _resolve_angular_mode,
        )

        _resolve_angular_mode("LEGACY_SEED_CONE")
