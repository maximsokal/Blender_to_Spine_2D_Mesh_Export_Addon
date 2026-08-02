"""Regressions for Blender float32 active-camera view matrices."""

from __future__ import annotations

import pytest

from Blender_to_Spine2D_Mesh_Exporter.domain.camera_projection import (
    A1CameraProjectionFrame,
    A1CameraProjectionKind,
)


_IDENTITY_PROJECTION = (
    1.0,
    0.0,
    0.0,
    0.0,
    0.0,
    1.0,
    0.0,
    0.0,
    0.0,
    0.0,
    1.0,
    0.0,
    0.0,
    0.0,
    0.0,
    1.0,
)

# Rounded float32-style Z rotation. Its first row is mathematically unit
# length but differs from 1.0 by more than the previous 1e-8 tolerance.
_COS = 0.93232733
_SIN = 0.36161542
_FLOAT32_ROTATED_VIEW = (
    _COS,
    _SIN,
    0.0,
    0.0,
    -_SIN,
    _COS,
    0.0,
    0.0,
    0.0,
    0.0,
    1.0,
    0.0,
    0.0,
    0.0,
    0.0,
    1.0,
)


def _frame(view_matrix: tuple[float, ...]) -> A1CameraProjectionFrame:
    return A1CameraProjectionFrame(
        camera_id="RotatedCamera",
        kind=A1CameraProjectionKind.PERSPECTIVE,
        texture_width=128,
        texture_height=128,
        clip_start=0.1,
        clip_end=100.0,
        view_matrix=view_matrix,
        projection_matrix=_IDENTITY_PROJECTION,
    )


def test_float32_rotation_residual_is_accepted() -> None:
    residual = abs(_COS * _COS + _SIN * _SIN - 1.0)

    assert residual > 1.0e-8
    assert residual < 1.0e-6

    frame = _frame(_FLOAT32_ROTATED_VIEW)

    assert frame.view_matrix == _FLOAT32_ROTATED_VIEW


def test_meaningful_scale_remains_rejected() -> None:
    scaled = list(_FLOAT32_ROTATED_VIEW)
    scaled[0] *= 1.00001

    with pytest.raises(ValueError, match="unit length"):
        _frame(tuple(scaled))


def test_meaningful_shear_remains_rejected() -> None:
    sheared = list(_FLOAT32_ROTATED_VIEW)
    sheared[4] += 1.0e-4

    with pytest.raises(
        ValueError,
        match="unit length|orthogonal|right-handed",
    ):
        _frame(tuple(sheared))
