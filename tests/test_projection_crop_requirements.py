"""Pure regressions for geometry-required camera projection crops."""

from __future__ import annotations

import pytest

from Blender_to_Spine2D_Mesh_Exporter.blender_adapter.a1_depth_projection_finalization import (
    A1DepthProjectionFinalizationError,
    _crop_uv,
)
from Blender_to_Spine2D_Mesh_Exporter.domain.baking import (
    CameraProjectionLayout,
    ProjectionContourMode,
    ProjectionCropBounds,
    ProjectionPixelPoint,
    ProjectionUvBounds,
    expand_projection_layout_to_uv_bounds,
)


def _alpha_only_layout() -> CameraProjectionLayout:
    return CameraProjectionLayout(
        full_width=1024,
        full_height=1024,
        crop=ProjectionCropBounds(
            minimum_x=500,
            minimum_y=500,
            maximum_x=540,
            maximum_y=540,
        ),
        hull=(
            ProjectionPixelPoint(504, 504),
            ProjectionPixelPoint(536, 504),
            ProjectionPixelPoint(536, 536),
            ProjectionPixelPoint(504, 536),
        ),
        alpha_threshold=0.01,
        padding_pixels=4,
        frame_count=1,
        visible_pixel_count=256,
        contour_mode=ProjectionContourMode.CONVEX_HULL,
    )


def test_geometry_uv_outside_alpha_crop_reproduces_manual_failure_then_passes() -> None:
    layout = _alpha_only_layout()
    uv = (0.104, 0.50)

    with pytest.raises(
        A1DepthProjectionFinalizationError,
        match="lies outside its camera render crop",
    ):
        _crop_uv(uv, layout, field_name="manual.reserve.uvs[0]")

    expanded = expand_projection_layout_to_uv_bounds(
        layout,
        ProjectionUvBounds.from_uvs(
            (uv, (0.23, 0.64)),
            field_name="manual.reserve.required_uvs",
        ),
    )
    cropped = _crop_uv(
        uv,
        expanded,
        field_name="manual.reserve.uvs[0]",
    )

    assert expanded.crop.minimum_x < layout.crop.minimum_x
    assert expanded.crop.maximum_x == layout.crop.maximum_x
    assert 0.0 <= cropped[0] <= 1.0
    assert 0.0 <= cropped[1] <= 1.0


def test_expansion_preserves_alpha_contour_and_coverage_diagnostics() -> None:
    layout = _alpha_only_layout()

    expanded = expand_projection_layout_to_uv_bounds(
        layout,
        ProjectionUvBounds(
            minimum_u=0.1,
            minimum_v=0.1,
            maximum_u=0.9,
            maximum_v=0.9,
        ),
    )

    assert expanded is not layout
    assert expanded.hull == layout.hull
    assert expanded.visible_pixel_count == layout.visible_pixel_count
    assert expanded.alpha_threshold == layout.alpha_threshold
    assert expanded.contour_mode is layout.contour_mode
    assert expanded.crop.minimum_x <= 102
    assert expanded.crop.maximum_x >= 922
    assert expanded.crop.minimum_y <= 102
    assert expanded.crop.maximum_y >= 922


def test_required_uv_pixel_crop_uses_spine_to_blender_v_orientation() -> None:
    bounds = ProjectionUvBounds(
        minimum_u=0.25,
        minimum_v=0.75,
        maximum_u=0.50,
        maximum_v=1.00,
    )

    crop = bounds.pixel_crop(
        width=200,
        height=100,
        padding_pixels=0,
    )

    assert crop == ProjectionCropBounds(
        minimum_x=50,
        minimum_y=0,
        maximum_x=100,
        maximum_y=25,
    )


def test_degenerate_required_uv_bounds_still_own_one_pixel() -> None:
    crop = ProjectionUvBounds(
        minimum_u=1.0,
        minimum_v=0.0,
        maximum_u=1.0,
        maximum_v=0.0,
    ).pixel_crop(width=64, height=64)

    assert crop.width == 1
    assert crop.height == 1
    assert crop.maximum_x == 64
    assert crop.maximum_y == 64
