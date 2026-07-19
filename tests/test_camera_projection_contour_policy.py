from dataclasses import replace
import inspect

import pytest

import Blender_to_Spine2D_Mesh_Exporter.blender_adapter.camera_projection_postprocess as postprocess
from Blender_to_Spine2D_Mesh_Exporter.domain.baking import (
    BakeExecutionSettings,
    ProjectionContourMode,
)


def test_projection_contour_defaults_to_simplified_concave():
    settings = BakeExecutionSettings()

    assert (
        settings.projection_contour_mode
        is ProjectionContourMode.SIMPLIFIED_CONCAVE
    )
    assert (
        settings.projection_contour_simplify_tolerance_pixels
        == 1.0
    )


def test_projection_contour_policy_accepts_explicit_convex_compatibility_mode():
    settings = replace(
        BakeExecutionSettings(),
        projection_contour_mode=ProjectionContourMode.CONVEX_HULL,
        projection_contour_simplify_tolerance_pixels=0.0,
    )

    assert (
        settings.projection_contour_mode
        is ProjectionContourMode.CONVEX_HULL
    )
    assert (
        settings.projection_contour_simplify_tolerance_pixels
        == 0.0
    )


def test_projection_contour_mode_requires_typed_enum():
    with pytest.raises(TypeError, match="projection_contour_mode"):
        BakeExecutionSettings(
            projection_contour_mode="SIMPLIFIED_CONCAVE"
        )


@pytest.mark.parametrize(
    "value",
    (
        -0.01,
        float("inf"),
        float("-inf"),
        float("nan"),
        "1",
        None,
        True,
    ),
)
def test_projection_contour_tolerance_rejects_invalid_values(value):
    with pytest.raises(ValueError, match="simplify_tolerance"):
        BakeExecutionSettings(
            projection_contour_simplify_tolerance_pixels=value,
        )


def test_b4_postprocess_passes_contour_policy_to_union_accumulator():
    source = inspect.getsource(
        postprocess.build_projection_union_accumulator
    )

    assert "projection_contour_mode" in source
    assert (
        "projection_contour_simplify_tolerance_pixels"
        in source
    )
    assert (
        "contour_mode=execution_settings.projection_contour_mode"
        in source
    )
