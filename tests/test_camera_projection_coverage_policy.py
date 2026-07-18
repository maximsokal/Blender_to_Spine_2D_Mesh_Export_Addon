import inspect

import pytest

import Blender_to_Spine2D_Mesh_Exporter.blender_adapter.camera_projection_executor_core as executor_core
from Blender_to_Spine2D_Mesh_Exporter.domain.baking import (
    BakeExecutionSettings,
    ProjectionAlphaUnionAccumulator,
    ProjectionCoverageMode,
    ProjectionCoveragePolicy,
)


def test_production_execution_defaults_to_coverage_weighted_cleanup():
    settings = BakeExecutionSettings()

    assert (
        settings.projection_coverage_policy.mode
        is ProjectionCoverageMode.HYSTERESIS_MORPHOLOGY
    )
    assert settings.projection_coverage_policy.core_alpha_threshold == 0.5
    assert settings.projection_coverage_policy.minimum_component_pixels == 2
    assert settings.projection_coverage_policy.maximum_hole_pixels == 1


def test_execution_requires_typed_coverage_policy():
    with pytest.raises(TypeError, match="projection_coverage_policy"):
        BakeExecutionSettings(projection_coverage_policy="HYSTERESIS_MORPHOLOGY")


def test_pure_accumulator_retains_binary_compatibility_default():
    accumulator = ProjectionAlphaUnionAccumulator(
        width=3,
        height=3,
        alpha_threshold=0.9,
        padding_pixels=0,
    )
    accumulator.add_mask(bytes((1, 0, 0, 0, 0, 0, 0, 0, 1)))
    layout = accumulator.build_layout()

    assert layout.coverage_mode is ProjectionCoverageMode.BINARY_THRESHOLD
    assert layout.visible_pixel_count == 2
    assert layout.coverage_removed_component_pixel_count == 0


def test_executor_decodes_coverage_and_passes_typed_policy_to_union():
    source = inspect.getsource(executor_core._render_to_reservations)

    assert "read_staged_alpha_coverage" in source
    assert "add_coverage" in source
    assert "coverage_policy=execution_settings.projection_coverage_policy" in source


def test_explicit_coverage_threshold_policy_is_available_without_morphology():
    policy = ProjectionCoveragePolicy(
        mode=ProjectionCoverageMode.COVERAGE_THRESHOLD,
        core_alpha_threshold=0.0,
        minimum_component_pixels=1,
        maximum_hole_pixels=0,
    )

    assert policy.mode is ProjectionCoverageMode.COVERAGE_THRESHOLD
