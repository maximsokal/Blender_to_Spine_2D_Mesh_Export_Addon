from dataclasses import replace

import pytest

from Blender_to_Spine2D_Mesh_Exporter.domain.baking import (
    ProjectionAlphaUnionAccumulator,
    ProjectionCoverageMode,
    ProjectionCoveragePolicy,
    build_projection_coverage_mask,
)


def _coverage(width, height, values):
    result = bytearray(width * height)
    for (x, y), value in values.items():
        result[y * width + x] = value
    return bytes(result)


def _visible_coordinates(mask, width):
    return {
        (index % width, index // width)
        for index, value in enumerate(mask)
        if value
    }


def test_default_policy_enables_hysteresis_and_conservative_morphology():
    policy = ProjectionCoveragePolicy()

    assert policy.mode is ProjectionCoverageMode.HYSTERESIS_MORPHOLOGY
    assert policy.core_alpha_threshold == 0.5
    assert policy.minimum_component_pixels == 2
    assert policy.maximum_hole_pixels == 1


@pytest.mark.parametrize(
    ("field_name", "value", "error_type"),
    (
        ("mode", "HYSTERESIS_MORPHOLOGY", TypeError),
        ("core_alpha_threshold", -0.1, ValueError),
        ("core_alpha_threshold", 1.1, ValueError),
        ("core_alpha_threshold", float("nan"), ValueError),
        ("core_alpha_threshold", True, TypeError),
        ("minimum_component_pixels", 0, ValueError),
        ("minimum_component_pixels", True, TypeError),
        ("maximum_hole_pixels", -1, ValueError),
        ("maximum_hole_pixels", False, TypeError),
    ),
)
def test_policy_rejects_invalid_values(field_name, value, error_type):
    with pytest.raises(error_type):
        replace(ProjectionCoveragePolicy(), **{field_name: value})


def test_hysteresis_keeps_connected_antialias_fringe_and_drops_weak_noise():
    width = 8
    coverage = _coverage(
        width,
        4,
        {
            (2, 1): 255,
            (1, 1): 24,
            (3, 1): 12,
            (7, 3): 18,
        },
    )
    result = build_projection_coverage_mask(
        coverage,
        width=width,
        height=4,
        fringe_alpha_threshold=1.0 / 255.0,
        policy=ProjectionCoveragePolicy(
            minimum_component_pixels=1,
            maximum_hole_pixels=0,
        ),
    )

    assert _visible_coordinates(result.mask, width) == {(1, 1), (2, 1), (3, 1)}
    assert result.raw_nonzero_pixel_count == 4
    assert result.strong_pixel_count == 1
    assert not result.used_weak_only_fallback


def test_translucent_only_object_uses_weak_coverage_fallback():
    width = 5
    coverage = _coverage(
        width,
        3,
        {(1, 1): 40, (2, 1): 55, (3, 1): 35},
    )
    result = build_projection_coverage_mask(
        coverage,
        width=width,
        height=3,
        fringe_alpha_threshold=1.0 / 255.0,
        policy=ProjectionCoveragePolicy(
            core_alpha_threshold=0.75,
            minimum_component_pixels=2,
            maximum_hole_pixels=0,
        ),
    )

    assert _visible_coordinates(result.mask, width) == {(1, 1), (2, 1), (3, 1)}
    assert result.strong_pixel_count == 0
    assert result.used_weak_only_fallback


def test_component_cleanup_removes_one_pixel_speck_but_keeps_main_component():
    width = 8
    coverage = _coverage(
        width,
        4,
        {(1, 1): 255, (2, 1): 255, (7, 3): 255},
    )
    result = build_projection_coverage_mask(
        coverage,
        width=width,
        height=4,
        fringe_alpha_threshold=1.0 / 255.0,
        policy=ProjectionCoveragePolicy(
            minimum_component_pixels=2,
            maximum_hole_pixels=0,
        ),
    )

    assert _visible_coordinates(result.mask, width) == {(1, 1), (2, 1)}
    assert result.component_count_before_cleanup == 2
    assert result.component_count_after_cleanup == 1
    assert result.removed_component_pixel_count == 1


def test_largest_one_pixel_object_is_never_deleted():
    result = build_projection_coverage_mask(
        _coverage(3, 3, {(1, 1): 255}),
        width=3,
        height=3,
        fringe_alpha_threshold=1.0 / 255.0,
        policy=ProjectionCoveragePolicy(
            minimum_component_pixels=8,
            maximum_hole_pixels=0,
        ),
    )

    assert result.visible_pixel_count == 1
    assert result.removed_component_pixel_count == 0


def test_diagonal_antialias_stroke_is_one_foreground_component():
    width = 5
    coordinates = {(1, 1), (2, 2), (3, 3)}
    result = build_projection_coverage_mask(
        _coverage(width, 5, {coordinate: 255 for coordinate in coordinates}),
        width=width,
        height=5,
        fringe_alpha_threshold=1.0 / 255.0,
        policy=ProjectionCoveragePolicy(
            minimum_component_pixels=3,
            maximum_hole_pixels=0,
        ),
    )

    assert _visible_coordinates(result.mask, width) == coordinates
    assert result.component_count_before_cleanup == 1
    assert result.component_count_after_cleanup == 1


def test_single_pixel_pinhole_is_filled_without_closing_open_background():
    width = 5
    ring = {
        (x, y): 255
        for y in range(1, 4)
        for x in range(1, 4)
        if (x, y) != (2, 2)
    }
    result = build_projection_coverage_mask(
        _coverage(width, 5, ring),
        width=width,
        height=5,
        fringe_alpha_threshold=1.0 / 255.0,
        policy=ProjectionCoveragePolicy(
            minimum_component_pixels=1,
            maximum_hole_pixels=1,
        ),
    )

    assert result.visible_pixel_count == 9
    assert (2, 2) in _visible_coordinates(result.mask, width)
    assert result.filled_hole_pixel_count == 1
    assert (0, 0) not in _visible_coordinates(result.mask, width)


def test_morphology_does_not_bridge_two_valid_components():
    width = 8
    coordinates = {(1, 1), (1, 2), (6, 1), (6, 2)}
    result = build_projection_coverage_mask(
        _coverage(width, 4, {coordinate: 255 for coordinate in coordinates}),
        width=width,
        height=4,
        fringe_alpha_threshold=1.0 / 255.0,
        policy=ProjectionCoveragePolicy(
            minimum_component_pixels=2,
            maximum_hole_pixels=4,
        ),
    )

    assert _visible_coordinates(result.mask, width) == coordinates
    assert result.component_count_after_cleanup == 2
    assert result.filled_hole_pixel_count == 0


def test_binary_mask_mode_preserves_nonzero_compatibility_values():
    width = 4
    binary = _coverage(width, 2, {(0, 0): 1, (3, 1): 1})
    result = build_projection_coverage_mask(
        binary,
        width=width,
        height=2,
        fringe_alpha_threshold=0.9,
        policy=ProjectionCoveragePolicy(
            mode=ProjectionCoverageMode.BINARY_THRESHOLD,
            core_alpha_threshold=0.0,
            minimum_component_pixels=1,
            maximum_hole_pixels=0,
        ),
    )

    assert _visible_coordinates(result.mask, width) == {(0, 0), (3, 1)}


def test_coverage_threshold_mode_uses_normalized_byte_threshold():
    width = 3
    result = build_projection_coverage_mask(
        bytes((1, 127, 128)),
        width=width,
        height=1,
        fringe_alpha_threshold=0.5,
        policy=ProjectionCoveragePolicy(
            mode=ProjectionCoverageMode.COVERAGE_THRESHOLD,
            core_alpha_threshold=0.0,
            minimum_component_pixels=1,
            maximum_hole_pixels=0,
        ),
    )

    assert _visible_coordinates(result.mask, width) == {(2, 0)}


def test_accumulator_max_unions_coverage_before_one_global_cleanup():
    policy = ProjectionCoveragePolicy(
        core_alpha_threshold=0.5,
        minimum_component_pixels=1,
        maximum_hole_pixels=0,
    )
    accumulator = ProjectionAlphaUnionAccumulator(
        width=5,
        height=3,
        alpha_threshold=1.0 / 255.0,
        padding_pixels=0,
        coverage_policy=policy,
    )
    accumulator.add_coverage(
        _coverage(5, 3, {(2, 1): 255, (1, 1): 10}),
        frame_index=0,
    )
    accumulator.add_coverage(
        _coverage(5, 3, {(2, 1): 128, (3, 1): 20}),
        frame_index=1,
    )

    layout = accumulator.build_layout()

    assert layout.frame_count == 2
    assert layout.coverage_mode is ProjectionCoverageMode.HYSTERESIS_MORPHOLOGY
    assert layout.coverage_raw_nonzero_pixel_count == 3
    assert layout.coverage_strong_pixel_count == 1
    assert layout.visible_pixel_count == 3
    assert layout.coverage_component_count_after_cleanup == 1
