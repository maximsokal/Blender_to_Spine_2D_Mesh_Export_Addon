"""Pure tests for the approved Normal / UV Segments projection domain."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
import math

import pytest

from Blender_to_Spine2D_Mesh_Exporter.domain.projection import (
    A1AxisDepthRange,
    A1AxisProjectionBasis,
    A1ProjectedPoint,
    A1ProjectionDirection,
    A1ProjectionError,
    calculate_a1_axis_depth_range,
    project_a1_axis_points,
    resolve_a1_axis_projection_basis,
    resolve_a1_projection_direction,
)


@pytest.mark.parametrize(
    ("direction", "expected"),
    (
        (A1ProjectionDirection.POSITIVE_X, (3.0, 5.0, 2.0)),
        (A1ProjectionDirection.NEGATIVE_X, (-3.0, 5.0, -2.0)),
        (A1ProjectionDirection.POSITIVE_Y, (-2.0, 5.0, 3.0)),
        (A1ProjectionDirection.NEGATIVE_Y, (2.0, 5.0, -3.0)),
        (A1ProjectionDirection.POSITIVE_Z, (2.0, 3.0, 5.0)),
        (A1ProjectionDirection.NEGATIVE_Z, (-2.0, 3.0, -5.0)),
    ),
)
def test_axis_projection_matches_approved_basis_table(
    direction: A1ProjectionDirection,
    expected: tuple[float, float, float],
) -> None:
    basis = resolve_a1_axis_projection_basis(direction)

    projected = basis.project_point((2.0, 3.0, 5.0))

    assert projected.canonical_position == expected
    assert basis.project_vector((2.0, 3.0, 5.0)) == expected
    assert basis.direction is direction


def test_positive_z_preserves_current_world_xyz_orientation() -> None:
    basis = resolve_a1_axis_projection_basis(A1ProjectionDirection.POSITIVE_Z)

    assert basis.project_point((-7.5, 2.25, -3.0)) == A1ProjectedPoint(
        u=-7.5,
        v=2.25,
        depth=-3.0,
    )


@pytest.mark.parametrize(
    ("direction", "label", "axis_aligned"),
    (
        (A1ProjectionDirection.POSITIVE_X, "+X", True),
        (A1ProjectionDirection.NEGATIVE_X, "-X", True),
        (A1ProjectionDirection.POSITIVE_Y, "+Y", True),
        (A1ProjectionDirection.NEGATIVE_Y, "-Y", True),
        (A1ProjectionDirection.POSITIVE_Z, "+Z", True),
        (A1ProjectionDirection.NEGATIVE_Z, "-Z", True),
        (A1ProjectionDirection.ACTIVE_CAMERA, "Active Camera", False),
    ),
)
def test_projection_direction_has_stable_label_and_kind(
    direction: A1ProjectionDirection,
    label: str,
    axis_aligned: bool,
) -> None:
    assert direction.label == label
    assert direction.axis_aligned is axis_aligned
    assert resolve_a1_projection_direction(direction.value) is direction
    assert resolve_a1_projection_direction(f"  {direction.value.lower()}  ") is direction


def test_projection_direction_rejects_invalid_values_without_fallback() -> None:
    with pytest.raises(TypeError, match="A1ProjectionDirection or str"):
        resolve_a1_projection_direction(None)

    with pytest.raises(ValueError, match="cannot be empty"):
        resolve_a1_projection_direction("   ")

    with pytest.raises(ValueError, match="Unsupported projection direction"):
        resolve_a1_projection_direction("TOP")


def test_active_camera_cannot_be_resolved_as_axis_basis() -> None:
    with pytest.raises(A1ProjectionError, match="evaluated Blender camera"):
        resolve_a1_axis_projection_basis(A1ProjectionDirection.ACTIVE_CAMERA)


def test_projected_point_localizes_against_projected_object_origin() -> None:
    origin = A1ProjectedPoint(u=10.0, v=-4.0, depth=8.0)
    vertex = A1ProjectedPoint(u=13.5, v=2.0, depth=5.0)

    assert vertex.relative_to(origin) == A1ProjectedPoint(
        u=3.5,
        v=6.0,
        depth=-3.0,
    )


def test_projected_point_normalizes_negative_zero() -> None:
    point = A1ProjectedPoint(u=-0.0, v=0.0, depth=-0.0)

    assert point.canonical_position == (0.0, 0.0, 0.0)
    assert math.copysign(1.0, point.u) == 1.0
    assert math.copysign(1.0, point.depth) == 1.0


def test_axis_depth_range_uses_nearest_max_and_farthest_min() -> None:
    points = (
        A1ProjectedPoint(0.0, 0.0, -4.0),
        A1ProjectedPoint(0.0, 0.0, 3.5),
        A1ProjectedPoint(0.0, 0.0, 1.0),
    )

    assert calculate_a1_axis_depth_range(points) == A1AxisDepthRange(
        nearest=3.5,
        farthest=-4.0,
    )


def test_axis_depth_range_rejects_empty_or_mixed_values() -> None:
    with pytest.raises(ValueError, match="cannot be empty"):
        calculate_a1_axis_depth_range(())

    with pytest.raises(TypeError, match="A1ProjectedPoint values"):
        calculate_a1_axis_depth_range((A1ProjectedPoint(0.0, 0.0, 0.0), object()))

    with pytest.raises(ValueError, match="nearest must be"):
        A1AxisDepthRange(nearest=-1.0, farthest=2.0)


def test_project_axis_points_preserves_input_order() -> None:
    world_points = ((1.0, 2.0, 3.0), (-4.0, 5.0, 6.0))

    result = project_a1_axis_points(
        world_points,
        A1ProjectionDirection.NEGATIVE_Y,
    )

    assert result == (
        A1ProjectedPoint(1.0, 3.0, -2.0),
        A1ProjectedPoint(-4.0, 6.0, -5.0),
    )


def test_projection_rejects_non_finite_and_malformed_points() -> None:
    basis = resolve_a1_axis_projection_basis(A1ProjectionDirection.POSITIVE_Z)

    with pytest.raises(TypeError, match="three-component"):
        basis.project_point((1.0, 2.0))

    with pytest.raises(TypeError, match="finite number"):
        basis.project_point((1.0, True, 3.0))

    with pytest.raises(ValueError, match="must be finite"):
        basis.project_point((1.0, float("nan"), 3.0))

    with pytest.raises(ValueError, match="cannot be empty"):
        project_a1_axis_points((), A1ProjectionDirection.POSITIVE_Z)


def test_axis_basis_rejects_non_orthonormal_or_left_handed_axes() -> None:
    with pytest.raises(ValueError, match="unit vectors"):
        A1AxisProjectionBasis(
            direction=A1ProjectionDirection.POSITIVE_Z,
            u_axis=(2.0, 0.0, 0.0),
            v_axis=(0.0, 1.0, 0.0),
            depth_axis=(0.0, 0.0, 1.0),
        )

    with pytest.raises(ValueError, match="right-handed"):
        A1AxisProjectionBasis(
            direction=A1ProjectionDirection.POSITIVE_Z,
            u_axis=(1.0, 0.0, 0.0),
            v_axis=(0.0, 1.0, 0.0),
            depth_axis=(0.0, 0.0, -1.0),
        )

    with pytest.raises(ValueError, match="ACTIVE_CAMERA"):
        A1AxisProjectionBasis(
            direction=A1ProjectionDirection.ACTIVE_CAMERA,
            u_axis=(1.0, 0.0, 0.0),
            v_axis=(0.0, 1.0, 0.0),
            depth_axis=(0.0, 0.0, 1.0),
        )


def test_projection_contracts_are_immutable() -> None:
    point = A1ProjectedPoint(1.0, 2.0, 3.0)
    basis = resolve_a1_axis_projection_basis(A1ProjectionDirection.POSITIVE_Z)

    with pytest.raises(FrozenInstanceError):
        point.u = 9.0  # type: ignore[misc]

    with pytest.raises(FrozenInstanceError):
        basis.u_axis = (0.0, 1.0, 0.0)  # type: ignore[misc]
