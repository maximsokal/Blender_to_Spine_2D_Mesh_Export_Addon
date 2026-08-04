"""Pure settings contracts for Depth Camera Projection parallax reserve."""

from __future__ import annotations

from math import pi, radians

import pytest

from Blender_to_Spine2D_Mesh_Exporter.domain.baking import (
    BakeExecutionSettings,
    DepthParallaxSettings,
)


def test_default_parallax_setting_preserves_front_only_behavior() -> None:
    settings = DepthParallaxSettings()

    assert settings.horizon_angle_radians == 0.0
    assert not settings.enabled
    assert BakeExecutionSettings().depth_parallax == settings


def test_positive_angle_is_stored_in_radians_and_enables_reserve() -> None:
    settings = DepthParallaxSettings(radians(35.0))

    assert settings.horizon_angle_radians == pytest.approx(radians(35.0))
    assert settings.enabled


@pytest.mark.parametrize("value", (True, "30", None))
def test_non_numeric_horizon_values_are_rejected(value: object) -> None:
    with pytest.raises(TypeError, match="horizon_angle_radians"):
        DepthParallaxSettings(value)  # type: ignore[arg-type]


@pytest.mark.parametrize("value", (-0.001, pi / 2.0, pi, float("inf"), float("nan")))
def test_out_of_range_horizon_values_are_rejected(value: float) -> None:
    with pytest.raises(ValueError, match="horizon_angle_radians"):
        DepthParallaxSettings(value)


def test_bake_execution_rejects_untyped_parallax_settings() -> None:
    with pytest.raises(TypeError, match="depth_parallax"):
        BakeExecutionSettings(depth_parallax=object())  # type: ignore[arg-type]
