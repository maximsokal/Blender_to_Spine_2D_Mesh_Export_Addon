from __future__ import annotations

from dataclasses import replace
import inspect

import pytest

import Blender_to_Spine2D_Mesh_Exporter.blender_adapter.camera_projection_executor_core as executor_core
from Blender_to_Spine2D_Mesh_Exporter.domain.baking import BakeExecutionSettings


def test_projection_alpha_threshold_preserves_legacy_default():
    settings = BakeExecutionSettings()

    assert settings.projection_alpha_threshold == 1.0 / 255.0


def test_projection_alpha_threshold_accepts_explicit_output_policy():
    settings = replace(
        BakeExecutionSettings(),
        projection_alpha_threshold=0.125,
    )

    assert settings.projection_alpha_threshold == 0.125


@pytest.mark.parametrize(
    "value",
    (
        -0.0001,
        1.0001,
        float("inf"),
        float("-inf"),
        float("nan"),
        "0.5",
        None,
        False,
        True,
    ),
)
def test_projection_alpha_threshold_rejects_invalid_values(value):
    with pytest.raises(ValueError, match="projection_alpha_threshold"):
        BakeExecutionSettings(projection_alpha_threshold=value)


def test_b4_executor_uses_one_execution_threshold_for_mask_and_layout():
    source = inspect.getsource(executor_core._render_to_reservations)
    module_source = inspect.getsource(executor_core)

    assert "_ALPHA_THRESHOLD" not in module_source
    assert "execution_settings.projection_alpha_threshold" in source
    assert "alpha_threshold=alpha_threshold" in source
    assert "threshold=alpha_threshold" in source
