from __future__ import annotations

from dataclasses import replace
import inspect

import pytest

import Blender_to_Spine2D_Mesh_Exporter.blender_adapter.camera_projection_postprocess as postprocess
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
    ("value", "error_type"),
    (
        (-0.0001, ValueError),
        (1.0001, ValueError),
        (float("inf"), ValueError),
        (float("-inf"), ValueError),
        (float("nan"), ValueError),
        ("0.5", TypeError),
        (None, TypeError),
        (False, TypeError),
        (True, TypeError),
    ),
)
def test_projection_alpha_threshold_rejects_invalid_values(value, error_type):
    with pytest.raises(error_type, match="projection_alpha_threshold"):
        BakeExecutionSettings(projection_alpha_threshold=value)


def test_b4_postprocess_uses_one_execution_threshold_for_coverage_layout():
    builder_source = inspect.getsource(
        postprocess.build_projection_union_accumulator
    )
    processor_source = inspect.getsource(
        postprocess.process_projection_outputs
    )
    module_source = inspect.getsource(postprocess)

    assert "_ALPHA_THRESHOLD" not in module_source
    assert (
        "execution_settings.projection_alpha_threshold"
        in builder_source
    )
    assert "alpha_threshold=float(" in builder_source
    assert "read_staged_alpha_coverage" in processor_source
