from math import inf, nan

import pytest

import Blender_to_Spine2D_Mesh_Exporter.domain.spine.curve_timeline_contract as curve_contract
from Blender_to_Spine2D_Mesh_Exporter.domain.spine.spine_scalar_contract import (
    require_finite_number,
)


@pytest.mark.parametrize("value", (0, -1, 2**128, 0.0, -3.5, 1e300))
def test_require_finite_number_returns_exact_value_and_type(value):
    result = require_finite_number(value, "field")

    assert result == value
    assert type(result) is type(value)


@pytest.mark.parametrize("value", (True, False, None, "1", (), [], complex(1, 0)))
def test_require_finite_number_rejects_non_numeric_values(value):
    with pytest.raises(TypeError, match="field must be a finite number"):
        require_finite_number(value, "field")


@pytest.mark.parametrize("value", (inf, -inf, nan))
def test_require_finite_number_rejects_non_finite_floats(value):
    with pytest.raises(ValueError, match="field must be finite"):
        require_finite_number(value, "field")


def test_curve_contract_holds_exact_shared_requirement_function():
    assert curve_contract._require_finite_number is require_finite_number


def test_curve_control_and_time_diagnostics_are_preserved():
    with pytest.raises(TypeError, match=r"curve\[0\] must be a finite number"):
        curve_contract.validate_curve_value(
            [True, 0, 1, 1],
            channel_count=1,
            path="curve",
        )

    with pytest.raises(
        ValueError,
        match=(
            r"animations\.idle\.bones\.root\.rotate\[0\]\.time "
            r"must be finite"
        ),
    ):
        curve_contract.validate_animation_curves(
            {
                "idle": {
                    "bones": {
                        "root": {
                            "rotate": [
                                {"time": float("inf")},
                            ]
                        }
                    }
                }
            },
            path="animations",
        )
