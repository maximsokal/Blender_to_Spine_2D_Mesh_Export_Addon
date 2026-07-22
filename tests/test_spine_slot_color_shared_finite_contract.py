from copy import deepcopy
from math import inf, nan

import pytest

import Blender_to_Spine2D_Mesh_Exporter.domain.spine.slot_color_timeline_contract as color_contract
from Blender_to_Spine2D_Mesh_Exporter.domain.spine.spine_scalar_contract import (
    require_finite_number,
)


def validate(animations):
    color_contract.validate_animation_slot_color_timelines(
        animations,
        slot_names=("slot",),
        path="document.animations",
    )


def test_color_contract_holds_exact_shared_requirement_function():
    assert color_contract._require_finite_number is require_finite_number


@pytest.mark.parametrize("value", (True, False, None, "0", (), {}))
def test_color_time_preserves_strict_numeric_type_diagnostic(value):
    with pytest.raises(
        TypeError,
        match=r"document\.animations\.idle\.slots\.slot\.rgba\[0\]\.time "
        r"must be a finite number",
    ):
        validate(
            {
                "idle": {
                    "slots": {
                        "slot": {
                            "rgba": [
                                {"time": value, "color": "FFFFFFFF"},
                            ]
                        }
                    }
                }
            }
        )


@pytest.mark.parametrize("value", (inf, -inf, nan))
def test_alpha_value_preserves_non_finite_diagnostic(value):
    with pytest.raises(
        ValueError,
        match=r"document\.animations\.idle\.slots\.slot\.alpha\[0\]\.value "
        r"must be finite",
    ):
        validate(
            {
                "idle": {
                    "slots": {
                        "slot": {
                            "alpha": [{"value": value}],
                        }
                    }
                }
            }
        )


@pytest.mark.parametrize("value", (-10, -1.5, 0, 1, 1.5, 100))
def test_alpha_value_keeps_unbounded_finite_semantics_without_mutation(value):
    animations = {
        "idle": {
            "slots": {
                "slot": {
                    "alpha": [{"value": value}],
                }
            }
        }
    }
    source = deepcopy(animations)

    validate(animations)

    assert animations == source
