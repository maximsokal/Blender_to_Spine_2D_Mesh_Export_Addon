from math import inf, nan

import pytest

from Blender_to_Spine2D_Mesh_Exporter.domain.spine.spine_scalar_contract import (
    is_finite_number,
    require_name,
)


@pytest.mark.parametrize("value", ("name", " name ", "a.b", "Имя"))
def test_require_name_returns_exact_non_empty_string(value):
    assert require_name(value) == value


@pytest.mark.parametrize("value", ("", " ", "\t", "\n"))
def test_require_name_rejects_empty_or_whitespace_only_strings(value):
    with pytest.raises(ValueError, match="field cannot be empty"):
        require_name(value, "field")


@pytest.mark.parametrize("value", (None, True, 1, 1.0, (), []))
def test_require_name_rejects_non_strings(value):
    with pytest.raises(TypeError, match="field must be str"):
        require_name(value, "field")


@pytest.mark.parametrize("value", (0, -1, 2**128, 0.0, -3.5, 1e300))
def test_is_finite_number_accepts_non_boolean_finite_numbers(value):
    assert is_finite_number(value) is True


@pytest.mark.parametrize(
    "value",
    (True, False, None, "1", (), [], complex(1, 0), inf, -inf, nan),
)
def test_is_finite_number_rejects_booleans_non_numbers_and_non_finite_floats(value):
    assert is_finite_number(value) is False
