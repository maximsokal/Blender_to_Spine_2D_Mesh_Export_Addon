from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPINE = ROOT / "Blender_to_Spine2D_Mesh_Exporter" / "domain" / "spine"
SCALAR = SPINE / "spine_scalar_contract.py"
MODEL = SPINE / "model.py"
ANIMATION = SPINE / "animation_model_contract.py"
SETUP_SLOT = SPINE / "setup_slot_contract.py"
SETUP_ATTACHMENT = SPINE / "setup_attachment_contract.py"
CURVE = SPINE / "curve_timeline_contract.py"
SLOT_COLOR = SPINE / "slot_color_timeline_contract.py"


def read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def test_scalar_contract_is_blender_and_model_independent():
    source = read(SCALAR)

    assert "import bpy" not in source
    assert "from .model import" not in source
    assert "from .animation_model_contract import" not in source
    assert "def require_name(" in source
    assert "def is_finite_number(" in source
    assert "def require_finite_number(" in source
    assert (
        '__all__ = ["is_finite_number", "require_finite_number", "require_name"]'
        in source
    )


def test_name_contract_preserves_spelling_and_exact_diagnostics():
    source = read(SCALAR)
    helper = source[source.index("def require_name(") : source.index("def is_finite_number(")]

    assert "isinstance(value, str)" in helper
    assert "if not value.strip():" in helper
    assert 'raise TypeError(f"{field_name} must be str")' in helper
    assert 'raise ValueError(f"{field_name} cannot be empty")' in helper
    assert "return value" in helper
    assert ".lower()" not in helper
    assert ".upper()" not in helper


def test_finite_number_contract_excludes_bool_and_non_finite_float():
    source = read(SCALAR)
    helper = source[
        source.index("def is_finite_number(") : source.index("def require_finite_number(")
    ]

    assert "isinstance(value, bool)" in helper
    assert "not isinstance(value, (int, float))" in helper
    assert "return isinstance(value, int) or isfinite(value)" in helper


def test_required_finite_number_reuses_predicate_and_preserves_diagnostics():
    source = read(SCALAR)
    helper = source[source.index("def require_finite_number(") : source.index("__all__")]

    assert "isinstance(value, bool)" in helper
    assert "not isinstance(value, (int, float))" in helper
    assert 'raise TypeError(f"{field_name} must be a finite number")' in helper
    assert "if not is_finite_number(value):" in helper
    assert 'raise ValueError(f"{field_name} must be finite")' in helper
    assert "return value" in helper


def test_model_and_animation_alias_shared_scalar_helpers():
    for path in (MODEL, ANIMATION):
        source = read(path)
        assert "from .spine_scalar_contract import (" in source
        assert "is_finite_number as _is_finite_number" in source
        assert "require_name as _require_name" in source
        assert "def _require_name(" not in source
        assert "def _is_finite_number(" not in source
        assert "from math import isfinite" not in source


def test_setup_indexes_alias_shared_name_helper_only():
    for path in (SETUP_SLOT, SETUP_ATTACHMENT):
        source = read(path)
        assert (
            "from .spine_scalar_contract import require_name as _require_name"
            in source
        )
        assert "def _require_name(" not in source
        assert "_require_name(" in source
        assert "_is_finite_number" not in source


def test_output_timeline_contracts_alias_shared_finite_requirement():
    for path in (CURVE, SLOT_COLOR):
        source = read(path)
        assert (
            "from .spine_scalar_contract import "
            "require_finite_number as _require_finite_number"
            in source
        )
        assert "def _require_finite_number(" not in source
        assert "from math import isfinite" not in source
