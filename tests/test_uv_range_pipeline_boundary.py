from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "Blender_to_Spine2D_Mesh_Exporter"
UV_PREPARATION = PACKAGE / "blender_adapter" / "a1_uv_preparation.py"
DOCUMENT_PREPARATION = PACKAGE / "blender_adapter" / "a1_document_preparation.py"
DOCUMENT_ASSEMBLY = PACKAGE / "application" / "a1_document_assembly.py"
CAMERA_PROJECTION = PACKAGE / "application" / "a1_camera_projection.py"
UV_MODEL = PACKAGE / "domain" / "uv" / "model.py"
UV_RANGE = PACKAGE / "domain" / "uv" / "range.py"


def read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def test_uv_range_contract_is_owned_by_domain_uv():
    source = read(UV_RANGE)

    assert "class UvRangePolicy" in source
    assert "class UvRangeReport" in source
    assert "class UvRangeError" in source
    assert "def inspect_uv_range(" in source
    assert "def enforce_uv_range(" in source
    assert "The function intentionally does not clamp coordinates" in source


def test_unwrap_settings_own_explicit_policy_and_epsilon():
    source = read(UV_MODEL)

    assert "range_policy: UvRangePolicy" in source
    assert "range_epsilon: float" in source
    assert '"range_policy": UvRangePolicy' in source
    assert 'if numeric_values["range_epsilon"] < 0.0' in source


def test_shared_unwrap_is_diagnostic_until_bake_mode_is_known():
    source = read(UV_PREPARATION)

    inspect_index = source.index("range_report = inspect_uv_range(")
    propagate_index = source.index("uv_regions = propagate_texturing_uv_to_regions(")
    assert inspect_index < propagate_index
    assert "enforce_uv_range(" not in source
    assert "cannot yet know whether material planning" in source
    assert "uv_outside_range_tolerance" in source
    assert "uv.range_policy is WARN_ONLY" in source


def test_propagated_object_bake_regions_are_checked_before_attachment_projection():
    source = read(DOCUMENT_ASSEMBLY)

    validate_index = source.index("enforce_uv_range(")
    projection_index = source.index("projection = project_triangulated_disk_attachment(")
    assert validate_index < projection_index
    assert "policy=settings.uv_range_policy" in source
    assert "epsilon=settings.uv_range_epsilon" in source


def test_document_preparation_propagates_policy_without_reconstructing_it():
    source = read(DOCUMENT_PREPARATION)

    assert "uv_range_policy=source.settings.uv.range_policy" in source
    assert "uv_range_epsilon=source.settings.uv.range_epsilon" in source


def test_camera_projection_keeps_exporter_generated_uv_strict():
    source = read(CAMERA_PROJECTION)

    validate_index = source.index("enforce_uv_range(")
    return_index = source.index("return snapshot")
    assert validate_index < return_index
    assert "policy=UvRangePolicy.REQUIRE_UNIT_SQUARE" in source
    assert "epsilon=0.0" in source
