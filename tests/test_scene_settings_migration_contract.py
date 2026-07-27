from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "Blender_to_Spine2D_Mesh_Exporter"


def test_scene_schema_property_is_hidden_and_starts_unmigrated():
    source = (PACKAGE / "blender_adapter" / "scene_properties.py").read_text(
        encoding="utf-8"
    )

    assert '"spine2d_settings_schema_version"' in source
    assert "default=0" in source
    assert 'options={"HIDDEN"}' in source
    assert 'default="AUTO"' in source
    assert "update=_update_seam_maker_mode" in source


def test_schema_three_repairs_custom_values_marked_current_during_registration():
    migration_source = (
        PACKAGE / "blender_adapter" / "scene_settings_migration.py"
    ).read_text(encoding="utf-8")
    property_source = (
        PACKAGE / "blender_adapter" / "scene_properties.py"
    ).read_text(encoding="utf-8")

    guard = "if current >= CURRENT_SETTINGS_SCHEMA_VERSION:"
    reset = 'scene.spine2d_seam_maker_mode = "AUTO"'
    marker = "scene.spine2d_settings_schema_version = CURRENT_SETTINGS_SCHEMA_VERSION"

    assert "CURRENT_SETTINGS_SCHEMA_VERSION = 3" in migration_source
    assert (
        migration_source.index(guard)
        < migration_source.index(reset)
        < migration_source.index(marker)
    )
    assert "Schema 3 repairs Scenes" in migration_source
    assert "_extension_registration_active()" in property_source
    assert "migration_file_loading()" in property_source


def test_root_registers_migration_after_scene_rna_and_before_ui():
    source = (PACKAGE / "__init__.py").read_text(encoding="utf-8")

    assert source.index('"Scene RNA properties"') < source.index(
        '"Scene settings migration"'
    ) < source.index('"UI"')
