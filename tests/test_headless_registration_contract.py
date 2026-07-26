from pathlib import Path


PACKAGE = Path(__file__).resolve().parents[1] / "Blender_to_Spine2D_Mesh_Exporter"


def test_root_registration_does_not_index_enabled_addon_preferences_directly():
    source = (PACKAGE / "__init__.py").read_text(encoding="utf-8")

    assert "preferences.addons[__name__]" not in source
    assert "def _initialize_registered_logging()" in source
    assert "prefs = config._addon_preferences()" in source
    assert "if prefs is None:" in source
    assert "config.setup_logging()" in source
    assert "_initialize_registered_logging()" in source
