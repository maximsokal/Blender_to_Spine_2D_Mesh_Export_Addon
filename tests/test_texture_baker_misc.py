"""Legacy texture baker remains repository-only and excluded from Rewrite runtime."""

from pathlib import Path
import tomllib


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "Blender_to_Spine2D_Mesh_Exporter"


def test_legacy_texture_baker_sources_are_not_imported_by_rewrite_entry_point():
    entry = (PACKAGE / "__init__.py").read_text(encoding="utf-8")
    assert "texture_baker" not in entry
    assert "texture_baker_integration" not in entry


def test_legacy_texture_baker_sources_are_excluded_from_extension_package():
    with (PACKAGE / "blender_manifest.toml").open("rb") as stream:
        manifest = tomllib.load(stream)
    excluded = frozenset(manifest["build"]["paths_exclude_pattern"])
    assert "/texture_baker.py" in excluded
    assert "/texture_baker_integration.py" in excluded


def test_legacy_texture_baker_files_remain_available_for_explicit_migration_only():
    assert (PACKAGE / "texture_baker.py").is_file()
    assert (PACKAGE / "texture_baker_integration.py").is_file()
