"""Static contract for the Blender Extensions Platform submission package."""

from __future__ import annotations

from pathlib import Path
import tomllib


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "Blender_to_Spine2D_Mesh_Exporter"
MANIFEST = PACKAGE / "blender_manifest.toml"
LICENSE = ROOT / "LICENSE"
INIT = PACKAGE / "__init__.py"
ADDON_PREFERENCES = PACKAGE / "addon_preferences.py"

EXPECTED_TAGS = {"Import-Export", "Mesh", "UV", "Animation"}
EXPECTED_WEBSITE = "https://github.com/maximsokal/Blender_to_Spine_2D_Mesh_Export_Addon"


def _manifest() -> dict[str, object]:
    with MANIFEST.open("rb") as stream:
        return tomllib.load(stream)


def test_submission_manifest_has_public_listing_metadata() -> None:
    manifest = _manifest()

    assert manifest["schema_version"] == "1.0.0"
    assert manifest["id"] == "blender_to_spine2d_mesh_exporter"
    assert manifest["version"] == "0.130.0"
    assert manifest["name"] == "Blender to Spine2D Mesh Exporter"
    assert manifest["maintainer"] == "Maxim Sokolenko"
    assert manifest["website"] == EXPECTED_WEBSITE
    assert manifest["type"] == "add-on"
    assert manifest["blender_version_min"] == "5.2.0"
    assert manifest["platforms"] == ["windows-x64"]
    assert set(manifest["tags"]) == EXPECTED_TAGS
    assert manifest["license"] == ["SPDX:GPL-3.0-or-later"]
    assert manifest["copyright"] == ["2025-2026 Maxim Sokolenko"]
    assert str(manifest["tagline"]).strip()


def test_submission_permission_reason_is_store_safe() -> None:
    permissions = _manifest()["permissions"]

    assert set(permissions) == {"files"}
    reason = permissions["files"]
    assert isinstance(reason, str)
    assert reason == reason.strip()
    assert 1 <= len(reason) <= 64
    assert not reason.endswith((".", "!", "?"))


def test_submission_package_uses_extension_namespace_contract() -> None:
    init_source = INIT.read_text(encoding="utf-8")
    preferences_source = ADDON_PREFERENCES.read_text(encoding="utf-8")

    assert "bl_info" not in init_source
    assert "ADDON_ID = __package__" in preferences_source
    assert "sys.path" not in init_source
    assert "sys.path" not in preferences_source


def test_submission_has_gpl_license_source() -> None:
    source = LICENSE.read_text(encoding="utf-8")

    assert "GNU GENERAL PUBLIC LICENSE" in source
    assert "Version 3" in source
    assert "Maxim Sokolenko" in source
