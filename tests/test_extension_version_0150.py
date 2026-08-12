"""Release-version contract for the unpublished 0.150.0 submission candidate."""

from __future__ import annotations

from pathlib import Path
import tomllib


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "Blender_to_Spine2D_Mesh_Exporter" / "blender_manifest.toml"


def test_extension_manifest_uses_version_0_150_0() -> None:
    with MANIFEST.open("rb") as stream:
        manifest = tomllib.load(stream)

    assert manifest["id"] == "blender_to_spine2d_mesh_exporter"
    assert manifest["version"] == "0.150.0"
