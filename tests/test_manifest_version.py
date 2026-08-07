"""Release-version contract for the Blender extension manifest."""

from __future__ import annotations

from pathlib import Path
import tomllib


MANIFEST = (
    Path(__file__).resolve().parents[1]
    / "Blender_to_Spine2D_Mesh_Exporter"
    / "blender_manifest.toml"
)
EXPECTED_RELEASE_VERSION = "0.130.0"


def test_extension_manifest_version_is_current_release() -> None:
    """Keep the packaged manifest pinned to the reviewed release candidate."""

    with MANIFEST.open("rb") as stream:
        manifest = tomllib.load(stream)

    assert manifest["version"] == EXPECTED_RELEASE_VERSION
