"""Release-version contract for the Blender extension manifest."""

from __future__ import annotations

from pathlib import Path
import re


MANIFEST = (
    Path(__file__).resolve().parents[1]
    / "Blender_to_Spine2D_Mesh_Exporter"
    / "blender_manifest.toml"
)


def test_extension_manifest_version_is_0_41_1():
    source = MANIFEST.read_text(encoding="utf-8")
    match = re.search(r'^version\s*=\s*"([^"]+)"\s*$', source, re.MULTILINE)

    assert match is not None
    assert match.group(1) == "0.41.1"
