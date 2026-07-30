"""Release-scope contract for extension candidate 0.47.10."""

from __future__ import annotations

from pathlib import Path
import tomllib

from Blender_to_Spine2D_Mesh_Exporter.domain.spine.export_capabilities import (
    SpineJsonExportScope,
    require_spine_json_export_capability,
)
from Blender_to_Spine2D_Mesh_Exporter.domain.spine.rig_profiles import A1RigProfile
from Blender_to_Spine2D_Mesh_Exporter.domain.spine.version_target import SpineJsonTarget


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "Blender_to_Spine2D_Mesh_Exporter" / "blender_manifest.toml"


def test_release_manifest_is_04710() -> None:
    with MANIFEST.open("rb") as stream:
        manifest = tomllib.load(stream)

    assert manifest["version"] == "0.47.10"
    assert manifest["blender_version_min"] == "5.2.0"


def test_release_spine41_scope_is_limited_and_has_no_open_scale_limitation() -> None:
    capability = require_spine_json_export_capability(
        SpineJsonTarget.SPINE_4_1,
        A1RigProfile.TWO_AXIS_ROTATION_SCALE,
        SpineJsonExportScope.SINGLE_OBJECT,
    )

    assert capability.scopes == frozenset(
        {
            SpineJsonExportScope.SINGLE_OBJECT,
            SpineJsonExportScope.STANDALONE_MULTI_OBJECT,
        }
    )
    assert capability.limitations == ()


def test_release_spine41_exact_version_remains_4124() -> None:
    assert SpineJsonTarget.SPINE_4_1.exact_version == "4.1.24"
    assert SpineJsonTarget.SPINE_4_1.descriptor.serializer_ready is True
