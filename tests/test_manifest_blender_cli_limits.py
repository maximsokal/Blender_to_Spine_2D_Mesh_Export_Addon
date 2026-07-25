"""Local guards for Blender Extension manifest limits enforced by the CLI."""

from __future__ import annotations

from pathlib import Path

from tools import prepare_package


MAX_PERMISSION_DESCRIPTION_LENGTH = 64


def test_repository_permission_descriptions_fit_blender_cli_limit():
    root = Path(__file__).resolve().parents[1]
    manifest = prepare_package._read_manifest(
        root / "Blender_to_Spine2D_Mesh_Exporter"
    )

    permissions = manifest.get("permissions", {})
    assert permissions
    assert all(
        isinstance(description, str)
        and description.strip()
        and len(description) <= MAX_PERMISSION_DESCRIPTION_LENGTH
        for description in permissions.values()
    )
