"""Static regressions for Blender APIs removed or changed before Blender 5.2."""

from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "Blender_to_Spine2D_Mesh_Exporter"
REWRITE_DIRECTORIES = (
    PACKAGE / "application",
    PACKAGE / "blender_adapter",
    PACKAGE / "domain",
    PACKAGE / "infrastructure",
)
ACTIVE_ROOT_FILES = (
    PACKAGE / "__init__.py",
    PACKAGE / "addon_preferences.py",
    PACKAGE / "single_object_operator.py",
    PACKAGE / "ui.py",
)


def _rewrite_sources() -> tuple[Path, ...]:
    files = list(ACTIVE_ROOT_FILES)
    for directory in REWRITE_DIRECTORIES:
        files.extend(directory.rglob("*.py"))
    return tuple(sorted(set(files), key=lambda path: path.as_posix().casefold()))


def _occurrences(token: str) -> tuple[str, ...]:
    findings: list[str] = []
    for path in _rewrite_sources():
        source = path.read_text(encoding="utf-8")
        for line_number, line in enumerate(source.splitlines(), start=1):
            if token in line:
                findings.append(f"{path.relative_to(ROOT)}:{line_number}: {line.strip()}")
    return tuple(findings)


def test_rewrite_contains_no_removed_eevee_next_runtime_id():
    assert _occurrences("BLENDER_EEVEE_NEXT") == ()


def test_rewrite_does_not_use_deprecated_material_or_world_use_nodes():
    assert _occurrences(".use_nodes") == ()


def test_rewrite_does_not_use_removed_scene_compositor_node_tree():
    assert _occurrences("scene.node_tree") == ()


def test_extension_entry_point_contains_no_legacy_metadata_or_uninstall_ops():
    entry = (PACKAGE / "__init__.py").read_text(encoding="utf-8")
    preferences = (PACKAGE / "addon_preferences.py").read_text(encoding="utf-8")

    assert "bl_info" not in entry
    assert "addon_disable" not in preferences
    assert "addon_remove" not in preferences


def test_working_color_space_is_captured_through_blender_52_interop_id():
    context = (PACKAGE / "domain" / "baking" / "context.py").read_text(
        encoding="utf-8"
    )
    resources = (
        PACKAGE / "blender_adapter" / "scene_bake_resources.py"
    ).read_text(encoding="utf-8")

    assert "working_space_interop_id" in context
    assert '"lin_rec709_scene"' in context
    assert "bpy.data" not in resources
    assert 'getattr(bpy, "data", None)' in resources
    assert '"working_space_interop_id"' in resources


def test_generated_palette_uses_display_rgb_and_color_managed_attribute_writes():
    materials = (
        PACKAGE / "blender_adapter" / "bake_materials.py"
    ).read_text(encoding="utf-8")
    generated_ui = (
        PACKAGE / "blender_adapter" / "generated_material_ui.py"
    ).read_text(encoding="utf-8")

    assert 'subtype="COLOR_GAMMA"' in generated_ui
    assert 'subtype="COLOR"' not in generated_ui
    assert "def _assign_generated_display_color(" in materials
    assert "attribute_value.color_srgb = resolved" in materials
    assert "attribute.data[mesh_loop_index].color = color" not in materials
    assert "Blender 5.2 FloatColorAttributeValue.color_srgb is unavailable" in materials
