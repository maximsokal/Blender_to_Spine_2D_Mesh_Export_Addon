"""Static regressions for APIs and runtime bridges retired before Blender 5.2."""

from __future__ import annotations

from pathlib import Path
import tomllib


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
    PACKAGE / "config.py",
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
                findings.append(
                    f"{path.relative_to(ROOT)}:{line_number}: {line.strip()}"
                )
    return tuple(findings)


def test_rewrite_contains_no_removed_or_fuzzy_renderer_identifiers():
    assert _occurrences("BLENDER_EEVEE_NEXT") == ()
    assert _occurrences('if "CYCLE" in normalized') == ()
    assert _occurrences('if "EEVEE" in normalized') == ()
    assert _occurrences('if "CYCLE" in target') == ()
    assert _occurrences('if "EEVEE" in target') == ()


def test_rewrite_does_not_mutate_material_or_world_use_nodes():
    assert _occurrences(".use_nodes =") == ()


def test_rewrite_does_not_use_removed_scene_compositor_node_tree():
    assert _occurrences("scene.node_tree") == ()


def test_rewrite_does_not_use_removed_action_fcurve_collections():
    assert _occurrences(".fcurves") == ()
    assert _occurrences("action.groups") == ()
    assert _occurrences("action.pose_markers") == ()


def test_rewrite_does_not_use_retired_mesh_edge_flags():
    assert _occurrences(".use_seam") == ()
    assert _occurrences(".use_edge_sharp") == ()


def test_rewrite_does_not_use_legacy_mesh_color_normal_or_tessface_api():
    forbidden = (
        ".vertex_colors",
        ".uv_textures",
        ".tessfaces",
        ".use_auto_smooth",
        ".calc_normals(",
        ".calc_normals_split(",
        ".calc_tessface(",
    )
    for token in forbidden:
        assert _occurrences(token) == (), token


def test_rewrite_does_not_use_legacy_uv_loop_data_coordinates():
    assert _occurrences("uv_layers.active.data") == ()
    assert _occurrences("layer.data[mesh_loop_index].uv") == ()
    assert _occurrences("layer.data[loop_index].uv") == ()


def test_rewrite_does_not_use_old_node_tree_group_interface_collections():
    assert _occurrences("node_tree.inputs") == ()
    assert _occurrences("node_tree.outputs") == ()


def test_rewrite_does_not_use_removed_opengl_or_old_operator_override_surface():
    assert _occurrences("import bgl") == ()
    assert _occurrences("from bgl import") == ()
    assert _occurrences(".cycles_visibility") == ()
    assert _occurrences("bpy.context.copy()") == ()


def test_extension_entry_point_contains_no_legacy_metadata_or_uninstall_ops():
    entry = (PACKAGE / "__init__.py").read_text(encoding="utf-8")
    preferences = (PACKAGE / "addon_preferences.py").read_text(encoding="utf-8")

    assert "bl_info" not in entry
    assert "addon_disable" not in preferences
    assert "addon_remove" not in preferences


def test_runtime_surface_contains_no_legacy_import_bridge():
    assert _occurrences("legacy_loader") == ()
    assert _occurrences("legacy_multi_facade") == ()
    assert _occurrences("load_legacy_single_backend") == ()
    assert _occurrences("MULTI_BACKEND_PROPERTY") == ()
    assert _occurrences("resolve_multi_backend") == ()
    assert _occurrences("a1_ui_rna") == ()
    assert not (PACKAGE / "blender_adapter" / "a1_ui_rna.py").exists()


def test_runtime_scene_properties_have_one_blender_52_owner():
    entry = (PACKAGE / "__init__.py").read_text(encoding="utf-8")
    config = (PACKAGE / "config.py").read_text(encoding="utf-8")
    properties = (
        PACKAGE / "blender_adapter" / "scene_properties.py"
    ).read_text(encoding="utf-8")

    assert "scene_properties.PROPERTIES" in entry
    assert "config.PROPERTIES" not in entry
    assert "PROPERTIES =" not in config
    assert "def register()" not in config
    assert "def unregister()" not in config
    assert 'self["spine2d_' not in config
    assert 'self["spine2d_' not in properties
    assert "config.get_texture_size" not in entry
    assert _occurrences('getattr(scene, "get"') == ()


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


def test_blender_52_package_excludes_every_pre_rewrite_module():
    with (PACKAGE / "blender_manifest.toml").open("rb") as stream:
        manifest = tomllib.load(stream)
    excluded = frozenset(manifest["build"]["paths_exclude_pattern"])

    required = {
        "/Legacy/",
        "/legacy_loader.py",
        "/legacy_multi_facade.py",
        "/json_export.py",
        "/json_merger.py",
        "/main.py",
        "/multi_object_export.py",
        "/plane_cut.py",
        "/seam_marker.py",
        "/texture_baker.py",
        "/texture_baker_integration.py",
        "/utils.py",
        "/uv_operations.py",
    }
    assert required <= excluded
