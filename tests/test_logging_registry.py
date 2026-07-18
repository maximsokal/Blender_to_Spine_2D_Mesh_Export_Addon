from pathlib import Path

from Blender_to_Spine2D_Mesh_Exporter.infrastructure.logging_registry import (
    discover_python_modules,
    merge_module_levels,
    resolve_logger_name,
)


def test_discovery_includes_nested_python_files(tmp_path: Path):
    (tmp_path / "__init__.py").write_text("", encoding="utf-8")
    (tmp_path / "config.py").write_text("", encoding="utf-8")
    nested = tmp_path / "blender_adapter"
    nested.mkdir()
    (nested / "__init__.py").write_text("", encoding="utf-8")
    (nested / "a1_ui_bridge.py").write_text("", encoding="utf-8")

    modules = discover_python_modules(
        tmp_path,
        root_display_name="Blender_to_Spine2D_Mesh_Exporter",
    )

    assert modules == (
        "Blender_to_Spine2D_Mesh_Exporter",
        "blender_adapter",
        "blender_adapter.a1_ui_bridge",
        "config",
    )


def test_merge_preserves_relative_and_full_runtime_logger_levels():
    discovered = (
        "Blender_to_Spine2D_Mesh_Exporter",
        "blender_adapter.a1_ui_bridge",
        "config",
    )
    merged = merge_module_levels(
        discovered,
        {
            "bl_ext.user_default.Blender_to_Spine2D_Mesh_Exporter.config": "INFO",
            "blender_adapter.a1_ui_bridge": "DEBUG",
        },
        package_root="bl_ext.user_default.Blender_to_Spine2D_Mesh_Exporter",
        root_display_name="Blender_to_Spine2D_Mesh_Exporter",
    )

    assert tuple((item.module_name, item.level) for item in merged) == (
        ("Blender_to_Spine2D_Mesh_Exporter", "ERROR"),
        ("blender_adapter.a1_ui_bridge", "DEBUG"),
        ("config", "INFO"),
    )


def test_resolve_logger_name_uses_actual_runtime_package_prefix():
    package_root = "bl_ext.user_default.Blender_to_Spine2D_Mesh_Exporter"
    assert resolve_logger_name(
        "blender_adapter.a1_ui_bridge",
        package_root=package_root,
        root_display_name="Blender_to_Spine2D_Mesh_Exporter",
    ) == f"{package_root}.blender_adapter.a1_ui_bridge"
