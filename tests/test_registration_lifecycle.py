import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "Blender_to_Spine2D_Mesh_Exporter"
INFRASTRUCTURE = PACKAGE / "infrastructure"


def _source(name: str) -> str:
    return (PACKAGE / name).read_text(encoding="utf-8")


def _tree(name: str) -> ast.Module:
    path = PACKAGE / name
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def _function(tree: ast.Module, name: str) -> ast.FunctionDef:
    return next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == name
    )


def _called_names(function: ast.FunctionDef) -> set[str]:
    result = set()
    for node in ast.walk(function):
        if not isinstance(node, ast.Call):
            continue
        if isinstance(node.func, ast.Name):
            result.add(node.func.id)
        elif isinstance(node.func, ast.Attribute):
            result.add(node.func.attr)
    return result


def _function_source(name: str, function_name: str) -> str:
    source = _source(name)
    function = _function(_tree(name), function_name)
    segment = ast.get_source_segment(source, function)
    assert segment is not None
    return segment


def test_registration_infrastructure_is_blender_independent():
    """Shared rollback helpers may remain for owners with mixed Blender resources."""

    path = INFRASTRUCTURE / "blender_registration.py"
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    imported = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    assert "bpy" not in imported
    definitions = {
        node.name
        for node in tree.body
        if isinstance(node, (ast.ClassDef, ast.FunctionDef))
    }
    assert {
        "RegistrationCleanupError",
        "register_classes_transactionally",
        "register_rna_properties_transactionally",
        "unregister_all_best_effort",
    }.issubset(definitions)


def test_runtime_operators_route_directly_to_rewrite_services():
    single = _source("single_object_operator.py")
    ui = _source("ui.py")
    automatic = _source("auto_readiness.py")
    assert "export_active_object_a1" in single
    assert "load_legacy_single_backend" not in single
    assert "DEFAULT_SINGLE_BACKEND" not in single
    assert "export_active_object_a1" in ui
    assert "export_selected_objects_a1" in ui
    assert "export_active_object_a1" in automatic
    assert "export_selected_objects_a1" in automatic
    assert "DEFAULT_MULTI_BACKEND" not in ui
    assert "resolve_multi_backend" not in ui


def test_simple_class_owners_use_normal_blender_registration_pattern():
    """Simple class-only owners must not use the rejected generic cleanup framework."""

    for name in (
        "addon_preferences.py",
        "single_object_operator.py",
        "rig_ui.py",
        "ui_layout.py",
    ):
        source = _source(name)
        register_calls = _called_names(_function(_tree(name), "register"))
        unregister_calls = _called_names(_function(_tree(name), "unregister"))

        assert "register_class" in register_calls, name
        assert "unregister_class" in unregister_calls, name
        assert "register_classes_transactionally" not in source, name
        assert "unregister_all_best_effort" not in source, name
        assert "class_cleanup_actions" not in source, name


def test_mixed_resource_owners_keep_targeted_rollback_helpers():
    """Only owners with classes plus RNA/handlers retain transactional cleanup."""

    expectations = {
        "ui.py": {
            "register_classes_transactionally",
            "register_rna_properties_transactionally",
            "unregister_all_best_effort",
        },
        "blender_adapter/generated_material_ui.py": {
            "register_classes_transactionally",
            "register_rna_properties_transactionally",
            "unregister_all_best_effort",
        },
    }
    for name, expected_calls in expectations.items():
        tree = _tree(name)
        combined = _called_names(_function(tree, "register")) | _called_names(
            _function(tree, "unregister")
        )
        assert expected_calls.issubset(combined), name


def test_ui_rna_ownership_registers_classes_before_pointer_properties():
    source = _source("ui.py")
    assert source.index("register_classes_transactionally(") < source.index(
        "register_rna_properties_transactionally("
    )
    assert 'name="spine2d_bake_settings"' in source
    assert 'name="spine2d_connect_settings"' in source

    unregister_source = _function_source("ui.py", "unregister")
    assert unregister_source.index("rna_property_cleanup_actions") < unregister_source.index(
        "class_cleanup_actions"
    )


def test_root_uses_explicit_standard_registration_order():
    source = _source("__init__.py")
    register_source = _function_source("__init__.py", "register")
    unregister_source = _function_source("__init__.py", "unregister")

    assert "REGISTRATION_STEPS" not in source
    assert "ExtensionRegistrationState" not in source
    assert "unregister_all_best_effort" not in source
    assert "repolish_ui" not in source
    assert "for name, value in CONFIG_RNA_PROPERTIES" in source
    assert "setattr(bpy.types.Scene, name, value)" in source
    assert "delattr(bpy.types.Scene, name)" in source

    registration_calls = [
        "addon_preferences.register()",
        "_register_config_rna()",
        "scene_settings_migration.register()",
        "ui.register()",
        "rig_ui.register()",
        "a1_readiness_invalidation.register()",
        "auto_readiness.register()",
        "generated_material_ui.register()",
        "ui_layout.register()",
        "single_object_operator.register()",
    ]
    positions = [register_source.index(call) for call in registration_calls]
    assert positions == sorted(positions)

    unregistration_calls = [
        "single_object_operator.unregister()",
        "ui_layout.unregister()",
        "generated_material_ui.unregister()",
        "auto_readiness.unregister()",
        "a1_readiness_invalidation.unregister()",
        "rig_ui.unregister()",
        "ui.unregister()",
        "scene_settings_migration.unregister()",
        "_unregister_config_rna()",
        "addon_preferences.unregister()",
    ]
    positions = [unregister_source.index(call) for call in unregistration_calls]
    assert positions == sorted(positions)


def test_non_blender_root_lifecycle_is_harmless_noop():
    source = _source("__init__.py")
    assert "Outside Blender there is nothing to register." in source
    assert "Outside Blender there is nothing to unregister." in source
    assert 'raise RuntimeError("Blender bpy module is required' not in source


def test_auto_readiness_does_not_install_background_polling_lifecycle():
    source = _source("auto_readiness.py")
    register_source = _function_source("auto_readiness.py", "register")
    unregister_source = _function_source("auto_readiness.py", "unregister")

    # Compatibility helpers may remain while old tests/probes are migrated, but the
    # installed lifecycle must not schedule automatic analysis or depsgraph callbacks.
    assert "_register_timer()" not in register_source
    assert "_install_handlers()" not in register_source
    assert "_unregister_timer()" not in unregister_source
    assert "_remove_handlers()" not in unregister_source
    assert "do not install timers" in register_source
    assert "do not install timers" in source


def test_preferences_own_and_release_one_shot_redraw_timer():
    source = _source("addon_preferences.py")
    unregister_source = _function_source("addon_preferences.py", "unregister")

    assert "bpy.app.timers one-shot callback" in source
    assert "_cancel_deferred_view3d_redraw()" in unregister_source


def test_root_startup_imports_no_legacy_implementation_module():
    source = _source("__init__.py")
    for forbidden in (
        "from . import main",
        "from . import json_export",
        "from . import texture_baker",
        "from . import plane_cut",
        "install_legacy_multi_facade",
        "load_legacy_single_backend",
    ):
        assert forbidden not in source
