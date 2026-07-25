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
        node for node in ast.walk(tree)
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


def test_registration_infrastructure_is_blender_independent():
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
    assert "export_active_object_a1" in single
    assert "load_legacy_single_backend" not in single
    assert "DEFAULT_SINGLE_BACKEND" not in single
    assert "export_active_object_a1" in ui
    assert "export_selected_objects_a1" in ui
    assert "DEFAULT_MULTI_BACKEND" not in ui
    assert "resolve_multi_backend" not in ui


def test_registration_owners_use_only_needed_transaction_helpers():
    expectations = {
        "addon_preferences.py": {
            "register_classes_transactionally",
            "unregister_all_best_effort",
        },
        "single_object_operator.py": {
            "register_classes_transactionally",
            "unregister_all_best_effort",
        },
        "repolish_ui.py": {
            "register_classes_transactionally",
            "unregister_all_best_effort",
        },
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

    unregister = _function(_tree("ui.py"), "unregister")
    unregister_source = ast.get_source_segment(source, unregister)
    assert unregister_source.index("rna_property_cleanup_actions") < unregister_source.index(
        "class_cleanup_actions"
    )


def test_root_owns_scene_properties_and_orders_all_runtime_dependencies():
    tree = _tree("__init__.py")
    source = _source("__init__.py")
    assert "for name, prop in scene_properties.PROPERTIES" in source
    assert "register_rna_properties_transactionally(CONFIG_RNA_PROPERTIES)" in source
    assert "bpy.utils.register_class" not in source
    assert "bpy.utils.unregister_class" not in source

    steps_assignment = next(
        node for node in ast.walk(tree)
        if isinstance(node, ast.AnnAssign)
        and isinstance(node.target, ast.Name)
        and node.target.id == "REGISTRATION_STEPS"
    )
    labels = [ast.literal_eval(item.elts[0]) for item in steps_assignment.value.elts]
    assert labels == [
        "addon preferences",
        "Scene RNA properties",
        "UI",
        "Re-Polish UI",
        "generated material UI",
        "single-object operator",
    ]
    assert "unregister_all_best_effort" in _called_names(_function(tree, "register"))
    assert "unregister_all_best_effort" in _called_names(_function(tree, "unregister"))


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
