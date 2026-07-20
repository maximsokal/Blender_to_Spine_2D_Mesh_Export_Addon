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


def _assigned_string(tree: ast.Module, name: str) -> str:
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        if any(isinstance(target, ast.Name) and target.id == name for target in node.targets):
            return ast.literal_eval(node.value)
    raise AssertionError(f"assignment {name!r} not found")


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


def test_single_backend_has_one_rewrite_default_and_unknown_values_fail_closed():
    tree = _tree("single_object_operator.py")
    assert _assigned_string(tree, "DEFAULT_SINGLE_BACKEND") == "REWRITE"
    resolver = _function(tree, "resolve_single_backend")
    resolver_source = ast.get_source_segment(_source("single_object_operator.py"), resolver)
    assert "DEFAULT_SINGLE_BACKEND" in resolver_source
    assert "return DEFAULT_SINGLE_BACKEND" in resolver_source

    execute = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef)
        and node.name == "execute"
        and any(
            isinstance(parent, ast.ClassDef)
            and parent.name == "OBJECT_OT_SaveUVAsJSON"
            and node in parent.body
            for parent in ast.walk(tree)
        )
    )
    assert "resolve_single_backend" in _called_names(execute)
    source = _source("single_object_operator.py")
    assert 'getattr(context.scene, SINGLE_BACKEND_PROPERTY, "LEGACY")' not in source
    assert "default=DEFAULT_SINGLE_BACKEND" in source


def test_multi_backend_has_one_rewrite_default_and_unknown_values_fail_closed():
    tree = _tree("ui.py")
    assert _assigned_string(tree, "DEFAULT_MULTI_BACKEND") == "REWRITE"
    resolver = _function(tree, "resolve_multi_backend")
    resolver_source = ast.get_source_segment(_source("ui.py"), resolver)
    assert "DEFAULT_MULTI_BACKEND" in resolver_source
    assert "return DEFAULT_MULTI_BACKEND" in resolver_source
    source = _source("ui.py")
    assert 'getattr(scene, "spine2d_multi_export_backend", "LEGACY")' not in source
    assert "default=DEFAULT_MULTI_BACKEND" in source


def test_nested_registration_owners_use_common_transaction_helpers():
    expectations = {
        "addon_preferences.py": {
            "register_classes_transactionally",
            "unregister_all_best_effort",
        },
        "single_object_operator.py": {
            "register_classes_transactionally",
            "register_rna_properties_transactionally",
            "unregister_all_best_effort",
        },
        "ui.py": {
            "register_classes_transactionally",
            "register_rna_properties_transactionally",
            "unregister_all_best_effort",
        },
    }
    for name, expected_calls in expectations.items():
        tree = _tree(name)
        register_calls = _called_names(_function(tree, "register"))
        unregister_calls = _called_names(_function(tree, "unregister"))
        combined = register_calls | unregister_calls
        assert expected_calls.issubset(combined), name


def test_ui_rna_ownership_keeps_property_groups_before_pointer_properties():
    source = _source("ui.py")
    class_position = source.index("register_classes_transactionally(")
    property_position = source.index("register_rna_properties_transactionally(")
    assert class_position < property_position
    assert 'name="spine2d_bake_settings"' in source
    assert 'name="spine2d_connect_settings"' in source

    unregister = _function(_tree("ui.py"), "unregister")
    unregister_source = ast.get_source_segment(source, unregister)
    assert unregister_source.index("rna_property_cleanup_actions") < unregister_source.index(
        "class_cleanup_actions"
    )


def test_root_owns_config_properties_and_orders_registration_dependencies():
    tree = _tree("__init__.py")
    source = _source("__init__.py")
    assert "for name, prop in config.PROPERTIES" in source
    assert "register_rna_properties_transactionally(CONFIG_RNA_PROPERTIES)" in source
    assert "bpy.utils.register_class" not in source
    assert "bpy.utils.unregister_class" not in source

    steps_assignment = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.AnnAssign)
        and isinstance(node.target, ast.Name)
        and node.target.id == "REGISTRATION_STEPS"
    )
    assert isinstance(steps_assignment.value, ast.Tuple)
    labels = [ast.literal_eval(item.elts[0]) for item in steps_assignment.value.elts]
    assert labels == [
        "addon preferences",
        "config RNA properties",
        "UI",
        "single-object operator",
    ]

    register = _function(tree, "register")
    unregister = _function(tree, "unregister")
    assert "unregister_all_best_effort" in _called_names(register)
    assert "unregister_all_best_effort" in _called_names(unregister)


def test_root_startup_still_keeps_legacy_implementation_lazy():
    source = _source("__init__.py")
    assert "install_legacy_multi_facade()" in source
    for forbidden in (
        "from . import main",
        "from . import json_export",
        "from . import texture_baker",
        "from . import plane_cut",
    ):
        assert forbidden not in source
