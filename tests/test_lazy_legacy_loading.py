import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "Blender_to_Spine2D_Mesh_Exporter"
MANIFEST = PACKAGE / "blender_manifest.toml"


def _tree(name: str) -> ast.Module:
    path = PACKAGE / name
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def _imported_leaf_names(name: str) -> set[str]:
    result = set()
    for node in ast.walk(_tree(name)):
        if isinstance(node, ast.ImportFrom):
            result.update(alias.name for alias in node.names)
        elif isinstance(node, ast.Import):
            result.update(alias.name.rsplit(".", 1)[-1] for alias in node.names)
    return result


def test_root_startup_imports_only_rewrite_registration_boundaries():
    tree = _tree("__init__.py")
    assignments = {
        target.id: node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        for target in node.targets
        if isinstance(target, ast.Name)
    }
    modules = assignments["MODULES"]
    assert isinstance(modules, ast.Tuple)
    assert [item.id for item in modules.elts] == [
        "addon_preferences",
        "scene_settings_migration",
        "ui",
        "rig_ui",
        "a1_readiness_invalidation",
        "auto_readiness",
        "repolish_ui",
        "generated_material_ui",
        "single_object_operator",
    ]


def test_runtime_operators_do_not_import_legacy_loaders():
    for filename in ("single_object_operator.py", "ui.py"):
        names = _imported_leaf_names(filename)
        assert "main" not in names
        assert "json_export" not in names
        assert "legacy_loader" not in names
        assert "load_legacy_single_backend" not in names
        assert "load_legacy_multi_backend" not in names


def test_retained_pre_rewrite_sources_are_excluded_from_extension_package():
    manifest = MANIFEST.read_text(encoding="utf-8")
    for path in (
        "/legacy_loader.py",
        "/main.py",
        "/multi_object_export.py",
        "/json_export.py",
        "/texture_baker.py",
        "/plane_cut.py",
    ):
        assert f'"{path}"' in manifest


def test_root_contains_no_automatic_legacy_fallback():
    source = (PACKAGE / "__init__.py").read_text(encoding="utf-8")
    assert "install_legacy_multi_facade" not in source
