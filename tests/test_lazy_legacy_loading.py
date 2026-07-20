import ast
import importlib.util
import sys
import types
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "Blender_to_Spine2D_Mesh_Exporter"


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


def _load_loader(package_dir: Path | None = None):
    package_name = "lazytest"
    package = types.ModuleType(package_name)
    package.__path__ = [str(package_dir or PACKAGE)]
    sys.modules[package_name] = package
    path = PACKAGE / "legacy_loader.py"
    name = package_name + ".legacy_loader"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_root_startup_imports_only_runtime_registration_boundaries():
    tree = _tree("__init__.py")
    source = (PACKAGE / "__init__.py").read_text(encoding="utf-8")
    for forbidden in (
        "main",
        "plane_cut",
        "uv_operations",
        "json_export",
        "json_merger",
        "texture_baker",
        "texture_baker_integration",
        "seam_marker",
    ):
        assert f"        {forbidden}," not in source
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
        "ui",
        "single_object_operator",
    ]
    assert "install_legacy_multi_facade" in source


def test_single_operator_does_not_import_legacy_implementation_at_module_load():
    names = _imported_leaf_names("single_object_operator.py")
    assert "main" not in names
    assert "json_export" not in names
    assert "load_legacy_single_backend" in names


def test_original_multi_source_remains_untouched_and_is_loaded_by_private_alias(tmp_path):
    source = (
        "from .dependency import VALUE\n"
        "def export_selected_objects(value=VALUE):\n"
        "    return value\n"
        "PRIVATE_NAME = 'legacy'\n"
    )
    (tmp_path / "multi_object_export.py").write_text(source, encoding="utf-8")
    package = types.ModuleType("lazytest")
    package.__path__ = [str(tmp_path)]
    sys.modules["lazytest"] = package
    dependency = types.ModuleType("lazytest.dependency")
    dependency.VALUE = "ok"
    sys.modules["lazytest.dependency"] = dependency

    loader = _load_loader(tmp_path)
    loader._MULTI_SOURCE_PATH = tmp_path / "multi_object_export.py"
    facade = loader.install_legacy_multi_facade()
    assert facade.__spine2d_lazy_legacy__ is True
    assert "lazytest._legacy_multi_object_export_impl" not in sys.modules
    assert facade.export_selected_objects() == "ok"
    assert facade.PRIVATE_NAME == "legacy"
    assert "lazytest._legacy_multi_object_export_impl" in sys.modules


def test_single_and_multi_legacy_routes_load_independently(monkeypatch):
    loader = _load_loader()
    main = types.ModuleType("main")
    main.save_uv_as_json = lambda *args, **kwargs: "result.json"
    json_export = types.ModuleType("json_export")
    multi = types.ModuleType("legacy_multi")
    multi.export_selected_objects = lambda *args, **kwargs: "multi.json"
    calls = []

    def fake_import(name, package):
        calls.append((name, package))
        return {".main": main, ".json_export": json_export}[name]

    monkeypatch.setattr(loader, "import_module", fake_import)
    single_backend = loader.load_legacy_single_backend()
    assert single_backend.main is main
    assert calls == [(".main", "lazytest"), (".json_export", "lazytest")]

    monkeypatch.setattr(loader, "_load_legacy_multi_module", lambda: multi)
    calls.clear()
    multi_backend = loader.load_legacy_multi_backend()
    assert multi_backend.export_selected_objects() == "multi.json"
    assert calls == []


def test_legacy_loader_fails_closed_when_entrypoint_is_missing(monkeypatch):
    loader = _load_loader()
    invalid = types.ModuleType("invalid")
    monkeypatch.setattr(loader, "import_module", lambda _name, _package: invalid)
    with pytest.raises(RuntimeError, match="save_uv_as_json"):
        loader.load_legacy_single_backend()
    monkeypatch.setattr(loader, "_load_legacy_multi_module", lambda: invalid)
    with pytest.raises(TypeError, match="callable"):
        loader.load_legacy_multi_backend()
