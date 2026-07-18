import ast
from pathlib import Path


ROOT = Path(__file__).parents[1] / "Blender_to_Spine2D_Mesh_Exporter"


def _source(relative: str) -> str:
    return (ROOT / relative).read_text(encoding="utf-8")


def _tree(relative: str) -> ast.Module:
    path = ROOT / relative
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def test_addon_no_longer_uses_a_hardcoded_logging_module_list():
    source = _source("__init__.py")
    assert "MODULE_NAMES_FOR_LOGGING" not in source
    assert "spine2d.refresh_logging_modules" in source
    assert "module_filter" in source


def test_config_discovers_nested_python_files_and_keeps_levels_independent():
    source = _source("config.py")
    assert "discover_python_modules" in source
    assert "synchronize_logging_preferences" in source
    assert "self.name ==" not in source
    assert "module_setting.level = new_level" not in source
    assert "preserve_failed_work_files" in source
    assert "recover_stale_work_files" in source


def test_atomic_cleanup_has_events_and_does_not_silently_pass():
    source = _source("infrastructure/atomic_files.py")
    tree = _tree("infrastructure/atomic_files.py")
    assert "ExportEventKind.CLEANUP_FAILED" in source
    assert "recover_stale_atomic_work_files" in source
    assert "preserve_failed_work_files" in source

    silent_handlers = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.ExceptHandler):
            continue
        if len(node.body) == 1 and isinstance(node.body[0], ast.Pass):
            silent_handlers.append(node.lineno)
    assert not silent_handlers, f"silent cleanup handlers at lines {silent_handlers}"


def test_logging_formatter_does_not_mutate_record_name():
    source = _source("config.py")
    assert "record.name =" not in source
    assert "record.short_name" in source
