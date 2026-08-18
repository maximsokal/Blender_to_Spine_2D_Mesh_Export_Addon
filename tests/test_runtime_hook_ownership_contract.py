"""Keep shipped Blender runtime hooks explicitly owned and reversible."""

from __future__ import annotations

import ast
from fnmatch import fnmatchcase
from pathlib import Path
import tomllib


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "Blender_to_Spine2D_Mesh_Exporter"
MANIFEST = PACKAGE / "blender_manifest.toml"


def _call_name(call: ast.Call) -> str:
    parts: list[str] = []
    node = call.func
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
    return ".".join(reversed(parts))


def _exclude_patterns() -> tuple[str, ...]:
    manifest = tomllib.loads(MANIFEST.read_text(encoding="utf-8"))
    return tuple(
        str(pattern).replace("\\", "/")
        for pattern in manifest.get("build", {}).get("paths_exclude_pattern", ())
    )


def _is_excluded(path: Path) -> bool:
    relative = path.relative_to(PACKAGE).as_posix()
    rooted = f"/{relative}"
    for pattern in _exclude_patterns():
        if pattern.startswith("/"):
            if pattern.endswith("/"):
                directory = pattern.rstrip("/")
                if rooted == directory or rooted.startswith(f"{directory}/"):
                    return True
            elif fnmatchcase(rooted, pattern):
                return True
            continue
        if pattern.endswith("/"):
            if pattern.rstrip("/") in path.relative_to(PACKAGE).parts:
                return True
            continue
        if fnmatchcase(path.name, pattern) or fnmatchcase(relative, pattern):
            return True
    return False


def _runtime_sources() -> tuple[Path, ...]:
    return tuple(
        sorted(path for path in PACKAGE.rglob("*.py") if not _is_excluded(path))
    )


def _function_source(path: Path, name: str) -> str:
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(path))
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == name
    )
    segment = ast.get_source_segment(source, function)
    assert segment is not None
    return segment


def test_rewrite_has_no_unowned_keymap_draw_or_preview_allocations():
    forbidden_suffixes = (
        "keymaps.new",
        "keymap_items.new",
        "draw_handler_add",
        "previews.new",
    )
    findings: list[str] = []
    for path in _runtime_sources():
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            name = _call_name(node)
            if name.endswith(forbidden_suffixes):
                findings.append(f"{path.relative_to(ROOT)}:{node.lineno}:{name}")
    assert findings == []


def test_blender_timer_surfaces_are_explicitly_classified_and_owned():
    timer_sources = {
        path.relative_to(PACKAGE).as_posix()
        for path in _runtime_sources()
        if "timers" in path.read_text(encoding="utf-8")
        and "register" in path.read_text(encoding="utf-8")
    }

    # addon_preferences owns one one-shot event-loop redraw timer. auto_readiness
    # retains compatibility timer helpers, but its installed register/unregister
    # lifecycle deliberately never activates the old automatic polling service.
    assert timer_sources == {"addon_preferences.py", "auto_readiness.py"}

    preferences = PACKAGE / "addon_preferences.py"
    assert "_cancel_deferred_view3d_redraw()" in _function_source(
        preferences, "unregister"
    )

    automatic = PACKAGE / "auto_readiness.py"
    automatic_register = _function_source(automatic, "register")
    automatic_unregister = _function_source(automatic, "unregister")
    assert "_register_timer()" not in automatic_register
    assert "_install_handlers()" not in automatic_register
    assert "_unregister_timer()" not in automatic_unregister
    assert "_remove_handlers()" not in automatic_unregister


def test_each_handler_append_has_a_matching_remove_in_same_shipped_module():
    appends: dict[str, set[str]] = {}
    removals: dict[str, set[str]] = {}
    for path in _runtime_sources():
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(path))
        key = str(path.relative_to(ROOT))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
                continue
            owner = node.func.value
            if not isinstance(owner, ast.Attribute):
                continue
            owner_name = ast.unparse(owner)
            if "handlers" not in owner_name or not node.args:
                continue
            callback = ast.unparse(node.args[0])
            if node.func.attr == "append":
                appends.setdefault(key, set()).add(f"{owner_name}:{callback}")
            elif node.func.attr == "remove":
                removals.setdefault(key, set()).add(f"{owner_name}:{callback}")
    missing = {
        module: sorted(callbacks - removals.get(module, set()))
        for module, callbacks in appends.items()
        if callbacks - removals.get(module, set())
    }
    assert missing == {}
