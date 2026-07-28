"""Keep Blender runtime hooks explicitly owned and reversible."""

from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "Blender_to_Spine2D_Mesh_Exporter"
REWRITE_ROOTS = (
    PACKAGE / "application",
    PACKAGE / "blender_adapter",
    PACKAGE / "domain",
    PACKAGE / "infrastructure",
)
TOP_LEVEL = (
    PACKAGE / "__init__.py",
    PACKAGE / "addon_preferences.py",
    PACKAGE / "config.py",
    PACKAGE / "repolish_ui.py",
    PACKAGE / "rig_ui.py",
    PACKAGE / "single_object_operator.py",
    PACKAGE / "ui.py",
)


def _call_name(call: ast.Call) -> str:
    parts: list[str] = []
    node = call.func
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
    return ".".join(reversed(parts))


def _runtime_sources() -> tuple[Path, ...]:
    paths = list(TOP_LEVEL)
    for root in REWRITE_ROOTS:
        paths.extend(root.rglob("*.py"))
    return tuple(sorted(path for path in paths if path.is_file()))


def test_rewrite_has_no_unowned_timer_keymap_draw_or_preview_allocations():
    forbidden_suffixes = (
        "app.timers.register",
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


def test_each_handler_append_has_a_matching_remove_in_same_module():
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
