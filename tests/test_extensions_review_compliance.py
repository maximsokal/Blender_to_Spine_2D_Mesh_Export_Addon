from __future__ import annotations

import ast
import tomllib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "Blender_to_Spine2D_Mesh_Exporter"
MANIFEST = PACKAGE / "blender_manifest.toml"


def _production_python_files() -> tuple[Path, ...]:
    return tuple(
        sorted(
            path
            for path in PACKAGE.rglob("*.py")
            if "Legacy" not in path.parts and "__pycache__" not in path.parts
        )
    )


def _imported_roots(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                roots.add(alias.name.partition(".")[0])
        elif isinstance(node, ast.ImportFrom) and node.module:
            roots.add(node.module.partition(".")[0])
    return roots


def test_manifest_matches_extensions_review_metadata() -> None:
    manifest = tomllib.loads(MANIFEST.read_text(encoding="utf-8"))

    assert manifest["id"] == "blender_to_spine2d_mesh_exporter"
    assert manifest["version"] == "0.155.0"
    assert manifest["name"] == "Spine2D Mesh Exporter"
    assert manifest["tags"] == ["Import-Export"]
    assert "platforms" not in manifest
    assert "Blender" not in manifest["name"]
    assert "Blender" not in manifest["tagline"]


def test_runtime_does_not_import_threading_or_queue() -> None:
    offenders: list[str] = []
    for path in _production_python_files():
        imported = _imported_roots(path)
        blocked = sorted(imported.intersection({"threading", "queue"}))
        if blocked:
            offenders.append(
                f"{path.relative_to(ROOT).as_posix()}: {', '.join(blocked)}"
            )

    assert offenders == [], "Forbidden Blender runtime imports:\n" + "\n".join(offenders)


def test_development_pipeline_trace_session_is_not_shipped() -> None:
    offenders = [
        path.relative_to(ROOT).as_posix()
        for path in _production_python_files()
        if "PipelineTraceSession" in path.read_text(encoding="utf-8")
    ]
    assert offenders == []


def test_repolish_advertisement_is_not_shipped() -> None:
    assert not (PACKAGE / "repolish_ui.py").exists()

    offenders: list[str] = []
    for path in _production_python_files():
        text = path.read_text(encoding="utf-8").lower()
        if "re-polish" in text or "repolish" in text:
            offenders.append(path.relative_to(ROOT).as_posix())

    assert offenders == []


def test_root_registration_has_no_registration_state_machine() -> None:
    source = (PACKAGE / "__init__.py").read_text(encoding="utf-8")

    assert "ExtensionRegistrationState" not in source
    assert "_REGISTRATION_STATE" not in source
    assert "RegistrationCleanupAction" not in source
    assert "register_rna_properties_transactionally" not in source
    assert "repolish_ui" not in source
    assert 'raise RuntimeError("Blender bpy module is required' not in source


def test_ui_layout_uses_child_panels_without_panel_replacement() -> None:
    source = (PACKAGE / "ui_layout.py").read_text(encoding="utf-8")

    assert "_ORIGINAL_PANEL_REMOVED" not in source
    assert "_restore_original_panel" not in source
    assert "bpy.utils.unregister_class(ui.OBJECT_PT_Spine2DMeshPanel)" not in source
    assert "bl_parent_id = _PARENT_PANEL_ID" in source


def test_manifest_build_excludes_repository_development_surfaces() -> None:
    manifest = tomllib.loads(MANIFEST.read_text(encoding="utf-8"))
    patterns = set(manifest["build"]["paths_exclude_pattern"])

    required = {
        "__pycache__/",
        "*.py[cod]",
        "/.git/",
        "/.github/",
        "/tests/",
        "/docs/",
        "/*.zip",
        "/Legacy/",
    }
    assert required.issubset(patterns)
