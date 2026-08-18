from __future__ import annotations

import ast
from fnmatch import fnmatchcase
import tomllib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "Blender_to_Spine2D_Mesh_Exporter"
MANIFEST = PACKAGE / "blender_manifest.toml"
PIPELINE_TRACE_RUNTIME_FILES = frozenset(
    {
        "/infrastructure/pipeline_trace.py",
        "/infrastructure/pipeline_trace_model.py",
        "/infrastructure/pipeline_trace_report.py",
        "/infrastructure/pipeline_trace_values.py",
    }
)


def _manifest_exclude_patterns() -> tuple[str, ...]:
    manifest = tomllib.loads(MANIFEST.read_text(encoding="utf-8"))
    raw_patterns = manifest.get("build", {}).get("paths_exclude_pattern", ())
    if not isinstance(raw_patterns, list):
        raise AssertionError("manifest build.paths_exclude_pattern must be a list")
    patterns: list[str] = []
    for value in raw_patterns:
        if not isinstance(value, str) or not value.strip():
            raise AssertionError(
                "manifest build.paths_exclude_pattern entries must be non-empty strings"
            )
        patterns.append(value.replace("\\", "/"))
    return tuple(patterns)


def _manifest_excludes_package_path(path: Path) -> bool:
    """Return whether a package path is excluded by the manifest build rules.

    The compliance scan only needs Python files under the extension source root.  It
    supports the rooted file/directory patterns used by this manifest plus ordinary
    filename globs such as ``*.py[cod]``.  The final Blender-built ZIP inventory is a
    separate release gate and remains authoritative for packaging behavior.
    """

    if not isinstance(path, Path):
        raise TypeError("path must be pathlib.Path")
    try:
        relative = path.relative_to(PACKAGE).as_posix()
    except ValueError as exc:
        raise ValueError(f"path is outside extension package: {path}") from exc

    rooted = f"/{relative}"
    for pattern in _manifest_exclude_patterns():
        if pattern.startswith("/"):
            if pattern.endswith("/"):
                directory = pattern.rstrip("/")
                if rooted == directory or rooted.startswith(f"{directory}/"):
                    return True
            elif fnmatchcase(rooted, pattern):
                return True
            continue

        if pattern.endswith("/"):
            directory_name = pattern.rstrip("/")
            if directory_name in path.relative_to(PACKAGE).parts:
                return True
            continue

        if fnmatchcase(path.name, pattern) or fnmatchcase(relative, pattern):
            return True

    return False


def _shipped_python_files() -> tuple[Path, ...]:
    return tuple(
        sorted(
            path
            for path in PACKAGE.rglob("*.py")
            if not _manifest_excludes_package_path(path)
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
    for path in _shipped_python_files():
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
        for path in _shipped_python_files()
        if "PipelineTraceSession" in path.read_text(encoding="utf-8")
    ]
    assert offenders == []


def test_repolish_advertisement_is_not_shipped() -> None:
    assert not (PACKAGE / "repolish_ui.py").exists()

    offenders: list[str] = []
    for path in _shipped_python_files():
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
        *PIPELINE_TRACE_RUNTIME_FILES,
    }
    assert required.issubset(patterns)


def test_source_scan_matches_manifest_shipping_boundary() -> None:
    shipped = {f"/{path.relative_to(PACKAGE).as_posix()}" for path in _shipped_python_files()}

    assert PIPELINE_TRACE_RUNTIME_FILES.isdisjoint(shipped)
    assert "/legacy_loader.py" not in shipped
    assert "/infrastructure/atomic_work_state.py" in shipped
    assert "/infrastructure/export_diagnostics.py" in shipped
    assert "/infrastructure/export_events.py" in shipped
