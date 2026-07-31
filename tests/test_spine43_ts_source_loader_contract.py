"""Source contracts for the read-only official Spine 4.3 TypeScript loader."""

from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LOADER = ROOT / "tools" / "spine43_ts_source_loader.mjs"


def _source() -> str:
    return LOADER.read_text(encoding="utf-8")


def test_loader_requires_an_explicit_runtime_source_root() -> None:
    source = _source()

    assert "SPINE43_RUNTIME_SOURCE_ROOT" in source
    assert "resolveAllowedRoot()" in source
    assert "statSync(root).isDirectory()" in source


def test_loader_redirects_only_missing_relative_js_to_existing_ts() -> None:
    source = _source()

    assert "specifier.startsWith('./')" in source
    assert "specifier.startsWith('../')" in source
    assert "specifier.endsWith('.js')" in source
    assert "`${specifier.slice(0, -3)}.ts`" in source
    assert "existsSync(candidate)" in source
    assert "statSync(candidate).isFile()" in source
    assert "return await nextResolve(specifier, context);" in source
    assert "shortCircuit: true" in source


def test_loader_rejects_cross_drive_and_parent_escape_paths() -> None:
    source = _source()

    assert "isAbsolute" in source
    assert "rel !== '..'" in source
    assert "rel.startsWith" in source
    assert "isInsideAllowedRoot(parentPath)" in source
    assert "isInsideAllowedRoot(candidate)" in source


def test_loader_has_no_filesystem_write_apis() -> None:
    source = _source()

    assert "import { existsSync, statSync } from 'node:fs';" in source
    for forbidden in (
        "writeFileSync",
        "appendFileSync",
        "mkdirSync",
        "rmSync",
        "unlinkSync",
        "renameSync",
        "copyFileSync",
        "cpSync",
    ):
        assert forbidden not in source
