from __future__ import annotations

from pathlib import Path
import re
from urllib.parse import unquote


ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
EXAMPLES = ROOT / "examples"
MANIFEST = ROOT / "Blender_to_Spine2D_Mesh_Exporter" / "blender_manifest.toml"
README = ROOT / "README.md"

CYRILLIC = re.compile(r"[\u0400-\u04FF]")
MARKDOWN_LINK = re.compile(r"!?\[[^\]]*\]\(([^)]+)\)")
HTML_LINK = re.compile(r"(?:src|href)=[\"']([^\"']+)[\"']", re.IGNORECASE)


def _public_document_paths() -> tuple[Path, ...]:
    paths = [README, MANIFEST]
    paths.extend(sorted(DOCS.rglob("*.md")))
    paths.extend(sorted(DOCS.rglob("*.json")))
    paths.extend(sorted(EXAMPLES.rglob("*.md")))
    return tuple(dict.fromkeys(paths))


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8-sig")


def _local_targets(path: Path, source: str) -> tuple[str, ...]:
    targets = [match.group(1).strip() for match in MARKDOWN_LINK.finditer(source)]
    targets.extend(match.group(1).strip() for match in HTML_LINK.finditer(source))
    return tuple(targets)


def _resolve_local_target(document: Path, raw_target: str) -> Path | None:
    target = raw_target.strip()
    if not target or target.startswith(("#", "http://", "https://", "mailto:")):
        return None

    target = target.split(" ", 1)[0].strip("<>")
    target = unquote(target.split("#", 1)[0].split("?", 1)[0])
    if not target:
        return None
    return (document.parent / target).resolve(strict=False)


def test_public_documentation_contains_no_cyrillic_characters():
    violations: list[str] = []
    for path in _public_document_paths():
        source = _read(path)
        match = CYRILLIC.search(source)
        if match is None:
            continue
        line = source.count("\n", 0, match.start()) + 1
        violations.append(f"{path.relative_to(ROOT)}:{line}: {match.group(0)!r}")

    assert not violations, "Cyrillic characters found in public documentation:\n" + "\n".join(
        violations
    )


def test_temporary_rewrite_documents_are_not_public_docs():
    assert not tuple(DOCS.glob("REWRITE_*.md"))
    assert not (DOCS / "a1_fixture_manifest.example.json").exists()
    assert not (DOCS / "private-release-manifest.example.json").exists()


def test_readme_preserves_visual_assets_badges_counters_and_video():
    source = _read(README)
    required_fragments = (
        "assets/cover.png",
        "img.shields.io/badge/License-GPLv3",
        "img.shields.io/github/v/release/",
        "img.shields.io/github/downloads/",
        "Blender-5.2%2B",
        "patreon.com/MaximSokolenko",
        "youtube.com/watch?v=f_1Zc2qCz44",
        "img.youtube.com/vi/f_1Zc2qCz44/maxresdefault.jpg",
        "assets/ui_addon.png",
    )
    missing = tuple(fragment for fragment in required_fragments if fragment not in source)
    assert not missing, f"README visual contract is missing: {missing}"
    assert (ROOT / "assets" / "cover.png").is_file()
    assert (ROOT / "assets" / "ui_addon.png").is_file()


def test_public_documentation_relative_links_and_images_exist():
    missing: list[str] = []
    for document in _public_document_paths():
        if document.suffix.lower() not in {".md", ".toml"}:
            continue
        source = _read(document)
        for raw_target in _local_targets(document, source):
            resolved = _resolve_local_target(document, raw_target)
            if resolved is None:
                continue
            try:
                resolved.relative_to(ROOT.resolve())
            except ValueError:
                missing.append(
                    f"{document.relative_to(ROOT)} -> target escapes repository: {raw_target}"
                )
                continue
            if not resolved.exists():
                missing.append(f"{document.relative_to(ROOT)} -> {raw_target}")

    assert not missing, "Broken public documentation links:\n" + "\n".join(missing)


def test_documentation_matches_manifest_and_current_defaults():
    manifest_source = _read(MANIFEST)
    match = re.search(
        r'^version\s*=\s*"([^"]+)"\s*$',
        manifest_source,
        re.MULTILINE,
    )
    assert match is not None
    version = match.group(1)
    assert version == "0.47.1"

    for relative_path in (
        "docs/README.md",
        "docs/CHANGELOG.md",
        "docs/installation.md",
        "docs/testing.md",
    ):
        assert version in _read(ROOT / relative_path), relative_path

    settings = _read(DOCS / "settings-reference.md")
    assert "### Seam Maker" in settings
    assert "| Auto | Yes |" in settings
    assert "Normal - UV Segments" in settings


def test_public_docs_do_not_describe_blender_44_as_supported():
    violations: list[str] = []
    for path in _public_document_paths():
        source = _read(path)
        if "Blender 4.4" in source or "Blender-4.4" in source:
            violations.append(str(path.relative_to(ROOT)))

    assert not violations, f"Outdated Blender 4.4 support claims: {violations}"
