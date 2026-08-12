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
SEMANTIC_VERSION = re.compile(r"\d+\.\d+\.\d+")
EXTENSION_ZERO_VERSION = re.compile(r"\b0\.\d+\.\d+\b")


def _public_document_paths() -> tuple[Path, ...]:
    paths = [README, MANIFEST]
    paths.extend(sorted(DOCS.rglob("*.md")))
    paths.extend(sorted(DOCS.rglob("*.json")))
    paths.extend(sorted(EXAMPLES.rglob("*.md")))
    return tuple(dict.fromkeys(paths))


def _maintained_markdown_paths() -> tuple[Path, ...]:
    return tuple(
        path
        for path in _public_document_paths()
        if path.suffix.lower() == ".md"
    )


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8-sig")


def _manifest_version() -> str:
    source = _read(MANIFEST)
    match = re.search(
        r'^version\s*=\s*"([^"]+)"\s*$',
        source,
        re.MULTILINE,
    )
    assert match is not None, "Manifest version is missing"
    version = match.group(1)
    assert SEMANTIC_VERSION.fullmatch(version), version
    return version


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


def test_public_documentation_contains_no_cyrillic_characters() -> None:
    violations: list[str] = []
    for path in _public_document_paths():
        source = _read(path)
        match = CYRILLIC.search(source)
        if match is None:
            continue
        line = source.count("\n", 0, match.start()) + 1
        violations.append(f"{path.relative_to(ROOT)}:{line}: {match.group(0)!r}")

    assert not violations, (
        "Cyrillic characters found in public documentation:\n"
        + "\n".join(violations)
    )


def test_historical_release_documents_are_not_public_docs() -> None:
    forbidden: list[str] = []
    for path in _maintained_markdown_paths():
        relative = path.relative_to(ROOT)
        parts = relative.parts
        if len(parts) >= 2 and parts[0] == "docs" and parts[1] == "releases":
            forbidden.append(str(relative))
            continue
        if path.name == "CHANGELOG.md" or path.name.startswith("RELEASE_"):
            forbidden.append(str(relative))

    assert not forbidden, (
        "Historical release documents remain in the maintained documentation tree:\n"
        + "\n".join(forbidden)
    )


def test_public_docs_do_not_publish_superseded_extension_versions() -> None:
    current_version = _manifest_version()
    violations: list[str] = []

    for path in _maintained_markdown_paths():
        source = _read(path)
        for match in EXTENSION_ZERO_VERSION.finditer(source):
            version = match.group(0)
            if version == current_version:
                continue
            line = source.count("\n", 0, match.start()) + 1
            violations.append(
                f"{path.relative_to(ROOT)}:{line}: superseded extension version {version}"
            )

    assert not violations, (
        "Superseded extension versions found in current-product documentation:\n"
        + "\n".join(violations)
    )


def test_temporary_rewrite_documents_are_not_public_docs() -> None:
    assert not tuple(DOCS.glob("REWRITE_*.md"))
    assert not (DOCS / "a1_fixture_manifest.example.json").exists()
    assert not (DOCS / "private-release-manifest.example.json").exists()


def test_readme_presents_current_product_without_stale_ui_media() -> None:
    source = _read(README)

    required_fragments = (
        "assets/cover.png",
        "img.shields.io/badge/License-GPLv3",
        "img.shields.io/github/v/release/",
        "img.shields.io/github/downloads/",
        "Blender-5.2%2B",
        "patreon.com/MaximSokolenko",
        "## What the exporter does",
        "## Export modes",
        "Active Camera — Object Root Bone",
        "Active Camera — Camera Root Bone",
        "## Rig and animation controls",
        "## Geometry, segmentation, and UV",
        "## Materials and baking",
        "## Texture sequences",
        "## Analyze and diagnostics",
        "## Output and transaction safety",
        "## Source-scene safety",
    )
    missing = tuple(fragment for fragment in required_fragments if fragment not in source)
    assert not missing, f"README current-product contract is missing: {missing}"

    forbidden_fragments = (
        "img.youtube.com/",
        "youtube.com/watch?v=",
        "Click to watch the video",
        "assets/ui_addon.png",
        "## Interface",
    )
    stale = tuple(fragment for fragment in forbidden_fragments if fragment in source)
    assert not stale, f"README still contains stale media/UI presentation: {stale}"

    assert (ROOT / "assets" / "cover.png").is_file()


def test_public_documentation_relative_links_and_images_exist() -> None:
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


def test_documentation_matches_manifest_and_current_defaults() -> None:
    version = _manifest_version()

    for relative_path in (
        "README.md",
        "docs/README.md",
        "docs/installation.md",
        "docs/usage.md",
        "docs/settings-reference.md",
        "docs/rig-profiles.md",
        "docs/testing.md",
        "docs/submission.md",
        "examples/examples.md",
    ):
        assert version in _read(ROOT / relative_path), relative_path

    settings = _read(DOCS / "settings-reference.md")
    assert "### Seam Maker" in settings
    assert "| Auto | Yes |" in settings
    assert "Normal / UV Segments" in settings
    assert "Active Camera — Object Root Bone" in settings
    assert "Active Camera — Camera Root Bone" in settings
    assert "`ACTIVE_CAMERA`" in settings
    assert "`ACTIVE_CAMERA_CAMERA_ROOT`" in settings
    assert "`CAMERA_VIEW_NORMAL`" in settings
    assert "Parallax Horizon Angle" in settings

    rig_profiles = _read(DOCS / "rig-profiles.md")
    assert "*_camera_setup" in rig_profiles
    assert "CAMERA_VIEW_NORMAL" in rig_profiles
    assert "PREPROJECTED_SCREEN" in rig_profiles


def test_texture_size_is_documented_as_bake_owned_setting() -> None:
    settings = _read(DOCS / "settings-reference.md")
    export_start = settings.index("## Export")
    rig_start = settings.index("## Rig")
    bake_start = settings.index("## Bake")
    texture_start = settings.index("### Texture Size")

    assert "### Texture Size" not in settings[export_start:rig_start]
    assert texture_start > bake_start
    assert "Scene-level" in settings[texture_start:]
    assert "Paths and Spine 2D version" in settings[texture_start:]

    usage = _read(DOCS / "usage.md")
    assert "## Configure Bake" in usage
    assert "### Texture size" in usage
    assert usage.index("### Texture size") > usage.index("## Configure Bake")

    readme = _read(README)
    assert "The **Bake** foldout owns the scene-wide **Texture size** setting" in readme


def test_submission_document_matches_current_public_manifest() -> None:
    submission = _read(DOCS / "submission.md")
    manifest = _read(MANIFEST)
    current_version = _manifest_version()

    assert "Windows x64" in submission
    assert "Import-Export, Mesh, UV, Animation" in submission
    assert "License: GPL-3.0-or-later" in submission
    assert "Permission: Files" in submission
    assert f"blender_to_spine2d_mesh_exporter-{current_version}.zip" in submission
    assert 'platforms = ["windows-x64"]' in manifest
    assert 'website = "https://github.com/maximsokal/Blender_to_Spine_2D_Mesh_Export_Addon"' in manifest


def test_public_docs_do_not_describe_blender_44_as_supported() -> None:
    violations: list[str] = []
    for path in _public_document_paths():
        source = _read(path)
        if "Blender 4.4" in source or "Blender-4.4" in source:
            violations.append(str(path.relative_to(ROOT)))
    assert not violations, f"Outdated Blender 4.4 support claims: {violations}"
