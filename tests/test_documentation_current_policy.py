"""Regression contract for maintained current-product documentation."""

from __future__ import annotations

from pathlib import Path
import re
import tomllib


ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
EXAMPLES = ROOT / "examples"
MANIFEST = ROOT / "Blender_to_Spine2D_Mesh_Exporter" / "blender_manifest.toml"
CYRILLIC = re.compile(r"[\u0400-\u04FF]")


def _maintained_markdown_files() -> tuple[Path, ...]:
    files: set[Path] = {ROOT / "README.md"}
    if DOCS.exists():
        files.update(DOCS.rglob("*.md"))
    if EXAMPLES.exists():
        files.update(EXAMPLES.rglob("*.md"))
    return tuple(sorted(files))


def test_maintained_documentation_is_english_only() -> None:
    offenders: list[str] = []
    for path in _maintained_markdown_files():
        text = path.read_text(encoding="utf-8")
        match = CYRILLIC.search(text)
        if match is not None:
            line = text.count("\n", 0, match.start()) + 1
            offenders.append(f"{path.relative_to(ROOT)}:{line}")

    assert not offenders, "Cyrillic text found in maintained documentation: " + ", ".join(
        offenders
    )


def test_historical_release_documents_are_not_maintained() -> None:
    forbidden: list[str] = []
    for path in _maintained_markdown_files():
        relative = path.relative_to(ROOT)
        parts = relative.parts
        if len(parts) >= 2 and parts[0] == "docs" and parts[1] == "releases":
            forbidden.append(str(relative))
            continue
        if path.name == "CHANGELOG.md" or path.name.startswith("RELEASE_"):
            forbidden.append(str(relative))

    assert not forbidden, "Historical release documents remain in the maintained tree: " + ", ".join(
        forbidden
    )


def test_public_documentation_version_matches_manifest() -> None:
    with MANIFEST.open("rb") as stream:
        version = str(tomllib.load(stream)["version"])

    assert version == "0.128.0"

    required_current_documents = (
        ROOT / "README.md",
        DOCS / "README.md",
        DOCS / "installation.md",
        DOCS / "usage.md",
        DOCS / "settings-reference.md",
        DOCS / "rig-profiles.md",
        DOCS / "testing.md",
        EXAMPLES / "examples.md",
    )
    missing_version = tuple(
        str(path.relative_to(ROOT))
        for path in required_current_documents
        if version not in path.read_text(encoding="utf-8")
    )
    assert not missing_version, (
        f"Current extension version {version} is missing from maintained entry documents: "
        + ", ".join(missing_version)
    )
