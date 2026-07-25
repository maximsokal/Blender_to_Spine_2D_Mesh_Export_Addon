"""Validate the physical Blender extension ZIP, not only the CLI exit code."""

from __future__ import annotations

from pathlib import Path
import stat
import zipfile

import pytest

from tools import prepare_package


MANIFEST_TEXT = """schema_version = "1.0.0"
id = "blender_to_spine2d_mesh_exporter"
version = "0.23.0"
name = "Blender to Spine2D Mesh Exporter"
tagline = "Export Spine data"
maintainer = "Test Maintainer"
blender_version_min = "5.2.0"
type = "add-on"
license = ["SPDX:GPL-3.0-or-later"]

[permissions]
files = "Write exported JSON and textures"
"""


def _manifest() -> dict[str, object]:
    import tomllib

    return tomllib.loads(MANIFEST_TEXT)


def _write_archive(
    path: Path,
    *,
    prefix: str = "",
    manifest_text: str = MANIFEST_TEXT,
    extra: dict[str, bytes] | None = None,
) -> Path:
    base = f"{prefix.rstrip('/')}/" if prefix else ""
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(base + "__init__.py", b"def register():\n    pass\n")
        archive.writestr(base + "blender_manifest.toml", manifest_text.encode("utf-8"))
        archive.writestr(base + "runtime.py", b"VALUE = 1\n")
        for name, payload in (extra or {}).items():
            archive.writestr(base + name, payload)
    return path


@pytest.mark.parametrize("prefix", ("", "addon-source"))
def test_validate_archive_accepts_official_root_layouts(tmp_path, prefix):
    archive = _write_archive(tmp_path / "extension.zip", prefix=prefix)

    prepare_package._validate_built_archive(
        archive,
        source_manifest=_manifest(),
    )


@pytest.mark.parametrize(
    "member",
    (
        "../escape.py",
        "/absolute.py",
        "C:/drive.py",
        "tests/test_hidden.py",
        ".github/workflows/ci.yml",
        "Legacy/main.py",
        "main.py",
        "nested.zip",
        "__pycache__/module.pyc",
    ),
)
def test_validate_archive_rejects_unsafe_or_unshippable_members(tmp_path, member):
    archive = _write_archive(
        tmp_path / "extension.zip",
        extra={member: b"forbidden"},
    )

    with pytest.raises(prepare_package.PackageBuildError):
        prepare_package._validate_built_archive(
            archive,
            source_manifest=_manifest(),
        )


def test_validate_archive_rejects_case_colliding_paths(tmp_path):
    archive = _write_archive(
        tmp_path / "extension.zip",
        extra={"Runtime.py": b"other"},
    )

    with pytest.raises(prepare_package.PackageBuildError, match="case-colliding"):
        prepare_package._validate_built_archive(
            archive,
            source_manifest=_manifest(),
        )


def test_validate_archive_rejects_symbolic_link_member(tmp_path):
    archive_path = tmp_path / "extension.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("__init__.py", b"def register():\n    pass\n")
        archive.writestr("blender_manifest.toml", MANIFEST_TEXT)
        link = zipfile.ZipInfo("linked.py")
        link.create_system = 3
        link.external_attr = (stat.S_IFLNK | 0o777) << 16
        archive.writestr(link, "runtime.py")

    with pytest.raises(prepare_package.PackageBuildError, match="symbolic link"):
        prepare_package._validate_built_archive(
            archive_path,
            source_manifest=_manifest(),
        )


def test_validate_archive_rejects_manifest_drift(tmp_path):
    archive = _write_archive(
        tmp_path / "extension.zip",
        manifest_text=MANIFEST_TEXT.replace('version = "0.23.0"', 'version = "9.9.9"'),
    )

    with pytest.raises(prepare_package.PackageBuildError, match="does not match"):
        prepare_package._validate_built_archive(
            archive,
            source_manifest=_manifest(),
        )


def test_validate_archive_rejects_corrupt_or_missing_runtime_files(tmp_path):
    corrupt = tmp_path / "corrupt.zip"
    corrupt.write_bytes(b"not a zip")
    with pytest.raises(prepare_package.PackageBuildError, match="Unable to validate"):
        prepare_package._validate_built_archive(
            corrupt,
            source_manifest=_manifest(),
        )

    missing = tmp_path / "missing.zip"
    with zipfile.ZipFile(missing, "w") as archive:
        archive.writestr("blender_manifest.toml", MANIFEST_TEXT)
    with pytest.raises(prepare_package.PackageBuildError, match="missing"):
        prepare_package._validate_built_archive(
            missing,
            source_manifest=_manifest(),
        )


def test_build_extension_removes_invalid_archive(tmp_path, monkeypatch):
    source = tmp_path / "extension"
    source.mkdir()
    (source / "__init__.py").write_text("def register():\n    pass\n", encoding="utf-8")
    (source / "blender_manifest.toml").write_text(MANIFEST_TEXT, encoding="utf-8")
    blender = tmp_path / "blender"
    blender.write_bytes(b"fake")
    output = tmp_path / "dist" / "extension.zip"

    monkeypatch.setattr(prepare_package, "_validate_blender_version", lambda _value: None)

    def _run(command, *, label):
        if label == "Extension package build":
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_bytes(b"invalid zip")

    monkeypatch.setattr(prepare_package, "_run_command", _run)

    with pytest.raises(prepare_package.PackageBuildError):
        prepare_package.build_extension(
            blender=blender,
            source=source,
            output=output,
        )

    assert not output.exists()


@pytest.mark.parametrize(
    ("field", "replacement"),
    (
        ("name", 'name = ""'),
        ("tagline", 'tagline = "   "'),
        ("maintainer", 'maintainer = ""'),
        ("schema_version", 'schema_version = "2.0.0"'),
        ("type", 'type = "theme"'),
        ("license", 'license = []'),
    ),
)
def test_read_manifest_rejects_incomplete_release_metadata(tmp_path, field, replacement):
    source = tmp_path / "extension"
    source.mkdir()
    (source / "__init__.py").write_text("", encoding="utf-8")
    source_text = MANIFEST_TEXT
    if field == "license":
        source_text = source_text.replace('license = ["SPDX:GPL-3.0-or-later"]', replacement)
    else:
        original_line = next(line for line in source_text.splitlines() if line.startswith(field + " ="))
        source_text = source_text.replace(original_line, replacement)
    (source / "blender_manifest.toml").write_text(source_text, encoding="utf-8")

    with pytest.raises(prepare_package.PackageBuildError):
        prepare_package._read_manifest(source)


def test_repository_manifest_and_exclusions_pass_local_release_gate():
    root = Path(__file__).resolve().parents[1]
    source = root / "Blender_to_Spine2D_Mesh_Exporter"
    manifest = prepare_package._read_manifest(source)

    assert manifest["id"] == "blender_to_spine2d_mesh_exporter"
    assert manifest["blender_version_min"] == "5.2.0"
    patterns = tuple(manifest["build"]["paths_exclude_pattern"])
    for required in (
        "/.github/",
        "/tests/",
        "/docs/",
        "/Legacy/",
        "/main.py",
        "/texture_baker.py",
        "__pycache__/",
        "/*.zip",
    ):
        assert required in patterns
