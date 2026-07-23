"""Blender 5.2 official extension packaging regressions."""

from __future__ import annotations

from pathlib import Path

import pytest

from tools import prepare_package


def _write_source(directory: Path, *, minimum: str = "5.2.0") -> Path:
    source = directory / "extension"
    source.mkdir()
    (source / "__init__.py").write_text("def register():\n    pass\n", encoding="utf-8")
    (source / "blender_manifest.toml").write_text(
        "\n".join(
            (
                'schema_version = "1.0.0"',
                'id = "blender_to_spine2d_mesh_exporter"',
                'version = "0.23.0"',
                'name = "Blender to Spine2D Mesh Exporter"',
                'tagline = "Export Spine data"',
                'maintainer = "Test"',
                f'blender_version_min = "{minimum}"',
                'type = "add-on"',
                'license = ["SPDX:GPL-3.0-or-later"]',
            )
        )
        + "\n",
        encoding="utf-8",
    )
    return source


def test_read_manifest_requires_exact_blender_52_minimum(tmp_path):
    source = _write_source(tmp_path)

    manifest = prepare_package._read_manifest(source)

    assert manifest["id"] == "blender_to_spine2d_mesh_exporter"
    assert manifest["version"] == "0.23.0"
    assert manifest["blender_version_min"] == "5.2.0"


@pytest.mark.parametrize("minimum", ["4.4.0", "5.0.0", "5.1.0", ""])
def test_read_manifest_rejects_old_or_missing_minimum(tmp_path, minimum):
    source = _write_source(tmp_path, minimum=minimum)

    with pytest.raises(prepare_package.PackageBuildError, match="5.2.0"):
        prepare_package._read_manifest(source)


def test_resolve_source_directory_requires_manifest_and_init(tmp_path):
    missing_manifest = tmp_path / "missing_manifest"
    missing_manifest.mkdir()
    (missing_manifest / "__init__.py").write_text("", encoding="utf-8")

    with pytest.raises(prepare_package.PackageBuildError, match="blender_manifest.toml"):
        prepare_package._resolve_source_directory(missing_manifest)


def test_default_output_path_uses_manifest_id_and_version(tmp_path, monkeypatch):
    fake_tool = tmp_path / "repository" / "tools" / "prepare_package.py"
    fake_tool.parent.mkdir(parents=True)
    fake_tool.write_text("", encoding="utf-8")
    monkeypatch.setattr(prepare_package, "__file__", str(fake_tool))

    output = prepare_package._resolve_output_path(
        None,
        {
            "id": "blender_to_spine2d_mesh_exporter",
            "version": "0.23.0",
        },
    )

    assert output == (
        tmp_path
        / "repository"
        / "dist"
        / "blender_to_spine2d_mesh_exporter-0.23.0.zip"
    )
    assert output.parent.is_dir()


def test_build_extension_runs_validate_then_official_build(tmp_path, monkeypatch):
    source = _write_source(tmp_path)
    blender = tmp_path / "blender.exe"
    blender.write_bytes(b"fake")
    output = tmp_path / "dist" / "extension.zip"
    commands: list[tuple[str, tuple[str, ...]]] = []

    monkeypatch.setattr(
        prepare_package,
        "_validate_blender_version",
        lambda value: commands.append(("runtime", (str(value),))),
    )

    def run_command(command, *, label):
        commands.append((label, tuple(command)))
        if label == "Extension package build":
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_bytes(b"zip")

    monkeypatch.setattr(prepare_package, "_run_command", run_command)

    result = prepare_package.build_extension(
        blender=blender,
        source=source,
        output=output,
    )

    assert result == output
    assert commands[0] == ("runtime", (str(blender),))
    assert commands[1] == (
        "Extension manifest validation",
        (str(blender), "--command", "extension", "validate", str(source)),
    )
    assert commands[2] == (
        "Extension package build",
        (
            str(blender),
            "--command",
            "extension",
            "build",
            "--source-dir",
            str(source),
            "--output-filepath",
            str(output),
        ),
    )


def test_build_extension_rejects_missing_output_even_after_success(tmp_path, monkeypatch):
    source = _write_source(tmp_path)
    blender = tmp_path / "blender.exe"
    blender.write_bytes(b"fake")
    output = tmp_path / "dist" / "missing.zip"

    monkeypatch.setattr(prepare_package, "_validate_blender_version", lambda _value: None)
    monkeypatch.setattr(prepare_package, "_run_command", lambda *_args, **_kwargs: None)

    with pytest.raises(prepare_package.PackageBuildError, match="no non-empty archive"):
        prepare_package.build_extension(
            blender=blender,
            source=source,
            output=output,
        )


def test_runtime_validation_command_requires_blender_52(monkeypatch, tmp_path):
    blender = tmp_path / "blender.exe"
    blender.write_bytes(b"fake")
    captured: list[tuple[str, ...]] = []

    monkeypatch.setattr(
        prepare_package,
        "_run_command",
        lambda command, **_kwargs: captured.append(tuple(command)),
    )

    prepare_package._validate_blender_version(blender)

    assert len(captured) == 1
    command = captured[0]
    assert command[:3] == (str(blender), "--background", "--python-expr")
    assert "version >= (5, 2, 0)" in command[3]
