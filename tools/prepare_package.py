"""Build and validate the Blender 5.2+ extension with Blender's official CLI."""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tomllib


LOGGER = logging.getLogger("prepare_package")
EXTENSION_DIRECTORY_NAME = "Blender_to_Spine2D_Mesh_Exporter"
MINIMUM_BLENDER_VERSION = (5, 2, 0)


class PackageBuildError(RuntimeError):
    """Raised when the Blender extension package cannot be validated or built."""


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate and build the Blender 5.2+ extension archive."
    )
    parser.add_argument(
        "--blender",
        type=Path,
        default=None,
        help="Path to the Blender 5.2+ executable. Defaults to BLENDER_EXECUTABLE or PATH.",
    )
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=None,
        help="Extension source directory containing __init__.py and blender_manifest.toml.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output .zip path. Defaults to dist/<id>-<version>.zip.",
    )
    return parser.parse_args(argv)


def _resolve_blender_executable(explicit: Path | None) -> Path:
    candidates: list[str] = []
    if explicit is not None:
        candidates.append(str(explicit.expanduser()))
    environment = str(os.environ.get("BLENDER_EXECUTABLE", "") or "").strip()
    if environment:
        candidates.append(environment)
    discovered = shutil.which("blender")
    if discovered:
        candidates.append(discovered)

    for candidate in candidates:
        path = Path(candidate).expanduser().resolve(strict=False)
        if path.is_file():
            return path
    raise PackageBuildError(
        "Blender 5.2+ executable was not found. Pass --blender, set "
        "BLENDER_EXECUTABLE, or add Blender to PATH."
    )


def _resolve_source_directory(explicit: Path | None) -> Path:
    repository_root = Path(__file__).resolve().parents[1]
    source = (
        explicit.expanduser().resolve(strict=False)
        if explicit is not None
        else repository_root / EXTENSION_DIRECTORY_NAME
    )
    if not source.is_dir():
        raise PackageBuildError(f"Extension source directory does not exist: {source}")
    for required_name in ("__init__.py", "blender_manifest.toml"):
        required = source / required_name
        if not required.is_file():
            raise PackageBuildError(
                f"Extension source is missing required file: {required}"
            )
    return source


def _read_manifest(source: Path) -> dict[str, object]:
    manifest_path = source / "blender_manifest.toml"
    try:
        with manifest_path.open("rb") as stream:
            manifest = tomllib.load(stream)
    except Exception as exc:
        raise PackageBuildError(f"Unable to read {manifest_path}: {exc}") from exc

    minimum = str(manifest.get("blender_version_min", "") or "").strip()
    if minimum != "5.2.0":
        raise PackageBuildError(
            "blender_manifest.toml must declare blender_version_min = \"5.2.0\"; "
            f"found {minimum!r}"
        )
    for key in ("id", "version"):
        value = str(manifest.get(key, "") or "").strip()
        if not value:
            raise PackageBuildError(f"Manifest field {key!r} must be non-empty")
    return manifest


def _resolve_output_path(
    explicit: Path | None,
    manifest: dict[str, object],
) -> Path:
    repository_root = Path(__file__).resolve().parents[1]
    if explicit is None:
        extension_id = str(manifest["id"])
        version = str(manifest["version"])
        output = repository_root / "dist" / f"{extension_id}-{version}.zip"
    else:
        output = explicit.expanduser().resolve(strict=False)
    if output.suffix.casefold() != ".zip":
        raise PackageBuildError(f"Output path must end with .zip: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    return output


def _run_command(command: list[str], *, label: str) -> None:
    LOGGER.info("%s: %s", label, subprocess.list2cmdline(command))
    try:
        completed = subprocess.run(
            command,
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
    except OSError as exc:
        raise PackageBuildError(f"Unable to execute {label}: {exc}") from exc
    if completed.stdout:
        LOGGER.info("%s", completed.stdout.rstrip())
    if completed.returncode != 0:
        raise PackageBuildError(
            f"{label} failed with exit code {completed.returncode}"
        )


def _validate_blender_version(blender: Path) -> None:
    command = [str(blender), "--background", "--python-expr", (
        "import bpy,sys; "
        "version=tuple(bpy.app.version[:3]); "
        "print('BLENDER_VERSION=' + '.'.join(map(str, version))); "
        "sys.exit(0 if version >= (5, 2, 0) else 9)"
    )]
    _run_command(command, label="Blender 5.2 runtime validation")


def build_extension(
    *,
    blender: Path,
    source: Path,
    output: Path,
) -> Path:
    """Validate source and build one extension archive without mutating source files."""

    _validate_blender_version(blender)
    _run_command(
        [
            str(blender),
            "--command",
            "extension",
            "validate",
            str(source),
        ],
        label="Extension manifest validation",
    )
    if output.exists():
        try:
            output.unlink()
        except OSError as exc:
            raise PackageBuildError(
                f"Unable to replace existing archive {output}: {exc}"
            ) from exc
    _run_command(
        [
            str(blender),
            "--command",
            "extension",
            "build",
            "--source-dir",
            str(source),
            "--output-filepath",
            str(output),
        ],
        label="Extension package build",
    )
    if not output.is_file() or output.stat().st_size <= 0:
        raise PackageBuildError(
            f"Blender reported success but no non-empty archive was created: {output}"
        )
    return output


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = _parse_args(sys.argv[1:] if argv is None else argv)
    try:
        blender = _resolve_blender_executable(args.blender)
        source = _resolve_source_directory(args.source_dir)
        manifest = _read_manifest(source)
        output = _resolve_output_path(args.output, manifest)
        built = build_extension(
            blender=blender,
            source=source,
            output=output,
        )
        LOGGER.info("Blender 5.2+ extension archive created: %s", built)
        return 0
    except PackageBuildError:
        LOGGER.exception("Extension package build failed")
        return 1
    except Exception:
        LOGGER.exception("Unexpected extension package build failure")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
