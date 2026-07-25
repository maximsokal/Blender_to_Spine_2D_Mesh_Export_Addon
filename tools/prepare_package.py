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
import zipfile
from zipfile import ZipInfo


LOGGER = logging.getLogger("prepare_package")
EXTENSION_DIRECTORY_NAME = "Blender_to_Spine2D_Mesh_Exporter"
MINIMUM_BLENDER_VERSION = (5, 2, 0)


class PackageBuildError(RuntimeError):
    """Raised when the Blender extension package cannot be validated or built."""


_REQUIRED_ARCHIVE_FILES = frozenset({"__init__.py", "blender_manifest.toml"})
_FORBIDDEN_ARCHIVE_PARTS = frozenset({
    ".git", ".github", "tests", "tests_bpy", "docs", "Legacy", "__pycache__",
})
_FORBIDDEN_ARCHIVE_BASENAMES = frozenset({
    "legacy_loader.py",
    "legacy_multi_facade.py",
    "json_export.py",
    "json_merger.py",
    "main.py",
    "multi_object_export.py",
    "plane_cut.py",
    "seam_marker.py",
    "texture_baker.py",
    "texture_baker_integration.py",
    "utils.py",
    "uv_operations.py",
})


def _normalise_archive_name(info: ZipInfo) -> tuple[str, ...]:
    """Return one safe POSIX archive path or fail closed."""

    raw = str(info.filename or "").replace("\\", "/")
    if not raw or raw.startswith("/"):
        raise PackageBuildError(f"Archive contains an invalid absolute/empty path: {raw!r}")
    parts = tuple(part for part in raw.split("/") if part not in {"", "."})
    if not parts or any(part == ".." for part in parts):
        raise PackageBuildError(f"Archive path escapes its root: {raw!r}")
    if ":" in parts[0]:
        raise PackageBuildError(f"Archive contains a drive-qualified path: {raw!r}")
    unix_mode = (int(info.external_attr) >> 16) & 0o170000
    if unix_mode == 0o120000:
        raise PackageBuildError(f"Archive contains a symbolic link: {raw!r}")
    return parts


def _archive_root_prefix(paths: tuple[tuple[str, ...], ...]) -> tuple[str, ...]:
    """Resolve either root-level files or one optional enclosing directory."""

    files = tuple(path for path in paths if path)
    root_names = {path[0] for path in files}
    if _REQUIRED_ARCHIVE_FILES.issubset(root_names):
        return ()
    if len(root_names) != 1:
        raise PackageBuildError(
            "Extension archive must place required files at its root or inside "
            "one enclosing directory"
        )
    prefix = (next(iter(root_names)),)
    nested_names = {path[1] for path in files if len(path) >= 2}
    if not _REQUIRED_ARCHIVE_FILES.issubset(nested_names):
        raise PackageBuildError(
            "Extension archive is missing __init__.py or blender_manifest.toml"
        )
    return prefix


def _validate_built_archive(
    archive_path: Path,
    *,
    source_manifest: dict[str, object],
) -> None:
    """Validate the physical ZIP emitted by Blender before publishing it."""

    if not isinstance(archive_path, Path):
        raise TypeError("archive_path must be pathlib.Path")
    if not isinstance(source_manifest, dict):
        raise TypeError("source_manifest must be dict")
    try:
        with zipfile.ZipFile(archive_path, mode="r") as archive:
            infos = tuple(archive.infolist())
            if not infos:
                raise PackageBuildError("Extension archive is empty")
            bad_member = archive.testzip()
            if bad_member is not None:
                raise PackageBuildError(
                    f"Extension archive contains a corrupt member: {bad_member}"
                )

            normalised = tuple(_normalise_archive_name(info) for info in infos)
            folded_names: set[str] = set()
            for parts in normalised:
                folded = "/".join(parts).casefold()
                if folded in folded_names:
                    raise PackageBuildError(
                        "Extension archive contains duplicate/case-colliding paths: "
                        + "/".join(parts)
                    )
                folded_names.add(folded)

            root_prefix = _archive_root_prefix(normalised)
            prefix_length = len(root_prefix)
            relative_paths = tuple(
                parts[prefix_length:]
                for parts in normalised
                if len(parts) > prefix_length
            )
            for parts in relative_paths:
                folded_parts = {part.casefold() for part in parts}
                if folded_parts & {part.casefold() for part in _FORBIDDEN_ARCHIVE_PARTS}:
                    raise PackageBuildError(
                        "Extension archive contains a forbidden repository path: "
                        + "/".join(parts)
                    )
                if parts[-1].casefold() in {
                    name.casefold() for name in _FORBIDDEN_ARCHIVE_BASENAMES
                }:
                    raise PackageBuildError(
                        "Extension archive contains a retired runtime source: "
                        + "/".join(parts)
                    )
                if parts[-1].casefold().endswith((".pyc", ".pyo", ".zip")):
                    raise PackageBuildError(
                        "Extension archive contains a forbidden generated/nested file: "
                        + "/".join(parts)
                    )

            manifest_name = "/".join((*root_prefix, "blender_manifest.toml"))
            init_name = "/".join((*root_prefix, "__init__.py"))
            names = {info.filename.rstrip("/"): info for info in infos}
            if manifest_name not in names or init_name not in names:
                raise PackageBuildError(
                    "Extension archive is missing required runtime files"
                )
            if names[init_name].file_size <= 0:
                raise PackageBuildError("Extension archive contains an empty __init__.py")
            try:
                packaged_manifest = tomllib.loads(
                    archive.read(names[manifest_name]).decode("utf-8")
                )
            except Exception as exc:
                raise PackageBuildError(
                    f"Unable to parse packaged blender_manifest.toml: {exc}"
                ) from exc
            for key in ("schema_version", "id", "version", "type", "blender_version_min"):
                if packaged_manifest.get(key) != source_manifest.get(key):
                    raise PackageBuildError(
                        f"Packaged manifest field {key!r} does not match source manifest"
                    )
    except PackageBuildError:
        raise
    except (OSError, zipfile.BadZipFile, UnicodeError) as exc:
        raise PackageBuildError(
            f"Unable to validate built extension archive {archive_path}: {exc}"
        ) from exc


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

    required_text_fields = (
        "schema_version",
        "id",
        "version",
        "name",
        "tagline",
        "maintainer",
        "blender_version_min",
        "type",
    )
    for key in required_text_fields:
        value = manifest.get(key)
        if not isinstance(value, str) or not value.strip():
            if key == "blender_version_min":
                raise PackageBuildError(
                    "blender_manifest.toml must declare blender_version_min = \"5.2.0\""
                )
            raise PackageBuildError(f"Manifest field {key!r} must be a non-empty string")
        if value != value.strip():
            raise PackageBuildError(f"Manifest field {key!r} must not have outer whitespace")

    if manifest["schema_version"] != "1.0.0":
        raise PackageBuildError(
            "blender_manifest.toml must declare schema_version = \"1.0.0\""
        )
    if manifest["type"] != "add-on":
        raise PackageBuildError(
            "blender_manifest.toml must declare type = \"add-on\""
        )
    minimum = manifest["blender_version_min"]
    if minimum != "5.2.0":
        raise PackageBuildError(
            "blender_manifest.toml must declare blender_version_min = \"5.2.0\"; "
            f"found {minimum!r}"
        )

    licenses = manifest.get("license")
    if not isinstance(licenses, list) or not licenses:
        raise PackageBuildError("Manifest field 'license' must be a non-empty list")
    if not all(isinstance(value, str) and value.strip() for value in licenses):
        raise PackageBuildError("Manifest licenses must be non-empty strings")

    permissions = manifest.get("permissions", {})
    if not isinstance(permissions, dict):
        raise PackageBuildError("Manifest permissions must be a table")
    if not all(
        isinstance(key, str) and key.strip()
        and isinstance(value, str) and value.strip()
        for key, value in permissions.items()
    ):
        raise PackageBuildError(
            "Manifest permissions must map non-empty names to non-empty explanations"
        )
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
    manifest = _read_manifest(source)
    try:
        _validate_built_archive(output, source_manifest=manifest)
    except Exception:
        try:
            output.unlink(missing_ok=True)
        except OSError:
            LOGGER.exception("Unable to remove invalid extension archive: %s", output)
        raise
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
