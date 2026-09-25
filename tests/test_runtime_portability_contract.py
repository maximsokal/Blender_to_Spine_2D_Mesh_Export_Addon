"""Static portability contracts for the manifest-eligible extension runtime."""

from __future__ import annotations

import ast
from fnmatch import fnmatchcase
from pathlib import Path
import re
import tomllib


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "Blender_to_Spine2D_Mesh_Exporter"
MANIFEST = PACKAGE / "blender_manifest.toml"
_NATIVE_SUFFIXES = frozenset({".dll", ".pyd", ".so", ".dylib"})
_FORBIDDEN_PLATFORM_IMPORTS = frozenset(
    {
        "msvcrt",
        "winreg",
        "fcntl",
        "subprocess",
    }
)
_ALLOWED_CTYPES_FILES = frozenset(
    {
        "infrastructure/atomic_work_state.py",
    }
)
_WINDOWS_DRIVE_LITERAL = re.compile(r"^[A-Za-z]:[\\/]")
_WINDOWS_UNC_LITERAL = re.compile(r"^\\\\[^\\/\s]+[\\/][^\\/\s]+(?:[\\/]|$)")


def _exclude_patterns() -> tuple[str, ...]:
    manifest = tomllib.loads(MANIFEST.read_text(encoding="utf-8"))
    raw = manifest.get("build", {}).get("paths_exclude_pattern", ())
    assert isinstance(raw, list)
    return tuple(str(value).replace("\\", "/") for value in raw)


def _is_excluded(path: Path) -> bool:
    relative = path.relative_to(PACKAGE).as_posix()
    rooted = f"/{relative}"

    for pattern in _exclude_patterns():
        if pattern.startswith("/"):
            if pattern.endswith("/"):
                directory = pattern.rstrip("/")
                if rooted == directory or rooted.startswith(f"{directory}/"):
                    return True
            elif fnmatchcase(rooted, pattern):
                return True
            continue

        if pattern.endswith("/"):
            if pattern.rstrip("/") in path.relative_to(PACKAGE).parts:
                return True
            continue

        if fnmatchcase(path.name, pattern) or fnmatchcase(relative, pattern):
            return True

    return False


def _shipped_files() -> tuple[Path, ...]:
    return tuple(
        sorted(
            path
            for path in PACKAGE.rglob("*")
            if path.is_file() and not _is_excluded(path)
        )
    )


def _shipped_python_files() -> tuple[Path, ...]:
    return tuple(path for path in _shipped_files() if path.suffix == ".py")


def _imported_roots(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots.update(alias.name.partition(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            roots.add(node.module.partition(".")[0])
    return roots


def _string_literals(path: Path) -> tuple[tuple[int, str], ...]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    values: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            values.append((int(getattr(node, "lineno", 0) or 0), node.value))
    return tuple(values)


def test_manifest_does_not_reintroduce_a_platform_restriction() -> None:
    manifest = tomllib.loads(MANIFEST.read_text(encoding="utf-8"))
    assert "platforms" not in manifest


def test_shipped_runtime_has_no_subprocess_or_unconditional_os_specific_imports() -> None:
    offenders: list[str] = []

    for path in _shipped_python_files():
        relative = path.relative_to(PACKAGE).as_posix()
        imports = _imported_roots(path)
        forbidden = sorted(imports.intersection(_FORBIDDEN_PLATFORM_IMPORTS))
        if forbidden:
            offenders.append(f"{relative}: {', '.join(forbidden)}")

        if "ctypes" in imports and relative not in _ALLOWED_CTYPES_FILES:
            offenders.append(f"{relative}: ctypes")

    assert offenders == [], "Unexpected platform/process imports:\n" + "\n".join(offenders)


def test_windows_process_api_is_locally_guarded_and_has_posix_fallback() -> None:
    path = PACKAGE / "infrastructure" / "atomic_work_state.py"
    source = path.read_text(encoding="utf-8")

    assert 'if os.name == "nt":' in source
    assert "return _windows_process_is_alive(process_id)" in source
    assert "os.kill(process_id, 0)" in source
    assert 'if os.name == "posix" and Path("/proc").is_dir():' in source
    assert "_linux_process_start_marker(process_id)" in source
    assert "_windows_process_start_marker(process_id)" in source
    assert "import ctypes" in source
    assert "from ctypes import wintypes" in source


def test_windows_path_budget_is_optional_on_non_windows_hosts() -> None:
    path = PACKAGE / "infrastructure" / "atomic_work_path.py"
    source = path.read_text(encoding="utf-8")

    assert 'return WINDOWS_EXTERNAL_IO_PATH_BUDGET if os.name == "nt" else None' in source
    assert "path_budget is None" in source


def test_durable_io_uses_portable_filesystem_primitives_with_guarded_windows_limit() -> None:
    path = PACKAGE / "infrastructure" / "durable_io.py"
    source = path.read_text(encoding="utf-8")

    for required in (
        "os.replace(",
        "os.fsync(",
        "Path",
        'hasattr(os, "O_BINARY")',
        'hasattr(os, "O_DIRECTORY")',
        'os.name == "nt"',
    ):
        assert required in source


def test_shipped_python_contains_no_absolute_windows_drive_or_unc_path_literals() -> None:
    offenders: list[str] = []

    for path in _shipped_python_files():
        relative = path.relative_to(PACKAGE).as_posix()
        for line, value in _string_literals(path):
            if _WINDOWS_DRIVE_LITERAL.match(value) or _WINDOWS_UNC_LITERAL.match(value):
                offenders.append(f"{relative}:{line}: {value!r}")

    assert offenders == [], "Hard-coded Windows paths in shipped runtime:\n" + "\n".join(
        offenders
    )


def test_shipped_package_contains_no_native_platform_binaries() -> None:
    native = tuple(
        path.relative_to(PACKAGE).as_posix()
        for path in _shipped_files()
        if path.suffix.casefold() in _NATIVE_SUFFIXES
    )
    assert native == ()
