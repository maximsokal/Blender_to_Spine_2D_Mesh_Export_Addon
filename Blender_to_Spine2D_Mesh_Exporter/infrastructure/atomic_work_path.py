"""Path-budgeted filenames for atomic stage and backup work files.

Blender and OpenImageIO still cross Windows APIs that may reject paths near the
legacy ``MAX_PATH`` boundary even when Python itself can resolve the same path.
Atomic work filenames historically repeated the complete final stem and appended a
long crash-recovery token, making a valid final output path substantially longer
while it was staged.

This module keeps final filenames untouched. It only compacts transaction-owned
work filenames when the complete path exceeds a conservative external-I/O budget.
The transaction token remains present in stage filenames, so stale-work ownership
classification continues to work. Durable backup ownership remains recorded in
the crash journal.
"""

from __future__ import annotations

import os
from pathlib import Path

from .atomic_work_state import BACKUP_MARKER, STAGE_MARKER


# Leave headroom below the traditional 260 UTF-16-code-unit boundary used by a
# number of Blender/OpenImageIO Windows code paths. The final output path is not
# rewritten; only temporary transaction work names are budgeted.
WINDOWS_EXTERNAL_IO_PATH_BUDGET = 240


class AtomicWorkPathError(RuntimeError):
    """Raised when no safe sibling work filename fits beside a final output."""


def _validate_final_path(final_path: Path) -> Path:
    if not isinstance(final_path, Path):
        raise TypeError("final_path must be pathlib.Path")
    normalized = final_path.expanduser().resolve(strict=False)
    if not normalized.is_absolute():
        raise ValueError("final_path must resolve to an absolute path")
    if not normalized.name:
        raise ValueError("final_path must include a filename")
    return normalized


def _validate_token(token: str) -> str:
    if not isinstance(token, str) or not token:
        raise ValueError("token must be a non-empty string")
    if token != token.strip():
        raise ValueError("token cannot contain boundary whitespace")
    if any(separator in token for separator in ("/", "\\", ":")):
        raise ValueError("token must be filename-safe")
    return token


def _validate_reservation_index(reservation_index: int) -> int:
    if isinstance(reservation_index, bool) or not isinstance(reservation_index, int):
        raise TypeError("reservation_index must be int")
    if reservation_index < 0:
        raise ValueError("reservation_index must be non-negative")
    return reservation_index


def _validate_path_budget(path_budget: int | None) -> int | None:
    if path_budget is None:
        return None
    if isinstance(path_budget, bool) or not isinstance(path_budget, int):
        raise TypeError("path_budget must be int or None")
    if path_budget <= 0:
        raise ValueError("path_budget must be positive")
    return path_budget


def _windows_utf16_code_units(path: Path) -> int:
    """Return Windows path length in UTF-16 code units, excluding the NUL."""

    value = os.fspath(path)
    return len(value.encode("utf-16-le")) // 2


def _effective_budget(path_budget: int | None) -> int | None:
    explicit = _validate_path_budget(path_budget)
    if explicit is not None:
        return explicit
    return WINDOWS_EXTERNAL_IO_PATH_BUDGET if os.name == "nt" else None


def _fits(path: Path, path_budget: int | None) -> bool:
    return path_budget is None or _windows_utf16_code_units(path) <= path_budget


def _resolve_sibling(final_path: Path, filename: str) -> Path:
    return final_path.with_name(filename).resolve(strict=False)


def _raise_budget_error(
    *,
    final_path: Path,
    compact_path: Path,
    path_budget: int,
    work_kind: str,
) -> None:
    raise AtomicWorkPathError(
        f"{work_kind} path cannot fit the Windows external-I/O budget of "
        f"{path_budget} UTF-16 code units. Choose a shorter export directory. "
        f"final='{final_path}', compact_work='{compact_path}', "
        f"compact_length={_windows_utf16_code_units(compact_path)}"
    )


def build_atomic_stage_path(
    final_path: Path,
    token: str,
    *,
    reservation_index: int,
    path_budget: int | None = None,
) -> Path:
    """Return a same-directory stage path safe for Blender external image I/O.

    The readable historical filename is retained while it fits. On overflow the
    final stem is replaced with the deterministic reservation index. The complete
    transaction token and final suffix are retained, preserving uniqueness and
    stale-work token parsing without probabilistic hashes.
    """

    normalized = _validate_final_path(final_path)
    resolved_token = _validate_token(token)
    index = _validate_reservation_index(reservation_index)
    budget = _effective_budget(path_budget)

    readable = _resolve_sibling(
        normalized,
        f".{normalized.stem}{STAGE_MARKER}{resolved_token}{normalized.suffix}",
    )
    if _fits(readable, budget):
        return readable

    compact = _resolve_sibling(
        normalized,
        f".s{index:x}{STAGE_MARKER}{resolved_token}{normalized.suffix}",
    )
    if _fits(compact, budget):
        return compact

    assert budget is not None
    _raise_budget_error(
        final_path=normalized,
        compact_path=compact,
        path_budget=budget,
        work_kind="Atomic stage",
    )


def build_atomic_backup_path(
    final_path: Path,
    token: str,
    *,
    reservation_index: int,
    path_budget: int | None = None,
) -> Path:
    """Return a same-directory durable backup path within the Windows budget.

    Durable transaction journals store the exact final, stage, and backup mapping.
    A compact backup therefore does not need to repeat the complete final filename.
    The compact form begins with an empty legacy owner segment (two leading dots),
    preventing legacy filename-only recovery from inventing an incorrect final path
    if a journal is unavailable; such an orphan remains fail-closed for diagnosis.
    """

    normalized = _validate_final_path(final_path)
    resolved_token = _validate_token(token)
    index = _validate_reservation_index(reservation_index)
    budget = _effective_budget(path_budget)

    readable = _resolve_sibling(
        normalized,
        f".{normalized.name}{BACKUP_MARKER}{resolved_token}",
    )
    if _fits(readable, budget):
        return readable

    compact = _resolve_sibling(
        normalized,
        f".{BACKUP_MARKER}{resolved_token}.{index:x}",
    )
    if _fits(compact, budget):
        return compact

    assert budget is not None
    _raise_budget_error(
        final_path=normalized,
        compact_path=compact,
        path_budget=budget,
        work_kind="Atomic backup",
    )


__all__ = [
    "AtomicWorkPathError",
    "WINDOWS_EXTERNAL_IO_PATH_BUDGET",
    "build_atomic_backup_path",
    "build_atomic_stage_path",
]
