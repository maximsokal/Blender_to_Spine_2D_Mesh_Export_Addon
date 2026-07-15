"""Atomic staging and rollback for multi-frame texture outputs."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import os
from pathlib import Path
from typing import Iterator
from uuid import uuid4


class AtomicFileCommitError(RuntimeError):
    """Raised when staged outputs cannot be committed or rolled back safely."""


@dataclass(frozen=True, slots=True)
class AtomicOutputReservation:
    final_path: Path
    staged_path: Path


@dataclass(slots=True)
class _TransactionEntry:
    final_path: Path
    staged_path: Path
    backup_path: Path | None = None
    installed: bool = False


class AtomicFileTransaction:
    """Stage several files and expose them together only after full success."""

    def __init__(self) -> None:
        self._token = uuid4().hex
        self._entries: list[_TransactionEntry] = []
        self._final_paths: set[Path] = set()
        self._committed = False
        self._closed = False

    @property
    def committed(self) -> bool:
        return self._committed

    @property
    def reservations(self) -> tuple[AtomicOutputReservation, ...]:
        return tuple(
            AtomicOutputReservation(entry.final_path, entry.staged_path)
            for entry in self._entries
        )

    def reserve(self, final_path: Path) -> AtomicOutputReservation:
        if self._closed:
            raise RuntimeError("transaction is already closed")
        if not isinstance(final_path, Path):
            raise TypeError("final_path must be pathlib.Path")
        normalized = final_path.expanduser().resolve(strict=False)
        if normalized in self._final_paths:
            raise ValueError(f"final output is reserved twice: {normalized}")
        normalized.parent.mkdir(parents=True, exist_ok=True)
        staged_name = f".{normalized.name}.spine2d-stage-{self._token}"
        staged_path = normalized.with_name(staged_name)
        if staged_path.exists():
            staged_path.unlink()
        entry = _TransactionEntry(
            final_path=normalized,
            staged_path=staged_path,
        )
        self._entries.append(entry)
        self._final_paths.add(normalized)
        return AtomicOutputReservation(normalized, staged_path)

    def _validate_staged_files(self) -> None:
        if not self._entries:
            raise AtomicFileCommitError("transaction contains no output reservations")
        missing = tuple(
            entry.staged_path
            for entry in self._entries
            if not entry.staged_path.is_file()
        )
        if missing:
            raise AtomicFileCommitError(
                "staged output files are missing: " + ", ".join(str(path) for path in missing)
            )

    def _backup_existing_outputs(self) -> None:
        for entry in self._entries:
            if not entry.final_path.exists():
                continue
            backup_name = f".{entry.final_path.name}.spine2d-backup-{self._token}"
            backup_path = entry.final_path.with_name(backup_name)
            if backup_path.exists():
                backup_path.unlink()
            os.replace(entry.final_path, backup_path)
            entry.backup_path = backup_path

    def _install_staged_outputs(self) -> None:
        for entry in self._entries:
            os.replace(entry.staged_path, entry.final_path)
            entry.installed = True

    def _remove_backups(self) -> None:
        for entry in self._entries:
            backup = entry.backup_path
            if backup is not None and backup.exists():
                backup.unlink()
            entry.backup_path = None

    def _restore_after_failed_commit(self) -> tuple[str, ...]:
        failures: list[str] = []
        for entry in reversed(self._entries):
            try:
                if entry.installed and entry.final_path.exists():
                    entry.final_path.unlink()
            except Exception as exc:
                failures.append(f"remove {entry.final_path}: {exc}")
            try:
                if entry.backup_path is not None and entry.backup_path.exists():
                    os.replace(entry.backup_path, entry.final_path)
                    entry.backup_path = None
            except Exception as exc:
                failures.append(f"restore {entry.final_path}: {exc}")
            entry.installed = False
        return tuple(failures)

    def commit(self) -> tuple[Path, ...]:
        if self._closed:
            raise RuntimeError("transaction is already closed")
        try:
            self._validate_staged_files()
            self._backup_existing_outputs()
            self._install_staged_outputs()
            self._remove_backups()
        except Exception as exc:
            rollback_failures = self._restore_after_failed_commit()
            self.rollback()
            details = ""
            if rollback_failures:
                details = "; rollback failures: " + "; ".join(rollback_failures)
            raise AtomicFileCommitError(
                f"Unable to commit staged output files: {exc}{details}"
            ) from exc

        self._committed = True
        self._closed = True
        return tuple(entry.final_path for entry in self._entries)

    def rollback(self) -> None:
        if self._committed:
            return
        failures: list[str] = []
        for entry in reversed(self._entries):
            try:
                if entry.staged_path.exists():
                    entry.staged_path.unlink()
            except Exception as exc:
                failures.append(f"remove staged {entry.staged_path}: {exc}")
            try:
                if entry.backup_path is not None and entry.backup_path.exists():
                    if entry.final_path.exists():
                        entry.final_path.unlink()
                    os.replace(entry.backup_path, entry.final_path)
                    entry.backup_path = None
            except Exception as exc:
                failures.append(f"restore backup {entry.final_path}: {exc}")
        self._closed = True
        if failures:
            raise AtomicFileCommitError(
                "Unable to roll back staged output files: " + "; ".join(failures)
            )


@contextmanager
def atomic_file_transaction() -> Iterator[AtomicFileTransaction]:
    transaction = AtomicFileTransaction()
    try:
        yield transaction
    except Exception:
        transaction.rollback()
        raise
    finally:
        if not transaction.committed and not transaction._closed:
            transaction.rollback()
