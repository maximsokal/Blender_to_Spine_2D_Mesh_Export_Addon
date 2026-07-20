"""Atomic output staging, rollback, lifecycle events, and stale-work recovery."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum
import logging
import math
import os
from pathlib import Path
from typing import Iterator

from .atomic_work_state import (
    AtomicRecoveryReason,
    AtomicWorkFileState,
    BACKUP_MARKER,
    DEFAULT_STALE_WORK_FILE_AGE_SECONDS,
    STAGE_MARKER,
    assess_atomic_work_file,
    claim_atomic_final_path,
    create_atomic_work_token_metadata,
    register_atomic_transaction,
    unregister_atomic_transaction,
)
from .export_diagnostics import get_export_diagnostics_policy
from .export_events import (
    ExportEvent,
    ExportEventDispatcher,
    ExportEventKind,
    GLOBAL_EXPORT_EVENTS,
)


logger = logging.getLogger(__name__)


class AtomicFileCommitError(RuntimeError):
    """Raised when staged outputs cannot be committed or rolled back safely."""


class AtomicRecoveryAction(str, Enum):
    """Filesystem action taken for one stale work file."""

    REMOVED_STAGE = "REMOVED_STAGE"
    PRESERVED_STAGE = "PRESERVED_STAGE"
    RESTORED_BACKUP = "RESTORED_BACKUP"
    REMOVED_REDUNDANT_BACKUP = "REMOVED_REDUNDANT_BACKUP"


@dataclass(frozen=True, slots=True)
class AtomicOutputReservation:
    final_path: Path
    staged_path: Path

    def __post_init__(self) -> None:
        for field_name, path in (
            ("final_path", self.final_path),
            ("staged_path", self.staged_path),
        ):
            if not isinstance(path, Path):
                raise TypeError(f"{field_name} must be pathlib.Path")
            if not path.is_absolute():
                raise ValueError(f"{field_name} must be absolute")
            if path != path.expanduser().resolve(strict=False):
                raise ValueError(f"{field_name} must be normalized")
        if self.final_path == self.staged_path:
            raise ValueError("final_path and staged_path must be different")
        if self.final_path.parent != self.staged_path.parent:
            raise ValueError("final_path and staged_path must share one directory")
        if STAGE_MARKER not in self.staged_path.name:
            raise ValueError("staged_path must contain the atomic stage marker")


@dataclass(frozen=True, slots=True)
class AtomicRecoveryRecord:
    """One explicit stale-work mutation and the reason that authorized it."""

    work_path: Path
    action: AtomicRecoveryAction
    reason: AtomicRecoveryReason
    final_path: Path | None = None
    age_seconds: float = 0.0

    def __post_init__(self) -> None:
        if not isinstance(self.work_path, Path):
            raise TypeError("work_path must be pathlib.Path")
        if not isinstance(self.action, AtomicRecoveryAction):
            raise TypeError("action must be AtomicRecoveryAction")
        if not isinstance(self.reason, AtomicRecoveryReason):
            raise TypeError("reason must be AtomicRecoveryReason")
        if self.final_path is not None and not isinstance(self.final_path, Path):
            raise TypeError("final_path must be pathlib.Path or None")
        if isinstance(self.age_seconds, bool) or not isinstance(
            self.age_seconds, (int, float)
        ):
            raise TypeError("age_seconds must be a finite number")
        if not math.isfinite(float(self.age_seconds)) or self.age_seconds < 0.0:
            raise ValueError("age_seconds must be finite and non-negative")


@dataclass(frozen=True, slots=True)
class AtomicCleanupReport:
    removed_paths: tuple[Path, ...] = ()
    preserved_paths: tuple[Path, ...] = ()
    restored_paths: tuple[Path, ...] = ()
    failures: tuple[str, ...] = ()
    recovery_records: tuple[AtomicRecoveryRecord, ...] = ()

    def __post_init__(self) -> None:
        for field_name in (
            "removed_paths",
            "preserved_paths",
            "restored_paths",
            "failures",
            "recovery_records",
        ):
            if not isinstance(getattr(self, field_name), tuple):
                raise TypeError(f"{field_name} must be tuple")
        if not all(
            isinstance(path, Path)
            for path in (
                *self.removed_paths,
                *self.preserved_paths,
                *self.restored_paths,
            )
        ):
            raise TypeError("cleanup report paths must be pathlib.Path values")
        if not all(isinstance(failure, str) for failure in self.failures):
            raise TypeError("failures must contain strings")
        if not all(
            isinstance(record, AtomicRecoveryRecord)
            for record in self.recovery_records
        ):
            raise TypeError(
                "recovery_records must contain AtomicRecoveryRecord values"
            )

    @property
    def success(self) -> bool:
        return not self.failures


@dataclass(slots=True)
class _TransactionEntry:
    final_path: Path
    staged_path: Path
    backup_path: Path | None = None
    installed: bool = False


def _resolve_optional_bool(
    value: bool | None,
    default: bool,
    field_name: str,
) -> bool:
    if value is None:
        return default
    if not isinstance(value, bool):
        raise TypeError(f"{field_name} must be bool or None")
    return value


def _validate_stale_age(value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError("minimum_stale_age_seconds must be a finite number")
    resolved = float(value)
    if not math.isfinite(resolved) or resolved < 0.0:
        raise ValueError(
            "minimum_stale_age_seconds must be finite and non-negative"
        )
    return resolved


def _emit(
    dispatcher: ExportEventDispatcher,
    kind: ExportEventKind,
    operation_id: str,
    message: str,
    *,
    path: Path | None = None,
    context: dict[str, object] | None = None,
) -> None:
    dispatcher.emit(
        ExportEvent(
            kind=kind,
            operation_id=operation_id,
            message=message,
            path=path,
            context={} if context is None else context,
        )
    )


def _backup_final_path(path: Path) -> Path | None:
    if not path.name.startswith(".") or BACKUP_MARKER not in path.name:
        return None
    original_name = path.name[1:].split(BACKUP_MARKER, 1)[0]
    return path.with_name(original_name) if original_name else None


def _recovery_context(assessment) -> dict[str, object]:
    return {
        "recovery_reason": assessment.reason.value,
        "work_file_age_seconds": assessment.age_seconds,
        "work_token": assessment.token or "",
    }


def recover_stale_atomic_work_files(
    directory: Path,
    *,
    preserve_failed_work_files: bool | None = None,
    minimum_stale_age_seconds: float = DEFAULT_STALE_WORK_FILE_AGE_SECONDS,
    dispatcher: ExportEventDispatcher = GLOBAL_EXPORT_EVENTS,
    operation_id: str = "startup-recovery",
) -> AtomicCleanupReport:
    """Restore abandoned backups and remove/preserve confirmed stale stage files."""

    if not isinstance(directory, Path):
        raise TypeError("directory must be pathlib.Path")
    if not isinstance(operation_id, str) or not operation_id.strip():
        raise ValueError("operation_id must be a non-empty string")
    minimum_age = _validate_stale_age(minimum_stale_age_seconds)
    root = directory.expanduser().resolve(strict=False)
    if not root.exists():
        return AtomicCleanupReport()
    if not root.is_dir():
        raise ValueError(f"recovery path is not a directory: {root}")

    policy = get_export_diagnostics_policy()
    preserve = _resolve_optional_bool(
        preserve_failed_work_files,
        policy.preserve_failed_work_files,
        "preserve_failed_work_files",
    )
    removed: list[Path] = []
    preserved: list[Path] = []
    restored: list[Path] = []
    failures: list[str] = []
    recovery_records: list[AtomicRecoveryRecord] = []

    candidates = sorted(
        (
            path
            for path in root.iterdir()
            if path.is_file()
            and (STAGE_MARKER in path.name or BACKUP_MARKER in path.name)
        ),
        key=lambda path: path.name,
    )
    for path in candidates:
        assessment = assess_atomic_work_file(
            path,
            minimum_stale_age_seconds=minimum_age,
        )
        if assessment.state is not AtomicWorkFileState.STALE:
            continue
        context = _recovery_context(assessment)
        try:
            if BACKUP_MARKER in path.name:
                final_path = _backup_final_path(path)
                if final_path is None:
                    raise ValueError("cannot resolve backup owner")
                if final_path.exists():
                    path.unlink()
                    removed.append(path)
                    action = AtomicRecoveryAction.REMOVED_REDUNDANT_BACKUP
                    recovery_records.append(
                        AtomicRecoveryRecord(
                            work_path=path,
                            action=action,
                            reason=assessment.reason,
                            final_path=final_path,
                            age_seconds=assessment.age_seconds,
                        )
                    )
                    _emit(
                        dispatcher,
                        ExportEventKind.STALE_WORK_RECOVERED,
                        operation_id,
                        (
                            "Removed stale backup because the final output "
                            "already exists"
                        ),
                        path=path,
                        context={
                            **context,
                            "recovery_action": action.value,
                        },
                    )
                else:
                    os.replace(path, final_path)
                    restored.append(final_path)
                    action = AtomicRecoveryAction.RESTORED_BACKUP
                    recovery_records.append(
                        AtomicRecoveryRecord(
                            work_path=path,
                            action=action,
                            reason=assessment.reason,
                            final_path=final_path,
                            age_seconds=assessment.age_seconds,
                        )
                    )
                    _emit(
                        dispatcher,
                        ExportEventKind.BACKUP_RESTORED,
                        operation_id,
                        "Restored backup left by interrupted commit",
                        path=final_path,
                        context={
                            **context,
                            "recovery_action": action.value,
                        },
                    )
                continue

            if preserve:
                preserved.append(path)
                action = AtomicRecoveryAction.PRESERVED_STAGE
                recovery_records.append(
                    AtomicRecoveryRecord(
                        work_path=path,
                        action=action,
                        reason=assessment.reason,
                        age_seconds=assessment.age_seconds,
                    )
                )
                _emit(
                    dispatcher,
                    ExportEventKind.WORK_FILE_PRESERVED,
                    operation_id,
                    "Preserved stale stage file by diagnostics policy",
                    path=path,
                    context={
                        **context,
                        "recovery_action": action.value,
                    },
                )
            else:
                path.unlink()
                removed.append(path)
                action = AtomicRecoveryAction.REMOVED_STAGE
                recovery_records.append(
                    AtomicRecoveryRecord(
                        work_path=path,
                        action=action,
                        reason=assessment.reason,
                        age_seconds=assessment.age_seconds,
                    )
                )
                _emit(
                    dispatcher,
                    ExportEventKind.STALE_WORK_RECOVERED,
                    operation_id,
                    "Removed stale stage file from interrupted export",
                    path=path,
                    context={
                        **context,
                        "recovery_action": action.value,
                    },
                )
        except Exception as exc:
            failure = f"recover {path}: {exc}"
            failures.append(failure)
            _emit(
                dispatcher,
                ExportEventKind.CLEANUP_FAILED,
                operation_id,
                failure,
                path=path,
                context={
                    **context,
                    "exception_type": type(exc).__name__,
                },
            )

    return AtomicCleanupReport(
        removed_paths=tuple(removed),
        preserved_paths=tuple(preserved),
        restored_paths=tuple(restored),
        failures=tuple(failures),
        recovery_records=tuple(recovery_records),
    )


class AtomicFileTransaction:
    """Stage several files and expose them together only after full success."""

    def __init__(
        self,
        *,
        operation_name: str = "export",
        preserve_failed_work_files: bool | None = None,
        recover_stale_work_files: bool | None = None,
        minimum_stale_age_seconds: float = DEFAULT_STALE_WORK_FILE_AGE_SECONDS,
        dispatcher: ExportEventDispatcher = GLOBAL_EXPORT_EVENTS,
    ) -> None:
        if not isinstance(operation_name, str) or not operation_name.strip():
            raise ValueError("operation_name must be a non-empty string")
        if operation_name != operation_name.strip():
            raise ValueError(
                "operation_name cannot contain boundary whitespace"
            )
        policy = get_export_diagnostics_policy()
        self._preserve_failed_work_files = _resolve_optional_bool(
            preserve_failed_work_files,
            policy.preserve_failed_work_files,
            "preserve_failed_work_files",
        )
        self._recover_stale_work_files = _resolve_optional_bool(
            recover_stale_work_files,
            policy.recover_stale_work_files,
            "recover_stale_work_files",
        )
        self._minimum_stale_age_seconds = _validate_stale_age(
            minimum_stale_age_seconds
        )
        self._metadata = create_atomic_work_token_metadata()
        self._token = self._metadata.token
        self._operation_id = f"{operation_name}:{self._token}"
        self._dispatcher = dispatcher
        self._entries: list[_TransactionEntry] = []
        self._final_paths: set[Path] = set()
        self._recovered_directories: set[Path] = set()
        self._committed = False
        self._closed = False
        register_atomic_transaction(self._metadata)
        self._event(
            ExportEventKind.TRANSACTION_STARTED,
            "Started atomic output transaction",
            context={
                "process_id": self._metadata.process_id,
                "process_start_marker": self._metadata.process_start_marker,
                "created_ns": self._metadata.created_ns,
            },
        )

    @property
    def committed(self) -> bool:
        return self._committed

    @property
    def reservations(self) -> tuple[AtomicOutputReservation, ...]:
        return tuple(
            AtomicOutputReservation(
                entry.final_path,
                entry.staged_path,
            )
            for entry in self._entries
        )

    def _event(
        self,
        kind: ExportEventKind,
        message: str,
        *,
        path: Path | None = None,
        context: dict[str, object] | None = None,
    ) -> None:
        _emit(
            self._dispatcher,
            kind,
            self._operation_id,
            message,
            path=path,
            context=context,
        )

    def _close(self) -> None:
        if self._closed:
            return
        self._closed = True
        unregister_atomic_transaction(self._token)

    def _recover_directory_once(self, directory: Path) -> None:
        if (
            not self._recover_stale_work_files
            or directory in self._recovered_directories
        ):
            return
        report = recover_stale_atomic_work_files(
            directory,
            preserve_failed_work_files=self._preserve_failed_work_files,
            minimum_stale_age_seconds=self._minimum_stale_age_seconds,
            dispatcher=self._dispatcher,
            operation_id=self._operation_id,
        )
        self._recovered_directories.add(directory)
        if report.failures:
            raise AtomicFileCommitError(
                "Unable to recover stale work files: "
                + "; ".join(report.failures)
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
        self._recover_directory_once(normalized.parent)
        staged_path = normalized.with_name(
            f".{normalized.stem}{STAGE_MARKER}{self._token}{normalized.suffix}"
        ).resolve(strict=False)
        reservation = AtomicOutputReservation(normalized, staged_path)

        # The stage filename contains this transaction's unique token. It cannot
        # belong to another active transaction, so remove an abandoned same-token
        # file before publishing the process-local final-path claim.
        if staged_path.exists():
            staged_path.unlink()
        claim_atomic_final_path(normalized, self._token)
        self._entries.append(_TransactionEntry(normalized, staged_path))
        self._final_paths.add(normalized)
        self._event(
            ExportEventKind.OUTPUT_RESERVED,
            "Reserved atomic output",
            path=normalized,
            context={"staged_path": str(staged_path)},
        )
        return reservation

    def _validate_staged_files(self) -> None:
        if not self._entries:
            raise AtomicFileCommitError(
                "transaction contains no output reservations"
            )
        missing = tuple(
            entry.staged_path
            for entry in self._entries
            if not entry.staged_path.is_file()
        )
        if missing:
            raise AtomicFileCommitError(
                "staged output files are missing: "
                + ", ".join(str(path) for path in missing)
            )

    def _backup_existing_outputs(self) -> None:
        for entry in self._entries:
            if entry.final_path.exists():
                backup = entry.final_path.with_name(
                    f".{entry.final_path.name}{BACKUP_MARKER}{self._token}"
                )
                if backup.exists():
                    backup.unlink()
                os.replace(entry.final_path, backup)
                entry.backup_path = backup

    def _install_staged_outputs(self) -> None:
        for entry in self._entries:
            os.replace(entry.staged_path, entry.final_path)
            entry.installed = True

    def _restore_installed_outputs(self) -> list[str]:
        failures: list[str] = []
        for entry in reversed(self._entries):
            try:
                if entry.installed and entry.final_path.exists():
                    entry.final_path.unlink()
            except Exception as exc:
                failures.append(f"remove {entry.final_path}: {exc}")
            try:
                if (
                    entry.backup_path is not None
                    and entry.backup_path.exists()
                ):
                    os.replace(entry.backup_path, entry.final_path)
                    entry.backup_path = None
                    self._event(
                        ExportEventKind.BACKUP_RESTORED,
                        "Restored output backup",
                        path=entry.final_path,
                    )
            except Exception as exc:
                failures.append(f"restore {entry.final_path}: {exc}")
            entry.installed = False
        return failures

    def _remove_backups_after_commit(self) -> tuple[str, ...]:
        failures: list[str] = []
        for entry in self._entries:
            backup = entry.backup_path
            if backup is None:
                continue
            try:
                if backup.exists():
                    backup.unlink()
                    self._event(
                        ExportEventKind.WORK_FILE_REMOVED,
                        "Removed commit backup",
                        path=backup,
                    )
                entry.backup_path = None
            except Exception as exc:
                failures.append(f"remove backup {backup}: {exc}")
        return tuple(failures)

    def commit(self) -> tuple[Path, ...]:
        if self._closed:
            raise RuntimeError("transaction is already closed")
        self._event(
            ExportEventKind.COMMIT_STARTED,
            f"Committing {len(self._entries)} outputs",
        )
        try:
            self._validate_staged_files()
            self._backup_existing_outputs()
            self._install_staged_outputs()
        except Exception as exc:
            self._event(
                ExportEventKind.TRANSACTION_FAILED,
                f"Atomic commit failed: {exc}",
                context={"exception_type": type(exc).__name__},
            )
            failures = self._restore_installed_outputs()
            try:
                failures.extend(self.rollback().failures)
            except Exception as rollback_exc:
                failures.append(str(rollback_exc))
            suffix = (
                ""
                if not failures
                else "; rollback failures: " + "; ".join(failures)
            )
            raise AtomicFileCommitError(
                f"Unable to commit staged output files: {exc}{suffix}"
            ) from exc

        self._committed = True
        self._close()
        cleanup_failures = self._remove_backups_after_commit()
        if cleanup_failures:
            self._event(
                ExportEventKind.CLEANUP_FAILED,
                "Committed outputs but could not remove every backup: "
                + "; ".join(cleanup_failures),
                context={"failure_count": len(cleanup_failures)},
            )
        self._event(
            ExportEventKind.COMMIT_SUCCEEDED,
            f"Committed {len(self._entries)} outputs",
            context={
                "backup_cleanup_failure_count": len(cleanup_failures)
            },
        )
        return tuple(entry.final_path for entry in self._entries)

    def rollback(self) -> AtomicCleanupReport:
        if self._committed or self._closed:
            return AtomicCleanupReport()
        self._event(
            ExportEventKind.ROLLBACK_STARTED,
            f"Rolling back {len(self._entries)} outputs",
        )
        removed: list[Path] = []
        preserved: list[Path] = []
        restored: list[Path] = []
        failures = self._restore_installed_outputs()

        for entry in reversed(self._entries):
            try:
                if entry.staged_path.exists():
                    if self._preserve_failed_work_files:
                        preserved.append(entry.staged_path)
                        self._event(
                            ExportEventKind.WORK_FILE_PRESERVED,
                            "Preserved failed stage output",
                            path=entry.staged_path,
                        )
                    else:
                        entry.staged_path.unlink()
                        removed.append(entry.staged_path)
                        self._event(
                            ExportEventKind.WORK_FILE_REMOVED,
                            "Removed failed stage output",
                            path=entry.staged_path,
                        )
            except Exception as exc:
                failures.append(
                    f"remove staged {entry.staged_path}: {exc}"
                )
            try:
                if (
                    entry.backup_path is not None
                    and entry.backup_path.exists()
                ):
                    if entry.final_path.exists():
                        entry.final_path.unlink()
                    os.replace(entry.backup_path, entry.final_path)
                    entry.backup_path = None
                    restored.append(entry.final_path)
            except Exception as exc:
                failures.append(
                    f"restore backup {entry.final_path}: {exc}"
                )

        self._close()
        report = AtomicCleanupReport(
            removed_paths=tuple(removed),
            preserved_paths=tuple(preserved),
            restored_paths=tuple(restored),
            failures=tuple(failures),
        )
        if failures:
            message = (
                "Unable to roll back staged files: "
                + "; ".join(failures)
            )
            self._event(
                ExportEventKind.CLEANUP_FAILED,
                message,
                context={"failure_count": len(failures)},
            )
            raise AtomicFileCommitError(message)
        self._event(
            ExportEventKind.ROLLBACK_SUCCEEDED,
            "Atomic rollback completed",
            context={
                "removed_count": len(removed),
                "preserved_count": len(preserved),
                "restored_count": len(restored),
            },
        )
        return report


@contextmanager
def atomic_file_transaction(
    *,
    operation_name: str = "export",
    preserve_failed_work_files: bool | None = None,
    recover_stale_work_files: bool | None = None,
    minimum_stale_age_seconds: float = DEFAULT_STALE_WORK_FILE_AGE_SECONDS,
    dispatcher: ExportEventDispatcher = GLOBAL_EXPORT_EVENTS,
) -> Iterator[AtomicFileTransaction]:
    transaction = AtomicFileTransaction(
        operation_name=operation_name,
        preserve_failed_work_files=preserve_failed_work_files,
        recover_stale_work_files=recover_stale_work_files,
        minimum_stale_age_seconds=minimum_stale_age_seconds,
        dispatcher=dispatcher,
    )
    primary_exception: BaseException | None = None
    try:
        yield transaction
    except BaseException as exc:
        primary_exception = exc
        try:
            transaction.rollback()
        except Exception as cleanup_exc:
            note = f"Atomic cleanup also failed: {cleanup_exc}"
            if hasattr(exc, "add_note"):
                exc.add_note(note)
            logger.exception(note)
        raise
    finally:
        if not transaction.committed and not transaction._closed:
            try:
                transaction.rollback()
            except Exception:
                if primary_exception is None:
                    raise
                logger.exception(
                    "Atomic cleanup failed while preserving primary exception"
                )


__all__ = [
    "AtomicCleanupReport",
    "AtomicFileCommitError",
    "AtomicFileTransaction",
    "AtomicOutputReservation",
    "AtomicRecoveryAction",
    "AtomicRecoveryRecord",
    "atomic_file_transaction",
    "recover_stale_atomic_work_files",
]
