"""Crash-durable, cross-process implementation of the public atomic transaction."""

from __future__ import annotations

from contextlib import contextmanager
import logging
from pathlib import Path
from typing import Iterator

from .atomic_files import (
    AtomicCleanupReport,
    AtomicFileCommitError,
    AtomicFileTransaction as _BaseAtomicFileTransaction,
    AtomicOutputReservation,
    AtomicRecoveryAction,
    AtomicRecoveryRecord,
    recover_stale_atomic_work_files as _recover_legacy_work_files,
)
from .atomic_journal import (
    AtomicCommitJournal,
    AtomicJournalEntry,
    AtomicJournalPhase,
    recover_interrupted_atomic_journals,
)
from .atomic_work_path import (
    build_atomic_backup_path,
    build_atomic_stage_path,
)
from .atomic_work_state import (
    DEFAULT_STALE_WORK_FILE_AGE_SECONDS,
    claim_atomic_final_path,
    unregister_atomic_transaction,
)
from .durable_io import durable_replace, durable_unlink, fsync_file
from .export_events import (
    ExportEventDispatcher,
    ExportEventKind,
    GLOBAL_EXPORT_EVENTS,
)
from .interprocess_lock import InterprocessFileLock


logger = logging.getLogger(__name__)


class DurableAtomicFileTransaction(_BaseAtomicFileTransaction):
    """Atomic output transaction durable across processes and process crashes."""

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._durable_locks: dict[Path, InterprocessFileLock] = {}
        self._durable_journal: AtomicCommitJournal | None = None

    def _release_durable_resources(self) -> tuple[str, ...]:
        failures: list[str] = []
        for path, lock in tuple(self._durable_locks.items()):
            try:
                lock.release()
            except Exception as exc:
                failures.append(f"release interprocess lock {path}: {exc}")
        self._durable_locks.clear()
        return tuple(failures)

    def _close(self) -> tuple[str, ...]:
        if self._closed:
            return ()
        self._closed = True
        unregister_atomic_transaction(self._token)
        return self._release_durable_resources()

    def _recover_directory_once(self, directory: Path) -> None:
        if (
            not self._recover_stale_work_files
            or directory in self._recovered_directories
        ):
            return
        journal_report = recover_interrupted_atomic_journals(directory)
        if journal_report.failures:
            raise AtomicFileCommitError(
                "Unable to recover interrupted commit journals: "
                + "; ".join(journal_report.failures)
            )
        report = _recover_legacy_work_files(
            directory,
            preserve_failed_work_files=self._preserve_failed_work_files,
            minimum_stale_age_seconds=self._minimum_stale_age_seconds,
            dispatcher=self._dispatcher,
            operation_id=self._operation_id,
        )
        self._recovered_directories.add(directory)
        if report.failures:
            raise AtomicFileCommitError(
                "Unable to recover stale work files: " + "; ".join(report.failures)
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
        staged_path = build_atomic_stage_path(
            normalized,
            self._token,
            reservation_index=len(self._entries),
        )
        reservation = AtomicOutputReservation(normalized, staged_path)
        if staged_path.exists():
            durable_unlink(staged_path)

        lock = InterprocessFileLock(
            normalized,
            token=self._token,
            minimum_stale_age_seconds=self._minimum_stale_age_seconds,
        )
        lock.acquire()
        try:
            claim_atomic_final_path(normalized, self._token)
        except Exception:
            lock.release()
            raise
        self._durable_locks[normalized] = lock
        from .atomic_files import _TransactionEntry

        self._entries.append(_TransactionEntry(normalized, staged_path))
        self._final_paths.add(normalized)
        self._event(
            ExportEventKind.OUTPUT_RESERVED,
            "Reserved durable atomic output",
            path=normalized,
            context={
                "staged_path": str(staged_path),
                "lock_path": str(lock.lock_path),
            },
        )
        return reservation

    def _prepare_durable_journal(self) -> None:
        entries: list[AtomicJournalEntry] = []
        for reservation_index, entry in enumerate(self._entries):
            backup = build_atomic_backup_path(
                entry.final_path,
                self._token,
                reservation_index=reservation_index,
            )
            entry.backup_path = backup
            had_original = entry.final_path.exists()
            entries.append(
                AtomicJournalEntry(
                    final_path=entry.final_path,
                    staged_path=entry.staged_path,
                    backup_path=backup,
                    had_original=had_original,
                )
            )
        self._durable_journal = AtomicCommitJournal(
            token=self._token,
            process_id=self._metadata.process_id,
            process_start_marker=self._metadata.process_start_marker,
            entries=entries,
        )
        self._durable_journal.write_phase(AtomicJournalPhase.PREPARED)

    def _backup_existing_outputs(self) -> None:
        for entry in self._entries:
            backup = entry.backup_path
            if backup is None:
                raise RuntimeError("durable journal did not assign a backup path")
            if backup.exists():
                durable_unlink(backup)
            if entry.final_path.exists():
                durable_replace(entry.final_path, backup)

    def _install_staged_outputs(self) -> None:
        for entry in self._entries:
            durable_replace(entry.staged_path, entry.final_path)
            entry.installed = True

    def _restore_installed_outputs(self) -> list[str]:
        failures: list[str] = []
        for entry in reversed(self._entries):
            try:
                if entry.installed and entry.final_path.exists():
                    durable_unlink(entry.final_path)
            except Exception as exc:
                failures.append(f"remove {entry.final_path}: {exc}")
            try:
                if entry.backup_path is not None and entry.backup_path.exists():
                    durable_replace(entry.backup_path, entry.final_path)
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
                    durable_unlink(backup)
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
            for entry in self._entries:
                fsync_file(entry.staged_path)
            self._prepare_durable_journal()
            self._backup_existing_outputs()
            assert self._durable_journal is not None
            self._durable_journal.write_phase(AtomicJournalPhase.BACKED_UP)
            self._install_staged_outputs()
            self._durable_journal.write_phase(AtomicJournalPhase.INSTALLED)
        except Exception as exc:
            failures = self._restore_installed_outputs()
            try:
                failures.extend(self.rollback().failures)
            except Exception as rollback_exc:
                failures.append(f"{type(rollback_exc).__name__}: {rollback_exc}")
            temporary = tuple(
                str(path)
                for entry in self._entries
                for path in (entry.staged_path, entry.backup_path)
                if path is not None and path.exists()
            )
            self._event(
                ExportEventKind.TRANSACTION_FAILED,
                f"Durable atomic commit failed: {exc}",
                context={
                    "stage": "commit",
                    "exception_type": type(exc).__name__,
                    "output_paths": tuple(str(e.final_path) for e in self._entries),
                    "rollback_result": "SUCCEEDED" if not failures else "FAILED",
                    "rollback_failure_count": len(failures),
                    "rollback_failures": tuple(failures),
                    "temporary_resources": temporary,
                },
            )
            suffix = "" if not failures else "; rollback failures: " + "; ".join(failures)
            raise AtomicFileCommitError(
                f"Unable to commit staged output files: {exc}{suffix}"
            ) from exc

        self._committed = True
        failures = list(self._remove_backups_after_commit())
        if self._durable_journal is not None:
            failures.extend(self._durable_journal.remove())
        failures.extend(self._close())
        if failures:
            self._event(
                ExportEventKind.CLEANUP_FAILED,
                "Committed outputs but durable cleanup failed: " + "; ".join(failures),
                context={"failure_count": len(failures)},
            )
        self._event(
            ExportEventKind.COMMIT_SUCCEEDED,
            f"Committed {len(self._entries)} outputs",
            context={"backup_cleanup_failure_count": len(failures)},
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
                    else:
                        durable_unlink(entry.staged_path)
                        removed.append(entry.staged_path)
            except Exception as exc:
                failures.append(f"remove staged {entry.staged_path}: {exc}")
            try:
                if entry.backup_path is not None and entry.backup_path.exists():
                    if entry.final_path.exists():
                        durable_unlink(entry.final_path)
                    durable_replace(entry.backup_path, entry.final_path)
                    entry.backup_path = None
                    restored.append(entry.final_path)
            except Exception as exc:
                failures.append(f"restore backup {entry.final_path}: {exc}")
        if not failures and self._durable_journal is not None:
            failures.extend(self._durable_journal.remove())
        failures.extend(self._close())
        report = AtomicCleanupReport(
            removed_paths=tuple(removed),
            preserved_paths=tuple(preserved),
            restored_paths=tuple(restored),
            failures=tuple(failures),
        )
        if failures:
            message = "Unable to roll back staged files: " + "; ".join(failures)
            self._event(
                ExportEventKind.CLEANUP_FAILED,
                message,
                context={"failure_count": len(failures)},
            )
            raise AtomicFileCommitError(message)
        self._event(
            ExportEventKind.ROLLBACK_SUCCEEDED,
            "Durable atomic rollback completed",
            context={
                "removed_count": len(removed),
                "preserved_count": len(preserved),
                "restored_count": len(restored),
            },
        )
        return report


@contextmanager
def durable_atomic_file_transaction(
    *,
    operation_name: str = "export",
    preserve_failed_work_files: bool | None = None,
    recover_stale_work_files: bool | None = None,
    minimum_stale_age_seconds: float = DEFAULT_STALE_WORK_FILE_AGE_SECONDS,
    dispatcher: ExportEventDispatcher = GLOBAL_EXPORT_EVENTS,
) -> Iterator[DurableAtomicFileTransaction]:
    transaction = DurableAtomicFileTransaction(
        operation_name=operation_name,
        preserve_failed_work_files=preserve_failed_work_files,
        recover_stale_work_files=recover_stale_work_files,
        minimum_stale_age_seconds=minimum_stale_age_seconds,
        dispatcher=dispatcher,
    )
    primary: BaseException | None = None
    try:
        yield transaction
    except BaseException as exc:
        primary = exc
        try:
            transaction.rollback()
        except Exception as cleanup_exc:
            if hasattr(exc, "add_note"):
                exc.add_note(f"Durable atomic cleanup also failed: {cleanup_exc}")
            logger.exception("Durable atomic cleanup also failed")
        raise
    finally:
        if not transaction.committed and not transaction._closed:
            try:
                transaction.rollback()
            except Exception:
                if primary is None:
                    raise
                logger.exception(
                    "Durable atomic cleanup failed while preserving primary exception"
                )


__all__ = [
    "DurableAtomicFileTransaction",
    "durable_atomic_file_transaction",
]
