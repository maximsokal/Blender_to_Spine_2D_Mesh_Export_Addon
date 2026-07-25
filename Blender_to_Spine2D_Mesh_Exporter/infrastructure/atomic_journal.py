"""Durable commit journals and deterministic interrupted-transaction recovery."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import json
from pathlib import Path
import time
from typing import Iterable, Mapping

from .process_identity import process_identity_is_alive
from .durable_io import durable_replace, durable_unlink, fsync_directory, fsync_file, write_json_durable

JOURNAL_PREFIX = ".spine2d-journal-"
JOURNAL_SUFFIX = ".json"

class AtomicJournalError(RuntimeError):
    """Raised when a durable transaction journal cannot be trusted or recovered."""

class AtomicJournalPhase(str, Enum):
    PREPARED = "PREPARED"
    BACKED_UP = "BACKED_UP"
    INSTALLED = "INSTALLED"

@dataclass(frozen=True, slots=True)
class AtomicJournalEntry:
    final_path: Path
    staged_path: Path
    backup_path: Path
    had_original: bool

    def __post_init__(self) -> None:
        for field_name in ("final_path", "staged_path", "backup_path"):
            value = getattr(self, field_name)
            if not isinstance(value, Path):
                raise TypeError(f"{field_name} must be pathlib.Path")
            if not value.is_absolute() or value != value.resolve(strict=False):
                raise ValueError(f"{field_name} must be absolute and normalized")
        if not isinstance(self.had_original, bool):
            raise TypeError("had_original must be bool")
        if len({self.final_path, self.staged_path, self.backup_path}) != 3:
            raise ValueError("journal entry paths must be distinct")

    def to_mapping(self) -> dict[str, object]:
        return {"final_path": str(self.final_path), "staged_path": str(self.staged_path), "backup_path": str(self.backup_path), "had_original": self.had_original}

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> "AtomicJournalEntry":
        if not isinstance(value, Mapping):
            raise TypeError("journal entry must be a mapping")
        return cls(final_path=Path(str(value["final_path"])).resolve(strict=False), staged_path=Path(str(value["staged_path"])).resolve(strict=False), backup_path=Path(str(value["backup_path"])).resolve(strict=False), had_original=bool(value["had_original"]))

@dataclass(frozen=True, slots=True)
class AtomicJournalRecoveryReport:
    recovered_tokens: tuple[str, ...] = ()
    deferred_tokens: tuple[str, ...] = ()
    removed_paths: tuple[Path, ...] = ()
    restored_paths: tuple[Path, ...] = ()
    failures: tuple[str, ...] = ()

    @property
    def success(self) -> bool:
        return not self.failures

class AtomicCommitJournal:
    """Replicate one transaction journal into every affected output directory."""
    def __init__(self, *, token: str, process_id: int, process_start_marker: str, entries: Iterable[AtomicJournalEntry]) -> None:
        if not isinstance(token, str) or not token:
            raise ValueError("token must be a non-empty string")
        if isinstance(process_id, bool) or not isinstance(process_id, int):
            raise TypeError("process_id must be int")
        if process_id <= 0:
            raise ValueError("process_id must be positive")
        if not isinstance(process_start_marker, str) or not process_start_marker:
            raise ValueError("process_start_marker must be a non-empty string")
        resolved_entries = tuple(entries)
        if not resolved_entries or not all(isinstance(entry, AtomicJournalEntry) for entry in resolved_entries):
            raise ValueError("entries must contain AtomicJournalEntry values")
        self.token = token
        self.process_id = process_id
        self.process_start_marker = process_start_marker
        self.entries = resolved_entries
        self.created_ns = time.time_ns()
        directories = tuple(sorted({entry.final_path.parent for entry in self.entries}, key=lambda path: str(path)))
        self.paths = tuple(directory / f"{JOURNAL_PREFIX}{self.token}{JOURNAL_SUFFIX}" for directory in directories)
        self.phase: AtomicJournalPhase | None = None

    def _payload(self, phase: AtomicJournalPhase) -> dict[str, object]:
        return {"schema_version": 1, "token": self.token, "process_id": self.process_id, "process_start_marker": self.process_start_marker, "created_ns": self.created_ns, "phase": phase.value, "journal_paths": [str(path) for path in self.paths], "entries": [entry.to_mapping() for entry in self.entries]}

    def write_phase(self, phase: AtomicJournalPhase) -> None:
        if not isinstance(phase, AtomicJournalPhase):
            raise TypeError("phase must be AtomicJournalPhase")
        try:
            for path in self.paths:
                path.parent.mkdir(parents=True, exist_ok=True)
                write_json_durable(path, self._payload(phase))
        except Exception as exc:
            raise AtomicJournalError(f"unable to persist transaction journal phase {phase.value}: {exc}") from exc
        self.phase = phase

    def remove(self) -> tuple[str, ...]:
        failures: list[str] = []
        for path in self.paths:
            try:
                durable_unlink(path, missing_ok=True)
            except Exception as exc:
                failures.append(f"remove journal {path}: {exc}")
        return tuple(failures)

def _parse_journal(path: Path):
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, Mapping) or payload.get("schema_version") != 1:
            raise ValueError("unsupported journal schema")
        token = str(payload["token"])
        process_id = int(payload["process_id"])
        marker = str(payload["process_start_marker"])
        phase = AtomicJournalPhase(str(payload["phase"]))
        journal_paths = tuple(Path(str(item)).resolve(strict=False) for item in payload["journal_paths"])
        entries = tuple(AtomicJournalEntry.from_mapping(item) for item in payload["entries"])
    except Exception as exc:
        raise AtomicJournalError(f"invalid transaction journal '{path}': {exc}") from exc
    if not token or not entries or path.resolve(strict=False) not in journal_paths:
        raise AtomicJournalError(f"incomplete transaction journal: {path}")
    return token, process_id, marker, phase, journal_paths, entries

def _journal_consensus(initial):
    token, process_id, marker, phase, journal_paths, entries = initial
    phases = [phase]
    for candidate in journal_paths:
        if not candidate.exists():
            continue
        try:
            parsed = _parse_journal(candidate)
        except AtomicJournalError:
            continue
        other_token, other_pid, other_marker, other_phase, _, other_entries = parsed
        if other_token != token or other_pid != process_id or other_marker != marker or other_entries != entries:
            raise AtomicJournalError(f"replicated journal disagreement for transaction {token}")
        phases.append(other_phase)
    order = {AtomicJournalPhase.PREPARED: 0, AtomicJournalPhase.BACKED_UP: 1, AtomicJournalPhase.INSTALLED: 2}
    return token, process_id, marker, min(phases, key=order.__getitem__), journal_paths, entries

def _rollback_entry(entry: AtomicJournalEntry, removed: list[Path], restored: list[Path]) -> None:
    if entry.backup_path.exists():
        if entry.final_path.exists():
            durable_unlink(entry.final_path)
            removed.append(entry.final_path)
        durable_replace(entry.backup_path, entry.final_path)
        restored.append(entry.final_path)
    elif not entry.had_original and entry.final_path.exists() and not entry.staged_path.exists():
        durable_unlink(entry.final_path)
        removed.append(entry.final_path)
    if entry.staged_path.exists():
        durable_unlink(entry.staged_path)
        removed.append(entry.staged_path)

def _finalize_entry(entry: AtomicJournalEntry, removed: list[Path], restored: list[Path]) -> None:
    if not entry.final_path.exists():
        if entry.backup_path.exists():
            durable_replace(entry.backup_path, entry.final_path)
            restored.append(entry.final_path)
        else:
            raise AtomicJournalError(f"installed journal lost both final and backup: {entry.final_path}")
    else:
        fsync_file(entry.final_path)
        fsync_directory(entry.final_path.parent)
    for path in (entry.staged_path, entry.backup_path):
        if path.exists():
            durable_unlink(path)
            removed.append(path)

def recover_interrupted_atomic_journals(directory: Path) -> AtomicJournalRecoveryReport:
    if not isinstance(directory, Path):
        raise TypeError("directory must be pathlib.Path")
    root = directory.expanduser().resolve(strict=False)
    if not root.exists():
        return AtomicJournalRecoveryReport()
    if not root.is_dir():
        raise ValueError(f"journal recovery path is not a directory: {root}")
    recovered: list[str] = []
    deferred: list[str] = []
    removed: list[Path] = []
    restored: list[Path] = []
    failures: list[str] = []
    visited: set[str] = set()
    for path in tuple(sorted(root.glob(f"{JOURNAL_PREFIX}*{JOURNAL_SUFFIX}"))):
        try:
            token, process_id, marker, phase, journal_paths, entries = _journal_consensus(_parse_journal(path))
            if token in visited:
                continue
            visited.add(token)
            if process_identity_is_alive(process_id, marker):
                deferred.append(token)
                continue
            if phase in {AtomicJournalPhase.PREPARED, AtomicJournalPhase.BACKED_UP}:
                for entry in reversed(entries):
                    _rollback_entry(entry, removed, restored)
            else:
                for entry in entries:
                    _finalize_entry(entry, removed, restored)
            for journal_path in journal_paths:
                if journal_path.exists():
                    durable_unlink(journal_path)
                    removed.append(journal_path)
            recovered.append(token)
        except Exception as exc:
            failures.append(f"recover journal {path}: {exc}")
    return AtomicJournalRecoveryReport(recovered_tokens=tuple(recovered), deferred_tokens=tuple(deferred), removed_paths=tuple(removed), restored_paths=tuple(restored), failures=tuple(failures))

__all__ = ["AtomicCommitJournal", "AtomicJournalEntry", "AtomicJournalError", "AtomicJournalPhase", "AtomicJournalRecoveryReport", "JOURNAL_PREFIX", "JOURNAL_SUFFIX", "recover_interrupted_atomic_journals"]
