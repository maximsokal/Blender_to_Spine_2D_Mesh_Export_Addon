from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from Blender_to_Spine2D_Mesh_Exporter.infrastructure import AtomicFileTransaction
from Blender_to_Spine2D_Mesh_Exporter.infrastructure.atomic_journal import AtomicCommitJournal, AtomicJournalEntry, AtomicJournalPhase, recover_interrupted_atomic_journals
from Blender_to_Spine2D_Mesh_Exporter.infrastructure.atomic_work_state import create_atomic_work_token_metadata
from Blender_to_Spine2D_Mesh_Exporter.infrastructure.interprocess_lock import lock_path_for_resource


def _journal(tmp_path: Path, *, token: str, process_id: int = 999_999_999, marker: str = "dead-process", second_directory: Path | None = None):
    first_final = (tmp_path / "first.json").resolve(); first_stage = (tmp_path / ".first.spine2d-stage-token.json").resolve(); first_backup = (tmp_path / ".first.json.spine2d-backup-token").resolve()
    entries = [AtomicJournalEntry(first_final, first_stage, first_backup, True)]
    if second_directory is not None:
        second_directory.mkdir(parents=True, exist_ok=True)
        entries.append(AtomicJournalEntry((second_directory / "second.json").resolve(), (second_directory / ".second.spine2d-stage-token.json").resolve(), (second_directory / ".second.json.spine2d-backup-token").resolve(), True))
    return AtomicCommitJournal(token=token, process_id=process_id, process_start_marker=marker, entries=entries)


def test_commit_fsyncs_and_removes_lock_journal_stage_and_backup(tmp_path, monkeypatch):
    final = (tmp_path / "hero.json").resolve(); final.write_bytes(b"old"); fsync_calls: list[int] = []; real_fsync = os.fsync
    def recording_fsync(descriptor: int) -> None:
        fsync_calls.append(descriptor); real_fsync(descriptor)
    monkeypatch.setattr(os, "fsync", recording_fsync)
    transaction = AtomicFileTransaction(recover_stale_work_files=False); reservation = transaction.reserve(final); reservation.staged_path.write_bytes(b"new")
    assert transaction.commit() == (final,)
    assert final.read_bytes() == b"new"; assert fsync_calls; assert not lock_path_for_resource(final).exists(); assert not tuple(tmp_path.glob(".spine2d-journal-*.json")); assert not tuple(tmp_path.glob("*spine2d-stage*")); assert not tuple(tmp_path.glob("*spine2d-backup*"))


def test_prepared_journal_rolls_back_partial_install(tmp_path):
    journal = _journal(tmp_path, token="rollback-prepared"); entry = journal.entries[0]; entry.final_path.write_bytes(b"old"); entry.staged_path.write_bytes(b"new"); journal.write_phase(AtomicJournalPhase.PREPARED); os.replace(entry.final_path, entry.backup_path); os.replace(entry.staged_path, entry.final_path)
    report = recover_interrupted_atomic_journals(tmp_path)
    assert report.success; assert report.recovered_tokens == (journal.token,); assert entry.final_path.read_bytes() == b"old"; assert not entry.backup_path.exists(); assert not entry.staged_path.exists(); assert not any(path.exists() for path in journal.paths)


def test_installed_journal_finishes_commit_and_removes_backup(tmp_path):
    journal = _journal(tmp_path, token="finish-installed"); entry = journal.entries[0]; entry.final_path.write_bytes(b"old"); entry.staged_path.write_bytes(b"new"); journal.write_phase(AtomicJournalPhase.PREPARED); os.replace(entry.final_path, entry.backup_path); journal.write_phase(AtomicJournalPhase.BACKED_UP); os.replace(entry.staged_path, entry.final_path); journal.write_phase(AtomicJournalPhase.INSTALLED)
    report = recover_interrupted_atomic_journals(tmp_path)
    assert report.success; assert entry.final_path.read_bytes() == b"new"; assert not entry.backup_path.exists(); assert not any(path.exists() for path in journal.paths)


def test_torn_multi_directory_phase_uses_earliest_consensus_and_rolls_back(tmp_path):
    other = tmp_path / "other"; journal = _journal(tmp_path, token="torn-phase", second_directory=other)
    for entry in journal.entries: entry.final_path.write_bytes(b"old"); entry.staged_path.write_bytes(b"new")
    journal.write_phase(AtomicJournalPhase.BACKED_UP)
    for entry in journal.entries: os.replace(entry.final_path, entry.backup_path); os.replace(entry.staged_path, entry.final_path)
    payload = journal._payload(AtomicJournalPhase.INSTALLED); journal.paths[0].write_text(json.dumps(payload), encoding="utf-8")
    report = recover_interrupted_atomic_journals(tmp_path)
    assert report.success
    for entry in journal.entries: assert entry.final_path.read_bytes() == b"old"; assert not entry.backup_path.exists()


def test_live_process_journal_is_deferred(tmp_path):
    metadata = create_atomic_work_token_metadata(); journal = _journal(tmp_path, token="live-owner", process_id=metadata.process_id, marker=metadata.process_start_marker); entry = journal.entries[0]; entry.final_path.write_bytes(b"old"); entry.staged_path.write_bytes(b"new"); journal.write_phase(AtomicJournalPhase.PREPARED)
    report = recover_interrupted_atomic_journals(tmp_path)
    assert report.deferred_tokens == (journal.token,); assert entry.staged_path.exists(); assert all(path.exists() for path in journal.paths); journal.remove(); entry.staged_path.unlink()
