from __future__ import annotations

from pathlib import Path
import time

import pytest

from Blender_to_Spine2D_Mesh_Exporter.infrastructure import (
    AtomicFileTransaction,
    AtomicOutputReservation,
    AtomicRecoveryAction,
    AtomicRecoveryReason,
    AtomicWorkFileState,
    AtomicWorkTokenMetadata,
    assess_atomic_work_file,
    recover_stale_atomic_work_files,
)
from Blender_to_Spine2D_Mesh_Exporter.infrastructure.atomic_work_state import (
    BACKUP_MARKER,
    STAGE_MARKER,
)
import Blender_to_Spine2D_Mesh_Exporter.infrastructure.atomic_work_state as work_state


def test_reservation_requires_absolute_normalized_distinct_sibling_paths(tmp_path):
    final = (tmp_path / "texture.png").resolve()
    staged = (tmp_path / f".texture{STAGE_MARKER}token.png").resolve()

    assert AtomicOutputReservation(final, staged).final_path == final

    with pytest.raises(ValueError, match="absolute"):
        AtomicOutputReservation(
            Path("texture.png"),
            Path(f".texture{STAGE_MARKER}token.png"),
        )
    with pytest.raises(ValueError, match="different"):
        AtomicOutputReservation(final, final)
    with pytest.raises(ValueError, match="share one directory"):
        AtomicOutputReservation(
            final,
            (tmp_path / "other" / f".texture{STAGE_MARKER}token.png").resolve(),
        )
    with pytest.raises(ValueError, match="stage marker"):
        AtomicOutputReservation(final, (tmp_path / ".texture.tmp").resolve())


def test_process_local_registry_rejects_parallel_final_path(tmp_path):
    final = tmp_path / "same.png"
    first = AtomicFileTransaction(recover_stale_work_files=False)
    second = AtomicFileTransaction(recover_stale_work_files=False)
    try:
        first.reserve(final)
        with pytest.raises(RuntimeError, match="another active transaction"):
            second.reserve(final)
    finally:
        first.rollback()
        second.rollback()

    third = AtomicFileTransaction(recover_stale_work_files=False)
    try:
        third.reserve(final)
    finally:
        third.rollback()


def test_active_process_local_stage_is_never_recovered(tmp_path):
    transaction = AtomicFileTransaction(recover_stale_work_files=False)
    reservation = transaction.reserve(tmp_path / "active.png")
    reservation.staged_path.write_bytes(b"active")
    try:
        assessment = assess_atomic_work_file(
            reservation.staged_path,
            minimum_stale_age_seconds=0.0,
        )
        report = recover_stale_atomic_work_files(
            tmp_path,
            minimum_stale_age_seconds=0.0,
        )

        assert assessment.state is AtomicWorkFileState.ACTIVE
        assert (
            assessment.reason
            is AtomicRecoveryReason.PROCESS_LOCAL_ACTIVE_TOKEN
        )
        assert report.removed_paths == ()
        assert report.restored_paths == ()
        assert report.recovery_records == ()
        assert reservation.staged_path.exists()
    finally:
        transaction.rollback()


def test_preserved_stage_from_closed_current_process_token_is_recoverable(tmp_path):
    transaction = AtomicFileTransaction(
        preserve_failed_work_files=True,
        recover_stale_work_files=False,
    )
    reservation = transaction.reserve(tmp_path / "failed.png")
    reservation.staged_path.write_bytes(b"failed")

    rollback = transaction.rollback()
    assert rollback.preserved_paths == (reservation.staged_path,)
    assert reservation.staged_path.exists()

    report = recover_stale_atomic_work_files(
        tmp_path,
        minimum_stale_age_seconds=0.0,
    )

    assert report.removed_paths == (reservation.staged_path,)
    assert len(report.recovery_records) == 1
    record = report.recovery_records[0]
    assert record.action is AtomicRecoveryAction.REMOVED_STAGE
    assert (
        record.reason
        is AtomicRecoveryReason.UNREGISTERED_CURRENT_PROCESS_TOKEN
    )
    assert not reservation.staged_path.exists()


def test_minimum_stale_age_defers_recent_closed_token(tmp_path):
    transaction = AtomicFileTransaction(
        preserve_failed_work_files=True,
        recover_stale_work_files=False,
    )
    reservation = transaction.reserve(tmp_path / "recent.png")
    reservation.staged_path.write_bytes(b"recent")
    transaction.rollback()

    assessment = assess_atomic_work_file(
        reservation.staged_path,
        minimum_stale_age_seconds=3600.0,
    )
    report = recover_stale_atomic_work_files(
        tmp_path,
        minimum_stale_age_seconds=3600.0,
    )

    assert assessment.state is AtomicWorkFileState.DEFERRED
    assert (
        assessment.reason
        is AtomicRecoveryReason.MINIMUM_STALE_AGE_NOT_REACHED
    )
    assert report.removed_paths == ()
    assert report.recovery_records == ()
    assert reservation.staged_path.exists()
    reservation.staged_path.unlink()


def test_pid_reuse_is_detected_by_process_start_marker(tmp_path, monkeypatch):
    now_ns = time.time_ns()
    metadata = AtomicWorkTokenMetadata(
        process_id=4242,
        process_start_marker="linux-old",
        created_ns=now_ns - 10_000_000_000,
        nonce="pidreuse",
    )
    path = tmp_path / f".reused{STAGE_MARKER}{metadata.token}.png"
    path.write_bytes(b"stale")

    monkeypatch.setattr(
        work_state,
        "_process_is_alive",
        lambda process_id: process_id == 4242,
    )
    monkeypatch.setattr(
        work_state,
        "read_process_start_marker",
        lambda _process_id: "linux-new",
    )

    assessment = assess_atomic_work_file(
        path,
        minimum_stale_age_seconds=0.0,
        now_ns=now_ns,
    )

    assert assessment.state is AtomicWorkFileState.STALE
    assert (
        assessment.reason
        is AtomicRecoveryReason.OWNER_PROCESS_IDENTITY_MISMATCH
    )


def test_matching_live_process_identity_remains_active(tmp_path, monkeypatch):
    now_ns = time.time_ns()
    metadata = AtomicWorkTokenMetadata(
        process_id=4343,
        process_start_marker="linux-same",
        created_ns=now_ns - 10_000_000_000,
        nonce="active",
    )
    path = tmp_path / f".active{STAGE_MARKER}{metadata.token}.png"
    path.write_bytes(b"active")

    monkeypatch.setattr(
        work_state,
        "_process_is_alive",
        lambda process_id: process_id == 4343,
    )
    monkeypatch.setattr(
        work_state,
        "read_process_start_marker",
        lambda _process_id: "linux-same",
    )

    assessment = assess_atomic_work_file(
        path,
        minimum_stale_age_seconds=0.0,
        now_ns=now_ns,
    )

    assert assessment.state is AtomicWorkFileState.ACTIVE
    assert assessment.reason is AtomicRecoveryReason.OWNER_PROCESS_ACTIVE


def test_unavailable_foreign_process_identity_fails_closed_as_active(
    tmp_path,
    monkeypatch,
):
    now_ns = time.time_ns()
    metadata = AtomicWorkTokenMetadata(
        process_id=4444,
        process_start_marker="unknown-platform-marker",
        created_ns=now_ns - 10_000_000_000,
        nonce="unknown",
    )
    path = tmp_path / f".unknown{STAGE_MARKER}{metadata.token}.png"
    path.write_bytes(b"active-unverified")

    monkeypatch.setattr(work_state, "_process_is_alive", lambda _pid: True)
    monkeypatch.setattr(
        work_state,
        "read_process_start_marker",
        lambda _process_id: None,
    )

    assessment = assess_atomic_work_file(
        path,
        minimum_stale_age_seconds=0.0,
        now_ns=now_ns,
    )

    assert assessment.state is AtomicWorkFileState.ACTIVE
    assert (
        assessment.reason
        is AtomicRecoveryReason.OWNER_PROCESS_IDENTITY_UNAVAILABLE
    )


def test_stale_backup_restore_records_explicit_reason(tmp_path, monkeypatch):
    now_ns = time.time_ns()
    metadata = AtomicWorkTokenMetadata(
        process_id=4545,
        process_start_marker="linux-exited",
        created_ns=now_ns - 10_000_000_000,
        nonce="backup",
    )
    final = tmp_path / "output.json"
    backup = tmp_path / f".{final.name}{BACKUP_MARKER}{metadata.token}"
    backup.write_bytes(b"old")

    monkeypatch.setattr(work_state, "_process_is_alive", lambda _pid: False)

    report = recover_stale_atomic_work_files(
        tmp_path,
        minimum_stale_age_seconds=0.0,
    )

    assert final.read_bytes() == b"old"
    assert report.restored_paths == (final,)
    assert len(report.recovery_records) == 1
    record = report.recovery_records[0]
    assert record.action is AtomicRecoveryAction.RESTORED_BACKUP
    assert record.reason is AtomicRecoveryReason.OWNER_PROCESS_EXITED
    assert record.final_path == final


def test_redundant_stale_backup_is_removed_without_overwriting_final(
    tmp_path,
    monkeypatch,
):
    now_ns = time.time_ns()
    metadata = AtomicWorkTokenMetadata(
        process_id=4646,
        process_start_marker="linux-exited",
        created_ns=now_ns - 10_000_000_000,
        nonce="redundant",
    )
    final = tmp_path / "output.json"
    final.write_bytes(b"current")
    backup = tmp_path / f".{final.name}{BACKUP_MARKER}{metadata.token}"
    backup.write_bytes(b"old")

    monkeypatch.setattr(work_state, "_process_is_alive", lambda _pid: False)

    report = recover_stale_atomic_work_files(
        tmp_path,
        minimum_stale_age_seconds=0.0,
    )

    assert final.read_bytes() == b"current"
    assert not backup.exists()
    assert report.removed_paths == (backup,)
    assert (
        report.recovery_records[0].action
        is AtomicRecoveryAction.REMOVED_REDUNDANT_BACKUP
    )


def test_token_metadata_round_trip_and_contracts():
    metadata = AtomicWorkTokenMetadata(
        process_id=123,
        process_start_marker="linux-ab",
        created_ns=456,
        nonce="nonce",
    )

    assert AtomicWorkTokenMetadata.parse(metadata.token) == metadata
    assert AtomicWorkTokenMetadata.parse("broken") is None

    with pytest.raises(TypeError):
        AtomicWorkTokenMetadata(True, "linux-ab", 456, "nonce")
    with pytest.raises(TypeError):
        AtomicWorkTokenMetadata(123, "linux-ab", True, "nonce")
    with pytest.raises(ValueError, match="filename-safe"):
        AtomicWorkTokenMetadata(123, "linux~ab", 456, "nonce")


@pytest.mark.parametrize("value", (True, -1.0, float("nan"), float("inf")))
def test_stale_age_policy_rejects_invalid_values(tmp_path, value):
    with pytest.raises((TypeError, ValueError)):
        recover_stale_atomic_work_files(
            tmp_path,
            minimum_stale_age_seconds=value,
        )
