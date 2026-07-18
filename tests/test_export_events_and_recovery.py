from pathlib import Path

import pytest

from Blender_to_Spine2D_Mesh_Exporter.infrastructure.atomic_files import (
    AtomicFileTransaction,
    atomic_file_transaction,
    recover_stale_atomic_work_files,
)
from Blender_to_Spine2D_Mesh_Exporter.infrastructure.export_events import (
    ExportEventDispatcher,
    ExportEventKind,
)


def test_failed_transaction_removes_stage_file_when_preservation_is_disabled(
    tmp_path: Path,
):
    final_path = tmp_path / "result.json"
    with pytest.raises(RuntimeError, match="forced"):
        with atomic_file_transaction(
            preserve_failed_work_files=False,
            recover_stale_work_files=False,
        ) as transaction:
            reservation = transaction.reserve(final_path)
            reservation.staged_path.write_text("partial", encoding="utf-8")
            raise RuntimeError("forced")

    assert not final_path.exists()
    assert not tuple(tmp_path.glob(".*.spine2d-stage-*"))


def test_failed_transaction_preserves_stage_file_only_in_debug_mode(tmp_path: Path):
    final_path = tmp_path / "result.json"
    transaction = AtomicFileTransaction(
        preserve_failed_work_files=True,
        recover_stale_work_files=False,
    )
    reservation = transaction.reserve(final_path)
    reservation.staged_path.write_text("partial", encoding="utf-8")

    report = transaction.rollback()

    assert report.preserved_paths == (reservation.staged_path,)
    assert reservation.staged_path.read_text(encoding="utf-8") == "partial"
    assert not final_path.exists()


def test_recovery_removes_stale_stage_and_restores_missing_final_backup(tmp_path: Path):
    stale_stage = tmp_path / ".result.spine2d-stage-oldtoken.json"
    stale_stage.write_text("partial", encoding="utf-8")
    backup = tmp_path / ".result.json.spine2d-backup-oldtoken"
    backup.write_text("previous", encoding="utf-8")

    report = recover_stale_atomic_work_files(
        tmp_path,
        preserve_failed_work_files=False,
    )

    assert stale_stage in report.removed_paths
    assert (tmp_path / "result.json") in report.restored_paths
    assert (tmp_path / "result.json").read_text(encoding="utf-8") == "previous"
    assert not stale_stage.exists()
    assert not backup.exists()


def test_dispatcher_reports_transaction_events(tmp_path: Path):
    dispatcher = ExportEventDispatcher()
    events = []
    dispatcher.subscribe(events.append)
    final_path = tmp_path / "result.json"

    with atomic_file_transaction(
        operation_name="unit-test",
        preserve_failed_work_files=False,
        recover_stale_work_files=False,
        dispatcher=dispatcher,
    ) as transaction:
        reservation = transaction.reserve(final_path)
        reservation.staged_path.write_text("ok", encoding="utf-8")
        transaction.commit()

    kinds = tuple(event.kind for event in events)
    assert kinds[0] is ExportEventKind.TRANSACTION_STARTED
    assert ExportEventKind.OUTPUT_RESERVED in kinds
    assert ExportEventKind.COMMIT_STARTED in kinds
    assert kinds[-1] is ExportEventKind.COMMIT_SUCCEEDED
