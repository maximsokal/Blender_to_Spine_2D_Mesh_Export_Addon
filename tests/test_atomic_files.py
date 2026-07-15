import os

import pytest

from Blender_to_Spine2D_Mesh_Exporter.infrastructure import (
    AtomicFileCommitError,
    AtomicFileTransaction,
    atomic_file_transaction,
)


def test_commit_installs_all_staged_files(tmp_path):
    first_final = tmp_path / "first.png"
    second_final = tmp_path / "second.png"

    with atomic_file_transaction() as transaction:
        first = transaction.reserve(first_final)
        second = transaction.reserve(second_final)
        first.staged_path.write_bytes(b"first")
        second.staged_path.write_bytes(b"second")
        committed = transaction.commit()

    assert committed == (first_final.resolve(), second_final.resolve())
    assert first_final.read_bytes() == b"first"
    assert second_final.read_bytes() == b"second"
    assert transaction.committed


def test_missing_staged_file_preserves_existing_output(tmp_path):
    final = tmp_path / "texture.png"
    final.write_bytes(b"old")
    transaction = AtomicFileTransaction()
    transaction.reserve(final)

    with pytest.raises(AtomicFileCommitError):
        transaction.commit()

    assert final.read_bytes() == b"old"
    assert not transaction.committed


def test_context_rollback_removes_staged_files(tmp_path):
    final = tmp_path / "texture.png"
    staged_path = None

    with pytest.raises(RuntimeError):
        with atomic_file_transaction() as transaction:
            reservation = transaction.reserve(final)
            staged_path = reservation.staged_path
            reservation.staged_path.write_bytes(b"partial")
            raise RuntimeError("bake failed")

    assert staged_path is not None
    assert not staged_path.exists()
    assert not final.exists()


def test_failed_second_install_restores_every_previous_output(tmp_path, monkeypatch):
    first_final = tmp_path / "first.png"
    second_final = tmp_path / "second.png"
    first_final.write_bytes(b"old-first")
    second_final.write_bytes(b"old-second")

    transaction = AtomicFileTransaction()
    first = transaction.reserve(first_final)
    second = transaction.reserve(second_final)
    first.staged_path.write_bytes(b"new-first")
    second.staged_path.write_bytes(b"new-second")

    real_replace = os.replace
    staged_install_count = 0

    def failing_replace(source, destination):
        nonlocal staged_install_count
        source_path = str(source)
        if "spine2d-stage" in source_path:
            staged_install_count += 1
            if staged_install_count == 2:
                raise OSError("simulated second install failure")
        return real_replace(source, destination)

    monkeypatch.setattr(os, "replace", failing_replace)

    with pytest.raises(AtomicFileCommitError):
        transaction.commit()

    assert first_final.read_bytes() == b"old-first"
    assert second_final.read_bytes() == b"old-second"
    assert not first.staged_path.exists()
    assert not second.staged_path.exists()


def test_duplicate_final_path_is_rejected(tmp_path):
    transaction = AtomicFileTransaction()
    transaction.reserve(tmp_path / "same.png")
    with pytest.raises(ValueError):
        transaction.reserve(tmp_path / "same.png")
