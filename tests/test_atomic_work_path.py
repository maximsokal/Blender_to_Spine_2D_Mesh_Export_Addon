"""Regression contracts for Windows-safe atomic stage and backup filenames."""

from __future__ import annotations

from pathlib import Path

import pytest

from Blender_to_Spine2D_Mesh_Exporter.infrastructure.atomic_files import (
    _backup_final_path,
)
from Blender_to_Spine2D_Mesh_Exporter.infrastructure.atomic_work_path import (
    AtomicWorkPathError,
    WINDOWS_EXTERNAL_IO_HEADROOM_CODE_UNITS,
    WINDOWS_EXTERNAL_IO_PATH_BUDGET,
    WINDOWS_LEGACY_MAX_PATH_CODE_UNITS,
    build_atomic_backup_path,
    build_atomic_stage_path,
)
from Blender_to_Spine2D_Mesh_Exporter.infrastructure.atomic_work_state import (
    BACKUP_MARKER,
    STAGE_MARKER,
    work_file_token,
)
from Blender_to_Spine2D_Mesh_Exporter.infrastructure.durable_atomic_transaction import (
    DurableAtomicFileTransaction,
)
import Blender_to_Spine2D_Mesh_Exporter.infrastructure.atomic_work_path as work_path_module


_TOKEN = (
    "v2~52104~windows-1dd2325020fea52~1785746976870310100~"
    "2038d9171fde4ebd95ec841bbd23b58d"
)


def _utf16_units(path: Path) -> int:
    return len(str(path).encode("utf-16-le")) // 2


def test_windows_budget_is_derived_from_legacy_limit_and_fixed_headroom() -> None:
    assert WINDOWS_LEGACY_MAX_PATH_CODE_UNITS == 260
    assert WINDOWS_EXTERNAL_IO_HEADROOM_CODE_UNITS == 12
    assert WINDOWS_EXTERNAL_IO_PATH_BUDGET == 248
    assert WINDOWS_EXTERNAL_IO_PATH_BUDGET == (
        WINDOWS_LEGACY_MAX_PATH_CODE_UNITS
        - WINDOWS_EXTERNAL_IO_HEADROOM_CODE_UNITS
    )


def test_stage_path_keeps_readable_final_stem_when_it_fits(tmp_path: Path) -> None:
    final_path = (tmp_path / "Hero_Baked_0001.png").resolve()
    staged = build_atomic_stage_path(
        final_path,
        _TOKEN,
        reservation_index=0,
        path_budget=1000,
    )

    assert staged.parent == final_path.parent
    assert staged.name == f".{final_path.stem}{STAGE_MARKER}{_TOKEN}.png"
    assert staged.suffix == final_path.suffix
    assert work_file_token(staged) == _TOKEN


def test_stage_path_compacts_only_the_repeated_final_stem(tmp_path: Path) -> None:
    final_path = (tmp_path / ("VeryLongExportObjectName_" * 5 + ".png")).resolve()
    expected = final_path.with_name(f".s0{STAGE_MARKER}{_TOKEN}.png").resolve()
    budget = _utf16_units(expected)

    staged = build_atomic_stage_path(
        final_path,
        _TOKEN,
        reservation_index=0,
        path_budget=budget,
    )

    assert staged == expected
    assert _utf16_units(staged) <= budget
    assert staged.parent == final_path.parent
    assert staged.suffix == ".png"
    assert work_file_token(staged) == _TOKEN
    assert final_path.name not in staged.name


def test_compact_stage_names_are_unique_without_probabilistic_hashes(
    tmp_path: Path,
) -> None:
    first_final = (tmp_path / ("A" * 100 + ".png")).resolve()
    second_final = (tmp_path / ("B" * 100 + ".png")).resolve()
    first_expected = first_final.with_name(f".s0{STAGE_MARKER}{_TOKEN}.png")
    second_expected = second_final.with_name(f".s1{STAGE_MARKER}{_TOKEN}.png")
    budget = max(_utf16_units(first_expected), _utf16_units(second_expected))

    first = build_atomic_stage_path(
        first_final,
        _TOKEN,
        reservation_index=0,
        path_budget=budget,
    )
    second = build_atomic_stage_path(
        second_final,
        _TOKEN,
        reservation_index=1,
        path_budget=budget,
    )

    assert first != second
    assert first.name.startswith(".s0")
    assert second.name.startswith(".s1")
    assert work_file_token(first) == work_file_token(second) == _TOKEN


def test_stage_path_fails_before_blender_io_when_even_compact_name_cannot_fit(
    tmp_path: Path,
) -> None:
    final_path = (tmp_path / ("C" * 100 + ".png")).resolve()
    compact = final_path.with_name(f".s0{STAGE_MARKER}{_TOKEN}.png")

    with pytest.raises(
        AtomicWorkPathError,
        match="Choose a shorter export directory",
    ):
        build_atomic_stage_path(
            final_path,
            _TOKEN,
            reservation_index=0,
            path_budget=_utf16_units(compact) - 1,
        )


def test_compact_backup_is_journal_owned_and_not_legacy_filename_mapped(
    tmp_path: Path,
) -> None:
    final_path = (tmp_path / ("ExistingOutput_" * 8 + ".png")).resolve()
    expected = final_path.with_name(f".{BACKUP_MARKER}{_TOKEN}.a").resolve()
    budget = _utf16_units(expected)

    backup = build_atomic_backup_path(
        final_path,
        _TOKEN,
        reservation_index=10,
        path_budget=budget,
    )

    assert backup == expected
    assert backup.parent == final_path.parent
    assert backup.name.startswith("..spine2d-backup-")
    assert _backup_final_path(backup) is None


def test_durable_transaction_commits_through_compact_stage_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first_final = (
        tmp_path / ("Connected_ThreeAxis_NormalUvObjectA_" * 3 + ".png")
    ).resolve()
    second_final = (
        tmp_path / ("Connected_ThreeAxis_NormalUvObjectB_" * 3 + ".png")
    ).resolve()

    transaction = DurableAtomicFileTransaction(recover_stale_work_files=False)
    try:
        # The helper still validates explicit budgets normally. This monkeypatch
        # emulates the platform-selected Windows budget on non-Windows pytest hosts
        # using the transaction's actual platform-specific ownership token.
        first_compact = first_final.with_name(
            f".s0{STAGE_MARKER}{transaction._token}.png"
        )
        second_compact = second_final.with_name(
            f".s1{STAGE_MARKER}{transaction._token}.png"
        )
        forced_budget = max(
            _utf16_units(first_compact),
            _utf16_units(second_compact),
        )
        monkeypatch.setattr(
            work_path_module,
            "_effective_budget",
            lambda _path_budget: forced_budget,
        )

        first = transaction.reserve(first_final)
        second = transaction.reserve(second_final)
        assert first.staged_path.name.startswith(".s0")
        assert second.staged_path.name.startswith(".s1")
        assert work_file_token(first.staged_path) == transaction._token
        assert work_file_token(second.staged_path) == transaction._token

        first.staged_path.write_bytes(b"first")
        second.staged_path.write_bytes(b"second")
        committed = transaction.commit()
    finally:
        if not transaction.committed:
            transaction.rollback()

    assert committed == (first_final, second_final)
    assert first_final.read_bytes() == b"first"
    assert second_final.read_bytes() == b"second"
    assert not first.staged_path.exists()
    assert not second.staged_path.exists()
