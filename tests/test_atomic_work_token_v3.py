"""Regression coverage for compact v3 atomic work ownership tokens."""

from __future__ import annotations

from pathlib import Path

import pytest

import Blender_to_Spine2D_Mesh_Exporter.infrastructure.atomic_work_state as state_module
from Blender_to_Spine2D_Mesh_Exporter.infrastructure.atomic_work_state import (
    AtomicRecoveryReason,
    AtomicWorkFileState,
    AtomicWorkTokenMetadata,
    STAGE_MARKER,
    assess_atomic_work_file,
    create_atomic_work_token_metadata,
    work_file_token,
)


_V3_NONCE = "0123456789abcdef0123456789abcdef"


def _stage_path(directory: Path, token: str) -> Path:
    return (directory / f".asset{STAGE_MARKER}{token}.png").resolve()


def test_production_factory_emits_compact_v3_round_trip() -> None:
    metadata = create_atomic_work_token_metadata()
    token = metadata.token
    parsed = AtomicWorkTokenMetadata.parse(token)

    assert metadata.token_version == "v3"
    assert token.startswith("v3~")
    assert len(token) < 80
    assert parsed is not None
    assert parsed.token_version == "v3"
    assert parsed.process_id == metadata.process_id
    assert parsed.created_ns == metadata.created_ns
    assert parsed.nonce == metadata.nonce
    assert parsed.process_start_marker is None
    assert parsed.process_start_marker_digest is not None
    assert metadata.process_start_marker is not None
    assert parsed.matches_process_start_marker(metadata.process_start_marker)
    assert parsed.token == token


def test_direct_metadata_constructor_retains_v2_compatibility() -> None:
    metadata = AtomicWorkTokenMetadata(
        process_id=12345,
        process_start_marker="windows-1dd232c6d2cb51d",
        created_ns=1785750163676482300,
        nonce="legacy-nonce",
    )
    expected = (
        "v2~12345~windows-1dd232c6d2cb51d~"
        "1785750163676482300~legacy-nonce"
    )

    assert metadata.token_version == "v2"
    assert metadata.token == expected
    assert AtomicWorkTokenMetadata.parse(expected) == metadata


def test_v3_parser_rejects_malformed_digest_nonce_and_hex_fields() -> None:
    valid = AtomicWorkTokenMetadata(
        process_id=0x1234,
        process_start_marker="windows-abcdef",
        created_ns=0x123456789ABC,
        nonce=_V3_NONCE,
        token_version="v3",
    ).token
    version, process_id, digest, created_ns, nonce = valid.split("~")

    malformed = (
        f"{version}~not-hex~{digest}~{created_ns}~{nonce}",
        f"{version}~{process_id}~short~{created_ns}~{nonce}",
        f"{version}~{process_id}~{digest}~not-hex~{nonce}",
        f"{version}~{process_id}~{digest}~{created_ns}~short",
        f"v4~{process_id}~{digest}~{created_ns}~{nonce}",
    )

    assert all(AtomicWorkTokenMetadata.parse(token) is None for token in malformed)


def test_v3_assessment_keeps_matching_live_process_active(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    marker = "windows-live-process-start"
    metadata = AtomicWorkTokenMetadata(
        process_id=0x7F12,
        process_start_marker=marker,
        created_ns=1_000_000_000,
        nonce=_V3_NONCE,
        token_version="v3",
    )
    path = _stage_path(tmp_path, metadata.token)
    path.write_bytes(b"active")

    monkeypatch.setattr(state_module, "_process_is_alive", lambda _pid: True)
    monkeypatch.setattr(
        state_module,
        "read_process_start_marker",
        lambda _pid: marker,
    )

    assessment = assess_atomic_work_file(
        path,
        minimum_stale_age_seconds=0.0,
        now_ns=2_000_000_000,
    )

    assert work_file_token(path) == metadata.token
    assert assessment.state is AtomicWorkFileState.ACTIVE
    assert assessment.reason is AtomicRecoveryReason.OWNER_PROCESS_ACTIVE
    assert assessment.metadata is not None
    assert assessment.metadata.token == metadata.token


def test_v3_assessment_detects_pid_reuse_from_marker_digest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    metadata = AtomicWorkTokenMetadata(
        process_id=0x7F13,
        process_start_marker="windows-original-process-start",
        created_ns=1_000_000_000,
        nonce=_V3_NONCE,
        token_version="v3",
    )
    path = _stage_path(tmp_path, metadata.token)
    path.write_bytes(b"stale")

    monkeypatch.setattr(state_module, "_process_is_alive", lambda _pid: True)
    monkeypatch.setattr(
        state_module,
        "read_process_start_marker",
        lambda _pid: "windows-reused-pid-process-start",
    )

    assessment = assess_atomic_work_file(
        path,
        minimum_stale_age_seconds=0.0,
        now_ns=2_000_000_000,
    )

    assert assessment.state is AtomicWorkFileState.STALE
    assert assessment.reason is AtomicRecoveryReason.OWNER_PROCESS_IDENTITY_MISMATCH
    assert assessment.recoverable


def test_v3_metadata_requires_full_128_bit_nonce() -> None:
    with pytest.raises(ValueError, match="32 lowercase hexadecimal"):
        AtomicWorkTokenMetadata(
            process_id=123,
            process_start_marker="windows-start",
            created_ns=456,
            nonce="abc",
            token_version="v3",
        )
