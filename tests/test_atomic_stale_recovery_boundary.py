import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
INFRASTRUCTURE = ROOT / "Blender_to_Spine2D_Mesh_Exporter" / "infrastructure"


def _source(name: str) -> str:
    return (INFRASTRUCTURE / name).read_text(encoding="utf-8")


def _tree(name: str) -> ast.Module:
    path = INFRASTRUCTURE / name
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def test_atomic_work_state_owns_process_identity_and_final_path_registry():
    source = _source("atomic_work_state.py")
    assert "AtomicWorkTokenMetadata" in source
    assert "process_start_marker" in source
    assert "created_ns" in source
    assert "_final_path_owners" in source
    assert "claim_atomic_final_path" in source
    assert "OWNER_PROCESS_IDENTITY_MISMATCH" in source
    assert "UNREGISTERED_CURRENT_PROCESS_TOKEN" in source
    assert "DEFAULT_STALE_WORK_FILE_AGE_SECONDS" in source


def test_atomic_transaction_no_longer_uses_pid_only_active_token_set():
    source = _source("atomic_files.py")
    assert "_ACTIVE_TOKENS" not in source
    assert "_process_is_alive" not in source
    assert "create_atomic_work_token_metadata" in source
    assert "register_atomic_transaction" in source
    assert "unregister_atomic_transaction" in source
    assert "claim_atomic_final_path" in source


def test_reservation_and_recovery_have_explicit_contracts():
    tree = _tree("atomic_files.py")
    classes = {
        node.name: node
        for node in tree.body
        if isinstance(node, ast.ClassDef)
    }
    reservation = classes["AtomicOutputReservation"]
    methods = {
        node.name
        for node in reservation.body
        if isinstance(node, ast.FunctionDef)
    }
    assert "__post_init__" in methods

    source = _source("atomic_files.py")
    assert "final_path and staged_path must be different" in source
    assert "must share one directory" in source
    assert "AtomicRecoveryRecord" in source
    assert '"recovery_reason"' in source
    assert '"recovery_action"' in source
    assert "minimum_stale_age_seconds" in source


def test_os_process_start_identity_uses_platform_specific_creation_markers():
    source = _source("atomic_work_state.py")
    assert '/proc/{process_id}/stat' in source
    assert "fields_after_command[19]" in source
    assert 'ctypes.WinDLL("kernel32"' in source
    assert "GetProcessTimes" in source
