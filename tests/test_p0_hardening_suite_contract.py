"""Fast source contracts preserving the P0 lifecycle and export hardening layer."""

from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "Blender_to_Spine2D_Mesh_Exporter"
TESTS = ROOT / "tests"
BPY_TESTS = ROOT / "tests_bpy"


def _source(path: Path) -> str:
    assert path.is_file(), str(path)
    source = path.read_text(encoding="utf-8")
    ast.parse(source, filename=str(path))
    return source


def test_root_registration_keeps_explicit_standard_owner_lifecycle():
    source = _source(PACKAGE / "__init__.py")

    for forbidden in (
        "class ExtensionRegistrationState",
        'DEGRADED = "DEGRADED"',
        "def get_registration_state",
        "REGISTRATION_STEPS",
        "Rewrite extension registration rollback",
    ):
        assert forbidden not in source

    registration_calls = (
        "addon_preferences.register()",
        "_register_config_rna()",
        "scene_settings_migration.register()",
        "ui.register()",
        "rig_ui.register()",
        "a1_readiness_invalidation.register()",
        "auto_readiness.register()",
        "generated_material_ui.register()",
        "ui_layout.register()",
        "single_object_operator.register()",
    )
    lines = tuple(line.strip() for line in source.splitlines())
    offsets = tuple(lines.index(call) for call in registration_calls)
    assert offsets == tuple(sorted(offsets))

    assert "def _initialize_registered_logging() -> bool:" in source
    assert "continuing with default logging" in source


def test_ui_entrypoints_keep_one_shared_export_reentrancy_guard():
    policy = _source(PACKAGE / "infrastructure" / "exclusive_operation.py")
    router = _source(PACKAGE / "blender_adapter" / "a1_ui_router.py")
    for required in (
        "class OperationAlreadyRunningError",
        "class ExclusiveOperationLease",
        "def exclusive_operation",
        "finally:",
        "_REGISTRY.release(lease)",
    ):
        assert required in policy
    assert router.count("with exclusive_operation(") == 2
    assert router.count("A1_EXPORT_OPERATION_KEY") >= 3


def test_operator_poll_and_adversarial_output_tests_remain_present():
    single = _source(PACKAGE / "single_object_operator.py")
    ui = _source(PACKAGE / "ui.py")
    assert "def poll" in single
    assert 'getattr(obj, "type", None) == "MESH"' in single
    assert 'getattr(obj, "data", None) is not None' in single
    assert ui.count("def poll") >= 3

    poll_tests = _source(TESTS / "test_operator_poll_matrix.py")
    collision_tests = _source(TESTS / "test_output_namespace_adversarial.py")
    fault_tests = _source(TESTS / "test_single_output_fault_matrix.py")
    for required in ("no_active", "camera", "missing_data"):
        assert required in poll_tests
    for required in ("CON", "LPT1", "trailing", "collision"):
        assert required in collision_tests
    for required in ("texture", "serialization", "commit", "range(8)"):
        assert required in fault_tests


def test_real_blender_source_fingerprint_and_context_rollback_remain_present():
    source = _source(BPY_TESTS / "test_runtime_invariants_real_bpy.py")
    for required in (
        "_object_fingerprint",
        "matrix_world",
        "users_collection",
        "read_uv_coordinates",
        "range(12)",
        "range(6)",
        "activate_object_for_operator",
        "test_real_operator_poll_matrix",
    ):
        assert required in source
