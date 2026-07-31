from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / "tools" / "spine42_runtime_step_trace.mjs"


def test_step_trace_identifies_first_non_finite_update_cache_item() -> None:
    source = TOOL.read_text(encoding="utf-8")

    assert "function traceUpdateCache(runtime, skeleton)" in source
    assert "initializeAppliedPose(skeleton)" in source
    assert "item.update(runtime.Physics.none)" in source
    assert "firstNonFiniteBone(skeleton)" in source
    assert "stage: 'UPDATE_CACHE_ITEM'" in source
    assert "cacheIndex" in source
    assert "cachePrefix" in source
    assert "nearbyBones" in source


def test_step_trace_reports_constraint_dependency_context() -> None:
    source = TOOL.read_text(encoding="utf-8")

    assert "function runtimeConstraintDiagnostics(skeleton)" in source
    assert "function updateItemDiagnostic(skeleton, item, cacheIndex)" in source
    assert "bones:" in source
    assert "target:" in source
    assert "order:" in source
    assert "local:" in source
    assert "relative:" in source


def test_step_trace_is_exact_spine42_and_read_only() -> None:
    source = TOOL.read_text(encoding="utf-8")

    assert "Expected Spine 4.2.43 JSON" in source
    assert "runtime.Physics.none" in source
    assert "readFileSync" in source
    for forbidden in (
        "writeFileSync",
        "appendFileSync",
        "unlinkSync",
        "renameSync",
        "copyFileSync",
        "rmSync",
    ):
        assert forbidden not in source
