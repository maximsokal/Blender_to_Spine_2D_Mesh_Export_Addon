from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / "tools" / "run_all_spine_runtime_oracles.py"


def test_runtime_collector_attempts_every_matrix_case_before_failure() -> None:
    source = TOOL.read_text(encoding="utf-8")

    assert "for case, record in zip(POSITIVE_CASES, case_records, strict=True):" in source
    assert "if completed.returncode != 0:" in source
    assert "failed_by_target[case.target] += 1" in source
    assert "total_failed = sum(failed_by_target.values())" in source
    assert 'print("SPINE_ALL_RUNTIME_ORACLES=FAIL")' in source


def test_runtime_collector_writes_per_case_and_matrix_reports() -> None:
    source = TOOL.read_text(encoding="utf-8")

    assert 'runtime_oracle_report.json' in source
    assert 'acceptance_summary.json' in source
    assert '"EXECUTE_ORACLE"' in source
    assert '"VALIDATE_ORACLE_REPORT"' in source
    assert '"totalFailed": total_failed' in source


def test_runtime_collector_keeps_external_runtimes_read_only() -> None:
    source = TOOL.read_text(encoding="utf-8")

    assert '"externalRuntimesReadOnly": True' in source
    for forbidden in (
        "writeFileSync",
        "appendFileSync",
        "copyFileSync",
        "unlinkSync",
        "renameSync",
    ):
        assert forbidden not in source
