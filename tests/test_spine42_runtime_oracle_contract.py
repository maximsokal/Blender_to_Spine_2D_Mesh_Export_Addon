from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ORACLE = ROOT / "tools" / "spine42_runtime_oracle.mjs"
CORE = ROOT / "tools" / "spine4x_legacy_runtime_oracle_core.mjs"


def test_spine42_oracle_pins_exact_version_and_supports_physics_signature() -> None:
    source = ORACLE.read_text(encoding="utf-8")
    core = CORE.read_text(encoding="utf-8")

    assert "expectedVersion: '4.2.43'" in source
    assert "expectedFamily: '4.2'" in source
    assert "runtime.Physics.none" in core
    assert "skeleton.updateWorldTransform(runtime.Physics.none)" in core
    assert "skeleton.updateWorldTransform();" in core


def test_spine42_oracle_validates_legacy_collections_not_unified_constraints() -> None:
    source = CORE.read_text(encoding="utf-8")

    assert "Unified constraints are not valid" in source
    assert "['ik', 'transform', 'path']" in source
    assert "Constraint orders must form 0..N-1" in source


def test_spine42_oracle_keeps_external_runtime_read_only() -> None:
    source = CORE.read_text(encoding="utf-8")

    assert "readFileSync" in source
    for forbidden_api in (
        "writeFileSync",
        "appendFileSync",
        "mkdirSync",
        "rmSync",
        "unlinkSync",
        "renameSync",
        "copyFileSync",
        "cpSync",
    ):
        assert forbidden_api not in source
