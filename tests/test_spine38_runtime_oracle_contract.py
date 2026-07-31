"""Source contracts for the exact read-only Spine 3.8 runtime oracle."""

from __future__ import annotations

from pathlib import Path

from Blender_to_Spine2D_Mesh_Exporter.domain.spine.two_axis_scale_profile import (
    TwoAxisScaleRigProfile,
)


ROOT = Path(__file__).resolve().parents[1]
ORACLE = ROOT / "tools" / "spine38_runtime_oracle.mjs"


def _source() -> str:
    return ORACLE.read_text(encoding="utf-8")


def test_oracle_uses_exact_version_and_windows_safe_esm_import() -> None:
    source = _source()

    assert "const EXPECTED_VERSION = '3.8.99';" in source
    assert "pathToFileURL(runtimeEntry).href" in source
    assert "await import(runtimeEntry)" not in source


def test_oracle_uses_exact_legacy_texture_atlas_api() -> None:
    source = _source()

    assert "new runtime.TextureAtlas(" in source
    assert "atlasText(collectAtlasRegions(document))" in source
    assert "() => oracleTexture()" in source
    assert "page.setTexture" not in source


def test_oracle_checks_legacy_mix_schema_and_runtime_execution() -> None:
    source = _source()

    for fragment in (
        "'rotateMix'",
        "'translateMix'",
        "'scaleMix'",
        "'shearMix'",
        "'mixRotate'",
        "skeleton.setToSetupPose();",
        "skeleton.updateCache();",
        "skeleton.updateWorldTransform();",
        "validateUpdateCache(skeleton, records)",
        "validateMatrices(skeleton)",
        "skeleton.getBounds(offset, size)",
    ):
        assert fragment in source


def test_oracle_bounds_do_not_require_vector2_wrapper_export() -> None:
    source = _source()

    for fragment in (
        "function mutableVector2()",
        "set(x, y)",
        "const offset = mutableVector2();",
        "const size = mutableVector2();",
        "const setupBounds = bounds(skeleton);",
    ):
        assert fragment in source

    assert "new runtime.Vector2()" not in source
    assert "    'Vector2'," not in source


def test_oracle_scale_response_matches_production_control_contract() -> None:
    source = _source()
    profile = TwoAxisScaleRigProfile()

    assert profile.scale_control_bone("Probe") == "Probe_scale"
    assert profile.scale_constraint("Probe") == profile.scale_control_bone("Probe")

    for fragment in (
        "function twoAxisScalePrefixes(document)",
        "_scale_spine41_bridge",
        "function scaleControlRecords(document)",
        "name !== target || !target.endsWith('_scale')",
        "Spine 3.8 2-Axis scale control inventory differs",
        "function scaleResponse(runtime, skeletonData, controls)",
        "disabledConstraint.scaleMix = 0;",
        "changedByConstraintBoneCount",
        "allControlsResponded",
        "constraintAffectsBounds",
        "fail('One or more Spine 3.8 scale controls did not affect constrained bones'",
        "scaleResponse: scaleResponseEvidence",
    ):
        assert fragment in source

    assert "name.endsWith('_scale_constraint')" not in source
    for hardcoded_fixture_name in (
        "Spine38TwoA",
        "Spine38TwoB",
        "Spine38TwoC",
    ):
        assert hardcoded_fixture_name not in source


def test_oracle_scale_response_uses_runtime_only_mutation() -> None:
    source = _source()

    for fragment in (
        "target.scaleX = setupScaleX * SCALE_RESPONSE_FACTOR;",
        "target.scaleY = setupScaleY * SCALE_RESPONSE_FACTOR;",
        "const setupSkeleton = runtimeSkeleton(runtime, skeletonData);",
        "const scaledSkeleton = runtimeSkeleton(runtime, skeletonData);",
        "const disabledSkeleton = runtimeSkeleton(runtime, skeletonData);",
    ):
        assert fragment in source

    assert "document.transform[" in source
    assert "document.bones[" in source
    assert "constraint.scaleMix = 0" not in source


def test_oracle_has_no_external_write_apis() -> None:
    source = _source()

    for forbidden in (
        "writeFile",
        "appendFile",
        "mkdir",
        "rmSync",
        "unlink",
        "rename",
        "copyFile",
        "npm",
        "npx",
    ):
        assert forbidden not in source
