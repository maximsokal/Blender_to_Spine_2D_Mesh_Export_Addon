"""Source contracts for the official Spine 4.3 runtime oracle."""

from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ORACLE = ROOT / "tools" / "spine43_runtime_oracle.mjs"


def _source() -> str:
    return ORACLE.read_text(encoding="utf-8")


def test_oracle_uses_only_read_apis_for_external_inputs() -> None:
    source = _source()

    assert "import { existsSync, readFileSync, statSync } from 'node:fs';" in source
    for forbidden in (
        "writeFileSync",
        "appendFileSync",
        "mkdirSync",
        "rmSync",
        "unlinkSync",
        "renameSync",
        "copyFileSync",
        "cpSync",
    ):
        assert forbidden not in source


def test_oracle_uses_exact_spine43_pose_and_physics_api() -> None:
    source = _source()

    assert "const EXPECTED_VERSION = '4.3.23';" in source
    assert "skeleton.setupPose();" in source
    assert "skeleton.updateCache();" in source
    assert "skeleton.updateWorldTransform(runtime.Physics.none);" in source
    assert "bone.appliedPose" in source
    assert "runtime.Physics.none" in source
    assert "setToSetupPose" not in source


def test_oracle_validates_unified_constraints_in_runtime_order_and_cache() -> None:
    source = _source()

    assert "document.constraints" in source
    assert "skeleton.constraints" in source
    assert "Runtime unified constraint inventory/order differs from JSON" in source
    assert "skeleton._updateCache" in source
    assert "everyConstraintScheduledExactlyOnce: true" in source
    assert "Constraint '${name}' must appear exactly once" in source


def test_oracle_binds_an_in_memory_atlas_before_parsing() -> None:
    source = _source()

    atlas = source.index(
        "const atlas = new runtime.TextureAtlas(createAtlasText(collectAtlasRegions(document)));"
    )
    bind = source.index("bindAtlasPageTextures(atlas);", atlas)
    loader = source.index("const loader = new runtime.AtlasAttachmentLoader(atlas);", bind)
    reader = source.index("const reader = new runtime.SkeletonJson(loader);", loader)
    skeleton = source.index("const skeleton = new runtime.Skeleton(skeletonData);", reader)

    assert atlas < bind < loader < reader < skeleton
    assert "page.setTexture(createOracleTexture(page.width, page.height));" in source
    assert "atlas.dispose();" in source


def test_oracle_requires_finite_matrices_setup_attachments_and_positive_bounds() -> None:
    source = _source()

    for fragment in (
        "validateBoneMatrices(skeleton)",
        "bone.appliedPose",
        "Runtime setup attachments differ from JSON setup attachments",
        "skeleton.getBounds(offset, size);",
        "Runtime setup bounds must have positive width and height",
        "allFinite: true",
        "setupRenderableAttachments: setupAttachments.length",
    ):
        assert fragment in source
