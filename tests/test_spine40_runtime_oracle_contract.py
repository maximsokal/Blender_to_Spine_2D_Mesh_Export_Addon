"""Source contracts for the read-only Spine 4.0 runtime oracle."""

from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ORACLE = ROOT / "tools" / "spine40_runtime_oracle.mjs"


def _source() -> str:
    return ORACLE.read_text(encoding="utf-8")


def test_oracle_requires_exact_spine_4064_json() -> None:
    source = _source()

    assert "const EXPECTED_VERSION = '4.0.64';" in source
    assert "skeletonMetadata.spine !== EXPECTED_VERSION" in source
    assert "version: skeletonData.version" in source


def test_oracle_treats_external_runtime_as_read_only() -> None:
    source = _source()

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


def test_oracle_loads_required_runtime_classes_and_matches_setup_attachments() -> None:
    source = _source()

    for export_name in (
        "'TextureAtlas'",
        "'AtlasAttachmentLoader'",
        "'SkeletonJson'",
        "'Skeleton'",
        "'Vector2'",
        "'RegionAttachment'",
        "'MeshAttachment'",
    ):
        assert export_name in source
    assert "attachment instanceof runtime.RegionAttachment" in source
    assert "attachment instanceof runtime.MeshAttachment" in source
    assert "Runtime setup renderable attachments differ" in source


def test_oracle_checks_constraint_schedule_matrices_and_positive_bounds() -> None:
    source = _source()

    assert "validateConstraintOrders(constraintRecords);" in source
    assert "skeleton.updateCache();" in source
    assert "everyConstraintScheduledExactlyOnce: true" in source
    assert "const bones = validateBoneMatrices(skeleton);" in source
    assert "allFinite: true" in source
    assert "Runtime setup bounds must have positive width and height" in source


def test_oracle_rejects_sequence_data_and_unknown_cli_options() -> None:
    source = _source()

    assert "Spine 4.0 acceptance does not permit setup attachment sequences" in source
    assert "fail(`Unknown oracle option: ${String(argument)}`);" in source
    assert "if (argument === '--full')" in source


def test_oracle_binds_atlas_textures_before_skeleton_json_read() -> None:
    source = _source()

    atlas_creation = source.index(
        "const atlas = new runtime.TextureAtlas(createAtlasText(atlasRegions));"
    )
    texture_binding = source.index("bindAtlasPageTextures(atlas);", atlas_creation)
    loader_creation = source.index(
        "const loader = new runtime.AtlasAttachmentLoader(atlas);",
        texture_binding,
    )

    assert atlas_creation < texture_binding < loader_creation
    assert "if (typeof page.setTexture === 'function') page.setTexture(texture);" in source
