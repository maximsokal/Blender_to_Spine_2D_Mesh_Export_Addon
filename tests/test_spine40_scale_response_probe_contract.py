"""Source contracts for the exact Spine 4.0 scale-response runtime probe."""

from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROBE = ROOT / "tools" / "spine40_scale_response_probe.mjs"


def _source() -> str:
    return PROBE.read_text(encoding="utf-8")


def test_probe_uses_only_read_apis_for_filesystem_inputs() -> None:
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


def test_probe_requires_exact_spine40_family_and_legacy_scale_topology() -> None:
    source = _source()

    assert "Spine 4.0 runtime entry" in source
    assert "version.startsWith('4.0')" in source
    assert "Expected Spine 4.0 JSON" in source
    assert "must remain relative-world" in source
    assert "must not use local evaluation" in source
    assert "const driver = `${prefix}_scale_rotate_X`;" in source
    assert "const unsafeDriver = `${prefix}_rotate_X`;" in source
    assert "constrainedBones.includes(driver)" in source
    assert "constrainedBones.includes(unsafeDriver)" in source


def test_probe_checks_uniform_scale_around_each_object_main_bone() -> None:
    source = _source()

    assert "const SCALE_FACTORS = Object.freeze([0.5, 1.5, 2.0]);" in source
    assert "pivotX: requireFinite(main.worldX" in source
    assert "pivotY: requireFinite(main.worldY" in source
    assert "setup.pivotX + (setup.bounds.x - setup.pivotX) * factor" in source
    assert "setup.pivotY + (setup.bounds.y - setup.pivotY) * factor" in source
    assert "setup.bounds.width * factor" in source
    assert "setup.bounds.height * factor" in source
    assert "allFinite: true" in source
    assert "allUniformAroundMain: true" in source


def test_probe_uses_exact_runtime_parser_and_in_memory_atlas() -> None:
    source = _source()

    atlas = source.index(
        "const atlas = new runtime.TextureAtlas(createAtlasText(collectAtlasRegions(document)));"
    )
    bind = source.index("bindAtlasPageTextures(atlas);", atlas)
    loader = source.index("const loader = new runtime.AtlasAttachmentLoader(atlas);", bind)
    parser = source.index("const reader = new runtime.SkeletonJson(loader);", loader)
    skeleton = source.index("const skeleton = new runtime.Skeleton(skeletonData);", parser)

    assert atlas < bind < loader < parser < skeleton
    assert "page.setTexture(createOracleTexture(page.width, page.height));" in source
    assert "control.scaleX *= factor;" in source
    assert "control.scaleY *= factor;" in source
