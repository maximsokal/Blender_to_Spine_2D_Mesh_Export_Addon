"""Source contracts for the exact read-only Spine 3.8 runtime oracle."""

from __future__ import annotations

from pathlib import Path


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
