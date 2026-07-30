"""Source contracts for the external Spine 4.1 runtime acceptance oracle."""

from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ORACLE = ROOT / "tools" / "spine41_runtime_oracle.mjs"


def _source() -> str:
    return ORACLE.read_text(encoding="utf-8")


def test_spine41_oracle_binds_textures_through_runtime_page_api() -> None:
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
    assert "function createOracleTexture(width, height)" in source
    assert "function bindAtlasPageTextures(atlas)" in source
    assert "page.setTexture(createOracleTexture(width, height));" in source
    assert "page.texture.getImage()" in source


def test_spine41_oracle_does_not_use_the_unsupported_textureatlas_callback() -> None:
    source = _source()

    assert "new runtime.TextureAtlas(\n    createAtlasText" not in source
    assert "() => ({\n      setFilters()" not in source


def test_spine41_oracle_treats_the_external_runtime_as_read_only() -> None:
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


def test_spine41_oracle_is_concise_by_default_and_full_only_on_request() -> None:
    source = _source()

    option_parse = source.index("const options = parseOutputOptions(process.argv.slice(4));")
    matrix_validation = source.index(
        "const boneSnapshots = validateBoneMatrices(skeleton);",
        option_parse,
    )
    summary_creation = source.index("const summary = {", matrix_validation)
    conditional_report = source.index(
        "const report = options.includeDetails",
        summary_creation,
    )

    assert option_parse < matrix_validation < summary_creation < conditional_report
    assert "if (argument === '--full')" in source
    assert "outputMode: options.includeDetails ? 'full' : 'summary'" in source
    assert "finiteBones: boneSnapshots.length" in source
    assert "everyConstraintScheduledExactlyOnce: true" in source
    assert "details: {" in source
    assert "bones: boneSnapshots" in source


def test_spine41_oracle_rejects_unknown_output_options() -> None:
    source = _source()

    assert "function parseOutputOptions(argumentsList)" in source
    assert "fail(`Unknown oracle option: ${String(argument)}`);" in source


def test_spine41_oracle_detects_renderables_by_runtime_class_identity() -> None:
    source = _source()

    assert "'RegionAttachment'" in source
    assert "'MeshAttachment'" in source
    assert "attachment instanceof runtime.RegionAttachment" in source
    assert "attachment instanceof runtime.MeshAttachment" in source
    assert "attachment.constructor?.name" not in source


def test_spine41_oracle_requires_runtime_setup_attachments_to_match_json() -> None:
    source = _source()

    expected_collection = source.index(
        "const expectedSetupAttachments = collectExpectedSetupRenderableAttachments(document);"
    )
    runtime_validation = source.index(
        "const setupAttachments = validateSetupAttachments(",
        expected_collection,
    )
    bounds_validation = source.index(
        "const bounds = setupBounds(runtime, skeleton, setupAttachments);",
        runtime_validation,
    )
    validation_function = source.index("function validateSetupAttachments(")
    deep_equal = source.index("assert.deepEqual(", validation_function)
    validation_return = source.index("return actualAttachments;", deep_equal)

    assert validation_function < deep_equal < validation_return
    assert expected_collection < runtime_validation < bounds_validation
    assert "setupRenderableAttachments: setupAttachments.length" in source
    assert "Runtime setup bounds must have positive width and height" in source
