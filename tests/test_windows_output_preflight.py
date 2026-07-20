from pathlib import Path

import pytest

from Blender_to_Spine2D_Mesh_Exporter.application import (
    A1OutputPathClaim,
    A1OutputPathKind,
    A1OutputPreflightSource,
    A1SingleObjectExportSettings,
    ExportSettings,
    preflight_a1_output_namespace,
    validate_a1_output_claims,
)
from Blender_to_Spine2D_Mesh_Exporter.domain.baking import (
    BakeSettings,
    TextureFormat,
    predict_bake_output_paths,
    sanitize_filename_stem,
    windows_path_identity,
)


def _single_settings(
    tmp_path: Path,
    *,
    output_stem: str,
    images_relative_path: str = "images",
    sequence_start_frame: int = 0,
    sequence_frame_count: int = 0,
) -> A1SingleObjectExportSettings:
    return A1SingleObjectExportSettings(
        export=ExportSettings(
            texture_width=64,
            texture_height=64,
            output_directory=tmp_path,
            images_relative_path=images_relative_path,
            sequence_start_frame=sequence_start_frame,
            sequence_frame_count=sequence_frame_count,
        ),
        output_stem=output_stem,
    )


@pytest.mark.parametrize(
    ("raw", "expected"),
    (
        ("CON", "CON_"),
        ("con.txt", "con_.txt"),
        ("PRN", "PRN_"),
        ("AUX.data", "AUX_.data"),
        ("NUL", "NUL_"),
        ("COM1", "COM1_"),
        ("com9.log", "com9_.log"),
        ("LPT1", "LPT1_"),
        ("lpt9.cache", "lpt9_.cache"),
        ("COM¹", "COM¹_"),
        ("LPT³.data", "LPT³_.data"),
        ("Safe.Name", "Safe.Name"),
    ),
)
def test_sanitize_filename_stem_handles_windows_reserved_names(raw, expected):
    assert sanitize_filename_stem(raw) == expected


def test_windows_path_identity_is_case_insensitive(tmp_path):
    first = tmp_path / "Images" / "Hero.PNG"
    second = tmp_path / "images" / "hero.png"

    assert windows_path_identity(first) == windows_path_identity(second)


def test_predict_bake_output_paths_matches_single_and_sequence_contract(tmp_path):
    single = BakeSettings(
        width=64,
        height=64,
        output_directory=tmp_path,
        output_stem="CON",
        texture_format=TextureFormat.PNG,
    )
    sequence = BakeSettings(
        width=64,
        height=64,
        output_directory=tmp_path,
        output_stem="Hero",
        texture_format=TextureFormat.WEBP,
        sequence_start_frame=7,
        sequence_frame_count=2,
        sequence_frame_digits=4,
    )

    assert predict_bake_output_paths(single) == (tmp_path / "CON__Baked.png",)
    assert predict_bake_output_paths(sequence) == (
        tmp_path / "Hero_Baked_0007.webp",
        tmp_path / "Hero_Baked_0008.webp",
    )


def test_generic_claim_validation_detects_json_texture_cross_collision(tmp_path):
    root = tmp_path.resolve(strict=False)
    json_path = root / "Result.JSON"
    texture_path = root / "result.json"

    with pytest.raises(ValueError, match="collision"):
        validate_a1_output_claims(
            root,
            (
                A1OutputPathClaim(
                    path=json_path,
                    owner="final document",
                    kind=A1OutputPathKind.JSON,
                ),
                A1OutputPathClaim(
                    path=texture_path,
                    owner="component Hero",
                    kind=A1OutputPathKind.TEXTURE,
                ),
            ),
        )


def test_preflight_detects_post_sanitization_collision_before_preparation(tmp_path):
    sources = (
        A1OutputPreflightSource(
            owner="first",
            object_name="First",
            settings=_single_settings(tmp_path, output_stem="A:B"),
        ),
        A1OutputPreflightSource(
            owner="second",
            object_name="Second",
            settings=_single_settings(tmp_path, output_stem="A?B"),
        ),
    )

    with pytest.raises(ValueError, match="collision"):
        preflight_a1_output_namespace(
            output_root=tmp_path,
            json_path=tmp_path / "combined.json",
            sources=sources,
        )


def test_preflight_detects_case_only_texture_collision(tmp_path):
    sources = (
        A1OutputPreflightSource(
            owner="upper",
            object_name="Upper",
            settings=_single_settings(tmp_path, output_stem="Hero"),
        ),
        A1OutputPreflightSource(
            owner="lower",
            object_name="Lower",
            settings=_single_settings(tmp_path, output_stem="hero"),
        ),
    )

    with pytest.raises(ValueError, match="collision"):
        preflight_a1_output_namespace(
            output_root=tmp_path,
            json_path=tmp_path / "combined.json",
            sources=sources,
        )


def test_preflight_returns_normalized_valid_namespace(tmp_path):
    result = preflight_a1_output_namespace(
        output_root=tmp_path,
        json_path=tmp_path / "combined.json",
        sources=(
            A1OutputPreflightSource(
                owner="hero",
                object_name="Hero",
                settings=_single_settings(
                    tmp_path,
                    output_stem="Hero",
                    sequence_start_frame=3,
                    sequence_frame_count=2,
                ),
            ),
            A1OutputPreflightSource(
                owner="enemy",
                object_name="Enemy",
                settings=_single_settings(tmp_path, output_stem="Enemy"),
            ),
        ),
    )

    assert result.json_path == (tmp_path / "combined.json").resolve(strict=False)
    assert result.texture_paths == (
        (tmp_path / "images" / "Hero_Baked_0003.png").resolve(strict=False),
        (tmp_path / "images" / "Hero_Baked_0004.png").resolve(strict=False),
        (tmp_path / "images" / "Enemy_Baked.png").resolve(strict=False),
    )
