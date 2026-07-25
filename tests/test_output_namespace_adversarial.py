from pathlib import Path

import pytest

from Blender_to_Spine2D_Mesh_Exporter.application import (
    A1OutputPathClaim,
    A1OutputPathKind,
    validate_a1_output_claims,
)
from Blender_to_Spine2D_Mesh_Exporter.domain.baking import sanitize_filename_stem


@pytest.mark.parametrize(
    ("source", "expected"),
    (
        ("CON", "CON_"),
        ("prn.json", "prn_.json"),
        ("AUX ", "AUX_"),
        ("LPT1.texture", "LPT1_.texture"),
        ('Object:A', "Object_A"),
        ('Object/A', "Object_A"),
        ("hero. ", "hero"),
    ),
)
def test_windows_filename_sanitization_handles_devices_and_unsafe_suffixes(source, expected):
    assert sanitize_filename_stem(source) == expected


def test_sanitization_collision_is_rejected_before_any_file_reservation(tmp_path: Path):
    root = tmp_path.resolve()
    first = root / f"{sanitize_filename_stem('Object:A')}_Baked.png"
    second = root / f"{sanitize_filename_stem('Object/A')}_Baked.png"

    with pytest.raises(ValueError, match="Windows output path collision"):
        validate_a1_output_claims(
            root,
            (
                A1OutputPathClaim(first, "Object:A", A1OutputPathKind.TEXTURE),
                A1OutputPathClaim(second, "Object/A", A1OutputPathKind.TEXTURE),
            ),
        )


def test_case_and_trailing_dot_collisions_are_rejected_on_non_windows_hosts(tmp_path: Path):
    root = tmp_path.resolve()
    claims = (
        A1OutputPathClaim(root / "Hero.PNG", "Hero", A1OutputPathKind.TEXTURE),
        A1OutputPathClaim(root / "hero.png. ", "hero", A1OutputPathKind.TEXTURE),
    )
    with pytest.raises(ValueError, match="Windows output path collision"):
        validate_a1_output_claims(root, claims)


def test_output_claim_cannot_escape_root_after_normalization(tmp_path: Path):
    root = (tmp_path / "export").resolve()
    with pytest.raises(ValueError, match="escapes output root"):
        validate_a1_output_claims(
            root,
            (
                A1OutputPathClaim(
                    root / ".." / "outside.json",
                    "document",
                    A1OutputPathKind.JSON,
                ),
            ),
        )
