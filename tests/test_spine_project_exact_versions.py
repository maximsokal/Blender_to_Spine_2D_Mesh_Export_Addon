"""Contracts for user-configurable exact Spine project patch versions."""

from __future__ import annotations

from pathlib import Path

import pytest

from Blender_to_Spine2D_Mesh_Exporter.application import ExportSettings
from Blender_to_Spine2D_Mesh_Exporter.domain.spine.version_target import (
    SpineJsonTarget,
    resolve_spine_json_exact_version,
    resolve_spine_json_target,
    spine_json_version_filename_token,
    validate_spine_json_exact_version_for_target,
)


_CUSTOM_EXACT_VERSIONS = (
    (SpineJsonTarget.SPINE_3_8, "3.8.42"),
    (SpineJsonTarget.SPINE_4_0, "4.0.17"),
    (SpineJsonTarget.SPINE_4_1, "4.1.7"),
    (SpineJsonTarget.SPINE_4_2, "4.2.11"),
    (SpineJsonTarget.SPINE_4_3, "4.3.5"),
)


@pytest.mark.parametrize(("target", "exact_version"), _CUSTOM_EXACT_VERSIONS)
def test_custom_exact_patch_resolves_to_family_and_survives_export_settings(
    target: SpineJsonTarget,
    exact_version: str,
) -> None:
    assert exact_version != target.exact_version
    assert resolve_spine_json_exact_version(exact_version) is target
    assert resolve_spine_json_target(exact_version) is target
    assert validate_spine_json_exact_version_for_target(target, exact_version) == exact_version
    assert spine_json_version_filename_token(exact_version) == f"spine_{exact_version}"

    settings = ExportSettings(
        texture_width=64,
        texture_height=64,
        output_directory=Path("."),
        spine_version=exact_version,
    )
    assert settings.spine_version == exact_version
    assert settings.spine_target is target


@pytest.mark.parametrize(
    "value",
    (
        "4.2",
        "v4.2.43",
        "4.2.43-beta",
        "04.2.43",
        "4.02.43",
        "4.2.043",
        "4.2.-1",
        "",
        "   ",
    ),
)
def test_exact_version_requires_canonical_major_minor_patch(value: str) -> None:
    with pytest.raises(ValueError):
        resolve_spine_json_exact_version(value)


@pytest.mark.parametrize("value", (None, 4.2, 42, object()))
def test_exact_version_rejects_non_string_values(value: object) -> None:
    with pytest.raises(TypeError, match="must be str"):
        resolve_spine_json_exact_version(value)


def test_exact_version_trims_surrounding_whitespace_to_canonical_value() -> None:
    value = "  4.2.11  "
    assert resolve_spine_json_exact_version(value) is SpineJsonTarget.SPINE_4_2
    assert (
        validate_spine_json_exact_version_for_target(
            SpineJsonTarget.SPINE_4_2,
            value,
        )
        == "4.2.11"
    )


def test_exact_version_must_match_selected_codec_family() -> None:
    with pytest.raises(ValueError, match="not selected family"):
        validate_spine_json_exact_version_for_target(
            SpineJsonTarget.SPINE_4_2,
            "4.1.24",
        )


def test_future_patch_inside_supported_family_is_not_artificially_capped() -> None:
    value = "4.2.9999"
    assert resolve_spine_json_exact_version(value) is SpineJsonTarget.SPINE_4_2
    assert (
        validate_spine_json_exact_version_for_target(
            SpineJsonTarget.SPINE_4_2,
            value,
        )
        == value
    )


@pytest.mark.parametrize("value", ("3.7.99", "4.4.0", "5.0.1"))
def test_unsupported_family_is_rejected_even_with_valid_semver(value: str) -> None:
    with pytest.raises(ValueError, match="supported families"):
        resolve_spine_json_exact_version(value)
