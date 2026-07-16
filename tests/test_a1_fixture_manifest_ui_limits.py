from pathlib import Path

import pytest

from tools.a1_fixture_manifest import (
    FixtureExportSettings,
    FixtureManifestError,
    FixtureSequenceSettings,
)


@pytest.mark.parametrize("value", (0, 63, 65, 4097, 512.5, True))
def test_texture_size_must_match_blender_property_exactly(value):
    with pytest.raises(FixtureManifestError, match="texture_size"):
        FixtureExportSettings(texture_size=value)


@pytest.mark.parametrize("value", (64, 1024, 4096, 1024.0))
def test_supported_even_texture_sizes_are_normalized_to_int(value):
    settings = FixtureExportSettings(texture_size=value)
    assert isinstance(settings.texture_size, int)
    assert settings.texture_size == int(value)


@pytest.mark.parametrize("value", (0, 90, 30.5, False))
def test_angle_limit_must_match_blender_int_property(value):
    with pytest.raises(FixtureManifestError, match="angle_limit"):
        FixtureExportSettings(angle_limit=value)


def test_integral_float_angle_is_normalized_without_rounding():
    settings = FixtureExportSettings(angle_limit=30.0)
    assert settings.angle_limit == 30
    assert isinstance(settings.angle_limit, int)


@pytest.mark.parametrize("value", (-1, 1.5, True))
def test_sequence_values_must_be_non_negative_integers(value):
    with pytest.raises(FixtureManifestError):
        FixtureSequenceSettings(start_frame=value)
    with pytest.raises(FixtureManifestError):
        FixtureSequenceSettings(frame_count=value)
