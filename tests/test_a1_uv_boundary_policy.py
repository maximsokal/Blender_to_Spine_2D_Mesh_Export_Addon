from dataclasses import replace
from pathlib import Path

import pytest

from Blender_to_Spine2D_Mesh_Exporter.application import (
    A1SingleObjectExportSettings,
    A1SourceUvBoundaryMode,
    ExportSettings,
)
from Blender_to_Spine2D_Mesh_Exporter.domain.geometry import SegmentationSettings


def make_settings(tmp_path: Path, **changes) -> A1SingleObjectExportSettings:
    base = A1SingleObjectExportSettings(
        export=ExportSettings(
            texture_width=64,
            texture_height=64,
            output_directory=tmp_path,
        ),
    )
    return replace(base, **changes)


def test_rewrite_default_disables_source_uv_boundary_segmentation(tmp_path):
    settings = make_settings(tmp_path)
    resolved = settings.resolved_geometry_settings().segmentation

    assert settings.source_uv_boundary_mode is A1SourceUvBoundaryMode.DISABLED
    assert resolved.split_uv_boundaries is False
    assert resolved.uv_layer_name is None


def test_disabled_mode_overrides_nested_active_uv_behavior(tmp_path):
    settings = make_settings(
        tmp_path,
        geometry=replace(
            make_settings(tmp_path).geometry,
            segmentation=SegmentationSettings(
                split_uv_boundaries=True,
                uv_layer_name=None,
            ),
        ),
    )

    resolved = settings.resolved_geometry_settings().segmentation

    assert resolved.split_uv_boundaries is False
    assert resolved.uv_layer_name is None


def test_explicit_source_uv_layer_is_the_only_named_source(tmp_path):
    settings = make_settings(
        tmp_path,
        source_uv_boundary_mode=A1SourceUvBoundaryMode.EXPLICIT_LAYER,
        source_uv_boundary_layer_name="SourceUV",
    )

    resolved = settings.resolved_geometry_settings().segmentation

    assert resolved.split_uv_boundaries is True
    assert resolved.uv_layer_name == "SourceUV"


def test_legacy_active_layer_behavior_requires_explicit_opt_in(tmp_path):
    settings = make_settings(
        tmp_path,
        source_uv_boundary_mode=A1SourceUvBoundaryMode.ACTIVE_LAYER_LEGACY,
    )

    resolved = settings.resolved_geometry_settings().segmentation

    assert resolved.split_uv_boundaries is True
    assert resolved.uv_layer_name is None


def test_explicit_layer_mode_requires_a_layer_name(tmp_path):
    with pytest.raises(ValueError, match="required for EXPLICIT_LAYER"):
        make_settings(
            tmp_path,
            source_uv_boundary_mode=A1SourceUvBoundaryMode.EXPLICIT_LAYER,
        )


def test_layer_name_is_rejected_when_mode_does_not_use_it(tmp_path):
    with pytest.raises(ValueError, match="only valid for EXPLICIT_LAYER"):
        make_settings(
            tmp_path,
            source_uv_boundary_layer_name="IgnoredUV",
        )


def test_uv_boundary_policy_does_not_override_custom_seam_mode(tmp_path):
    settings = A1SingleObjectExportSettings(
        export=ExportSettings(
            texture_width=64,
            texture_height=64,
            output_directory=tmp_path,
            seam_mode="CUSTOM",
        ),
        source_uv_boundary_mode=A1SourceUvBoundaryMode.EXPLICIT_LAYER,
        source_uv_boundary_layer_name="SourceUV",
    )

    resolved = settings.resolved_geometry_settings().segmentation

    assert resolved.split_uv_boundaries is True
    assert resolved.uv_layer_name == "SourceUV"
    assert resolved.split_by_angle is False
    assert resolved.respect_seams is True
