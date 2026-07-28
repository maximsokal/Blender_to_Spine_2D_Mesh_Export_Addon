from dataclasses import fields, replace
from pathlib import Path

import pytest

from Blender_to_Spine2D_Mesh_Exporter.application import (
    A1SingleObjectExportSettings,
    A1SourceUvBoundaryMode,
    ExportSettings,
)
from Blender_to_Spine2D_Mesh_Exporter.blender_adapter.a1_source_geometry_preparation import (
    _resolve_source_uv_boundary_layer,
)
from Blender_to_Spine2D_Mesh_Exporter.domain.geometry import (
    MeshSnapshot,
    SegmentationSettings,
)


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


def make_uv_only_snapshot(*, active_uv_layer: str | None) -> MeshSnapshot:
    return MeshSnapshot(
        snapshot_id="uv-policy",
        source_object_id="Object",
        object_name="Object",
        vertices=(),
        edges=(),
        loops=(),
        faces=(),
        uv_layer_names=("SourceUV", "OtherUV"),
        active_uv_layer=active_uv_layer,
        render_uv_layer=active_uv_layer,
    )


def test_appended_settings_preserve_uv_policy_and_rig_positional_order():
    names = tuple(field.name for field in fields(A1SingleObjectExportSettings))

    assert names[-6:] == (
        "source_uv_boundary_mode",
        "source_uv_boundary_layer_name",
        "material_source_policy",
        "generated_material_pattern",
        "generated_gray_color",
        "rig_setup_pose_mode",
    )


def test_disabled_mode_never_resolves_the_active_source_layer(tmp_path):
    snapshot = make_uv_only_snapshot(active_uv_layer="SourceUV")

    assert _resolve_source_uv_boundary_layer(
        snapshot,
        make_settings(tmp_path),
    ) is None


def test_explicit_mode_rejects_a_layer_missing_from_the_snapshot(tmp_path):
    settings = make_settings(
        tmp_path,
        source_uv_boundary_mode=A1SourceUvBoundaryMode.EXPLICIT_LAYER,
        source_uv_boundary_layer_name="MissingUV",
    )

    with pytest.raises(ValueError, match="is absent from snapshot"):
        _resolve_source_uv_boundary_layer(
            make_uv_only_snapshot(active_uv_layer="SourceUV"),
            settings,
        )


def test_legacy_mode_requires_and_resolves_the_active_layer(tmp_path):
    settings = make_settings(
        tmp_path,
        source_uv_boundary_mode=A1SourceUvBoundaryMode.ACTIVE_LAYER_LEGACY,
    )

    assert _resolve_source_uv_boundary_layer(
        make_uv_only_snapshot(active_uv_layer="OtherUV"),
        settings,
    ) == "OtherUV"

    with pytest.raises(ValueError, match="requires the source mesh"):
        _resolve_source_uv_boundary_layer(
            make_uv_only_snapshot(active_uv_layer=None),
            settings,
        )
