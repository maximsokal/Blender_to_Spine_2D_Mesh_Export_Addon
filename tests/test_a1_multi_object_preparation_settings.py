from dataclasses import replace
from pathlib import Path

import pytest

from Blender_to_Spine2D_Mesh_Exporter.application import (
    A1MultiObjectMode,
    A1SingleObjectExportSettings,
    ExportSettings,
    resolve_a1_multi_object_preparation_settings,
)
from Blender_to_Spine2D_Mesh_Exporter.domain.baking import A1TextureExportMode
from Blender_to_Spine2D_Mesh_Exporter.domain.projection import A1ProjectionDirection
from Blender_to_Spine2D_Mesh_Exporter.domain.spine.rig_profiles import (
    A1RigSetupPoseMode,
)


def _settings(
    *,
    use_world_location_for_main_bone: bool = True,
    projection_direction: A1ProjectionDirection = A1ProjectionDirection.POSITIVE_Z,
    texture_export_mode: A1TextureExportMode = A1TextureExportMode.NORMAL_UV_SEGMENTS,
    rig_setup_pose_mode: A1RigSetupPoseMode = A1RigSetupPoseMode.PRESERVE_COMPOSITION,
) -> A1SingleObjectExportSettings:
    base = A1SingleObjectExportSettings(
        export=ExportSettings(
            texture_width=128,
            texture_height=96,
            output_directory=Path("multi-object-settings-test-output"),
        ),
        prefix="SettingsPolicy",
        use_world_location_for_main_bone=use_world_location_for_main_bone,
        projection_direction=projection_direction,
        rig_setup_pose_mode=rig_setup_pose_mode,
    )
    if base.bake_execution.texture_export_mode is texture_export_mode:
        return base
    return replace(
        base,
        bake_execution=replace(
            base.bake_execution,
            texture_export_mode=texture_export_mode,
        ),
    )


@pytest.mark.parametrize(
    "projection_direction",
    (
        A1ProjectionDirection.POSITIVE_X,
        A1ProjectionDirection.NEGATIVE_X,
        A1ProjectionDirection.POSITIVE_Y,
        A1ProjectionDirection.NEGATIVE_Y,
        A1ProjectionDirection.POSITIVE_Z,
        A1ProjectionDirection.NEGATIVE_Z,
    ),
)
def test_standalone_signed_axis_normal_uv_uses_neutral_projected_object_root_setup(
    projection_direction: A1ProjectionDirection,
) -> None:
    settings = _settings(projection_direction=projection_direction)

    resolved = resolve_a1_multi_object_preparation_settings(
        settings,
        A1MultiObjectMode.STANDALONE,
    )

    assert resolved is not settings
    assert resolved.rig_setup_pose_mode is A1RigSetupPoseMode.CAMERA_VIEW_NORMAL
    assert (
        replace(
            resolved,
            rig_setup_pose_mode=A1RigSetupPoseMode.PRESERVE_COMPOSITION,
        )
        == settings
    )


def test_standalone_camera_projection_preserves_settings_identity() -> None:
    settings = _settings(texture_export_mode=A1TextureExportMode.CAMERA_PROJECTION)

    resolved = resolve_a1_multi_object_preparation_settings(
        settings,
        A1MultiObjectMode.STANDALONE,
    )

    assert resolved is settings


def test_standalone_active_camera_policy_is_owned_later_by_document_preparation() -> None:
    settings = _settings(projection_direction=A1ProjectionDirection.ACTIVE_CAMERA)

    resolved = resolve_a1_multi_object_preparation_settings(
        settings,
        A1MultiObjectMode.STANDALONE,
    )

    assert resolved is settings
    assert resolved.rig_setup_pose_mode is A1RigSetupPoseMode.PRESERVE_COMPOSITION


def test_standalone_respects_explicit_nondefault_setup_mode() -> None:
    settings = _settings(
        projection_direction=A1ProjectionDirection.POSITIVE_X,
        rig_setup_pose_mode=A1RigSetupPoseMode.NORMALIZED_SINGLE,
    )

    resolved = resolve_a1_multi_object_preparation_settings(
        settings,
        A1MultiObjectMode.STANDALONE,
    )

    assert resolved is settings


def test_connected_preparation_disables_only_absolute_world_placement():
    settings = _settings()

    resolved = resolve_a1_multi_object_preparation_settings(
        settings,
        A1MultiObjectMode.CONNECTED,
    )

    assert resolved is not settings
    assert not resolved.use_world_location_for_main_bone
    assert resolved.rig_setup_pose_mode is A1RigSetupPoseMode.PRESERVE_COMPOSITION
    assert replace(resolved, use_world_location_for_main_bone=True) == settings


def test_connected_preparation_reuses_already_compatible_settings():
    settings = _settings(use_world_location_for_main_bone=False)

    resolved = resolve_a1_multi_object_preparation_settings(
        settings,
        A1MultiObjectMode.CONNECTED,
    )

    assert resolved is settings


def test_mixed_preparation_requires_explicit_subgroup_mode():
    with pytest.raises(ValueError, match="must be resolved"):
        resolve_a1_multi_object_preparation_settings(
            _settings(),
            A1MultiObjectMode.MIXED,
        )


def test_multi_object_preparation_settings_reject_invalid_types():
    settings = _settings()

    with pytest.raises(TypeError, match="settings"):
        resolve_a1_multi_object_preparation_settings(
            object(),
            A1MultiObjectMode.CONNECTED,
        )
    with pytest.raises(TypeError, match="mode"):
        resolve_a1_multi_object_preparation_settings(settings, object())


def test_preparation_and_composition_share_one_settings_policy_owner():
    root = Path(__file__).resolve().parents[1]
    adapter_root = root / "Blender_to_Spine2D_Mesh_Exporter" / "blender_adapter"
    preparation_source = (adapter_root / "a1_multi_object_export.py").read_text(
        encoding="utf-8"
    )
    composition_source = (adapter_root / "a1_multi_object_composition.py").read_text(
        encoding="utf-8"
    )

    for source in (preparation_source, composition_source):
        assert "resolve_a1_multi_object_preparation_settings" in source
        assert "replace(source.settings, use_world_location_for_main_bone=False)" not in source
