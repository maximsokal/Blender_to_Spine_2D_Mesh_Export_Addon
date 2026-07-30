"""Production capability contracts for limited legacy 4.x and full Spine 4.2 scope."""

from __future__ import annotations

from pathlib import Path

import pytest

from Blender_to_Spine2D_Mesh_Exporter.application import (
    A1MultiObjectMode,
    A1SingleObjectExportSettings,
    ExportSettings,
    resolve_a1_multi_object_preparation_settings,
)
from Blender_to_Spine2D_Mesh_Exporter.domain.spine.export_capabilities import (
    SpineJsonExportCapabilityError,
    SpineJsonExportScope,
    registered_spine_json_export_capabilities,
    require_spine_json_export_capability,
)
from Blender_to_Spine2D_Mesh_Exporter.domain.spine.rig_profiles import A1RigProfile
from Blender_to_Spine2D_Mesh_Exporter.domain.spine.version_target import SpineJsonTarget


def _settings(
    tmp_path: Path,
    *,
    target: SpineJsonTarget,
    profile: A1RigProfile,
) -> A1SingleObjectExportSettings:
    return A1SingleObjectExportSettings(
        export=ExportSettings(
            texture_width=64,
            texture_height=64,
            output_directory=tmp_path,
            spine_version=target.exact_version,
            rig_profile=profile.value,
        ),
        prefix="Cone",
        output_stem="Cone",
    )


@pytest.mark.parametrize(
    "target",
    (SpineJsonTarget.SPINE_4_0, SpineJsonTarget.SPINE_4_1),
)
def test_legacy_four_x_two_axis_accepts_single_and_standalone_only(target) -> None:
    accepted = {
        SpineJsonExportScope.SINGLE_OBJECT,
        SpineJsonExportScope.STANDALONE_MULTI_OBJECT,
    }

    for scope in accepted:
        capability = require_spine_json_export_capability(
            target,
            A1RigProfile.TWO_AXIS_ROTATION_SCALE,
            scope,
        )
        assert capability.target is target
        assert capability.rig_profile is A1RigProfile.TWO_AXIS_ROTATION_SCALE
        assert capability.scopes == frozenset(accepted)

    capability = require_spine_json_export_capability(
        target,
        A1RigProfile.TWO_AXIS_ROTATION_SCALE,
        SpineJsonExportScope.SINGLE_OBJECT,
    )
    if target is SpineJsonTarget.SPINE_4_0:
        assert capability.limitations == (
            "Attachment and animation sequences are not supported by Spine 4.0.64.",
        )
    else:
        assert capability.limitations == ()


@pytest.mark.parametrize(
    "target",
    (SpineJsonTarget.SPINE_4_0, SpineJsonTarget.SPINE_4_1),
)
@pytest.mark.parametrize(
    "scope",
    (
        SpineJsonExportScope.CONNECTED_MULTI_OBJECT,
        SpineJsonExportScope.MIXED_MULTI_OBJECT,
    ),
)
def test_legacy_four_x_two_axis_rejects_connected_and_mixed(target, scope) -> None:
    with pytest.raises(SpineJsonExportCapabilityError, match=scope.value):
        require_spine_json_export_capability(
            target,
            A1RigProfile.TWO_AXIS_ROTATION_SCALE,
            scope,
        )


@pytest.mark.parametrize(
    "target",
    (SpineJsonTarget.SPINE_4_0, SpineJsonTarget.SPINE_4_1),
)
@pytest.mark.parametrize("scope", tuple(SpineJsonExportScope))
def test_legacy_four_x_rejects_three_axis_for_every_scope(target, scope) -> None:
    with pytest.raises(SpineJsonExportCapabilityError, match="3-Axis Rotation"):
        require_spine_json_export_capability(
            target,
            A1RigProfile.THREE_AXIS_ROTATION,
            scope,
        )


@pytest.mark.parametrize("scope", tuple(SpineJsonExportScope))
def test_spine42_preserves_all_existing_profiles_and_scopes(scope) -> None:
    for profile in A1RigProfile:
        capability = require_spine_json_export_capability(
            SpineJsonTarget.SPINE_4_2,
            profile,
            scope,
        )
        assert capability.target is SpineJsonTarget.SPINE_4_2
        assert capability.rig_profile is profile
        assert scope in capability.scopes


def test_capability_registry_is_immutable_and_contains_only_ready_pairs() -> None:
    capabilities = registered_spine_json_export_capabilities()

    assert set(capabilities) == {
        (
            SpineJsonTarget.SPINE_4_0,
            A1RigProfile.TWO_AXIS_ROTATION_SCALE,
        ),
        (
            SpineJsonTarget.SPINE_4_1,
            A1RigProfile.TWO_AXIS_ROTATION_SCALE,
        ),
        (
            SpineJsonTarget.SPINE_4_2,
            A1RigProfile.THREE_AXIS_ROTATION,
        ),
        (
            SpineJsonTarget.SPINE_4_2,
            A1RigProfile.TWO_AXIS_ROTATION_SCALE,
        ),
    }
    with pytest.raises(TypeError):
        capabilities[
            (
                SpineJsonTarget.SPINE_4_1,
                A1RigProfile.THREE_AXIS_ROTATION,
            )
        ] = next(iter(capabilities.values()))


@pytest.mark.parametrize(
    "target",
    (SpineJsonTarget.SPINE_4_0, SpineJsonTarget.SPINE_4_1),
)
def test_legacy_four_x_standalone_settings_pass_without_world_location_rewrite(
    tmp_path: Path,
    target: SpineJsonTarget,
) -> None:
    settings = _settings(
        tmp_path,
        target=target,
        profile=A1RigProfile.TWO_AXIS_ROTATION_SCALE,
    )

    resolved = resolve_a1_multi_object_preparation_settings(
        settings,
        A1MultiObjectMode.STANDALONE,
    )

    assert resolved is settings
    assert resolved.use_world_location_for_main_bone is True


@pytest.mark.parametrize(
    "target",
    (SpineJsonTarget.SPINE_4_0, SpineJsonTarget.SPINE_4_1),
)
def test_legacy_four_x_connected_settings_fail_before_connected_rewrite(
    tmp_path: Path,
    target: SpineJsonTarget,
) -> None:
    settings = _settings(
        tmp_path,
        target=target,
        profile=A1RigProfile.TWO_AXIS_ROTATION_SCALE,
    )

    with pytest.raises(
        SpineJsonExportCapabilityError,
        match="CONNECTED_MULTI_OBJECT",
    ):
        resolve_a1_multi_object_preparation_settings(
            settings,
            A1MultiObjectMode.CONNECTED,
        )
    assert settings.use_world_location_for_main_bone is True
