from pathlib import Path

import pytest

from Blender_to_Spine2D_Mesh_Exporter.blender_adapter import (
    scene_settings_migration as migration,
)
from Blender_to_Spine2D_Mesh_Exporter.blender_adapter.scene_settings_migration import (
    CURRENT_SETTINGS_SCHEMA_VERSION,
    clear_pre_registration_scene_state,
    migrate_scene_settings,
)
from Blender_to_Spine2D_Mesh_Exporter.domain.spine import A1RigProfile


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "Blender_to_Spine2D_Mesh_Exporter"


class _Scene:
    def __init__(
        self,
        *,
        schema: int,
        rig_profile: A1RigProfile,
        persisted_keys=(),
        seam_mode: str = "AUTO",
    ):
        self.name = "Fixture"
        self.spine2d_settings_schema_version = schema
        self.spine2d_rig_profile = rig_profile.value
        self.spine2d_seam_maker_mode = seam_mode
        self._persisted_keys = tuple(persisted_keys)

    def keys(self):
        return self._persisted_keys


class _RnaReboundScene:
    """Simulate Blender exposing RNA defaults over older raw ID-properties."""

    def __init__(self):
        self.name = "Schema 2 Rebound Fixture"
        self.spine2d_settings_schema_version = 0
        self.spine2d_seam_maker_mode = "AUTO"
        self.spine2d_rig_profile = A1RigProfile.TWO_AXIS_ROTATION_SCALE.value
        self._raw = {
            "spine2d_settings_schema_version": 2,
            "spine2d_seam_maker_mode": "CUSTOM",
        }

    def as_pointer(self):
        return id(self)

    def keys(self):
        return tuple(self._raw)

    def get(self, key, default=None):
        return self._raw.get(key, default)

    def __getitem__(self, key):
        return self._raw[key]


def test_scene_schema_property_is_hidden_and_fresh_default_is_two_axis():
    source = (PACKAGE / "blender_adapter" / "scene_properties.py").read_text(
        encoding="utf-8"
    )

    assert '"spine2d_settings_schema_version"' in source
    assert "default=0" in source
    assert 'options={"HIDDEN"}' in source
    assert 'default="AUTO"' in source
    assert "update=_update_seam_maker_mode" in source
    assert '"spine2d_rig_profile"' in source
    assert "default=A1RigProfile.TWO_AXIS_ROTATION_SCALE.value" in source
    assert "update=_update_rig_profile" in source


def test_schema_five_assigns_two_axis_only_to_genuinely_fresh_scene():
    scene = _Scene(
        schema=0,
        rig_profile=A1RigProfile.TWO_AXIS_ROTATION_SCALE,
        persisted_keys=(),
    )

    assert migrate_scene_settings(scene) is True
    assert scene.spine2d_settings_schema_version == CURRENT_SETTINGS_SCHEMA_VERSION
    assert scene.spine2d_rig_profile == A1RigProfile.TWO_AXIS_ROTATION_SCALE.value


def test_pre_profile_saved_scene_remains_on_compatibility_three_axis():
    scene = _Scene(
        schema=3,
        rig_profile=A1RigProfile.TWO_AXIS_ROTATION_SCALE,
        persisted_keys=(
            "spine2d_settings_schema_version",
            "spine2d_seam_maker_mode",
        ),
        seam_mode="CUSTOM",
    )

    assert migrate_scene_settings(scene) is True
    assert scene.spine2d_settings_schema_version == CURRENT_SETTINGS_SCHEMA_VERSION
    assert scene.spine2d_rig_profile == A1RigProfile.THREE_AXIS_ROTATION.value
    assert scene.spine2d_seam_maker_mode == "CUSTOM"


def test_pre_registration_snapshot_survives_rna_default_rebind():
    scene = _RnaReboundScene()
    clear_pre_registration_scene_state()
    try:
        assert migration._capture_pre_registration_scene_state_for_scenes((scene,)) == 1
        assert migrate_scene_settings(scene) is True
        assert scene.spine2d_settings_schema_version == CURRENT_SETTINGS_SCHEMA_VERSION
        assert scene.spine2d_seam_maker_mode == "AUTO"
        assert scene.spine2d_rig_profile == A1RigProfile.THREE_AXIS_ROTATION.value
    finally:
        clear_pre_registration_scene_state()


@pytest.mark.parametrize("profile", tuple(A1RigProfile))
def test_schema_four_preserves_the_users_selected_rig(profile):
    scene = _Scene(
        schema=4,
        rig_profile=profile,
        persisted_keys=(
            "spine2d_settings_schema_version",
            "spine2d_rig_profile",
        ),
    )

    assert migrate_scene_settings(scene) is True
    assert scene.spine2d_settings_schema_version == CURRENT_SETTINGS_SCHEMA_VERSION
    assert scene.spine2d_rig_profile == profile.value


def test_current_schema_is_idempotent():
    scene = _Scene(
        schema=CURRENT_SETTINGS_SCHEMA_VERSION,
        rig_profile=A1RigProfile.TWO_AXIS_ROTATION_SCALE,
        persisted_keys=(
            "spine2d_settings_schema_version",
            "spine2d_rig_profile",
        ),
    )

    assert migrate_scene_settings(scene) is False


def test_root_registers_snapshot_before_scene_rna_and_migration_before_ui():
    source = (PACKAGE / "__init__.py").read_text(encoding="utf-8")
    register_config_start = source.index("def _register_config_rna()")
    unregister_config_start = source.index("def _unregister_config_rna()")
    register_config = source[register_config_start:unregister_config_start]

    assert register_config.index("capture_pre_registration_scene_state()") < (
        register_config.index("register_rna_properties_transactionally(")
    )
    assert "clear_pre_registration_scene_state()" in register_config
    assert source.index('"Scene RNA properties"') < source.index(
        '"Scene settings migration"'
    ) < source.index('"UI"')
    assert source.index('"UI"') < source.index('"Rig UI"')
