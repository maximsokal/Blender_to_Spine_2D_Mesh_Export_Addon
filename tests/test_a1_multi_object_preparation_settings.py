from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from Blender_to_Spine2D_Mesh_Exporter.application import (
    A1MultiObjectExportSettings,
    A1MultiObjectMode,
    A1SingleObjectExportSettings,
    ExportSettings,
    resolve_a1_multi_object_preparation_settings,
)
from Blender_to_Spine2D_Mesh_Exporter.blender_adapter.a1_multi_object_composition import (
    _expected_prepared_settings,
    _resolve_composition_shared_pivot_world,
)
from Blender_to_Spine2D_Mesh_Exporter.blender_adapter.a1_multi_object_contracts import (
    A1MultiObjectSource,
)
from Blender_to_Spine2D_Mesh_Exporter.blender_adapter.a1_multi_object_export import (
    _settings_for_preparation,
)


def _settings(
    *,
    use_world_location_for_main_bone: bool = True,
) -> A1SingleObjectExportSettings:
    return A1SingleObjectExportSettings(
        export=ExportSettings(
            texture_width=128,
            texture_height=96,
            output_directory=Path("multi-object-settings-test-output"),
        ),
        prefix="SettingsPolicy",
        use_world_location_for_main_bone=use_world_location_for_main_bone,
    )


def _source(settings: A1SingleObjectExportSettings) -> A1MultiObjectSource:
    source_object = SimpleNamespace(type="MESH", name="SettingsSource", data=object())
    return A1MultiObjectSource(
        source_object=source_object,
        component_id="settings-source",
        settings=settings,
    )


def _multi_settings(*, shared_pivot_enabled: bool) -> A1MultiObjectExportSettings:
    return A1MultiObjectExportSettings(
        output_directory=Path("multi-object-composition-test-output"),
        output_stem="settings-composition",
        mode=A1MultiObjectMode.STANDALONE,
        shared_pivot_enabled=shared_pivot_enabled,
    )


def test_standalone_preparation_preserves_settings_identity():
    settings = _settings()

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


def test_shared_pivot_disabled_keeps_exact_standalone_settings_object():
    settings = _settings()

    resolved = _settings_for_preparation(
        _source(settings),
        A1MultiObjectMode.STANDALONE,
    )

    assert resolved is settings
    assert resolved.shared_pivot_world is None


def test_shared_pivot_replacement_changes_only_export_pivot_field():
    settings = _settings()
    pivot = (1.25, -4.5, 9.0)

    resolved = _settings_for_preparation(
        _source(settings),
        A1MultiObjectMode.STANDALONE,
        shared_pivot_world=pivot,
    )

    assert resolved is not settings
    assert settings.shared_pivot_world is None
    assert resolved.shared_pivot_world == pivot
    assert replace(resolved, shared_pivot_world=None) == settings
    assert resolved.rig_setup_pose_mode is settings.rig_setup_pose_mode
    assert resolved.projection_direction is settings.projection_direction
    assert resolved.export is settings.export


def test_composition_accepts_one_identical_transaction_shared_pivot():
    pivot = (1.25, -4.5, 9.0)
    first = replace(_settings(), shared_pivot_world=pivot)
    second = replace(_settings(), shared_pivot_world=pivot)

    resolved = _resolve_composition_shared_pivot_world(
        (first, second),
        _multi_settings(shared_pivot_enabled=True),
    )

    assert resolved == pivot


def test_composition_rejects_missing_shared_pivot_when_enabled():
    pivot = (1.25, -4.5, 9.0)

    with pytest.raises(ValueError, match="missing shared_pivot_world"):
        _resolve_composition_shared_pivot_world(
            (replace(_settings(), shared_pivot_world=pivot), _settings()),
            _multi_settings(shared_pivot_enabled=True),
        )


def test_composition_rejects_different_shared_pivots_when_enabled():
    with pytest.raises(ValueError, match="one identical transaction pivot"):
        _resolve_composition_shared_pivot_world(
            (
                replace(_settings(), shared_pivot_world=(1.0, 2.0, 3.0)),
                replace(_settings(), shared_pivot_world=(1.0, 2.0, 4.0)),
            ),
            _multi_settings(shared_pivot_enabled=True),
        )


def test_composition_rejects_unexpected_shared_pivot_when_disabled():
    with pytest.raises(ValueError, match="Shared Pivot is disabled"):
        _resolve_composition_shared_pivot_world(
            (
                replace(_settings(), shared_pivot_world=(1.0, 2.0, 3.0)),
                _settings(),
            ),
            _multi_settings(shared_pivot_enabled=False),
        )


def test_composition_legacy_path_requires_no_shared_pivot():
    resolved = _resolve_composition_shared_pivot_world(
        (_settings(), _settings()),
        _multi_settings(shared_pivot_enabled=False),
    )

    assert resolved is None


def test_composition_expected_settings_add_only_transaction_shared_pivot():
    source_settings = _settings()
    source = _source(source_settings)
    pivot = (1.25, -4.5, 9.0)

    expected = _expected_prepared_settings(
        source,
        A1MultiObjectMode.STANDALONE,
        pivot,
    )

    assert expected is not source_settings
    assert expected.shared_pivot_world == pivot
    assert replace(expected, shared_pivot_world=None) == source_settings
    assert expected.export is source_settings.export
    assert expected.rig_setup_pose_mode is source_settings.rig_setup_pose_mode
    assert expected.projection_direction is source_settings.projection_direction


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
