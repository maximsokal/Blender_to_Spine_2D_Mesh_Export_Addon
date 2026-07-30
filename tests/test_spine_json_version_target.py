from pathlib import Path

import pytest

from Blender_to_Spine2D_Mesh_Exporter.application import ExportSettings
from Blender_to_Spine2D_Mesh_Exporter.domain.spine.version_target import (
    DEFAULT_SPINE_JSON_TARGET,
    DEFAULT_SPINE_JSON_VERSION,
    SpineJsonTarget,
    SpineJsonTargetUnavailableError,
    require_spine_json_target_serializable,
    resolve_spine_json_exact_version,
    resolve_spine_json_target,
    spine_json_target_enum_items,
)


EXPECTED_TARGETS = (
    (SpineJsonTarget.SPINE_3_8, "3.8", "3.8.99"),
    (SpineJsonTarget.SPINE_4_0, "4.0", "4.0.64"),
    (SpineJsonTarget.SPINE_4_1, "4.1", "4.1.24"),
    (SpineJsonTarget.SPINE_4_2, "4.2", "4.2.43"),
    (SpineJsonTarget.SPINE_4_3, "4.3", "4.3.23"),
)



def test_registry_order_and_exact_versions_are_stable() -> None:
    assert tuple(
        (target, target.family, target.exact_version)
        for target in SpineJsonTarget
    ) == EXPECTED_TARGETS
    assert DEFAULT_SPINE_JSON_TARGET is SpineJsonTarget.SPINE_4_2
    assert DEFAULT_SPINE_JSON_VERSION == "4.2.43"



def test_blender_enum_items_are_derived_from_the_registry() -> None:
    assert spine_json_target_enum_items() == tuple(
        (target.value, target.label, target.description)
        for target, _family, _exact in EXPECTED_TARGETS
    )


@pytest.mark.parametrize("target,family,exact", EXPECTED_TARGETS)
def test_ui_resolver_accepts_identifier_family_and_exact_version(
    target: SpineJsonTarget,
    family: str,
    exact: str,
) -> None:
    assert resolve_spine_json_target(target) is target
    assert resolve_spine_json_target(target.value) is target
    assert resolve_spine_json_target(family) is target
    assert resolve_spine_json_target(exact) is target
    assert resolve_spine_json_exact_version(exact) is target


@pytest.mark.parametrize("value", ("", "latest", "4.4", "4.2-custom", object()))
def test_unknown_targets_fail_closed(value: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        resolve_spine_json_target(value)


@pytest.mark.parametrize("value", ("3.8", "4.0", "4.1", "4.2", "4.3", "latest"))
def test_application_exact_version_resolver_rejects_family_only_strings(
    value: str,
) -> None:
    with pytest.raises(ValueError):
        resolve_spine_json_exact_version(value)



def test_export_settings_default_to_exact_spine_four_two(tmp_path: Path) -> None:
    settings = ExportSettings(
        texture_width=128,
        texture_height=128,
        output_directory=tmp_path,
    )

    assert settings.spine_version == "4.2.43"
    assert settings.spine_target is SpineJsonTarget.SPINE_4_2


@pytest.mark.parametrize("value", ("4.2", "4.4.0", "latest", "4.2-custom"))
def test_export_settings_reject_unregistered_exact_versions(
    tmp_path: Path,
    value: str,
) -> None:
    with pytest.raises(ValueError):
        ExportSettings(
            texture_width=128,
            texture_height=128,
            output_directory=tmp_path,
            spine_version=value,
        )



def test_spine_four_one_and_four_two_are_production_serializable() -> None:
    ready = {
        SpineJsonTarget.SPINE_4_1,
        SpineJsonTarget.SPINE_4_2,
    }

    for target in SpineJsonTarget:
        if target in ready:
            assert require_spine_json_target_serializable(target) is target
            continue

        with pytest.raises(SpineJsonTargetUnavailableError):
            require_spine_json_target_serializable(target)



def test_spine_four_one_is_selectable_with_limited_scope_description() -> None:
    target = resolve_spine_json_target("4.1.24")

    assert target is SpineJsonTarget.SPINE_4_1
    assert target.descriptor.serializer_ready is True
    assert "limited" in target.description.lower()
    assert "standalone" in target.description.lower()
    assert "connected" in target.description.lower()
    assert require_spine_json_target_serializable(target) is target
