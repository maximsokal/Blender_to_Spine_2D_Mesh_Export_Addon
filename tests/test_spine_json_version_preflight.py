"""Fail-closed preflight contracts for target codec and rig capability readiness."""

from __future__ import annotations

from pathlib import Path

import pytest

from Blender_to_Spine2D_Mesh_Exporter.application import (
    A1SingleObjectExportSettings,
    A1SingleObjectStage,
    ExportSettings,
)
from Blender_to_Spine2D_Mesh_Exporter.blender_adapter import a1_object_preparation
from Blender_to_Spine2D_Mesh_Exporter.blender_adapter.a1_preparation_contracts import (
    A1ObjectPreparationError,
)
from Blender_to_Spine2D_Mesh_Exporter.domain.spine.export_capabilities import (
    SpineJsonExportCapabilityError,
)
from Blender_to_Spine2D_Mesh_Exporter.domain.spine.rig_profiles import A1RigProfile
from Blender_to_Spine2D_Mesh_Exporter.domain.spine.version_target import (
    SpineJsonTarget,
    SpineJsonTargetUnavailableError,
)


def _settings(
    root: Path,
    target: SpineJsonTarget,
    profile: A1RigProfile = A1RigProfile.THREE_AXIS_ROTATION,
) -> A1SingleObjectExportSettings:
    return A1SingleObjectExportSettings(
        export=ExportSettings(
            texture_width=64,
            texture_height=64,
            output_directory=root,
            spine_version=target.exact_version,
            rig_profile=profile.value,
        )
    )


@pytest.mark.parametrize(
    "target",
    tuple(
        target
        for target in SpineJsonTarget
        if not target.descriptor.serializer_ready
    ),
)
def test_unready_target_is_rejected_before_geometry(
    tmp_path: Path,
    monkeypatch,
    target: SpineJsonTarget,
) -> None:
    geometry_called = False

    def unexpected_geometry(*_args, **_kwargs):
        nonlocal geometry_called
        geometry_called = True
        raise AssertionError("geometry must not run for an unready target codec")

    monkeypatch.setattr(
        a1_object_preparation,
        "prepare_a1_source_geometry",
        unexpected_geometry,
    )

    with pytest.raises(A1ObjectPreparationError) as exc_info:
        a1_object_preparation.prepare_a1_object(
            object(),
            _settings(tmp_path, target),
        )

    error = exc_info.value
    assert error.stage is A1SingleObjectStage.VALIDATE_REQUEST
    assert isinstance(error.cause, SpineJsonTargetUnavailableError)
    assert geometry_called is False


@pytest.mark.parametrize(
    "target,profile",
    (
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
    ),
)
def test_supported_target_rig_pair_reaches_geometry_preparation(
    tmp_path: Path,
    monkeypatch,
    target: SpineJsonTarget,
    profile: A1RigProfile,
) -> None:
    sentinel = RuntimeError(f"geometry reached for {target.value}/{profile.value}")

    def stop_at_geometry(*_args, **_kwargs):
        raise sentinel

    monkeypatch.setattr(
        a1_object_preparation,
        "prepare_a1_source_geometry",
        stop_at_geometry,
    )

    with pytest.raises(A1ObjectPreparationError) as exc_info:
        a1_object_preparation.prepare_a1_object(
            object(),
            _settings(tmp_path, target, profile),
        )

    error = exc_info.value
    assert error.stage is A1SingleObjectStage.VALIDATE_REQUEST
    assert error.cause is sentinel


@pytest.mark.parametrize(
    "target",
    (SpineJsonTarget.SPINE_4_0, SpineJsonTarget.SPINE_4_1),
)
def test_legacy_four_x_three_axis_is_rejected_before_geometry(
    tmp_path: Path,
    monkeypatch,
    target: SpineJsonTarget,
) -> None:
    geometry_called = False

    def unexpected_geometry(*_args, **_kwargs):
        nonlocal geometry_called
        geometry_called = True
        raise AssertionError("geometry must not run for an unsupported rig profile")

    monkeypatch.setattr(
        a1_object_preparation,
        "prepare_a1_source_geometry",
        unexpected_geometry,
    )

    with pytest.raises(A1ObjectPreparationError) as exc_info:
        a1_object_preparation.prepare_a1_object(
            object(),
            _settings(
                tmp_path,
                target,
                A1RigProfile.THREE_AXIS_ROTATION,
            ),
        )

    error = exc_info.value
    assert error.stage is A1SingleObjectStage.VALIDATE_REQUEST
    assert isinstance(error.cause, SpineJsonExportCapabilityError)
    assert "3-Axis Rotation" in str(error.cause)
    assert geometry_called is False
