from pathlib import Path

import pytest

from Blender_to_Spine2D_Mesh_Exporter.application import (
    ExportIssue,
    ExportRequest,
    ExportResult,
    ExportSettings,
)
from Blender_to_Spine2D_Mesh_Exporter.application.contracts import IssueSeverity
from Blender_to_Spine2D_Mesh_Exporter.domain.spine import A1RigProfile


def test_export_request_is_immutable_and_validated(tmp_path: Path):
    settings = ExportSettings(
        texture_width=1024,
        texture_height=1024,
        output_directory=tmp_path,
    )
    request = ExportRequest(("Cube",), "Cube", settings)

    assert request.settings.rig_profile == A1RigProfile.THREE_AXIS_ROTATION.value
    with pytest.raises(ValueError, match="active_object_id"):
        ExportRequest(("Cube",), "Other", settings)


def test_export_settings_accept_only_supported_rig_profiles(tmp_path: Path):
    selected = ExportSettings(
        texture_width=512,
        texture_height=512,
        output_directory=tmp_path,
        rig_profile=A1RigProfile.TWO_AXIS_ROTATION_SCALE.value,
    )
    assert selected.rig_profile == A1RigProfile.TWO_AXIS_ROTATION_SCALE.value

    with pytest.raises(ValueError, match="Unsupported rig profile"):
        ExportSettings(
            texture_width=512,
            texture_height=512,
            output_directory=tmp_path,
            rig_profile="UNKNOWN_RIG",
        )


def test_failed_result_requires_error_issue():
    with pytest.raises(ValueError, match="must contain at least one ERROR"):
        ExportResult(success=False)

    result = ExportResult(
        success=False,
        issues=(
            ExportIssue(
                severity=IssueSeverity.ERROR,
                stage="validation",
                code="INVALID_MESH",
                message="Mesh is invalid",
                object_id="Cube",
            ),
        ),
    )
    assert result.success is False
