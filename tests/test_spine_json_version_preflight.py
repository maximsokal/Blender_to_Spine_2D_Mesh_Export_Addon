"""Fail-closed preflight contracts for target codecs that are not production ready."""

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
from Blender_to_Spine2D_Mesh_Exporter.domain.spine.version_target import (
    SpineJsonTarget,
    SpineJsonTargetUnavailableError,
)


def _settings(root: Path, target: SpineJsonTarget) -> A1SingleObjectExportSettings:
    return A1SingleObjectExportSettings(
        export=ExportSettings(
            texture_width=64,
            texture_height=64,
            output_directory=root,
            spine_version=target.exact_version,
        )
    )


@pytest.mark.parametrize(
    "target",
    tuple(
        target
        for target in SpineJsonTarget
        if target is not SpineJsonTarget.SPINE_4_2
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
