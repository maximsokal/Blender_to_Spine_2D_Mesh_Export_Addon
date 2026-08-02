"""Typed routing tests for Active Camera Perspective and Orthographic layers."""

from __future__ import annotations

import pytest

from Blender_to_Spine2D_Mesh_Exporter.blender_adapter.a1_document_preparation import (
    _active_camera_layer_kind,
)
from Blender_to_Spine2D_Mesh_Exporter.domain.spine import (
    A1CameraLayerProjectionKind,
)


def test_active_camera_statistics_route_perspective_layer() -> None:
    assert _active_camera_layer_kind(
        {"active_camera_type": "PERSPECTIVE"}
    ) is A1CameraLayerProjectionKind.PERSPECTIVE


def test_active_camera_statistics_route_orthographic_layer() -> None:
    assert _active_camera_layer_kind(
        {"active_camera_type": "ORTHOGRAPHIC"}
    ) is A1CameraLayerProjectionKind.ORTHOGRAPHIC


def test_active_camera_statistics_reject_missing_kind() -> None:
    with pytest.raises(ValueError, match="did not provide active_camera_type"):
        _active_camera_layer_kind({})


def test_active_camera_statistics_reject_unknown_kind() -> None:
    with pytest.raises(ValueError, match="Unsupported Active Camera type"):
        _active_camera_layer_kind({"active_camera_type": "PANORAMIC"})
