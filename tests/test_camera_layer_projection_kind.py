"""Typed routing tests for Active Camera Perspective and Orthographic layers."""

from __future__ import annotations

import pytest

from Blender_to_Spine2D_Mesh_Exporter.blender_adapter.a1_document_preparation import (
    _active_camera_layer_kind,
)
from Blender_to_Spine2D_Mesh_Exporter.domain.camera_projection import (
    A1CameraProjectionKind,
)
from Blender_to_Spine2D_Mesh_Exporter.domain.spine import (
    A1CameraLayerProjectionKind,
)


def test_typed_perspective_routes_to_perspective_layer() -> None:
    assert _active_camera_layer_kind(
        A1CameraProjectionKind.PERSPECTIVE
    ) is A1CameraLayerProjectionKind.PERSPECTIVE


def test_typed_orthographic_routes_to_orthographic_layer() -> None:
    assert _active_camera_layer_kind(
        A1CameraProjectionKind.ORTHOGRAPHIC
    ) is A1CameraLayerProjectionKind.ORTHOGRAPHIC


def test_missing_typed_camera_kind_fails_closed() -> None:
    with pytest.raises(ValueError, match="did not provide camera_projection_kind"):
        _active_camera_layer_kind(None)


def test_untyped_camera_kind_fails_closed() -> None:
    with pytest.raises(TypeError, match="must be A1CameraProjectionKind"):
        _active_camera_layer_kind("PERSPECTIVE")  # type: ignore[arg-type]
