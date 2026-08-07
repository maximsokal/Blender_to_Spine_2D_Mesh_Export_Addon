"""Typed routing tests for Active Camera Perspective and Orthographic layers."""

from __future__ import annotations

import pytest

from Blender_to_Spine2D_Mesh_Exporter.blender_adapter.a1_document_preparation import (
    _active_camera_projection_kind,
    _camera_layer_projection_kind,
)
from Blender_to_Spine2D_Mesh_Exporter.domain.camera_projection import (
    A1CameraProjectionKind,
)
from Blender_to_Spine2D_Mesh_Exporter.domain.spine import (
    A1CameraLayerProjectionKind,
)


def test_active_camera_perspective_kind_is_preserved() -> None:
    """Validation must preserve the typed Perspective camera model."""

    assert _active_camera_projection_kind(
        A1CameraProjectionKind.PERSPECTIVE
    ) is A1CameraProjectionKind.PERSPECTIVE


def test_active_camera_orthographic_kind_is_preserved() -> None:
    """Validation must preserve the typed Orthographic camera model."""

    assert _active_camera_projection_kind(
        A1CameraProjectionKind.ORTHOGRAPHIC
    ) is A1CameraProjectionKind.ORTHOGRAPHIC


def test_missing_active_camera_kind_fails_closed() -> None:
    """Active Camera preparation cannot continue without evaluated camera kind."""

    with pytest.raises(
        ValueError,
        match="did not provide camera_projection_kind",
    ):
        _active_camera_projection_kind(None)


def test_untyped_active_camera_kind_fails_closed() -> None:
    """Stringly typed camera models must never enter document preparation."""

    with pytest.raises(
        TypeError,
        match="value must be A1CameraProjectionKind or None",
    ):
        _active_camera_projection_kind("PERSPECTIVE")  # type: ignore[arg-type]


def test_typed_perspective_routes_to_perspective_layer() -> None:
    """A validated Perspective camera maps to Perspective rigid-layer semantics."""

    camera_kind = _active_camera_projection_kind(A1CameraProjectionKind.PERSPECTIVE)

    assert _camera_layer_projection_kind(
        camera_kind
    ) is A1CameraLayerProjectionKind.PERSPECTIVE


def test_typed_orthographic_routes_to_orthographic_layer() -> None:
    """A validated Orthographic camera maps to Orthographic rigid-layer semantics."""

    camera_kind = _active_camera_projection_kind(A1CameraProjectionKind.ORTHOGRAPHIC)

    assert _camera_layer_projection_kind(
        camera_kind
    ) is A1CameraLayerProjectionKind.ORTHOGRAPHIC


def test_missing_camera_layer_kind_fails_closed() -> None:
    """The rigid-layer converter accepts only an already validated camera kind."""

    with pytest.raises(
        TypeError,
        match="value must be A1CameraProjectionKind",
    ):
        _camera_layer_projection_kind(None)  # type: ignore[arg-type]


def test_untyped_camera_layer_kind_fails_closed() -> None:
    """The rigid-layer converter rejects untyped camera-model values."""

    with pytest.raises(
        TypeError,
        match="value must be A1CameraProjectionKind",
    ):
        _camera_layer_projection_kind("PERSPECTIVE")  # type: ignore[arg-type]
