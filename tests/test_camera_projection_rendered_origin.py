"""Coordinate-system regression for rendered Camera Projection Object Origin."""

from __future__ import annotations

from types import SimpleNamespace

from Blender_to_Spine2D_Mesh_Exporter.blender_adapter import (
    a1_projection_finalization as finalization,
)
from Blender_to_Spine2D_Mesh_Exporter.domain.camera_projection import (
    A1CameraProjectionKind,
)
from Blender_to_Spine2D_Mesh_Exporter.domain.projection import A1ProjectedPoint
from Blender_to_Spine2D_Mesh_Exporter.domain.spine import (
    A1CameraLayerProjectionKind,
)


class _Frame:
    def __init__(self, kind: A1CameraProjectionKind) -> None:
        self.kind = kind
        self.calls: list[tuple[object, str]] = []

    def project_world_point(
        self,
        point: object,
        *,
        field_name: str,
    ) -> A1ProjectedPoint:
        self.calls.append((point, field_name))
        return A1ProjectedPoint(u=12.5, v=-7.25, depth=-8.0)


def test_rendered_origin_maps_image_y_and_perspective_kind(
    monkeypatch,
) -> None:
    scene = object()
    depsgraph = object()
    source_object = object()
    frame = _Frame(A1CameraProjectionKind.PERSPECTIVE)
    prepared = SimpleNamespace(source_object=source_object)
    plan = SimpleNamespace(settings=SimpleNamespace(width=128, height=96))

    monkeypatch.setattr(
        finalization,
        "_resolved_scene_and_depsgraph",
        lambda context, resolved_scene: (scene, depsgraph),
    )
    monkeypatch.setattr(
        finalization,
        "resolve_a1_active_camera_projection_frame",
        lambda resolved_scene, **kwargs: frame,
    )
    monkeypatch.setattr(
        finalization,
        "_evaluated_object_origin",
        lambda obj, resolved_scene, graph: (1.0, 2.0, 3.0),
    )

    main_position, depth, layer_kind = finalization._rendered_camera_main_position(
        prepared,
        plan,
        context=object(),
        scene=scene,
    )

    assert main_position == (12.5, 7.25)
    assert depth == -8.0
    assert layer_kind is A1CameraLayerProjectionKind.PERSPECTIVE
    assert frame.calls == [
        (
            (1.0, 2.0, 3.0),
            "rendered_camera_projection_object_origin",
        )
    ]


def test_rendered_origin_maps_orthographic_kind(monkeypatch) -> None:
    scene = object()
    depsgraph = object()
    frame = _Frame(A1CameraProjectionKind.ORTHOGRAPHIC)
    prepared = SimpleNamespace(source_object=object())
    plan = SimpleNamespace(settings=SimpleNamespace(width=64, height=64))

    monkeypatch.setattr(
        finalization,
        "_resolved_scene_and_depsgraph",
        lambda context, resolved_scene: (scene, depsgraph),
    )
    monkeypatch.setattr(
        finalization,
        "resolve_a1_active_camera_projection_frame",
        lambda resolved_scene, **kwargs: frame,
    )
    monkeypatch.setattr(
        finalization,
        "_evaluated_object_origin",
        lambda obj, resolved_scene, graph: (0.0, 0.0, -5.0),
    )

    _main_position, _depth, layer_kind = (
        finalization._rendered_camera_main_position(
            prepared,
            plan,
            context=object(),
            scene=scene,
        )
    )

    assert layer_kind is A1CameraLayerProjectionKind.ORTHOGRAPHIC
