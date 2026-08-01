"""Validate Active Camera layout for Normal / UV Segments in Blender 5.2.

The worker exercises production evaluated-geometry preparation and standalone composition
for Perspective and Orthographic cameras. Expected screen points come from Blender's
independent ``world_to_camera_view`` helper configured temporarily to the export-texture
canvas. Expected camera depth is calculated from captured Blender matrix and vertex tuples
using Python affine arithmetic, matching the numeric input model used by production while
remaining independent from the production projection implementation. The production Scene
render dimensions remain intentionally different.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import dataclass
import json
from math import isfinite
from pathlib import Path
import sys
import traceback
from typing import Iterator

import bpy
from bpy_extras.object_utils import world_to_camera_view
from mathutils import Vector


SCRIPT_DIRECTORY = Path(__file__).resolve().parent
REPOSITORY_ROOT = SCRIPT_DIRECTORY.parents[1]
for path in (SCRIPT_DIRECTORY, REPOSITORY_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from Blender_to_Spine2D_Mesh_Exporter.application import (  # noqa: E402
    A1MultiObjectExportSettings,
    A1MultiObjectMode,
    A1SingleObjectExportSettings,
    A1SourceGeometryMode,
    ExportSettings,
)
from Blender_to_Spine2D_Mesh_Exporter.blender_adapter import (  # noqa: E402
    A1MultiObjectSource,
    prepare_a1_multi_object,
)
from Blender_to_Spine2D_Mesh_Exporter.blender_adapter.a1_multi_object_composition import (  # noqa: E402
    compose_a1_multi_object_document,
)
from Blender_to_Spine2D_Mesh_Exporter.domain.baking import (  # noqa: E402
    A1TextureExportMode,
    BakeExecutionSettings,
    BakeMode,
)
from Blender_to_Spine2D_Mesh_Exporter.domain.geometry import (  # noqa: E402
    calculate_a1_projected_snapshot_depth_range,
)
from Blender_to_Spine2D_Mesh_Exporter.domain.projection import (  # noqa: E402
    A1ProjectionDirection,
)
from Blender_to_Spine2D_Mesh_Exporter.domain.spine.rig_profiles import (  # noqa: E402
    A1RigProfile,
    A1RigSetupPoseMode,
)
from Blender_to_Spine2D_Mesh_Exporter.domain.spine.version_target import (  # noqa: E402
    SpineJsonTarget,
)
from Blender_to_Spine2D_Mesh_Exporter.domain.uv import UvUnwrapSettings  # noqa: E402
from run_bake_integration import (  # noqa: E402
    _activate_only,
    _assert,
    _clear_scene,
    _configure_cycles_scene,
    _create_emission_material,
)


_TEXTURE_WIDTH = 160
_TEXTURE_HEIGHT = 96
_SCENE_RENDER_WIDTH = 901
_SCENE_RENDER_HEIGHT = 577
_MATRIX_TOLERANCE = 1.0e-7
_SCREEN_TOLERANCE = 1.0e-8
_MAIN_BONE_TOLERANCE = 0.011
_DEPTH_TOLERANCE = 1.0e-8
_DEPTH_EXPECTATION_MODEL = "CAPTURED_TUPLE_AFFINE"


@dataclass(frozen=True)
class _ObjectSpecification:
    name: str
    component_id: str
    location: tuple[float, float, float]
    local_minimum: tuple[float, float, float]
    local_maximum: tuple[float, float, float]
    rotation: tuple[float, float, float]
    scale: tuple[float, float, float]


@dataclass(frozen=True)
class _SourceState:
    vertices: tuple[tuple[float, float, float], ...]
    matrix_world: tuple[float, ...]
    location: tuple[float, float, float]
    rotation_euler: tuple[float, float, float]
    scale: tuple[float, float, float]
    parent_name: str | None


@dataclass(frozen=True)
class _ExpectedVertex:
    index: int
    screen_x: float
    screen_y: float
    camera_z: float


_SPECIFICATIONS = (
    # Origin is farthest, but the geometry extends toward the camera and becomes nearest.
    _ObjectSpecification(
        name="CameraAlpha",
        component_id="component_alpha",
        location=(-1.3, 0.6, -8.0),
        local_minimum=(-0.7, -0.6, 4.0),
        local_maximum=(0.9, 0.8, 5.0),
        rotation=(0.08, -0.05, 0.04),
        scale=(1.1, 0.9, 1.0),
    ),
    _ObjectSpecification(
        name="CameraBeta",
        component_id="component_beta",
        location=(0.8, -0.9, -5.0),
        local_minimum=(-0.8, -0.5, 0.0),
        local_maximum=(0.7, 0.9, 0.5),
        rotation=(-0.03, 0.06, -0.08),
        scale=(0.85, 1.15, 1.0),
    ),
    # Origin is nearest, while all geometry lies well behind it and outside the frame.
    _ObjectSpecification(
        name="CameraGamma",
        component_id="component_gamma",
        location=(10.0, 1.2, -4.0),
        local_minimum=(-0.6, -0.7, -4.0),
        local_maximum=(0.8, 0.6, -3.0),
        rotation=(0.05, 0.02, 0.07),
        scale=(1.0, 0.95, 1.1),
    ),
)


def _parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    arguments = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else ()
    return parser.parse_args(arguments)


def _prepare_output_directory(value: Path) -> Path:
    if not isinstance(value, Path):
        raise TypeError("output must be pathlib.Path")
    resolved = value.expanduser().resolve(strict=False)
    if resolved.exists() and not resolved.is_dir():
        raise ValueError(f"Output path is not a directory: {resolved}")
    if resolved.exists() and any(resolved.iterdir()):
        raise ValueError(f"Output directory must be empty: {resolved}")
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def _matrix_tuple(matrix: object) -> tuple[float, ...]:
    values = tuple(
        float(matrix[row][column])
        for row in range(4)
        for column in range(4)
    )
    if len(values) != 16 or not all(isfinite(value) for value in values):
        raise ValueError("matrix must contain sixteen finite values")
    return values


def _affine_transform_point(
    matrix: tuple[float, ...],
    point: tuple[float, float, float],
    *,
    field_name: str,
) -> tuple[float, float, float]:
    """Transform one captured point using Python float affine arithmetic."""

    if not isinstance(matrix, tuple) or len(matrix) != 16:
        raise TypeError(f"{field_name} matrix must be a 16-value tuple")
    if not isinstance(point, tuple) or len(point) != 3:
        raise TypeError(f"{field_name} point must be a 3-value tuple")
    values = tuple(float(value) for value in matrix)
    coordinates = tuple(float(value) for value in point)
    if not all(isfinite(value) for value in values + coordinates):
        raise ValueError(f"{field_name} contains non-finite values")
    x, y, z = coordinates
    transformed = (
        values[0] * x + values[1] * y + values[2] * z + values[3],
        values[4] * x + values[5] * y + values[6] * z + values[7],
        values[8] * x + values[9] * y + values[10] * z + values[11],
    )
    if not all(isfinite(value) for value in transformed):
        raise ValueError(f"{field_name} transform produced non-finite values")
    return transformed


def _matrix_translation(matrix: tuple[float, ...]) -> tuple[float, float, float]:
    if not isinstance(matrix, tuple) or len(matrix) != 16:
        raise TypeError("matrix must be a 16-value tuple")
    translation = (float(matrix[3]), float(matrix[7]), float(matrix[11]))
    if not all(isfinite(value) for value in translation):
        raise ValueError("matrix translation contains non-finite values")
    return translation


def _rotation_only_camera_view_matrix(
    camera: bpy.types.Object,
) -> tuple[float, ...]:
    """Capture the same scale-independent Blender camera frame as production."""

    matrix_world = getattr(camera, "matrix_world", None)
    if matrix_world is None:
        raise ValueError("camera.matrix_world is missing")
    location, rotation, _scale = matrix_world.decompose()
    rotation.normalize()
    camera_world = rotation.to_matrix().to_4x4()
    camera_world.translation = location
    return _matrix_tuple(camera_world.inverted())


def _cuboid_vertices(
    minimum: tuple[float, float, float],
    maximum: tuple[float, float, float],
) -> tuple[tuple[float, float, float], ...]:
    min_x, min_y, min_z = minimum
    max_x, max_y, max_z = maximum
    if not (min_x < max_x and min_y < max_y and min_z < max_z):
        raise ValueError("Cuboid minimum must be strictly below maximum")
    return (
        (min_x, min_y, min_z),
        (max_x, min_y, min_z),
        (max_x, max_y, min_z),
        (min_x, max_y, min_z),
        (min_x, min_y, max_z),
        (max_x, min_y, max_z),
        (max_x, max_y, max_z),
        (min_x, max_y, max_z),
    )


def _create_cuboid(specification: _ObjectSpecification) -> bpy.types.Object:
    mesh = bpy.data.meshes.new(f"{specification.name}Mesh")
    mesh.from_pydata(
        _cuboid_vertices(
            specification.local_minimum,
            specification.local_maximum,
        ),
        (),
        (
            (0, 3, 2, 1),
            (4, 5, 6, 7),
            (0, 1, 5, 4),
            (1, 2, 6, 5),
            (2, 3, 7, 6),
            (3, 0, 4, 7),
        ),
    )
    mesh.update(calc_edges=True)
    source_object = bpy.data.objects.new(specification.name, mesh)
    bpy.context.scene.collection.objects.link(source_object)
    source_object.location = specification.location
    source_object.rotation_euler = specification.rotation
    source_object.scale = specification.scale
    _create_emission_material(source_object)
    return source_object


def _create_camera(kind: str) -> bpy.types.Object:
    if kind not in {"PERSP", "ORTHO"}:
        raise ValueError("kind must be PERSP or ORTHO")
    data = bpy.data.cameras.new(f"Active{kind}CameraData")
    data.type = kind
    data.clip_start = 0.1
    data.clip_end = 100.0
    data.shift_x = 0.07
    data.shift_y = -0.04
    if kind == "PERSP":
        data.lens = 45.0
        data.sensor_width = 36.0
    else:
        data.ortho_scale = 6.0
    camera = bpy.data.objects.new(f"Active{kind}Camera", data)
    bpy.context.scene.collection.objects.link(camera)
    camera.location = (0.0, 0.0, 0.0)
    camera.rotation_euler = (0.0, 0.0, 0.0)
    bpy.context.scene.camera = camera
    return camera


def _capture_state(source_object: bpy.types.Object) -> _SourceState:
    bpy.context.view_layer.update()
    return _SourceState(
        vertices=tuple(
            tuple(float(value) for value in vertex.co)
            for vertex in source_object.data.vertices
        ),
        matrix_world=_matrix_tuple(source_object.matrix_world),
        location=tuple(float(value) for value in source_object.location),
        rotation_euler=tuple(float(value) for value in source_object.rotation_euler),
        scale=tuple(float(value) for value in source_object.scale),
        parent_name=(
            None
            if source_object.parent is None
            else str(source_object.parent.name_full or source_object.parent.name)
        ),
    )


def _assert_state_unchanged(
    source_object: bpy.types.Object,
    before: _SourceState,
    *,
    label: str,
) -> float:
    after = _capture_state(source_object)
    _assert(after.vertices == before.vertices, f"{label} mutated source vertices")
    _assert(after.location == before.location, f"{label} mutated Object.location")
    _assert(
        after.rotation_euler == before.rotation_euler,
        f"{label} mutated Object.rotation_euler",
    )
    _assert(after.scale == before.scale, f"{label} mutated Object.scale")
    _assert(after.parent_name == before.parent_name, f"{label} mutated Object.parent")
    maximum_delta = max(
        (
            abs(actual - expected)
            for actual, expected in zip(
                after.matrix_world,
                before.matrix_world,
                strict=True,
            )
        ),
        default=0.0,
    )
    _assert(
        maximum_delta <= _MATRIX_TOLERANCE,
        f"{label} mutated matrix_world: max_delta={maximum_delta}",
    )
    return maximum_delta


@contextmanager
def _temporary_export_render_dimensions(
    scene: bpy.types.Scene,
) -> Iterator[None]:
    before = (
        int(scene.render.resolution_x),
        int(scene.render.resolution_y),
        int(scene.render.resolution_percentage),
        float(scene.render.pixel_aspect_x),
        float(scene.render.pixel_aspect_y),
    )
    scene.render.resolution_x = _TEXTURE_WIDTH
    scene.render.resolution_y = _TEXTURE_HEIGHT
    scene.render.resolution_percentage = 100
    scene.render.pixel_aspect_x = 1.0
    scene.render.pixel_aspect_y = 1.0
    try:
        yield
    finally:
        (
            scene.render.resolution_x,
            scene.render.resolution_y,
            scene.render.resolution_percentage,
            scene.render.pixel_aspect_x,
            scene.render.pixel_aspect_y,
        ) = before


def _expected_screen_point(
    scene: bpy.types.Scene,
    camera: bpy.types.Object,
    world_point: tuple[float, float, float],
) -> tuple[float, float]:
    with _temporary_export_render_dimensions(scene):
        normalized = world_to_camera_view(scene, camera, Vector(world_point))
    return (
        (float(normalized.x) - 0.5) * float(_TEXTURE_WIDTH),
        (float(normalized.y) - 0.5) * float(_TEXTURE_HEIGHT),
    )


def _camera_z(
    camera_view_matrix: tuple[float, ...],
    world_point: tuple[float, float, float],
) -> float:
    return _affine_transform_point(
        camera_view_matrix,
        world_point,
        field_name="camera depth",
    )[2]


def _expected_vertices(
    scene: bpy.types.Scene,
    camera: bpy.types.Object,
    source_state: _SourceState,
    camera_view_matrix: tuple[float, ...],
) -> tuple[_ExpectedVertex, ...]:
    expected: list[_ExpectedVertex] = []
    for index, local_position in enumerate(source_state.vertices):
        world_point = _affine_transform_point(
            source_state.matrix_world,
            local_position,
            field_name=f"source vertex {index}",
        )
        screen_x, screen_y = _expected_screen_point(
            scene,
            camera,
            world_point,
        )
        expected.append(
            _ExpectedVertex(
                index=index,
                screen_x=screen_x,
                screen_y=screen_y,
                camera_z=_camera_z(camera_view_matrix, world_point),
            )
        )
    return tuple(expected)


def _settings(
    output_directory: Path,
    specification: _ObjectSpecification,
    camera_kind: str,
) -> A1SingleObjectExportSettings:
    return A1SingleObjectExportSettings(
        export=ExportSettings(
            texture_width=_TEXTURE_WIDTH,
            texture_height=_TEXTURE_HEIGHT,
            output_directory=output_directory,
            images_relative_path="images",
            spine_version=SpineJsonTarget.SPINE_4_2.exact_version,
            rig_profile=A1RigProfile.TWO_AXIS_ROTATION_SCALE.value,
            bake_margin=1,
        ),
        prefix=specification.name,
        output_stem=f"{specification.name}_{camera_kind.lower()}",
        source_geometry_mode=A1SourceGeometryMode.EVALUATED,
        uv=UvUnwrapSettings(layer_name="SpineBakeUV"),
        diffuse_mode=BakeMode.EMIT,
        procedural_mode=BakeMode.EMIT,
        bake_execution=BakeExecutionSettings(
            samples=1,
            texture_export_mode=A1TextureExportMode.NORMAL_UV_SEGMENTS,
        ),
        rig_setup_pose_mode=A1RigSetupPoseMode.PRESERVE_COMPOSITION,
        projection_direction=A1ProjectionDirection.ACTIVE_CAMERA,
    )


def _collapsed_owner_order(
    slot_names: tuple[str, ...],
    owner_by_slot: dict[str, str],
) -> tuple[str, ...]:
    owners: list[str] = []
    for slot_name in slot_names:
        owner = owner_by_slot.get(slot_name)
        if owner is None:
            raise AssertionError(f"Composed slot has no object owner: {slot_name}")
        if not owners or owners[-1] != owner:
            owners.append(owner)
    return tuple(owners)


def _run_camera_kind(
    output_root: Path,
    camera_kind: str,
) -> dict[str, object]:
    _clear_scene()
    _configure_cycles_scene()
    scene = bpy.context.scene
    scene.render.resolution_x = _SCENE_RENDER_WIDTH
    scene.render.resolution_y = _SCENE_RENDER_HEIGHT
    scene.render.resolution_percentage = 63
    scene.render.pixel_aspect_x = 1.25
    scene.render.pixel_aspect_y = 0.8
    camera = _create_camera(camera_kind)
    bpy.context.view_layer.update()
    camera_matrix_before = _matrix_tuple(camera.matrix_world)
    camera_view_matrix = _rotation_only_camera_view_matrix(camera)
    camera_data_before = (
        camera.data.type,
        float(camera.data.clip_start),
        float(camera.data.clip_end),
        float(camera.data.shift_x),
        float(camera.data.shift_y),
        float(camera.data.lens),
        float(camera.data.ortho_scale),
    )
    scene_render_before = (
        int(scene.render.resolution_x),
        int(scene.render.resolution_y),
        int(scene.render.resolution_percentage),
        float(scene.render.pixel_aspect_x),
        float(scene.render.pixel_aspect_y),
    )

    camera_root = output_root / camera_kind.lower()
    camera_root.mkdir(parents=True, exist_ok=True)
    sources: list[A1MultiObjectSource] = []
    state_by_component: dict[str, tuple[bpy.types.Object, _SourceState]] = {}
    expected_by_component: dict[str, tuple[_ExpectedVertex, ...]] = {}

    for specification in _SPECIFICATIONS:
        source_object = _create_cuboid(specification)
        _activate_only(source_object)
        bpy.context.view_layer.update()
        state = _capture_state(source_object)
        expected = _expected_vertices(
            scene,
            camera,
            state,
            camera_view_matrix,
        )
        sources.append(
            A1MultiObjectSource(
                source_object=source_object,
                component_id=specification.component_id,
                animation_namespace=specification.component_id,
                settings=_settings(camera_root, specification, camera_kind),
            )
        )
        state_by_component[specification.component_id] = (source_object, state)
        expected_by_component[specification.component_id] = expected

    multi_settings = A1MultiObjectExportSettings(
        output_directory=camera_root,
        output_stem=f"active_camera_{camera_kind.lower()}",
        mode=A1MultiObjectMode.STANDALONE,
        z_tolerance=1.0e-5,
    )
    prepared = prepare_a1_multi_object(
        tuple(sources),
        multi_settings,
        context=bpy.context,
        scene=scene,
    )
    composition = compose_a1_multi_object_document(
        prepared.sources,
        prepared.objects,
        multi_settings,
    )

    expected_order = tuple(
        component_id
        for component_id, expected in sorted(
            expected_by_component.items(),
            key=lambda item: max(vertex.camera_z for vertex in item[1]),
        )
    )
    origin_depth_by_component = {
        component_id: _camera_z(
            camera_view_matrix,
            _matrix_translation(state.matrix_world),
        )
        for component_id, (_source_object, state) in state_by_component.items()
    }
    origin_order = tuple(
        component_id
        for component_id, _depth in sorted(
            origin_depth_by_component.items(),
            key=lambda item: item[1],
        )
    )
    _assert(
        expected_order != origin_order,
        f"{camera_kind} fixture no longer distinguishes nearest vertex from origin",
    )

    owner_by_slot: dict[str, str] = {}
    component_slots: dict[str, tuple[str, ...]] = {}
    objects_report: list[dict[str, object]] = []
    maximum_matrix_delta = 0.0
    maximum_vertex_screen_delta = 0.0
    maximum_vertex_depth_delta = 0.0
    maximum_origin_delta = 0.0
    outside_frame_count = 0

    for source, item in zip(prepared.sources, prepared.objects, strict=True):
        source_object, state = state_by_component[source.component_id]
        expected_vertices = expected_by_component[source.component_id]
        maximum_matrix_delta = max(
            maximum_matrix_delta,
            _assert_state_unchanged(
                source_object,
                state,
                label=f"{camera_kind} {source.component_id}",
            ),
        )

        _assert(
            item.statistics["projection_kind"] == "ACTIVE_CAMERA",
            f"{source.component_id} did not use Active Camera projection",
        )
        _assert(
            item.statistics["texture_pipeline"] == "OBJECT_BAKE",
            f"{source.component_id} accidentally entered Camera Projection render path",
        )
        _assert(
            int(item.statistics["projection_canvas_width"]) == _TEXTURE_WIDTH
            and int(item.statistics["projection_canvas_height"]) == _TEXTURE_HEIGHT,
            f"{source.component_id} did not use export texture canvas",
        )
        _assert(
            int(item.statistics["active_camera_preprojection_triangulation"]) == 1,
            f"{source.component_id} did not triangulate before camera projection",
        )

        slots = tuple(slot.name for slot in item.document.slots)
        _assert(slots, f"{source.component_id} produced no slots")
        component_slots[source.component_id] = slots
        for slot_name in slots:
            previous = owner_by_slot.get(slot_name)
            _assert(
                previous is None,
                f"Slot {slot_name} is shared by {previous} and {source.component_id}",
            )
            owner_by_slot[slot_name] = source.component_id

        main_bone = next(
            bone
            for bone in item.document.bones
            if bone.name == item.rig.info.main_bone_name
        )
        actual_main = (
            0.0 if main_bone.x is None else float(main_bone.x),
            0.0 if main_bone.y is None else float(main_bone.y),
        )
        expected_origin = _expected_screen_point(
            scene,
            camera,
            _matrix_translation(state.matrix_world),
        )
        origin_delta = max(
            abs(actual - expected)
            for actual, expected in zip(actual_main, expected_origin, strict=True)
        )
        maximum_origin_delta = max(maximum_origin_delta, origin_delta)
        _assert(
            origin_delta <= _MAIN_BONE_TOLERANCE,
            f"{source.component_id} main origin mismatch: "
            f"actual={actual_main}, expected={expected_origin}, delta={origin_delta}",
        )

        scale = float(item.rig.info.uniform_scale)
        projected_by_index = {
            vertex.id.index: vertex for vertex in item.source_snapshot.vertices
        }
        for expected_vertex in expected_vertices:
            projected_vertex = projected_by_index[expected_vertex.index]
            final_screen = (
                actual_main[0] + float(projected_vertex.position[0]) * scale,
                actual_main[1] - float(projected_vertex.position[1]) * scale,
            )
            screen_delta = max(
                abs(final_screen[0] - expected_vertex.screen_x),
                abs(final_screen[1] - expected_vertex.screen_y),
            )
            depth = (
                float(item.source_snapshot.world_matrix[11])
                + float(projected_vertex.position[2])
            )
            depth_delta = abs(depth - expected_vertex.camera_z)
            maximum_vertex_screen_delta = max(
                maximum_vertex_screen_delta,
                screen_delta,
            )
            maximum_vertex_depth_delta = max(
                maximum_vertex_depth_delta,
                depth_delta,
            )
            _assert(
                screen_delta <= _MAIN_BONE_TOLERANCE,
                f"{source.component_id} vertex {expected_vertex.index} screen mismatch: "
                f"actual={final_screen}, "
                f"expected={(expected_vertex.screen_x, expected_vertex.screen_y)}, "
                f"delta={screen_delta}",
            )
            _assert(
                depth_delta <= _DEPTH_TOLERANCE,
                f"{source.component_id} vertex {expected_vertex.index} depth mismatch: "
                f"actual={depth}, expected={expected_vertex.camera_z}, "
                f"delta={depth_delta}, model={_DEPTH_EXPECTATION_MODEL}",
            )
            if (
                abs(expected_vertex.screen_x) > _TEXTURE_WIDTH / 2.0
                or abs(expected_vertex.screen_y) > _TEXTURE_HEIGHT / 2.0
            ):
                outside_frame_count += 1

        depth_range = calculate_a1_projected_snapshot_depth_range(
            item.source_snapshot
        )
        expected_nearest = max(
            expected_vertices,
            key=lambda value: (value.camera_z, -value.index),
        )
        _assert(
            depth_range.nearest_vertex_id.index == expected_nearest.index,
            f"{source.component_id} nearest vertex mismatch: "
            f"actual={depth_range.nearest_vertex_id.index}, "
            f"expected={expected_nearest.index}",
        )
        _assert(
            abs(depth_range.nearest_vertex_depth - expected_nearest.camera_z)
            <= _DEPTH_TOLERANCE,
            f"{source.component_id} nearest depth mismatch",
        )

        objects_report.append(
            {
                "componentId": source.component_id,
                "objectName": item.object_id,
                "mainPosition": list(actual_main),
                "expectedOrigin": list(expected_origin),
                "nearestVertexIndex": depth_range.nearest_vertex_id.index,
                "nearestVertexDepth": depth_range.nearest_vertex_depth,
                "farthestVertexIndex": depth_range.farthest_vertex_id.index,
                "farthestVertexDepth": depth_range.farthest_vertex_depth,
                "slotNames": list(slots),
            }
        )

    _assert(outside_frame_count > 0, f"{camera_kind} fixture has no out-of-frame vertex")

    final_slot_names = tuple(slot.name for slot in composition.document.slots)
    actual_order = _collapsed_owner_order(final_slot_names, owner_by_slot)
    _assert(
        actual_order == expected_order,
        f"{camera_kind} object-block order mismatch: "
        f"actual={actual_order}, expected={expected_order}",
    )
    for component_id in expected_order:
        actual_component_slots = tuple(
            slot_name
            for slot_name in final_slot_names
            if owner_by_slot[slot_name] == component_id
        )
        _assert(
            actual_component_slots == component_slots[component_id],
            f"{camera_kind} changed internal slot order for {component_id}",
        )

    _assert(
        _matrix_tuple(camera.matrix_world) == camera_matrix_before,
        f"{camera_kind} projection mutated camera matrix_world",
    )
    _assert(
        (
            camera.data.type,
            float(camera.data.clip_start),
            float(camera.data.clip_end),
            float(camera.data.shift_x),
            float(camera.data.shift_y),
            float(camera.data.lens),
            float(camera.data.ortho_scale),
        )
        == camera_data_before,
        f"{camera_kind} projection mutated Camera data",
    )
    _assert(
        (
            int(scene.render.resolution_x),
            int(scene.render.resolution_y),
            int(scene.render.resolution_percentage),
            float(scene.render.pixel_aspect_x),
            float(scene.render.pixel_aspect_y),
        )
        == scene_render_before,
        f"{camera_kind} projection mutated Scene render dimensions",
    )

    return {
        "cameraType": camera_kind,
        "cameraName": camera.name,
        "textureWidth": _TEXTURE_WIDTH,
        "textureHeight": _TEXTURE_HEIGHT,
        "sceneRenderWidth": _SCENE_RENDER_WIDTH,
        "sceneRenderHeight": _SCENE_RENDER_HEIGHT,
        "sourceInputOrder": [source.component_id for source in sources],
        "originDepthOrder": list(origin_order),
        "expectedComponentOrder": list(expected_order),
        "actualComponentOrder": list(actual_order),
        "outsideFrameVertexCount": outside_frame_count,
        "maximumMatrixDelta": maximum_matrix_delta,
        "maximumOriginDelta": maximum_origin_delta,
        "maximumVertexScreenDelta": maximum_vertex_screen_delta,
        "maximumVertexDepthDelta": maximum_vertex_depth_delta,
        "depthExpectationModel": _DEPTH_EXPECTATION_MODEL,
        "depthTolerance": _DEPTH_TOLERANCE,
        "objects": objects_report,
        "sourceUnchanged": True,
        "cameraUnchanged": True,
        "sceneRenderUnchanged": True,
    }


def run(output_directory: Path) -> Path:
    output_root = _prepare_output_directory(output_directory)
    camera_results = tuple(
        _run_camera_kind(output_root, camera_kind)
        for camera_kind in ("PERSP", "ORTHO")
    )
    report = {
        "status": "passed",
        "blenderVersion": bpy.app.version_string,
        "cameraCount": len(camera_results),
        "cameras": list(camera_results),
    }
    report_path = output_root / "active_camera_normal_uv_acceptance.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return report_path


def main() -> None:
    arguments = _parse_arguments()
    print(f"Blender version: {bpy.app.version_string}")
    print("[ACTIVE_CAMERA_NORMAL_UV] RUN Perspective + Orthographic standalone")
    report_path = run(arguments.output)
    print(f"[ACTIVE_CAMERA_NORMAL_UV] REPORT {report_path}")
    print("[ACTIVE_CAMERA_NORMAL_UV] PASS")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise
