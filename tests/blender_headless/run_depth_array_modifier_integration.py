"""Real Blender 5.2 regression for modifier-duplicated Depth topology.

The fixture evaluates a two-triangle source through an Array modifier with four copies.
Original stamped lineage therefore repeats across the evaluated mesh. The public Depth
preparation route must validate that provenance, canonicalize every evaluated element to
an independent local identity, and complete without changing the source object, modifier,
selection, Scene, or temporary Blender datablocks.
"""

from __future__ import annotations

from math import radians
from pathlib import Path
import sys
import tempfile
import traceback

import bpy
from mathutils import Vector


SCRIPT_DIRECTORY = Path(__file__).resolve().parent
REPOSITORY_ROOT = SCRIPT_DIRECTORY.parents[1]
for path in (SCRIPT_DIRECTORY, REPOSITORY_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from Blender_to_Spine2D_Mesh_Exporter.application import (  # noqa: E402
    A1SingleObjectExportSettings,
    A1SourceGeometryMode,
    ExportSettings,
)
from Blender_to_Spine2D_Mesh_Exporter.blender_adapter import (  # noqa: E402
    prepare_a1_object,
    read_evaluated_mesh_snapshot,
)
from Blender_to_Spine2D_Mesh_Exporter.blender_adapter.a1_object_preparation import (  # noqa: E402
    PreparedDepthA1Object,
)
from Blender_to_Spine2D_Mesh_Exporter.domain.baking import (  # noqa: E402
    A1TextureExportMode,
    BakeExecutionSettings,
    DepthCameraProjectionSettings,
    DepthParallaxSettings,
)
from Blender_to_Spine2D_Mesh_Exporter.domain.geometry import (  # noqa: E402
    ModifierLineagePolicy,
)
from Blender_to_Spine2D_Mesh_Exporter.domain.geometry.evaluated_identity import (  # noqa: E402
    rebase_mesh_snapshot_to_evaluated_identity,
)
from Blender_to_Spine2D_Mesh_Exporter.domain.projection import (  # noqa: E402
    A1ProjectionDirection,
)
from Blender_to_Spine2D_Mesh_Exporter.domain.spine import (  # noqa: E402
    A1RigProfile,
    SpineJsonTarget,
)
from Blender_to_Spine2D_Mesh_Exporter.domain.uv import UvUnwrapSettings  # noqa: E402
from run_bake_integration import (  # noqa: E402
    _activate_only,
    _assert,
    _capture_context,
    _clear_scene,
    _create_mesh_object,
    _create_sentinel,
    _material_fingerprint,
    _temporary_datablock_names,
)
from run_camera_projection_integration import (  # noqa: E402
    _configure_scene,
    _purge_orphan_scene_data,
)


_PREFIX = "DepthArrayModifier"
_COPY_COUNT = 4
_SOURCE_FACE_COUNT = 2
_EVALUATED_FACE_COUNT = _SOURCE_FACE_COUNT * _COPY_COUNT
_TARGET = SpineJsonTarget.SPINE_4_2


def _create_emission_material(name: str):
    material = bpy.data.materials.new(name=name)
    material.use_nodes = True
    nodes = material.node_tree.nodes
    nodes.clear()
    output = nodes.new(type="ShaderNodeOutputMaterial")
    emission = nodes.new(type="ShaderNodeEmission")
    emission.inputs["Color"].default_value = (0.12, 0.68, 1.0, 1.0)
    emission.inputs["Strength"].default_value = 1.5
    material.node_tree.links.new(
        emission.outputs["Emission"],
        output.inputs["Surface"],
    )
    return material


def _create_orthographic_camera(name: str):
    data = bpy.data.cameras.new(name=f"{name}_Data")
    data.type = "ORTHO"
    data.ortho_scale = 7.5
    data.clip_start = 0.1
    data.clip_end = 100.0
    camera = bpy.data.objects.new(name, data)
    bpy.context.scene.collection.objects.link(camera)
    camera.location = (2.25, 0.0, 8.0)
    camera.rotation_euler = (
        Vector((2.25, 0.0, 0.0)) - camera.location
    ).to_track_quat("-Z", "Y").to_euler()
    bpy.context.scene.camera = camera
    return camera


def _create_array_source(name: str):
    source = _create_mesh_object(
        name,
        (
            (-0.50, -0.50, 0.0),
            (0.50, -0.50, 0.0),
            (0.50, 0.50, 0.0),
            (-0.50, 0.50, 0.0),
        ),
        (
            (0, 1, 2),
            (0, 2, 3),
        ),
    )
    material = _create_emission_material(f"{name}_Material")
    source.data.materials.append(material)

    modifier = source.modifiers.new(name="DepthArrayCopies", type="ARRAY")
    modifier.count = _COPY_COUNT
    modifier.use_relative_offset = True
    modifier.relative_offset_displace = (1.5, 0.0, 0.0)
    modifier.use_constant_offset = False
    modifier.use_merge_vertices = False
    return source, material, modifier


def _settings(output_directory: Path) -> A1SingleObjectExportSettings:
    return A1SingleObjectExportSettings(
        export=ExportSettings(
            texture_width=128,
            texture_height=128,
            output_directory=output_directory,
            images_relative_path="images",
            spine_version=_TARGET.exact_version,
            rig_profile=A1RigProfile.TWO_AXIS_ROTATION_SCALE.value,
            bake_margin=1,
        ),
        prefix=_PREFIX,
        output_stem=_PREFIX,
        json_output_stem=_PREFIX,
        source_geometry_mode=A1SourceGeometryMode.EVALUATED,
        projection_direction=A1ProjectionDirection.ACTIVE_CAMERA,
        uv=UvUnwrapSettings(layer_name="SpineBakeUV"),
        bake_execution=BakeExecutionSettings(
            samples=1,
            texture_export_mode=A1TextureExportMode.DEPTH_CAMERA_PROJECTION,
            depth_projection=DepthCameraProjectionSettings(
                smoothing=0.0,
                edge_threshold_fraction=1.0,
                mesh_error_pixels=8.0,
                max_points=256,
            ),
            depth_parallax=DepthParallaxSettings(
                horizon_angle_radians=radians(0.0),
            ),
        ),
    )


def _object_fingerprint(source, modifier) -> tuple[object, ...]:
    return (
        source.name,
        source.data.name,
        tuple(tuple(float(value) for value in row) for row in source.matrix_world),
        len(source.data.vertices),
        len(source.data.edges),
        len(source.data.polygons),
        tuple(material.name if material is not None else None for material in source.data.materials),
        tuple(item.name for item in source.modifiers),
        modifier.name,
        modifier.type,
        int(modifier.count),
        bool(modifier.use_relative_offset),
        tuple(float(value) for value in modifier.relative_offset_displace),
        bool(modifier.use_constant_offset),
        bool(modifier.use_merge_vertices),
    )


def _camera_fingerprint(camera) -> tuple[object, ...]:
    return (
        camera.name,
        camera.data.name,
        tuple(tuple(float(value) for value in row) for row in camera.matrix_world),
        str(camera.data.type),
        float(camera.data.ortho_scale),
        float(camera.data.clip_start),
        float(camera.data.clip_end),
        bpy.context.scene.camera.name if bpy.context.scene.camera is not None else None,
    )


def _run() -> None:
    _clear_scene()
    _purge_orphan_scene_data()
    _configure_scene()
    bpy.context.scene.cycles.samples = 1
    bpy.context.scene.render.film_transparent = True
    bpy.context.scene.frame_set(1)

    source, material, modifier = _create_array_source(f"{_PREFIX}_Source")
    camera = _create_orthographic_camera(f"{_PREFIX}_Camera")
    sentinel = _create_sentinel()
    sentinel.location = (20.0, 0.0, 0.0)
    _activate_only(sentinel)
    source.select_set(False)
    bpy.context.view_layer.update()

    context_before = _capture_context()
    object_before = _object_fingerprint(source, modifier)
    camera_before = _camera_fingerprint(camera)
    material_before = _material_fingerprint(material)
    temporary_before = _temporary_datablock_names()
    scene_frame_before = int(bpy.context.scene.frame_current)

    depsgraph = bpy.context.evaluated_depsgraph_get()
    depsgraph.update()
    raw = read_evaluated_mesh_snapshot(
        source,
        scene=bpy.context.scene,
        depsgraph=depsgraph,
        source_object_id=source.name,
        snapshot_id=f"{source.name}:array-raw",
        uv_layer_names=("UVMap",),
        lineage_policy=ModifierLineagePolicy.ALLOW_SOURCE_DUPLICATION,
    )
    _assert(
        len(raw.snapshot.faces) == _EVALUATED_FACE_COUNT,
        f"Array evaluated face count changed: {len(raw.snapshot.faces)}",
    )
    _assert(
        len({face.source_id for face in raw.snapshot.faces}) == _SOURCE_FACE_COUNT,
        "raw Array provenance no longer contains repeated source face ids",
    )
    _assert(raw.lineage_report.valid, "permissive Array lineage report is invalid")
    _assert(
        raw.lineage_report.faces.duplicated_source_indices == (0, 1),
        "Array face duplication was not diagnosed",
    )

    rebased = rebase_mesh_snapshot_to_evaluated_identity(raw.snapshot)
    _assert(rebased.changed, "Array evaluated identity was not rebased")
    _assert(
        len({face.source_id for face in rebased.snapshot.faces})
        == _EVALUATED_FACE_COUNT,
        "evaluated Array faces do not own unique canonical source ids",
    )
    _assert(
        tuple(face.source_id.face_index for face in rebased.snapshot.faces)
        == tuple(range(_EVALUATED_FACE_COUNT)),
        "evaluated Array face identity does not follow dense polygon order",
    )
    _assert(
        len({vertex.source_id for vertex in rebased.snapshot.vertices})
        == len(rebased.snapshot.vertices),
        "evaluated Array vertices do not own unique canonical source ids",
    )

    with tempfile.TemporaryDirectory(prefix="spine2d_depth_array_") as directory:
        prepared = prepare_a1_object(
            source,
            _settings(Path(directory)),
            context=bpy.context,
            scene=bpy.context.scene,
        )

    _assert(
        isinstance(prepared, PreparedDepthA1Object),
        f"public Array preparation returned {type(prepared)!r}",
    )
    _assert(
        int(prepared.statistics.get("evaluated_identity_rebased", 0)) == 1,
        "public Depth preparation did not report evaluated identity rebasing",
    )
    _assert(
        int(
            prepared.statistics.get(
                "evaluated_identity_duplicate_face_source_ids",
                0,
            )
        )
        == _EVALUATED_FACE_COUNT - _SOURCE_FACE_COUNT,
        "public Depth preparation reported wrong duplicate face count",
    )
    _assert(
        any(issue.code == "EVALUATED_IDENTITY_REBASED" for issue in prepared.warnings),
        "public Depth preparation emitted no evaluated identity warning",
    )
    _assert(
        len(prepared.depth_parallax_package.front_face_indices)
        == _EVALUATED_FACE_COUNT,
        "front visibility did not retain every camera-visible Array face",
    )
    _assert(
        not prepared.depth_parallax_package.reserve_surfaces,
        "zero-horizon Array preparation unexpectedly created reserve surfaces",
    )

    _assert(_capture_context() == context_before, "selection or active context changed")
    _assert(
        _object_fingerprint(source, modifier) == object_before,
        "source object, mesh, or Array modifier changed",
    )
    _assert(_camera_fingerprint(camera) == camera_before, "active camera changed")
    _assert(_material_fingerprint(material) == material_before, "material changed")
    _assert(
        _temporary_datablock_names() == temporary_before,
        "temporary Blender datablocks leaked",
    )
    _assert(
        int(bpy.context.scene.frame_current) == scene_frame_before,
        "Scene frame changed",
    )

    print(
        "[DEPTH-ARRAY-MODIFIER] PASS "
        f"source_faces={_SOURCE_FACE_COUNT} "
        f"evaluated_faces={_EVALUATED_FACE_COUNT} "
        "lineage=validated identity=rebased preparation=public"
    )


def main() -> None:
    try:
        _run()
    except Exception:
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
