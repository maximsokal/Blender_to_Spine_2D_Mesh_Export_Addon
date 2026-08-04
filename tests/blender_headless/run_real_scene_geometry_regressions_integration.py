"""Blender 5.2 regressions captured from mushrooms and flower_shop 0.90.0.

The fixture reproduces both public multi-object PREPARE_OBJECTS failures:

* ``Plane.008`` owns one slightly warped n-gon and an Array modifier with four copies.
  Evaluated provenance repeats legitimately and the n-gon centroid-plane residue matches
  the real scene failure magnitude.
* ``banco`` owns one planar Blender n-gon. Depth projection triangulates it into two local
  faces that share one historical SourceFaceId; geometry coverage must preserve both
  occurrences without reporting an overlap.

Preparation must complete without mutating source objects, modifiers, materials, camera,
selection, frame, or temporary Blender datablocks.
"""

from __future__ import annotations

from collections import Counter
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
from Blender_to_Spine2D_Mesh_Exporter.blender_adapter.a1_object_preparation import (  # noqa: E402
    PreparedDepthA1Object,
)
from Blender_to_Spine2D_Mesh_Exporter.domain.baking import (  # noqa: E402
    A1TextureExportMode,
    BakeExecutionSettings,
    DepthCameraProjectionSettings,
    DepthParallaxSettings,
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


_TARGET = SpineJsonTarget.SPINE_4_2
_MUSHROOMS_OBJECT_NAME = "Plane.008"
_FLOWER_SHOP_OBJECT_NAME = "banco"
_MUSHROOMS_COMPONENT = "object_1:Plane.008"
_FLOWER_SHOP_COMPONENT = "object_1:banco"
_MUSHROOMS_PREFIX = "MushroomsPlane008"
_FLOWER_SHOP_PREFIX = "FlowerShopBanco"
_MULTI_STEM = "RealSceneGeometryRegressions"
_COPY_COUNT = 4
_WARP_HEIGHT = 0.0007581877679385422
_HORIZON_ANGLE = radians(20.0)


def _create_emission_material(
    name: str,
    color: tuple[float, float, float, float],
):
    material = bpy.data.materials.new(name=name)
    material.use_nodes = True
    nodes = material.node_tree.nodes
    nodes.clear()
    output = nodes.new(type="ShaderNodeOutputMaterial")
    emission = nodes.new(type="ShaderNodeEmission")
    emission.inputs["Color"].default_value = color
    emission.inputs["Strength"].default_value = 1.5
    material.node_tree.links.new(
        emission.outputs["Emission"],
        output.inputs["Surface"],
    )
    return material


def _create_orthographic_camera(name: str):
    data = bpy.data.cameras.new(name=f"{name}_Data")
    data.type = "ORTHO"
    data.ortho_scale = 11.0
    data.clip_start = 0.1
    data.clip_end = 100.0
    camera = bpy.data.objects.new(name, data)
    bpy.context.scene.collection.objects.link(camera)
    target = Vector((1.0, 0.0, 0.0))
    camera.location = (1.0, 0.0, 9.0)
    camera.rotation_euler = (
        target - camera.location
    ).to_track_quat("-Z", "Y").to_euler()
    bpy.context.scene.camera = camera
    return camera


def _create_mushrooms_source():
    source = _create_mesh_object(
        _MUSHROOMS_OBJECT_NAME,
        (
            (-0.5, -0.5, 0.0),
            (0.5, -0.5, 0.0),
            (0.5, 0.5, _WARP_HEIGHT),
            (-0.5, 0.5, 0.0),
        ),
        ((0, 1, 2, 3),),
    )
    source.location = (-1.75, 0.0, 0.0)
    material = _create_emission_material(
        "MushroomsPlane008Material",
        (0.12, 0.68, 1.0, 1.0),
    )
    source.data.materials.append(material)

    modifier = source.modifiers.new(name="MushroomsArrayCopies", type="ARRAY")
    modifier.count = _COPY_COUNT
    modifier.use_relative_offset = True
    modifier.relative_offset_displace = (1.35, 0.0, 0.0)
    modifier.use_constant_offset = False
    modifier.use_merge_vertices = False
    return source, material, modifier


def _create_flower_shop_source():
    source = _create_mesh_object(
        _FLOWER_SHOP_OBJECT_NAME,
        (
            (-0.75, -0.65, 0.0),
            (0.75, -0.65, 0.0),
            (0.75, 0.65, 0.0),
            (-0.75, 0.65, 0.0),
        ),
        ((0, 1, 2, 3),),
    )
    source.location = (3.6, 0.0, 0.0)
    material = _create_emission_material(
        "FlowerShopBancoMaterial",
        (1.0, 0.32, 0.08, 1.0),
    )
    source.data.materials.append(material)
    return source, material


def _single_settings(
    output_directory: Path,
    *,
    prefix: str,
) -> A1SingleObjectExportSettings:
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
        prefix=prefix,
        output_stem=prefix,
        json_output_stem=prefix,
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
                max_points=512,
            ),
            depth_parallax=DepthParallaxSettings(
                horizon_angle_radians=_HORIZON_ANGLE,
            ),
        ),
    )


def _object_fingerprint(source, modifier=None) -> tuple[object, ...]:
    modifier_values: tuple[object, ...] = ()
    if modifier is not None:
        modifier_values = (
            modifier.name,
            modifier.type,
            int(modifier.count),
            bool(modifier.use_relative_offset),
            tuple(float(value) for value in modifier.relative_offset_displace),
            bool(modifier.use_constant_offset),
            bool(modifier.use_merge_vertices),
        )
    return (
        source.name,
        source.data.name,
        tuple(tuple(float(value) for value in row) for row in source.matrix_world),
        tuple(tuple(float(value) for value in vertex.co) for vertex in source.data.vertices),
        tuple(tuple(int(value) for value in polygon.vertices) for polygon in source.data.polygons),
        tuple(item.name for item in source.modifiers),
        tuple(
            material.name if material is not None else None
            for material in source.data.materials
        ),
        modifier_values,
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


def _prepared_by_component(prepared) -> dict[str, PreparedDepthA1Object]:
    result: dict[str, PreparedDepthA1Object] = {}
    for source, item in zip(prepared.sources, prepared.objects, strict=True):
        _assert(
            isinstance(item, PreparedDepthA1Object),
            f"component {source.component_id} returned {type(item)!r}",
        )
        result[source.component_id] = item
    return result


def _run() -> None:
    _clear_scene()
    _purge_orphan_scene_data()
    _configure_scene()
    bpy.context.scene.cycles.samples = 1
    bpy.context.scene.render.film_transparent = True
    bpy.context.scene.frame_set(1)

    mushrooms, mushrooms_material, array_modifier = _create_mushrooms_source()
    banco, banco_material = _create_flower_shop_source()
    camera = _create_orthographic_camera("RealSceneRegressionCamera")
    sentinel = _create_sentinel()
    sentinel.location = (20.0, 0.0, 0.0)
    _activate_only(sentinel)
    mushrooms.select_set(False)
    banco.select_set(False)
    bpy.context.view_layer.update()

    context_before = _capture_context()
    mushrooms_before = _object_fingerprint(mushrooms, array_modifier)
    banco_before = _object_fingerprint(banco)
    camera_before = _camera_fingerprint(camera)
    materials_before = (
        _material_fingerprint(mushrooms_material),
        _material_fingerprint(banco_material),
    )
    temporary_before = _temporary_datablock_names()
    frame_before = int(bpy.context.scene.frame_current)

    with tempfile.TemporaryDirectory(
        prefix="spine2d_real_scene_geometry_regressions_"
    ) as directory:
        output_directory = Path(directory)
        sources = (
            A1MultiObjectSource(
                source_object=mushrooms,
                component_id=_MUSHROOMS_COMPONENT,
                settings=_single_settings(
                    output_directory,
                    prefix=_MUSHROOMS_PREFIX,
                ),
            ),
            A1MultiObjectSource(
                source_object=banco,
                component_id=_FLOWER_SHOP_COMPONENT,
                settings=_single_settings(
                    output_directory,
                    prefix=_FLOWER_SHOP_PREFIX,
                ),
            ),
        )
        settings = A1MultiObjectExportSettings(
            output_directory=output_directory,
            output_stem=_MULTI_STEM,
            mode=A1MultiObjectMode.STANDALONE,
        )
        prepared = prepare_a1_multi_object(
            sources,
            settings,
            context=bpy.context,
            scene=bpy.context.scene,
        )

        _assert(
            not tuple(path for path in output_directory.rglob("*") if path.is_file()),
            "preparation wrote output files",
        )

    items = _prepared_by_component(prepared)
    mushrooms_item = items[_MUSHROOMS_COMPONENT]
    banco_item = items[_FLOWER_SHOP_COMPONENT]

    _assert(
        int(mushrooms_item.statistics["depth_projection_source_triangle_count"])
        == _COPY_COUNT * 2,
        "Plane.008 Array n-gons did not produce eight projected triangles",
    )
    _assert(
        int(mushrooms_item.statistics["evaluated_identity_rebased"]) == 1,
        "Plane.008 repeated modifier lineage was not canonicalized",
    )
    _assert(
        any(
            issue.code == "EVALUATED_IDENTITY_REBASED"
            for issue in mushrooms_item.warnings
        ),
        "Plane.008 emitted no evaluated identity warning",
    )

    banco_lineage = Counter(
        face.source_id for face in banco_item.source_snapshot.faces
    )
    _assert(
        len(banco_item.source_snapshot.faces) == 2,
        "banco n-gon did not produce two local depth triangles",
    )
    _assert(
        tuple(sorted(banco_lineage.values())) == (2,),
        f"banco did not retain two occurrences of one SourceFaceId: {banco_lineage}",
    )
    prepared_banco_lineage = Counter(
        source_face_id
        for region in banco_item.geometry.regions
        for source_face_id in region.source_face_ids
    )
    _assert(
        prepared_banco_lineage == banco_lineage,
        "banco prepared regions changed SourceFaceId multiplicity",
    )

    _assert(_capture_context() == context_before, "selection or active context changed")
    _assert(
        _object_fingerprint(mushrooms, array_modifier) == mushrooms_before,
        "Plane.008 object, mesh, or Array modifier changed",
    )
    _assert(
        _object_fingerprint(banco) == banco_before,
        "banco object or mesh changed",
    )
    _assert(_camera_fingerprint(camera) == camera_before, "active camera changed")
    _assert(
        (
            _material_fingerprint(mushrooms_material),
            _material_fingerprint(banco_material),
        )
        == materials_before,
        "source materials changed",
    )
    _assert(
        _temporary_datablock_names() == temporary_before,
        "temporary Blender datablocks leaked",
    )
    _assert(
        int(bpy.context.scene.frame_current) == frame_before,
        "Scene frame changed",
    )

    print(
        "[REAL-SCENE-GEOMETRY-REGRESSIONS] PASS "
        "mushrooms=Plane.008 warp=0.0001895469 array=4 "
        "flower_shop=banco source_face_multiplicity=2 "
        "pipeline=public-multi-object"
    )


def main() -> None:
    try:
        _run()
    except Exception:
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
