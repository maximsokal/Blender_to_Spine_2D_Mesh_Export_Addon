"""Blender 5.2 regressions captured from mushrooms and flower_shop 0.90.0.

``Plane.008`` reproduces a slightly warped n-gon evaluated through Array x4.
``banco`` places a hidden reserve n-gon before a visible front n-gon, forcing FRONT and
reserve face-index collisions after triangulation. Public multi-object Depth preparation
must resolve both cases without changing the caller's Blender state or source datablocks.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import replace
from math import cos, radians, sin
from pathlib import Path
import sys
import tempfile
import traceback

import bpy


SCRIPT_DIRECTORY = Path(__file__).resolve().parent
REPOSITORY_ROOT = SCRIPT_DIRECTORY.parents[1]
for path in (SCRIPT_DIRECTORY, REPOSITORY_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from Blender_to_Spine2D_Mesh_Exporter.application import (  # noqa: E402
    A1MultiObjectExportSettings,
    A1MultiObjectMode,
)
from Blender_to_Spine2D_Mesh_Exporter.blender_adapter import (  # noqa: E402
    A1MultiObjectSource,
    prepare_a1_multi_object,
)
from Blender_to_Spine2D_Mesh_Exporter.blender_adapter.a1_object_preparation import (  # noqa: E402
    PreparedDepthA1Object,
)
from Blender_to_Spine2D_Mesh_Exporter.domain.baking import (  # noqa: E402
    DepthParallaxSettings,
)
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
import run_depth_array_modifier_integration as array_smoke  # noqa: E402


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
_RESERVE_FOLD_ANGLE = radians(15.0)


def _create_material(
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
    material = _create_material(
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
    """Create reserve face 0 and front face 1 with a true 15-degree dihedral."""

    flap_length = 0.75
    folded_x = 0.9 - cos(_RESERVE_FOLD_ANGLE) * flap_length
    folded_z = -sin(_RESERVE_FOLD_ANGLE) * flap_length
    source = _create_mesh_object(
        _FLOWER_SHOP_OBJECT_NAME,
        (
            (-0.9, -0.9, 0.0),
            (0.9, -0.9, 0.0),
            (0.9, 0.9, 0.0),
            (-0.9, 0.9, 0.0),
            (folded_x, -0.9, folded_z),
            (folded_x, 0.9, folded_z),
        ),
        (
            (1, 4, 5, 2),  # hidden reserve n-gon, evaluated source face 0
            (0, 1, 2, 3),  # visible front n-gon, evaluated source face 1
        ),
    )
    source.location = (3.4, 0.0, 0.0)
    material = _create_material(
        "FlowerShopBancoMaterial",
        (1.0, 0.32, 0.08, 1.0),
    )
    source.data.materials.append(material)
    return source, material


def _settings(output_directory: Path, prefix: str):
    base = array_smoke._settings(output_directory)
    return replace(
        base,
        prefix=prefix,
        output_stem=prefix,
        json_output_stem=prefix,
        export=replace(
            base.export,
            output_directory=output_directory,
        ),
        bake_execution=replace(
            base.bake_execution,
            depth_parallax=DepthParallaxSettings(
                horizon_angle_radians=_HORIZON_ANGLE,
            ),
        ),
    )


def _source_fingerprint(source, modifier=None) -> tuple[object, ...]:
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
        tuple(
            tuple(int(value) for value in polygon.vertices)
            for polygon in source.data.polygons
        ),
        tuple(item.name for item in source.modifiers),
        tuple(
            material.name if material is not None else None
            for material in source.data.materials
        ),
        modifier_values,
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


def _assert_unique_working_lineage(
    item: PreparedDepthA1Object,
    *,
    component_id: str,
) -> None:
    face_ids = tuple(face.source_id for face in item.source_snapshot.faces)
    loop_ids = tuple(loop.source_id for loop in item.source_snapshot.loops)
    _assert(
        len(face_ids) == len(set(face_ids)),
        f"{component_id} final union contains duplicate SourceFaceId values",
    )
    _assert(
        len(loop_ids) == len(set(loop_ids)),
        f"{component_id} final union contains duplicate SourceLoopId values",
    )
    union_vertices = {
        vertex.source_id for vertex in item.source_snapshot.vertices
    }
    subsets = (
        item.depth_parallax_package.front_snapshot,
        *tuple(
            surface.snapshot
            for surface in item.depth_parallax_package.reserve_surfaces
        ),
    )
    for subset in subsets:
        _assert(
            all(vertex.source_id in union_vertices for vertex in subset.vertices),
            f"{component_id} subset lost canonical union vertex lineage",
        )


def _run() -> None:
    _clear_scene()
    _purge_orphan_scene_data()
    _configure_scene()
    bpy.context.scene.cycles.samples = 1
    bpy.context.scene.render.film_transparent = True
    bpy.context.scene.frame_set(1)

    mushrooms, mushrooms_material, array_modifier = _create_mushrooms_source()
    banco, banco_material = _create_flower_shop_source()
    camera = array_smoke._create_orthographic_camera(
        "RealSceneRegressionCamera"
    )
    sentinel = _create_sentinel()
    sentinel.location = (20.0, 0.0, 0.0)
    _activate_only(sentinel)
    mushrooms.select_set(False)
    banco.select_set(False)
    bpy.context.view_layer.update()

    context_before = _capture_context()
    mushrooms_before = _source_fingerprint(mushrooms, array_modifier)
    banco_before = _source_fingerprint(banco)
    camera_before = array_smoke._camera_fingerprint(camera)
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
                settings=_settings(output_directory, _MUSHROOMS_PREFIX),
            ),
            A1MultiObjectSource(
                source_object=banco,
                component_id=_FLOWER_SHOP_COMPONENT,
                settings=_settings(output_directory, _FLOWER_SHOP_PREFIX),
            ),
        )
        prepared = prepare_a1_multi_object(
            sources,
            A1MultiObjectExportSettings(
                output_directory=output_directory,
                output_stem=_MULTI_STEM,
                mode=A1MultiObjectMode.STANDALONE,
            ),
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
    _assert_unique_working_lineage(
        mushrooms_item,
        component_id=_MUSHROOMS_COMPONENT,
    )

    banco_package = banco_item.depth_parallax_package
    _assert(
        banco_package.front_face_indices == (2, 3),
        f"banco front n-gon triangulation changed: {banco_package.front_face_indices}",
    )
    _assert(
        banco_package.reserve_face_indices == (0, 1),
        f"banco hidden reserve n-gon was not retained: "
        f"{banco_package.reserve_face_indices}",
    )
    _assert(
        len(banco_package.reserve_surfaces) >= 1,
        "banco positive horizon created no reserve attachment",
    )
    reserve_owned_faces = tuple(
        sorted(
            {
                face_index
                for surface in banco_package.reserve_surfaces
                for face_index in surface.source_face_indices
            }
        )
    )
    _assert(
        reserve_owned_faces == (0, 1),
        f"banco reserve render ownership changed: {reserve_owned_faces}",
    )
    _assert_unique_working_lineage(
        banco_item,
        component_id=_FLOWER_SHOP_COMPONENT,
    )

    banco_union_lineage = Counter(
        face.source_id for face in banco_item.source_snapshot.faces
    )
    banco_prepared_lineage = Counter(
        source_face_id
        for region in banco_item.geometry.regions
        for source_face_id in region.source_face_ids
    )
    _assert(
        banco_prepared_lineage == banco_union_lineage,
        "banco prepared regions changed canonical SourceFaceId multiplicity",
    )
    _assert(
        all(
            region.transfer_report.complete
            for region in banco_item.uv_regions.regions
        ),
        "banco canonical SourceLoopId values did not complete UV propagation",
    )

    _assert(_capture_context() == context_before, "selection or active context changed")
    _assert(
        _source_fingerprint(mushrooms, array_modifier) == mushrooms_before,
        "Plane.008 object, mesh, or Array modifier changed",
    )
    _assert(
        _source_fingerprint(banco) == banco_before,
        "banco object or mesh changed",
    )
    _assert(
        array_smoke._camera_fingerprint(camera) == camera_before,
        "active camera changed",
    )
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
        "flower_shop=banco source_face_collision=canonicalized "
        "uv_lineage=unique pipeline=public-multi-object"
    )


def main() -> None:
    try:
        _run()
    except Exception:
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
