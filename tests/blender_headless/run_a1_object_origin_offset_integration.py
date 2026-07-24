"""Blender 5.2 headless integration for Object Origin placement through A1.

Run from the repository root with Blender 5.2 or newer::

    blender --background --factory-startup --python \
        tests/blender_headless/run_a1_object_origin_offset_integration.py
"""

from __future__ import annotations

from dataclasses import replace
from math import radians
from pathlib import Path
import sys

import bpy
from mathutils import Euler, Matrix, Vector


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Blender_to_Spine2D_Mesh_Exporter.application import (  # noqa: E402
    A1MultiObjectExportSettings,
    A1MultiObjectMode,
    A1SingleObjectExportSettings,
    A1SourceGeometryMode,
    ExportSettings,
    calculate_a1_object_bake_main_position_pixels,
)
from Blender_to_Spine2D_Mesh_Exporter.blender_adapter.a1_multi_object_composition import (  # noqa: E402
    compose_a1_multi_object_document,
)
from Blender_to_Spine2D_Mesh_Exporter.blender_adapter.a1_multi_object_contracts import (  # noqa: E402
    A1MultiObjectSource,
)
from Blender_to_Spine2D_Mesh_Exporter.blender_adapter.a1_object_preparation import (  # noqa: E402
    prepare_a1_object,
)
from Blender_to_Spine2D_Mesh_Exporter.blender_adapter.mesh_writer import (  # noqa: E402
    temporary_mesh_object,
)
from Blender_to_Spine2D_Mesh_Exporter.domain.baking.generated_materials import (  # noqa: E402
    A1MaterialSourcePolicy,
)
from Blender_to_Spine2D_Mesh_Exporter.domain.spine import (  # noqa: E402
    ConnectedGroupBuildResult,
    ConnectedPlacementSpace,
)


_EPSILON = 2.1e-2
_TEXTURE_SIZE = 100
_OUTPUT_ROOT = Path(bpy.app.tempdir) / "spine2d-origin-offset-integration"


def _assert_vector_close(first: Vector, second: Vector, *, label: str) -> None:
    if (first - second).length > _EPSILON:
        raise AssertionError(
            f"{label} differs: first={tuple(first)}, second={tuple(second)}"
        )


def _assert_pair_close(
    first: tuple[float, float],
    second: tuple[float, float],
    *,
    label: str,
) -> None:
    if max(abs(first[0] - second[0]), abs(first[1] - second[1])) > _EPSILON:
        raise AssertionError(f"{label} differs: first={first}, second={second}")


def _datablock_counts() -> dict[str, int]:
    """Capture every Blender datablock family allocated by this preparation path."""

    return {
        "objects": len(bpy.data.objects),
        "meshes": len(bpy.data.meshes),
        "collections": len(bpy.data.collections),
        "textures": len(bpy.data.textures),
        "materials": len(bpy.data.materials),
        "images": len(bpy.data.images),
        "node_groups": len(bpy.data.node_groups),
    }


def _context_signature() -> tuple[str | None, tuple[str, ...], str | None]:
    """Return the active object, selection, and mode visible to Blender operators."""

    active = bpy.context.view_layer.objects.active
    active_name = None if active is None else str(active.name)
    active_mode = None if active is None else str(active.mode).upper()
    selected_names = tuple(sorted(str(obj.name) for obj in bpy.context.selected_objects))
    return active_name, selected_names, active_mode


def _create_source(
    name: str,
    *,
    vertices: tuple[tuple[float, float, float], ...],
    matrix_world: Matrix,
) -> bpy.types.Object:
    mesh = bpy.data.meshes.new(f"{name}_Mesh")
    source = bpy.data.objects.new(name, mesh)
    bpy.context.scene.collection.objects.link(source)
    mesh.from_pydata(vertices, (), ((0, 1, 2), (0, 2, 3)))
    mesh.update()
    source.matrix_world = matrix_world
    bpy.context.view_layer.update()
    return source


def _cleanup_object(obj: bpy.types.Object | None) -> None:
    if obj is None:
        return
    mesh = obj.data if isinstance(obj.data, bpy.types.Mesh) else None
    bpy.data.objects.remove(obj, do_unlink=True)
    if mesh is not None and mesh.users == 0:
        bpy.data.meshes.remove(mesh)


def _cleanup_texture(texture: bpy.types.Texture | None) -> None:
    if texture is None:
        return
    if texture.users != 0:
        raise AssertionError(
            f"Texture '{texture.name}' still has {texture.users} users during cleanup"
        )
    bpy.data.textures.remove(texture)


def _evaluated_world_positions(obj: bpy.types.Object) -> tuple[Vector, ...]:
    depsgraph = bpy.context.evaluated_depsgraph_get()
    evaluated = obj.evaluated_get(depsgraph)
    evaluated_mesh = evaluated.to_mesh(
        preserve_all_data_layers=True,
        depsgraph=depsgraph,
    )
    if evaluated_mesh is None:
        raise AssertionError(f"Unable to evaluate mesh for {obj.name}")
    try:
        return tuple(
            evaluated.matrix_world @ vertex.co for vertex in evaluated_mesh.vertices
        )
    finally:
        evaluated.to_mesh_clear()


def _settings(
    prefix: str,
    *,
    source_mode: A1SourceGeometryMode,
    use_world_location: bool,
) -> A1SingleObjectExportSettings:
    return A1SingleObjectExportSettings(
        export=ExportSettings(
            texture_width=_TEXTURE_SIZE,
            texture_height=_TEXTURE_SIZE,
            output_directory=_OUTPUT_ROOT,
            images_relative_path="images",
        ),
        prefix=prefix,
        output_stem=prefix,
        json_output_stem=prefix,
        source_geometry_mode=source_mode,
        material_source_policy=A1MaterialSourcePolicy.FORCE_GENERATED,
        use_world_location_for_main_bone=use_world_location,
    )


def _assert_bake_target_matches_world(prepared, expected_world: tuple[Vector, ...]) -> None:
    expected_by_source = {
        index: position.copy() for index, position in enumerate(expected_world)
    }
    snapshot = prepared.bake_target_snapshot
    with temporary_mesh_object(
        snapshot,
        scene=bpy.context.scene,
        name_prefix=f"__{prepared.prefix}_OriginTarget",
    ) as temporary:
        target = temporary.object
        for vertex in snapshot.vertices:
            expected = expected_by_source.get(vertex.source_id.vertex_index)
            if expected is None:
                raise AssertionError(
                    f"Unknown source vertex {vertex.source_id.vertex_index}"
                )
            actual = target.matrix_world @ target.data.vertices[vertex.id.index].co
            _assert_vector_close(
                expected,
                actual,
                label=(
                    f"{prepared.prefix} bake target source "
                    f"{vertex.source_id.vertex_index}"
                ),
            )


def _assert_spine_vertex_bones(prepared) -> None:
    """Check the origin-offset coordinate split before runtime Z controls deform it."""

    bones = {bone.name: bone for bone in prepared.document.bones}
    main = bones[prepared.rig.info.main_bone_name]
    matrix = prepared.source_snapshot.world_matrix
    translation_x = (
        float(matrix[3]) if prepared.settings.use_world_location_for_main_bone else 0.0
    )
    translation_y = (
        float(matrix[7]) if prepared.settings.use_world_location_for_main_bone else 0.0
    )
    scale = prepared.rig.info.uniform_scale

    for region, projection in zip(
        prepared.uv_regions.snapshots,
        prepared.document_assembly.projections,
        strict=True,
    ):
        vertex_by_id = region.vertex_by_id()
        request = projection.request
        for projected, key in zip(
            request.vertices,
            projection.ordered_vertex_keys,
            strict=True,
        ):
            vertex_bone = bones[
                prepared.rig.profile.vertex_bone(
                    request.vertex_prefix,
                    projected.index,
                )
            ]
            expected_bone = (
                round(float(projected.bone_position_pixels[0]), 2),
                round(float(projected.bone_position_pixels[1]), 2),
            )
            _assert_pair_close(
                (float(vertex_bone.x), float(vertex_bone.y)),
                expected_bone,
                label=f"{prepared.prefix} stored vertex bone {projected.index}",
            )

            source_vertex = vertex_by_id[key.vertex_id]
            expected_world_xy = (
                (translation_x + float(source_vertex.position[0])) * scale,
                translation_y * scale - float(source_vertex.position[1]) * scale,
            )
            actual_world_xy = (
                float(main.x) + float(vertex_bone.x),
                float(main.y) + float(vertex_bone.y),
            )
            _assert_pair_close(
                actual_world_xy,
                expected_world_xy,
                label=f"{prepared.prefix} reconstructed vertex {projected.index}",
            )


def _prepare(
    source: bpy.types.Object,
    *,
    prefix: str,
    source_mode: A1SourceGeometryMode,
    use_world_location: bool,
    expected_world: tuple[Vector, ...],
):
    prepared = prepare_a1_object(
        source,
        _settings(
            prefix,
            source_mode=source_mode,
            use_world_location=use_world_location,
        ),
        context=bpy.context,
        scene=bpy.context.scene,
    )
    _assert_bake_target_matches_world(prepared, expected_world)
    _assert_spine_vertex_bones(prepared)
    return prepared


def _assert_connected(anchor, other) -> None:
    """Exercise the real adapter routing from prepared objects to connected domain."""

    # Multi-object request settings retain the caller's standalone default. Preparation
    # owns the CONNECTED override that disables absolute world placement per object.
    sources = (
        A1MultiObjectSource(
            source_object=anchor.source_object,
            component_id="anchor",
            settings=replace(
                anchor.settings,
                use_world_location_for_main_bone=True,
            ),
        ),
        A1MultiObjectSource(
            source_object=other.source_object,
            component_id="other",
            settings=replace(
                other.settings,
                use_world_location_for_main_bone=True,
            ),
        ),
    )
    composition_settings = A1MultiObjectExportSettings(
        output_directory=_OUTPUT_ROOT,
        output_stem="origin_group",
        mode=A1MultiObjectMode.CONNECTED,
        connected_group_prefix="origin_group",
        anchor_component_id="anchor",
    )

    try:
        compose_a1_multi_object_document(
            sources,
            (other, anchor),
            composition_settings,
        )
    except ValueError as exc:
        if "does not match the prepared object's live source_object" not in str(exc):
            raise AssertionError(
                f"Swapped source/prepared pairing failed for the wrong reason: {exc}"
            ) from exc
    else:
        raise AssertionError("Swapped source/prepared pairing was accepted")

    result = compose_a1_multi_object_document(
        sources,
        (anchor, other),
        composition_settings,
    )
    if not isinstance(result, ConnectedGroupBuildResult):
        raise AssertionError(
            f"Connected composition returned unexpected type {type(result).__name__}"
        )

    placement_by_component = {
        placement.component_id: placement for placement in result.placements
    }
    if set(placement_by_component) != {"anchor", "other"}:
        raise AssertionError(
            "Connected composition did not preserve the component placement set"
        )
    if any(
        placement.placement_space
        is not ConnectedPlacementSpace.ANCHOR_RELATIVE_WORLD
        for placement in placement_by_component.values()
    ):
        raise AssertionError(
            "Object-bake prepared documents were routed to the wrong placement space"
        )

    bones = {bone.name: bone for bone in result.document.bones}
    anchor_local = calculate_a1_object_bake_main_position_pixels(
        anchor.source_snapshot,
        anchor.settings,
    )
    other_local = calculate_a1_object_bake_main_position_pixels(
        other.source_snapshot,
        other.settings,
    )
    relative_x = other.world_position[0] - anchor.world_position[0]
    relative_y = other.world_position[1] - anchor.world_position[1]
    expected_other = (
        other_local[0] + relative_x * result.uniform_scale,
        other_local[1] + relative_y * result.uniform_scale,
    )
    _assert_pair_close(
        (
            float(bones[anchor.rig.info.main_bone_name].x),
            float(bones[anchor.rig.info.main_bone_name].y),
        ),
        anchor_local,
        label="connected anchor main",
    )
    _assert_pair_close(
        (
            float(bones[other.rig.info.main_bone_name].x),
            float(bones[other.rig.info.main_bone_name].y),
        ),
        expected_other,
        label="connected other main",
    )


def main() -> None:
    if tuple(bpy.app.version) < (5, 2, 0):
        raise AssertionError(
            f"Blender 5.2+ is required, running {tuple(bpy.app.version)}"
        )

    initial_counts = _datablock_counts()
    initial_context = _context_signature()
    mirrored = None
    evaluated = None
    displace_texture = None
    try:
        mirrored = _create_source(
            "Spine2D_Origin_Mirrored",
            vertices=(
                (2.0, -3.0, 0.0),
                (5.0, -3.0, 0.0),
                (5.0, -1.0, 0.0),
                (2.0, -1.0, 0.0),
            ),
            matrix_world=(
                Matrix.Translation((7.0, -4.0, 2.0))
                @ Euler((0.0, 0.0, radians(31.0)), "XYZ").to_matrix().to_4x4()
                @ Matrix.Diagonal((-1.5, 0.75, 1.0, 1.0))
            ),
        )
        mirrored_world = _evaluated_world_positions(mirrored)
        mirrored_standalone = _prepare(
            mirrored,
            prefix="MirroredOrigin",
            source_mode=A1SourceGeometryMode.ORIGINAL,
            use_world_location=True,
            expected_world=mirrored_world,
        )
        if not any(
            issue.code == "MIRRORED_OBJECT_TRANSFORM"
            for issue in mirrored_standalone.warnings
        ):
            raise AssertionError("Mirrored Object transform warning was not emitted")

        evaluated = _create_source(
            "Spine2D_Origin_Evaluated",
            vertices=(
                (-4.0, 1.0, 0.0),
                (-1.0, 1.0, 0.0),
                (-1.0, 3.0, 0.0),
                (-4.0, 3.0, 0.0),
            ),
            matrix_world=(
                Matrix.Translation((-6.0, 8.0, -1.0))
                @ Euler(
                    (radians(9.0), radians(-13.0), radians(17.0)),
                    "XYZ",
                ).to_matrix().to_4x4()
                @ Matrix.Diagonal((1.25, 0.6, 1.4, 1.0))
            ),
        )
        displace_texture = bpy.data.textures.new(
            "Spine2D_Origin_Displace_Texture",
            type="BLEND",
        )
        displace_texture.progression = "LINEAR"
        modifier = evaluated.modifiers.new("ShiftBBoxX", "DISPLACE")
        modifier.texture = displace_texture
        modifier.texture_coords = "LOCAL"
        modifier.direction = "X"
        modifier.strength = 1.75
        modifier.mid_level = 0.0
        bpy.context.view_layer.update()

        evaluated_world = _evaluated_world_positions(evaluated)
        original_world = tuple(
            evaluated.matrix_world @ vertex.co for vertex in evaluated.data.vertices
        )
        if all(
            (current - original).length <= _EPSILON
            for current, original in zip(
                evaluated_world,
                original_world,
                strict=True,
            )
        ):
            raise AssertionError("Explicit Displace texture did not change geometry")

        evaluated_standalone = _prepare(
            evaluated,
            prefix="EvaluatedOrigin",
            source_mode=A1SourceGeometryMode.EVALUATED,
            use_world_location=True,
            expected_world=evaluated_world,
        )
        if int(evaluated_standalone.statistics["modifier_count"]) != 1:
            raise AssertionError("Evaluated modifier stack was not recorded")

        mirrored_connected = _prepare(
            mirrored,
            prefix="MirroredConnected",
            source_mode=A1SourceGeometryMode.ORIGINAL,
            use_world_location=False,
            expected_world=mirrored_world,
        )
        evaluated_connected = _prepare(
            evaluated,
            prefix="EvaluatedConnected",
            source_mode=A1SourceGeometryMode.EVALUATED,
            use_world_location=False,
            expected_world=evaluated_world,
        )
        _assert_connected(mirrored_connected, evaluated_connected)
    finally:
        _cleanup_object(evaluated)
        _cleanup_object(mirrored)
        _cleanup_texture(displace_texture)

    if any(
        obj.name.startswith("__") and "OriginTarget" in obj.name
        for obj in bpy.data.objects
    ):
        raise AssertionError("Temporary origin-offset target object leaked")

    final_counts = _datablock_counts()
    if final_counts != initial_counts:
        raise AssertionError(
            f"Blender datablock cleanup mismatch: initial={initial_counts}, "
            f"final={final_counts}"
        )
    final_context = _context_signature()
    if final_context != initial_context:
        raise AssertionError(
            f"Blender operator context was not restored: initial={initial_context}, "
            f"final={final_context}"
        )
    print("Blender 5.2 A1 Object Origin offset integration passed")


if __name__ == "__main__":
    main()
