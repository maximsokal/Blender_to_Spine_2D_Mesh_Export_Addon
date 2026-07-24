"""Blender 5.2 headless integration for Object Origin placement through A1.

Run from the repository root with Blender 5.2 or newer::

    blender --background --factory-startup --python \
        tests/blender_headless/run_a1_object_origin_offset_integration.py
"""

from __future__ import annotations

from math import radians
from pathlib import Path
import sys

import bpy
from mathutils import Euler, Matrix, Vector


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Blender_to_Spine2D_Mesh_Exporter.application import (  # noqa: E402
    A1SingleObjectExportSettings,
    A1SourceGeometryMode,
    ExportSettings,
    calculate_a1_object_bake_main_position_pixels,
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
from Blender_to_Spine2D_Mesh_Exporter.domain.spine.connected_group_assembly import (  # noqa: E402
    build_connected_group_document,
)
from Blender_to_Spine2D_Mesh_Exporter.domain.spine.connected_group_contracts import (  # noqa: E402
    ConnectedGroupSettings,
    ConnectedObjectDocument,
)


_EPSILON = 2.1e-2
_TEXTURE_SIZE = 100
_OUTPUT_ROOT = Path(bpy.app.tempdir) / "spine2d-origin-offset-integration"


def _assert_close(first: Vector, second: Vector, *, label: str) -> None:
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


def _create_source(
    name: str,
    *,
    vertices: tuple[tuple[float, float, float], ...],
    matrix_world: Matrix,
) -> bpy.types.Object:
    mesh = bpy.data.meshes.new(f"{name}_Mesh")
    source = bpy.data.objects.new(name, mesh)
    bpy.context.scene.collection.objects.link(source)
    mesh.from_pydata(
        vertices,
        (),
        (
            (0, 1, 2),
            (0, 2, 3),
        ),
    )
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


def _expected_world_by_source_index(
    positions: tuple[Vector, ...],
) -> dict[int, Vector]:
    return {index: position.copy() for index, position in enumerate(positions)}


def _assert_bake_target_matches_evaluated_geometry(
    prepared,
    expected_world_positions: tuple[Vector, ...],
) -> None:
    expected_by_source = _expected_world_by_source_index(expected_world_positions)
    target_snapshot = prepared.bake_target_snapshot
    with temporary_mesh_object(
        target_snapshot,
        scene=bpy.context.scene,
        name_prefix=f"__{prepared.prefix}_OriginTarget",
    ) as temporary:
        target = temporary.object
        if len(target.data.vertices) != len(target_snapshot.vertices):
            raise AssertionError("Temporary bake target vertex order changed")
        for snapshot_vertex in target_snapshot.vertices:
            source_index = snapshot_vertex.source_id.vertex_index
            expected = expected_by_source.get(source_index)
            if expected is None:
                raise AssertionError(
                    f"Bake target references unknown source vertex {source_index}"
                )
            actual = target.matrix_world @ target.data.vertices[
                snapshot_vertex.id.index
            ].co
            _assert_close(
                expected,
                actual,
                label=(
                    f"{prepared.prefix} bake target source vertex "
                    f"{source_index}/local {snapshot_vertex.id.index}"
                ),
            )


def _assert_spine_vertex_bones_reconstruct_world_xy(prepared) -> None:
    document_bones = {bone.name: bone for bone in prepared.document.bones}
    main = document_bones[prepared.rig.info.main_bone_name]
    matrix = prepared.source_snapshot.world_matrix
    translation_x = (
        float(matrix[3])
        if prepared.settings.use_world_location_for_main_bone
        else 0.0
    )
    translation_y = (
        float(matrix[7])
        if prepared.settings.use_world_location_for_main_bone
        else 0.0
    )
    scale = prepared.rig.info.uniform_scale

    if len(prepared.document_assembly.projections) != len(
        prepared.uv_regions.snapshots
    ):
        raise AssertionError("Projection count does not match prepared UV regions")

    for region, projection in zip(
        prepared.uv_regions.snapshots,
        prepared.document_assembly.projections,
        strict=True,
    ):
        vertex_by_id = region.vertex_by_id()
        request = projection.request
        for attachment_vertex, key in zip(
            request.vertices,
            projection.ordered_vertex_keys,
            strict=True,
        ):
            source_vertex = vertex_by_id[key.vertex_id]
            vertex_bone_name = prepared.rig.profile.vertex_bone(
                request.vertex_prefix,
                attachment_vertex.index,
            )
            vertex_bone = document_bones[vertex_bone_name]
            if float(vertex_bone.x) != round(
                float(attachment_vertex.bone_position_pixels[0]),
                2,
            ):
                raise AssertionError(
                    f"{prepared.prefix} vertex bone X does not match projection request"
                )
            if float(vertex_bone.y) != round(
                float(attachment_vertex.bone_position_pixels[1]),
                2,
            ):
                raise AssertionError(
                    f"{prepared.prefix} vertex bone Y does not match projection request"
                )
            actual = (
                float(main.x)
                + float(attachment_vertex.bone_position_pixels[0]),
                float(main.y)
                + float(attachment_vertex.bone_position_pixels[1]),
            )
            expected = (
                (translation_x + float(source_vertex.position[0])) * scale,
                translation_y * scale - float(source_vertex.position[1]) * scale,
            )
            _assert_pair_close(
                actual,
                expected,
                label=(
                    f"{prepared.prefix} Spine vertex bone "
                    f"{attachment_vertex.index}"
                ),
            )


def _prepare_and_assert(
    source: bpy.types.Object,
    *,
    prefix: str,
    source_mode: A1SourceGeometryMode,
    use_world_location: bool,
    expected_world_positions: tuple[Vector, ...],
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
    _assert_bake_target_matches_evaluated_geometry(
        prepared,
        expected_world_positions,
    )
    _assert_spine_vertex_bones_reconstruct_world_xy(prepared)
    return prepared


def _assert_connected_origin_conventions(anchor, other) -> None:
    result = build_connected_group_document(
        (
            ConnectedObjectDocument(
                component_id="anchor",
                prefix=anchor.prefix,
                document=anchor.document,
                world_position=anchor.world_position,
            ),
            ConnectedObjectDocument(
                component_id="other",
                prefix=other.prefix,
                document=other.document,
                world_position=other.world_position,
            ),
        ),
        ConnectedGroupSettings(
            texture_width=_TEXTURE_SIZE,
            texture_height=_TEXTURE_SIZE,
            group_prefix="origin_group",
            anchor_component_id="anchor",
        ),
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
    expected_anchor = anchor_local
    expected_other = (
        other_local[0] + relative_x * result.uniform_scale,
        other_local[1] + relative_y * result.uniform_scale,
    )
    anchor_main = bones[anchor.rig.info.main_bone_name]
    other_main = bones[other.rig.info.main_bone_name]
    _assert_pair_close(
        (float(anchor_main.x), float(anchor_main.y)),
        expected_anchor,
        label="connected anchor main",
    )
    _assert_pair_close(
        (float(other_main.x), float(other_main.y)),
        expected_other,
        label="connected other main",
    )


def main() -> None:
    if tuple(bpy.app.version) < (5, 2, 0):
        raise AssertionError(
            f"Blender 5.2+ is required, running {tuple(bpy.app.version)}"
        )

    initial_object_count = len(bpy.data.objects)
    initial_mesh_count = len(bpy.data.meshes)
    mirrored = None
    evaluated = None
    try:
        mirrored_matrix = (
            Matrix.Translation((7.0, -4.0, 2.0))
            @ Euler((0.0, 0.0, radians(31.0)), "XYZ").to_matrix().to_4x4()
            @ Matrix.Diagonal((-1.5, 0.75, 1.0, 1.0))
        )
        mirrored = _create_source(
            "Spine2D_Origin_Mirrored",
            vertices=(
                (2.0, -3.0, 0.0),
                (5.0, -3.0, 0.0),
                (5.0, -1.0, 0.0),
                (2.0, -1.0, 0.0),
            ),
            matrix_world=mirrored_matrix,
        )
        mirrored_world = _evaluated_world_positions(mirrored)
        mirrored_prepared = _prepare_and_assert(
            mirrored,
            prefix="MirroredOrigin",
            source_mode=A1SourceGeometryMode.ORIGINAL,
            use_world_location=True,
            expected_world_positions=mirrored_world,
        )
        if not any(
            issue.code == "MIRRORED_OBJECT_TRANSFORM"
            for issue in mirrored_prepared.warnings
        ):
            raise AssertionError("Mirrored Object transform warning was not emitted")

        evaluated_matrix = (
            Matrix.Translation((-6.0, 8.0, -1.0))
            @ Euler((radians(9.0), radians(-13.0), radians(17.0)), "XYZ")
            .to_matrix()
            .to_4x4()
            @ Matrix.Diagonal((1.25, 0.6, 1.4, 1.0))
        )
        evaluated = _create_source(
            "Spine2D_Origin_Evaluated",
            vertices=(
                (-4.0, 1.0, 0.0),
                (-1.0, 1.0, 0.0),
                (-1.0, 3.0, 0.0),
                (-4.0, 3.0, 0.0),
            ),
            matrix_world=evaluated_matrix,
        )
        modifier = evaluated.modifiers.new("ShiftBBoxX", "DISPLACE")
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
            raise AssertionError("Displace modifier did not change evaluated geometry")

        evaluated_prepared = _prepare_and_assert(
            evaluated,
            prefix="EvaluatedOrigin",
            source_mode=A1SourceGeometryMode.EVALUATED,
            use_world_location=True,
            expected_world_positions=evaluated_world,
        )
        if int(evaluated_prepared.statistics["modifier_count"]) != 1:
            raise AssertionError("Evaluated modifier stack was not recorded")

        mirrored_connected = _prepare_and_assert(
            mirrored,
            prefix="MirroredConnected",
            source_mode=A1SourceGeometryMode.ORIGINAL,
            use_world_location=False,
            expected_world_positions=mirrored_world,
        )
        evaluated_connected = _prepare_and_assert(
            evaluated,
            prefix="EvaluatedConnected",
            source_mode=A1SourceGeometryMode.EVALUATED,
            use_world_location=False,
            expected_world_positions=evaluated_world,
        )
        _assert_connected_origin_conventions(
            mirrored_connected,
            evaluated_connected,
        )
    finally:
        _cleanup_object(evaluated)
        _cleanup_object(mirrored)

    if any(
        obj.name.startswith("__") and "OriginTarget" in obj.name
        for obj in bpy.data.objects
    ):
        raise AssertionError("Temporary origin-offset target object leaked")
    if len(bpy.data.objects) != initial_object_count:
        raise AssertionError("Object cleanup did not restore initial count")
    if len(bpy.data.meshes) != initial_mesh_count:
        raise AssertionError("Mesh cleanup did not restore initial count")
    print("Blender 5.2 A1 Object Origin offset integration passed")


if __name__ == "__main__":
    main()
