"""Real grenade.blend end-to-end Normal/UV regression across every Mesh object.

This gate is intentionally tied to the artist-authored fixture that exposed the Bump
Only capability bug, the Blender-vs-domain polygon tessellation mismatch, and Blender's
zero-area loop-triangle output for ``Cylinder.002``. It must not synthesize replacement
geometry or materials.

The test first proves that source polygon 26 is recovered through Blender's own
``Mesh.loop_triangles`` while zero-area tessellation triangles are discarded and
``SourceFaceId``/``SourceLoopId`` lineage is preserved. It then exports every Mesh object
in the loaded grenade scene through the same standalone multi-object Normal / UV Segments
route used by the UI.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from math import sqrt
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
    export_a1_multi_object,
    prepare_a1_object,
    read_source_mesh_snapshot,
)
from Blender_to_Spine2D_Mesh_Exporter.blender_adapter.blender_loop_triangulation import (  # noqa: E402
    _ZERO_AREA_EPSILON,
)
from run_bake_integration import PNG_SIGNATURE, _assert  # noqa: E402
from run_grenade_bump_displacement_normal_uv_integration import (  # noqa: E402
    _assert_capability_route,
    _require_loaded_blend,
    _require_material,
    _require_source_object,
    _settings as _cube_settings,
)


_EXPECTED_TRIANGULATION_OBJECT = "Cylinder.002"
_EXPECTED_SOURCE_FACE_INDEX = 26
_MINIMUM_MESH_OBJECTS = 2


def _parse_arguments() -> argparse.Namespace:
    arguments = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser(
        description="Run full real grenade.blend Normal/UV multi-object export."
    )
    parser.add_argument(
        "--expected-blend",
        required=True,
        help="Exact grenade.blend path Blender must already have loaded.",
    )
    return parser.parse_args(arguments)


def _mesh_objects(scene) -> tuple:
    values = tuple(
        sorted(
            (
                obj
                for obj in scene.objects
                if getattr(obj, "type", None) == "MESH"
                and getattr(obj, "data", None) is not None
            ),
            key=lambda obj: obj.name_full,
        )
    )
    _assert(
        len(values) >= _MINIMUM_MESH_OBJECTS,
        f"grenade fixture contains too few Mesh objects: {len(values)}",
    )
    names = tuple(obj.name_full for obj in values)
    _assert(
        _EXPECTED_TRIANGULATION_OBJECT in names,
        f"grenade fixture lost {_EXPECTED_TRIANGULATION_OBJECT!r}: {names!r}",
    )
    return values


def _object_fingerprint(obj) -> tuple:
    uv_layers = getattr(obj.data, "uv_layers", None)
    layers = tuple(layer.name for layer in uv_layers) if uv_layers is not None else ()
    active = None
    if uv_layers is not None and getattr(uv_layers, "active", None) is not None:
        active = uv_layers.active.name
    return (
        obj.name_full,
        obj.data.name_full,
        tuple(tuple(float(value) for value in row) for row in obj.matrix_world),
        tuple(
            None if slot.material is None else slot.material.name_full
            for slot in obj.material_slots
        ),
        layers,
        active,
        len(obj.data.vertices),
        len(obj.data.edges),
        len(obj.data.polygons),
    )


def _scene_fingerprint(scene, objects: tuple) -> tuple:
    return (
        tuple(_object_fingerprint(obj) for obj in objects),
        None if scene.camera is None else scene.camera.name_full,
        int(scene.frame_current),
        str(scene.render.engine),
        tuple(sorted(obj.name_full for obj in bpy.context.selected_objects)),
        (
            None
            if bpy.context.view_layer.objects.active is None
            else bpy.context.view_layer.objects.active.name_full
        ),
    )


def _datablock_fingerprint() -> tuple:
    return (
        tuple(sorted(item.name_full for item in bpy.data.objects)),
        tuple(sorted(item.name_full for item in bpy.data.meshes)),
        tuple(sorted(item.name_full for item in bpy.data.materials)),
        tuple(sorted(item.name_full for item in bpy.data.images)),
    )


def _loop_triangle_cross_length(mesh, triangle) -> float:
    """Return twice the geometric area of one Blender loop triangle."""

    loop_indices = tuple(int(value) for value in triangle.loops)
    _assert(len(loop_indices) == 3, f"unexpected Blender triangle loops: {loop_indices}")
    positions = []
    for loop_index in loop_indices:
        vertex_index = int(mesh.loops[loop_index].vertex_index)
        co = mesh.vertices[vertex_index].co
        positions.append(tuple(float(co[index]) for index in range(3)))
    first, second, third = positions
    ab = tuple(second[index] - first[index] for index in range(3))
    ac = tuple(third[index] - first[index] for index in range(3))
    cross = (
        ab[1] * ac[2] - ab[2] * ac[1],
        ab[2] * ac[0] - ab[0] * ac[2],
        ab[0] * ac[1] - ab[1] * ac[0],
    )
    return sqrt(sum(component * component for component in cross))


def _assert_blender_tessellation_fallback(source) -> tuple[int, int, int]:
    mesh = source.data
    _assert(
        len(mesh.polygons) > _EXPECTED_SOURCE_FACE_INDEX,
        f"{source.name_full} lost polygon {_EXPECTED_SOURCE_FACE_INDEX}",
    )
    source_polygon = mesh.polygons[_EXPECTED_SOURCE_FACE_INDEX]
    source_corner_count = int(source_polygon.loop_total)
    _assert(
        source_corner_count >= 4,
        "expected grenade regression polygon to remain non-triangular; "
        f"corners={source_corner_count}",
    )

    mesh.calc_loop_triangles()
    raw_source_triangles = tuple(
        triangle
        for triangle in mesh.loop_triangles
        if int(triangle.polygon_index) == _EXPECTED_SOURCE_FACE_INDEX
    )
    expected_raw_count = source_corner_count - 2
    _assert(
        len(raw_source_triangles) == expected_raw_count,
        "Blender raw tessellation no longer provides source-face N-2 coverage: "
        f"face={_EXPECTED_SOURCE_FACE_INDEX}, corners={source_corner_count}, "
        f"expected={expected_raw_count}, actual={len(raw_source_triangles)}",
    )
    raw_areas = tuple(
        _loop_triangle_cross_length(mesh, triangle)
        for triangle in raw_source_triangles
    )
    degenerate_count = sum(
        1 for value in raw_areas if value <= _ZERO_AREA_EPSILON
    )
    non_degenerate_count = sum(
        1 for value in raw_areas if value > _ZERO_AREA_EPSILON
    )
    _assert(
        degenerate_count >= 1,
        "grenade regression polygon no longer contains a zero-area Blender triangle; "
        f"areas={raw_areas!r}",
    )
    _assert(
        non_degenerate_count >= 1,
        "grenade regression polygon became wholly degenerate; "
        f"areas={raw_areas!r}",
    )

    snapshot = read_source_mesh_snapshot(
        source,
        source_object_id=source.name_full,
        snapshot_id=f"{source.name_full}:grenade-real-fixture",
    )
    _assert(
        snapshot.snapshot_id.endswith(":blender-fallback"),
        "Cylinder.002 no longer exercises Blender loop-triangle fallback; "
        f"snapshot_id={snapshot.snapshot_id!r}",
    )
    _assert(
        all(len(face.loop_ids) == 3 for face in snapshot.faces),
        "Blender fallback snapshot contains a non-triangle face",
    )
    retained_source_triangles = tuple(
        face
        for face in snapshot.faces
        if face.source_id.face_index == _EXPECTED_SOURCE_FACE_INDEX
    )
    _assert(
        len(retained_source_triangles) == non_degenerate_count,
        "Blender fallback did not retain exactly the non-degenerate source triangles: "
        f"face={_EXPECTED_SOURCE_FACE_INDEX}, raw={len(raw_source_triangles)}, "
        f"degenerate={degenerate_count}, non_degenerate={non_degenerate_count}, "
        f"retained={len(retained_source_triangles)}",
    )
    source_loop_ids = tuple(
        loop.source_id
        for loop in snapshot.loops
        if loop.source_id.face_index == _EXPECTED_SOURCE_FACE_INDEX
    )
    _assert(source_loop_ids, "fallback lost SourceLoopId lineage for source face 26")
    _assert(
        all(item.object_id == source.name_full for item in source_loop_ids),
        "fallback SourceLoopId object lineage changed",
    )
    return source_corner_count, non_degenerate_count, degenerate_count


def _component_token(index: int) -> str:
    return f"grenade_{index:03d}"


def _sources(objects: tuple, output_directory: Path) -> tuple[A1MultiObjectSource, ...]:
    values: list[A1MultiObjectSource] = []
    for index, obj in enumerate(objects):
        token = _component_token(index)
        base = _cube_settings(output_directory)
        settings = replace(
            base,
            prefix=token,
            output_stem=token,
            json_output_stem=None,
        )
        values.append(
            A1MultiObjectSource(
                source_object=obj,
                component_id=token,
                animation_namespace=token,
                settings=settings,
            )
        )
    return tuple(values)


def _assert_multi_outputs(result, expected_texture_count: int) -> tuple[Path, ...]:
    _assert(
        bool(result.success),
        "real grenade multi-object Normal/UV export failed: "
        f"issues={result.issues!r}, statistics={dict(result.statistics)!r}",
    )
    outputs = tuple(Path(path).resolve(strict=False) for path in result.output_files)
    json_files = tuple(path for path in outputs if path.suffix.lower() == ".json")
    png_files = tuple(path for path in outputs if path.suffix.lower() == ".png")
    _assert(len(json_files) == 1, f"expected one multi-object JSON: {json_files!r}")
    _assert(
        len(png_files) == expected_texture_count,
        "grenade multi-object texture count differs from Mesh object count: "
        f"expected={expected_texture_count}, actual={len(png_files)}",
    )
    _assert(
        all(path.is_file() and path.stat().st_size > 8 for path in outputs),
        f"grenade multi-object export contains missing/empty files: {outputs!r}",
    )
    for path in png_files:
        _assert(
            path.read_bytes().startswith(PNG_SIGNATURE),
            f"grenade output is not PNG: {path}",
        )
    document = json.loads(json_files[0].read_text(encoding="utf-8"))
    _assert(isinstance(document, dict), "grenade multi-object JSON root must be mapping")
    _assert(bool(document.get("bones")), "grenade multi-object JSON contains no bones")
    _assert(bool(document.get("skins")), "grenade multi-object JSON contains no skins")
    return outputs


def _run(expected_blend: str) -> None:
    loaded = _require_loaded_blend(expected_blend)
    scene = bpy.context.scene
    _assert(scene.camera is not None, "grenade fixture must provide an active scene camera")
    objects = _mesh_objects(scene)
    triangulation_source = bpy.data.objects.get(_EXPECTED_TRIANGULATION_OBJECT)
    _assert(triangulation_source is not None, "Cylinder.002 lookup failed")

    before = _scene_fingerprint(scene, objects)
    datablocks_before = _datablock_fingerprint()
    (
        source_corner_count,
        fallback_triangle_count,
        degenerate_triangle_count,
    ) = _assert_blender_tessellation_fallback(triangulation_source)

    # Keep the earlier material regression inside the same real fixture gate. The full
    # multi-object export below then proves that capability + tessellation coexist.
    cube = _require_source_object()
    material = _require_material(cube)
    _assert_capability_route(cube, material)

    with tempfile.TemporaryDirectory(prefix="spine2d_grenade_full_") as temp_root:
        output_directory = Path(temp_root).resolve(strict=False)

        # Prepare the exact object/face that previously failed before running the full
        # scene. This localizes any future geometry regression immediately.
        cylinder_settings = replace(
            _cube_settings(output_directory),
            prefix="Grenade_Cylinder_002",
            output_stem="Grenade_Cylinder_002",
            json_output_stem="Grenade_Cylinder_002",
        )
        cylinder_prepared = prepare_a1_object(
            triangulation_source,
            cylinder_settings,
            context=bpy.context,
            scene=scene,
        )
        _assert(
            all(len(face.loop_ids) == 3 for face in cylinder_prepared.source_snapshot.faces),
            "prepared Cylinder.002 source geometry is not fully triangulated",
        )

        sources = _sources(objects, output_directory)
        multi_settings = A1MultiObjectExportSettings(
            output_directory=output_directory,
            output_stem="Grenade_Full_Normal_UV",
            mode=A1MultiObjectMode.STANDALONE,
        )
        result = export_a1_multi_object(
            sources,
            multi_settings,
            context=bpy.context,
            scene=scene,
        )
        outputs = _assert_multi_outputs(result, len(objects))

    _assert(
        _scene_fingerprint(scene, objects) == before,
        "full grenade export changed source object/scene/context state",
    )
    _assert(
        _datablock_fingerprint() == datablocks_before,
        "full grenade export leaked or removed Blender datablocks",
    )

    print(
        "[GRENADE-FULL-NORMAL-UV] PASS "
        f"blend={loaded} mesh_objects={len(objects)} "
        f"problem_object={_EXPECTED_TRIANGULATION_OBJECT!r} "
        f"source_face={_EXPECTED_SOURCE_FACE_INDEX} "
        f"source_corners={source_corner_count} "
        f"retained_blender_triangles={fallback_triangle_count} "
        f"filtered_zero_area_triangles={degenerate_triangle_count} "
        f"outputs={len(outputs)} source=unchanged",
        flush=True,
    )


def main() -> None:
    arguments = _parse_arguments()
    _run(arguments.expected_blend)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise
