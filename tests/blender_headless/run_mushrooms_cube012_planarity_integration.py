"""Real Blender 5.2 regression for the captured mushrooms ``Cube.012`` n-gon.

The real 0.90.0 scene failed before camera projection because one evaluated quad had a
maximum Newell-plane residue of ``7.565148185365435e-05`` at polygon scale
``0.13120450643194492``. The public multi-object Depth route must accept that
sub-per-mille warp while processing at least two real export sources, and must not mutate
objects, meshes, materials, selection, camera, Scene state, or temporary datablocks.
"""

from __future__ import annotations

from dataclasses import replace
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
    prepare_a1_multi_object,
)
from Blender_to_Spine2D_Mesh_Exporter.blender_adapter.a1_object_preparation import (  # noqa: E402
    PreparedDepthA1Object,
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
import run_depth_array_modifier_integration as depth_helpers  # noqa: E402


_OBJECT_NAME = "Cube.012"
_COMPONENT_ID = "object_1:Cube.012"
_PREFIX = "MushroomsCube012"

_CONTROL_OBJECT_NAME = "Cube012Control"
_CONTROL_COMPONENT_ID = "object_2:Cube012Control"
_CONTROL_PREFIX = "MushroomsCube012Control"

_MULTI_STEM = "MushroomsCube012Multi"
_SIDE_LENGTH = 0.09277534946637461
_WARP_HEIGHT = 0.00030260673225328884
_CAPTURED_MAXIMUM_PLANE_DISTANCE = 7.565148185365435e-05
_CAPTURED_POLYGON_SCALE = 0.13120450643194492
_CAPTURED_NORMALIZED_WARP = (
    _CAPTURED_MAXIMUM_PLANE_DISTANCE / _CAPTURED_POLYGON_SCALE
)

# Scale the complete object uniformly only to give the 128 px headless camera enough
# raster coverage. Uniform scaling preserves the normalized warp exactly.
_WORLD_SCALE = 20.0


def _create_source():
    """Create the exact local-space Cube.012 planarity fixture."""

    half = _SIDE_LENGTH / 2.0
    source = _create_mesh_object(
        _OBJECT_NAME,
        (
            (-half, -half, 0.0),
            (half, -half, 0.0),
            (half, half, _WARP_HEIGHT),
            (-half, half, 0.0),
        ),
        ((0, 1, 2, 3),),
    )
    source.location = (2.25, 0.0, 0.0)
    source.scale = (_WORLD_SCALE, _WORLD_SCALE, _WORLD_SCALE)
    material = depth_helpers._create_emission_material(
        "MushroomsCube012Material"
    )
    source.data.materials.append(material)
    return source, material


def _create_control_source():
    """Create the second public multi-object source required by the API contract."""

    source = _create_mesh_object(
        _CONTROL_OBJECT_NAME,
        (
            (-0.45, -0.45, 0.0),
            (0.45, -0.45, 0.0),
            (0.45, 0.45, 0.0),
            (-0.45, 0.45, 0.0),
        ),
        ((0, 1, 2, 3),),
    )
    source.location = (-0.25, 0.0, -0.15)
    material = depth_helpers._create_emission_material(
        "MushroomsCube012ControlMaterial"
    )
    source.data.materials.append(material)
    return source, material


def _settings(
    output_directory: Path,
    *,
    prefix: str,
):
    """Return independent Depth settings for one multi-object component."""

    if not isinstance(output_directory, Path):
        raise TypeError("output_directory must be pathlib.Path")
    if not isinstance(prefix, str) or not prefix.strip():
        raise ValueError("prefix must be a non-empty string")

    base = depth_helpers._settings(output_directory)
    return replace(
        base,
        prefix=prefix,
        output_stem=prefix,
        json_output_stem=prefix,
        export=replace(
            base.export,
            output_directory=output_directory,
        ),
    )


def _local_planarity_metrics(source) -> tuple[float, float]:
    """Measure the exact local-space metrics reported by the real traceback."""

    polygons = tuple(source.data.polygons)
    _assert(
        len(polygons) == 1,
        f"expected one Cube.012 polygon: {polygons}",
    )
    polygon = polygons[0]
    points = tuple(
        tuple(float(value) for value in source.data.vertices[index].co)
        for index in polygon.vertices
    )
    _assert(
        len(points) == 4,
        f"Cube.012 must remain a quad: {points}",
    )

    newell = [0.0, 0.0, 0.0]
    for index, current in enumerate(points):
        following = points[(index + 1) % len(points)]
        newell[0] += (
            (current[1] - following[1])
            * (current[2] + following[2])
        )
        newell[1] += (
            (current[2] - following[2])
            * (current[0] + following[0])
        )
        newell[2] += (
            (current[0] - following[0])
            * (current[1] + following[1])
        )

    magnitude = sqrt(
        sum(component * component for component in newell)
    )
    _assert(
        magnitude > 0.0,
        "Cube.012 Newell normal collapsed",
    )
    normal = tuple(component / magnitude for component in newell)
    centroid = tuple(
        sum(point[axis] for point in points) / float(len(points))
        for axis in range(3)
    )
    maximum_distance = max(
        abs(
            sum(
                (point[axis] - centroid[axis]) * normal[axis]
                for axis in range(3)
            )
        )
        for point in points
    )
    extents = tuple(
        max(point[axis] for point in points)
        - min(point[axis] for point in points)
        for axis in range(3)
    )
    polygon_scale = sqrt(
        sum(extent * extent for extent in extents)
    )
    return maximum_distance, polygon_scale


def _source_fingerprint(source) -> tuple[object, ...]:
    """Capture object and source Mesh state without evaluating or mutating it."""

    return (
        source.name,
        source.data.name,
        tuple(
            tuple(float(value) for value in row)
            for row in source.matrix_world
        ),
        tuple(
            tuple(float(value) for value in vertex.co)
            for vertex in source.data.vertices
        ),
        tuple(
            tuple(int(value) for value in polygon.vertices)
            for polygon in source.data.polygons
        ),
        tuple(
            material.name if material is not None else None
            for material in source.data.materials
        ),
        tuple(float(value) for value in source.location),
        tuple(float(value) for value in source.scale),
        tuple(float(value) for value in source.rotation_euler),
    )


def _prepared_by_component(prepared_multi) -> dict[str, PreparedDepthA1Object]:
    """Resolve prepared results by public component identity, never tuple position."""

    result: dict[str, PreparedDepthA1Object] = {}
    for source, prepared in zip(
        prepared_multi.sources,
        prepared_multi.objects,
        strict=True,
    ):
        _assert(
            isinstance(prepared, PreparedDepthA1Object),
            f"component {source.component_id} returned {type(prepared)!r}",
        )
        _assert(
            source.component_id not in result,
            f"duplicate prepared component {source.component_id}",
        )
        result[source.component_id] = prepared
    return result


def _run() -> None:
    _clear_scene()
    _purge_orphan_scene_data()
    _configure_scene()
    bpy.context.scene.cycles.samples = 1
    bpy.context.scene.render.film_transparent = True
    bpy.context.scene.frame_set(1)

    source, material = _create_source()
    control, control_material = _create_control_source()
    camera = depth_helpers._create_orthographic_camera(
        "MushroomsCube012Camera"
    )
    sentinel = _create_sentinel()
    sentinel.location = (20.0, 0.0, 0.0)
    _activate_only(sentinel)
    source.select_set(False)
    control.select_set(False)
    bpy.context.view_layer.update()

    maximum_distance, polygon_scale = _local_planarity_metrics(source)
    _assert(
        abs(
            maximum_distance
            - _CAPTURED_MAXIMUM_PLANE_DISTANCE
        )
        <= 1.0e-11,
        f"Cube.012 maximum distance fixture drifted: {maximum_distance}",
    )
    _assert(
        abs(polygon_scale - _CAPTURED_POLYGON_SCALE) <= 1.0e-8,
        f"Cube.012 polygon scale fixture drifted: {polygon_scale}",
    )
    _assert(
        abs(
            maximum_distance / polygon_scale
            - _CAPTURED_NORMALIZED_WARP
        )
        <= 1.0e-10,
        "Cube.012 normalized warp fixture drifted",
    )

    context_before = _capture_context()
    source_before = _source_fingerprint(source)
    control_before = _source_fingerprint(control)
    camera_before = depth_helpers._camera_fingerprint(camera)
    materials_before = (
        _material_fingerprint(material),
        _material_fingerprint(control_material),
    )
    temporary_before = _temporary_datablock_names()
    frame_before = int(bpy.context.scene.frame_current)

    with tempfile.TemporaryDirectory(
        prefix="spine2d_mushrooms_cube012_"
    ) as directory:
        output_directory = Path(directory)
        sources = (
            A1MultiObjectSource(
                source_object=source,
                component_id=_COMPONENT_ID,
                settings=_settings(
                    output_directory,
                    prefix=_PREFIX,
                ),
            ),
            A1MultiObjectSource(
                source_object=control,
                component_id=_CONTROL_COMPONENT_ID,
                settings=_settings(
                    output_directory,
                    prefix=_CONTROL_PREFIX,
                ),
            ),
        )
        _assert(
            len(sources) == 2,
            "Cube.012 regression must exercise the public two-object contract",
        )

        prepared_multi = prepare_a1_multi_object(
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
            not tuple(
                path
                for path in output_directory.rglob("*")
                if path.is_file()
            ),
            "Cube.012 preparation wrote output files",
        )

    _assert(
        len(prepared_multi.objects) == 2,
        f"expected two prepared objects: {prepared_multi.objects}",
    )
    prepared_by_component = _prepared_by_component(prepared_multi)
    _assert(
        set(prepared_by_component)
        == {_COMPONENT_ID, _CONTROL_COMPONENT_ID},
        f"prepared component set changed: {prepared_by_component}",
    )

    prepared = prepared_by_component[_COMPONENT_ID]
    control_prepared = prepared_by_component[_CONTROL_COMPONENT_ID]

    _assert(
        int(
            prepared.statistics[
                "depth_projection_source_triangle_count"
            ]
        )
        == 2,
        "Cube.012 n-gon did not produce two source triangles",
    )
    _assert(
        len(prepared.depth_parallax_package.front_face_indices) == 2,
        "Cube.012 camera-visible front did not retain both triangles",
    )
    _assert(
        not prepared.depth_parallax_package.reserve_surfaces,
        "Cube.012 zero-horizon unexpectedly created reserve surfaces",
    )

    _assert(
        int(
            control_prepared.statistics[
                "depth_projection_source_triangle_count"
            ]
        )
        == 2,
        "control n-gon did not produce two source triangles",
    )
    _assert(
        not control_prepared.depth_parallax_package.reserve_surfaces,
        "control zero-horizon unexpectedly created reserve surfaces",
    )

    _assert(
        _capture_context() == context_before,
        "selection or active context changed",
    )
    _assert(
        _source_fingerprint(source) == source_before,
        "Cube.012 source changed",
    )
    _assert(
        _source_fingerprint(control) == control_before,
        "control source changed",
    )
    _assert(
        depth_helpers._camera_fingerprint(camera) == camera_before,
        "active camera changed",
    )
    _assert(
        (
            _material_fingerprint(material),
            _material_fingerprint(control_material),
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
        "[MUSHROOMS-CUBE012-PLANARITY] PASS "
        "object=Cube.012 "
        "maximum_plane_distance=7.565148185365435e-05 "
        "polygon_scale=0.13120450643194492 "
        "normalized_warp=0.0005765920996996728 "
        "triangles=2 sources=2 pipeline=public-multi-object"
    )


def main() -> None:
    try:
        _run()
    except Exception:
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
