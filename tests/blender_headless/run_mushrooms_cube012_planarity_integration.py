"""Real Blender 5.2 regressions for captured mushrooms n-gon planarity failures.

The real 0.90.0 scene exposed two distinct evaluated quads:

* ``Cube.012``: maximum Newell-plane residue ``7.565148185365435e-05`` at
  polygon scale ``0.13120450643194492``;
* ``Plane.008``: maximum residue ``9.477343601658832e-05`` at the much smaller
  polygon scale ``0.023314310303391664``.

The public two-object Depth route must accept both bounded evaluation residues without
mutating objects, meshes, materials, selection, camera, Scene state, or temporary
Blender datablocks.
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


_CUBE_OBJECT_NAME = "Cube.012"
_CUBE_COMPONENT_ID = "object_1:Cube.012"
_CUBE_PREFIX = "MushroomsCube012"
_CUBE_SIDE_LENGTH = 0.09277534946637461
_CUBE_WARP_HEIGHT = 0.00030260673225328884
_CUBE_CAPTURED_MAXIMUM_PLANE_DISTANCE = 7.565148185365435e-05
_CUBE_CAPTURED_POLYGON_SCALE = 0.13120450643194492
_CUBE_CAPTURED_NORMALIZED_WARP = (
    _CUBE_CAPTURED_MAXIMUM_PLANE_DISTANCE
    / _CUBE_CAPTURED_POLYGON_SCALE
)

_PLANE_OBJECT_NAME = "Plane.008"
_PLANE_COMPONENT_ID = "object_2:Plane.008"
_PLANE_PREFIX = "MushroomsPlane008Small"
_PLANE_SIDE_LENGTH = 0.0164835268501562
_PLANE_WARP_HEIGHT = 0.000379143881919382
_PLANE_CAPTURED_MAXIMUM_PLANE_DISTANCE = 9.477343601658832e-05
_PLANE_CAPTURED_POLYGON_SCALE = 0.023314310303391664
_PLANE_CAPTURED_NORMALIZED_WARP = (
    _PLANE_CAPTURED_MAXIMUM_PLANE_DISTANCE
    / _PLANE_CAPTURED_POLYGON_SCALE
)

_MULTI_STEM = "MushroomsCapturedPlanarityMulti"
_CAMERA_ORTHO_SCALE = 0.35


def _create_warped_quad(
    *,
    object_name: str,
    side_length: float,
    warp_height: float,
    location_x: float,
    material_name: str,
):
    """Create one local-space square with one lifted corner and no modifiers."""

    if not isinstance(object_name, str) or not object_name.strip():
        raise ValueError("object_name must be a non-empty string")
    if isinstance(side_length, bool) or not isinstance(side_length, (int, float)):
        raise TypeError("side_length must be numeric")
    if isinstance(warp_height, bool) or not isinstance(warp_height, (int, float)):
        raise TypeError("warp_height must be numeric")
    if isinstance(location_x, bool) or not isinstance(location_x, (int, float)):
        raise TypeError("location_x must be numeric")
    if not isinstance(material_name, str) or not material_name.strip():
        raise ValueError("material_name must be a non-empty string")

    resolved_side = float(side_length)
    if resolved_side <= 0.0:
        raise ValueError("side_length must be positive")

    half = resolved_side / 2.0
    source = _create_mesh_object(
        object_name,
        (
            (-half, -half, 0.0),
            (half, -half, 0.0),
            (half, half, float(warp_height)),
            (-half, half, 0.0),
        ),
        ((0, 1, 2, 3),),
    )
    source.location = (float(location_x), 0.0, 0.0)
    material = depth_helpers._create_emission_material(material_name)
    source.data.materials.append(material)
    return source, material


def _create_cube012_source():
    return _create_warped_quad(
        object_name=_CUBE_OBJECT_NAME,
        side_length=_CUBE_SIDE_LENGTH,
        warp_height=_CUBE_WARP_HEIGHT,
        location_x=2.17,
        material_name="MushroomsCube012Material",
    )


def _create_plane008_source():
    return _create_warped_quad(
        object_name=_PLANE_OBJECT_NAME,
        side_length=_PLANE_SIDE_LENGTH,
        warp_height=_PLANE_WARP_HEIGHT,
        location_x=2.34,
        material_name="MushroomsPlane008SmallMaterial",
    )


def _settings(
    output_directory: Path,
    *,
    prefix: str,
):
    """Return independent zero-horizon Depth settings for one component."""

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


def _local_planarity_metrics(
    source,
    *,
    label: str,
) -> tuple[float, float]:
    """Measure the local Newell-plane distance and bounding-box diagonal."""

    if not isinstance(label, str) or not label.strip():
        raise ValueError("label must be a non-empty string")

    polygons = tuple(source.data.polygons)
    _assert(
        len(polygons) == 1,
        f"expected one {label} polygon: {polygons}",
    )
    polygon = polygons[0]
    points = tuple(
        tuple(float(value) for value in source.data.vertices[index].co)
        for index in polygon.vertices
    )
    _assert(
        len(points) == 4,
        f"{label} must remain a quad: {points}",
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
        f"{label} Newell normal collapsed",
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


def _assert_metrics(
    source,
    *,
    label: str,
    expected_maximum_distance: float,
    expected_scale: float,
    expected_normalized_warp: float,
) -> None:
    maximum_distance, polygon_scale = _local_planarity_metrics(
        source,
        label=label,
    )
    _assert(
        abs(maximum_distance - expected_maximum_distance) <= 1.0e-11,
        f"{label} maximum distance fixture drifted: {maximum_distance}",
    )
    _assert(
        abs(polygon_scale - expected_scale) <= 1.0e-8,
        f"{label} polygon scale fixture drifted: {polygon_scale}",
    )
    _assert(
        abs(
            maximum_distance / polygon_scale
            - expected_normalized_warp
        )
        <= 1.0e-10,
        f"{label} normalized warp fixture drifted",
    )


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


def _assert_prepared_quad(
    prepared: PreparedDepthA1Object,
    *,
    label: str,
) -> None:
    _assert(
        int(
            prepared.statistics[
                "depth_projection_source_triangle_count"
            ]
        )
        == 2,
        f"{label} n-gon did not produce two source triangles",
    )
    _assert(
        len(prepared.depth_parallax_package.front_face_indices) == 2,
        f"{label} camera-visible front did not retain both triangles",
    )
    _assert(
        not prepared.depth_parallax_package.reserve_surfaces,
        f"{label} zero-horizon unexpectedly created reserve surfaces",
    )


def _run() -> None:
    _clear_scene()
    _purge_orphan_scene_data()
    _configure_scene()
    bpy.context.scene.cycles.samples = 1
    bpy.context.scene.render.film_transparent = True
    bpy.context.scene.frame_set(1)

    cube, cube_material = _create_cube012_source()
    plane, plane_material = _create_plane008_source()
    camera = depth_helpers._create_orthographic_camera(
        "MushroomsCapturedPlanarityCamera"
    )
    camera.data.ortho_scale = _CAMERA_ORTHO_SCALE

    sentinel = _create_sentinel()
    sentinel.location = (20.0, 0.0, 0.0)
    _activate_only(sentinel)
    cube.select_set(False)
    plane.select_set(False)
    bpy.context.view_layer.update()

    _assert_metrics(
        cube,
        label=_CUBE_OBJECT_NAME,
        expected_maximum_distance=(
            _CUBE_CAPTURED_MAXIMUM_PLANE_DISTANCE
        ),
        expected_scale=_CUBE_CAPTURED_POLYGON_SCALE,
        expected_normalized_warp=_CUBE_CAPTURED_NORMALIZED_WARP,
    )
    _assert_metrics(
        plane,
        label=_PLANE_OBJECT_NAME,
        expected_maximum_distance=(
            _PLANE_CAPTURED_MAXIMUM_PLANE_DISTANCE
        ),
        expected_scale=_PLANE_CAPTURED_POLYGON_SCALE,
        expected_normalized_warp=_PLANE_CAPTURED_NORMALIZED_WARP,
    )

    context_before = _capture_context()
    cube_before = _source_fingerprint(cube)
    plane_before = _source_fingerprint(plane)
    camera_before = depth_helpers._camera_fingerprint(camera)
    materials_before = (
        _material_fingerprint(cube_material),
        _material_fingerprint(plane_material),
    )
    temporary_before = _temporary_datablock_names()
    frame_before = int(bpy.context.scene.frame_current)

    with tempfile.TemporaryDirectory(
        prefix="spine2d_mushrooms_captured_planarity_"
    ) as directory:
        output_directory = Path(directory)
        sources = (
            A1MultiObjectSource(
                source_object=cube,
                component_id=_CUBE_COMPONENT_ID,
                settings=_settings(
                    output_directory,
                    prefix=_CUBE_PREFIX,
                ),
            ),
            A1MultiObjectSource(
                source_object=plane,
                component_id=_PLANE_COMPONENT_ID,
                settings=_settings(
                    output_directory,
                    prefix=_PLANE_PREFIX,
                ),
            ),
        )
        _assert(
            len(sources) == 2,
            "captured planarity regression must exercise two objects",
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
            "captured planarity preparation wrote output files",
        )

    _assert(
        len(prepared_multi.objects) == 2,
        f"expected two prepared objects: {prepared_multi.objects}",
    )
    prepared_by_component = _prepared_by_component(prepared_multi)
    _assert(
        set(prepared_by_component)
        == {_CUBE_COMPONENT_ID, _PLANE_COMPONENT_ID},
        f"prepared component set changed: {prepared_by_component}",
    )

    _assert_prepared_quad(
        prepared_by_component[_CUBE_COMPONENT_ID],
        label=_CUBE_OBJECT_NAME,
    )
    _assert_prepared_quad(
        prepared_by_component[_PLANE_COMPONENT_ID],
        label=_PLANE_OBJECT_NAME,
    )

    _assert(
        _capture_context() == context_before,
        "selection or active context changed",
    )
    _assert(
        _source_fingerprint(cube) == cube_before,
        "Cube.012 source changed",
    )
    _assert(
        _source_fingerprint(plane) == plane_before,
        "Plane.008 source changed",
    )
    _assert(
        depth_helpers._camera_fingerprint(camera) == camera_before,
        "active camera changed",
    )
    _assert(
        (
            _material_fingerprint(cube_material),
            _material_fingerprint(plane_material),
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
        "[MUSHROOMS-CAPTURED-PLANARITY] PASS "
        "cube=Cube.012 cube_max=7.565148185365435e-05 "
        "cube_scale=0.13120450643194492 "
        "plane=Plane.008 plane_max=9.477343601658832e-05 "
        "plane_scale=0.023314310303391664 "
        "triangles=2+2 sources=2 pipeline=public-multi-object"
    )


def main() -> None:
    try:
        _run()
    except Exception:
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
