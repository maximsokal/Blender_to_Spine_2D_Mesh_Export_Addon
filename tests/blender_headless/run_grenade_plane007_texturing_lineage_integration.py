"""Real grenade.blend regression for repaired-edge texturing seam correspondence.

The artist-authored ``Plane.007`` fixture exposed a stale local-EdgeId bug after
non-manifold repair. Geometry preparation operates on an immutable repaired snapshot,
while UV texturing intentionally unwraps the unrepaired full-object snapshot. This gate
proves that the two topologies are reconciled through stable ``SourceEdgeId`` lineage and
that no repair-generated local edge leaks into the unwrap seam plan.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
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

from Blender_to_Spine2D_Mesh_Exporter.blender_adapter import (  # noqa: E402
    read_source_mesh_snapshot,
)
from Blender_to_Spine2D_Mesh_Exporter.blender_adapter.a1_source_geometry_preparation import (  # noqa: E402
    prepare_a1_source_geometry,
)
from Blender_to_Spine2D_Mesh_Exporter.blender_adapter.a1_uv_preparation import (  # noqa: E402
    prepare_a1_uv,
)
from Blender_to_Spine2D_Mesh_Exporter.domain.geometry import (  # noqa: E402
    build_edge_to_faces,
    split_non_manifold_edges,
)
from run_bake_integration import _assert  # noqa: E402
from run_grenade_bump_displacement_normal_uv_integration import (  # noqa: E402
    _require_loaded_blend,
    _settings as _base_settings,
)


_EXPECTED_OBJECT_NAME = "Plane.007"


def _parse_arguments() -> argparse.Namespace:
    arguments = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser(
        description="Run real grenade Plane.007 repaired-edge texturing regression."
    )
    parser.add_argument(
        "--expected-blend",
        required=True,
        help="Exact grenade.blend path Blender must already have loaded.",
    )
    return parser.parse_args(arguments)


def _require_source():
    source = bpy.data.objects.get(_EXPECTED_OBJECT_NAME)
    _assert(source is not None, f"missing grenade object: {_EXPECTED_OBJECT_NAME!r}")
    _assert(source.type == "MESH", f"{_EXPECTED_OBJECT_NAME} must be MESH")
    _assert(source.data is not None, f"{_EXPECTED_OBJECT_NAME} has no Mesh datablock")
    return source


def _matrix_fingerprint(matrix) -> tuple[tuple[float, ...], ...]:
    return tuple(tuple(float(value) for value in row) for row in matrix)


def _source_fingerprint(source, scene) -> tuple:
    uv_layers = source.data.uv_layers
    return (
        source.name_full,
        source.data.name_full,
        _matrix_fingerprint(source.matrix_world),
        tuple(
            None if slot.material is None else slot.material.name_full
            for slot in source.material_slots
        ),
        tuple(layer.name for layer in uv_layers),
        None if uv_layers.active is None else uv_layers.active.name,
        len(source.data.vertices),
        len(source.data.edges),
        len(source.data.polygons),
        None if scene.camera is None else scene.camera.name_full,
        int(scene.frame_current),
        str(scene.render.engine),
    )


def _datablock_fingerprint() -> tuple:
    return (
        tuple(sorted(item.name_full for item in bpy.data.objects)),
        tuple(sorted(item.name_full for item in bpy.data.meshes)),
        tuple(sorted(item.name_full for item in bpy.data.materials)),
        tuple(sorted(item.name_full for item in bpy.data.images)),
    )


def _internal_geometry_cut_ids(geometry) -> tuple[int, ...]:
    segmentation = tuple(
        boundary.edge_id.index
        for boundary in geometry.segmentation.boundary_edges
        if len(boundary.linked_face_ids) == 2
        and len(boundary.segment_ids) == 2
    )
    decomposition = tuple(
        cut.edge_id.index
        for cut in geometry.decomposition.cuts
        if len(cut.linked_face_ids) == 2
        and len(cut.region_ids) == 2
    )
    return tuple(sorted(set(segmentation) | set(decomposition)))


def _run(expected_blend: str) -> None:
    loaded = _require_loaded_blend(expected_blend)
    scene = bpy.context.scene
    _assert(scene.camera is not None, "grenade fixture must provide an active camera")
    source_object = _require_source()

    before = _source_fingerprint(source_object, scene)
    datablocks_before = _datablock_fingerprint()

    raw_snapshot = read_source_mesh_snapshot(
        source_object,
        source_object_id=source_object.name_full,
        snapshot_id=f"{source_object.name_full}:plane007-lineage-fixture",
    )
    raw_incidence = build_edge_to_faces(raw_snapshot)
    raw_non_manifold = tuple(
        sorted(
            edge_id.index
            for edge_id, face_ids in raw_incidence.items()
            if len(face_ids) > 2
        )
    )
    _assert(
        raw_non_manifold,
        "Plane.007 no longer exercises non-manifold edge repair",
    )

    repaired, report = split_non_manifold_edges(
        raw_snapshot,
        snapshot_id=raw_snapshot.snapshot_id,
    )
    _assert(report.changed, "Plane.007 repair report unexpectedly reports no change")
    _assert(
        report.created_edge_count > 0,
        "Plane.007 repair did not append any per-face edge copies",
    )
    _assert(
        tuple(edge.id for edge in repaired.edges[: len(raw_snapshot.edges)])
        == tuple(edge.id for edge in raw_snapshot.edges),
        "Plane.007 repair shifted original EdgeId values",
    )
    _assert(
        all(len(face_ids) <= 2 for face_ids in build_edge_to_faces(repaired).values()),
        "Plane.007 repair left a >2-face edge",
    )

    with tempfile.TemporaryDirectory(prefix="spine2d_grenade_plane007_") as temp_root:
        output_directory = Path(temp_root).resolve(strict=False)
        settings = replace(
            _base_settings(output_directory),
            prefix="Grenade_Plane_007",
            output_stem="Grenade_Plane_007",
            json_output_stem="Grenade_Plane_007",
        )

        source = prepare_a1_source_geometry(
            source_object,
            settings,
            scene=scene,
        )
        geometry_cut_ids = _internal_geometry_cut_ids(source.geometry)

        uv = prepare_a1_uv(
            source,
            context=bpy.context,
            scene=scene,
        )
        topology = uv.texturing_topology
        source_edge_ids = set(source.source_snapshot.edge_by_id())
        _assert(
            set(topology.all_seam_edge_ids).issubset(source_edge_ids),
            "Plane.007 texturing topology still contains edges absent from source "
            f"snapshot: seams={tuple(edge.id.index for edge in topology.snapshot.edges if edge.seam)!r}",
        )
        _assert(
            all(topology.snapshot.edge_by_id()[edge_id].seam for edge_id in topology.all_seam_edge_ids),
            "Plane.007 resolved seam IDs are not marked on the unwrap snapshot",
        )
        _assert(
            uv.unwrap_result.snapshot.active_uv_layer == settings.uv.layer_name,
            "Plane.007 unwrap did not activate the generated UV layer",
        )
        _assert(
            len(uv.uv_regions.regions) == len(source.geometry.regions),
            "Plane.007 UV propagation region count differs from geometry",
        )

    _assert(
        _source_fingerprint(source_object, scene) == before,
        "Plane.007 preparation changed source object or scene state",
    )
    _assert(
        _datablock_fingerprint() == datablocks_before,
        "Plane.007 preparation leaked or removed Blender datablocks",
    )

    print(
        "[GRENADE-PLANE-007-TEXTURING] PASS "
        f"blend={loaded} object={source_object.name_full!r} "
        f"source_edges={len(raw_snapshot.edges)} "
        f"non_manifold_edges={len(raw_non_manifold)} "
        f"created_repair_edges={report.created_edge_count} "
        f"geometry_internal_cuts={len(geometry_cut_ids)} "
        f"resolved_texturing_seams={len(topology.all_seam_edge_ids)} "
        "edge_ids=stable lineage=resolved source=unchanged",
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
