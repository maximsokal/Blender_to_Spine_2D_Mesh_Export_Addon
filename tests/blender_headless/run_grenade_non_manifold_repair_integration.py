"""Real grenade.blend regression for conservative non-manifold topology repair.

The artist-authored fixture contains object ``Cylinder.019`` whose Normal/UV segmentation
produced a non-manifold segment. This gate verifies the actual source file rather than a
synthetic mesh: source >2-face edges must be cut only in an immutable working snapshot,
all source lineage/UV payload must survive, and real A1 object preparation must complete.
"""

from __future__ import annotations

import argparse
from collections import Counter
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
    prepare_a1_object,
    read_source_mesh_snapshot,
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


_EXPECTED_OBJECT_NAME = "Cylinder.019"


def _parse_arguments() -> argparse.Namespace:
    arguments = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser(
        description="Run real grenade.blend non-manifold repair regression."
    )
    parser.add_argument(
        "--expected-blend",
        required=True,
        help="Exact grenade.blend path Blender must already have loaded.",
    )
    return parser.parse_args(arguments)


def _require_source_object():
    source = bpy.data.objects.get(_EXPECTED_OBJECT_NAME)
    _assert(source is not None, f"missing grenade object: {_EXPECTED_OBJECT_NAME!r}")
    _assert(source.type == "MESH", f"{_EXPECTED_OBJECT_NAME} must be MESH")
    _assert(len(source.data.polygons) > 0, f"{_EXPECTED_OBJECT_NAME} has no polygons")
    return source


def _matrix_fingerprint(matrix) -> tuple[tuple[float, ...], ...]:
    return tuple(tuple(float(value) for value in row) for row in matrix)


def _source_fingerprint(source, scene) -> tuple:
    return (
        source.name_full,
        source.data.name_full,
        _matrix_fingerprint(source.matrix_world),
        len(source.data.vertices),
        len(source.data.edges),
        len(source.data.loops),
        len(source.data.polygons),
        tuple(
            None if slot.material is None else slot.material.name_full
            for slot in source.material_slots
        ),
        tuple(layer.name for layer in source.data.uv_layers),
        None if scene.camera is None else scene.camera.name_full,
        int(scene.frame_current),
    )


def _lineage_counter(values) -> Counter:
    return Counter(values)


def _assert_repair(snapshot):
    before_edge_to_faces = build_edge_to_faces(snapshot)
    problem_edges = tuple(
        sorted(
            (
                (edge_id, face_ids)
                for edge_id, face_ids in before_edge_to_faces.items()
                if len(face_ids) > 2
            ),
            key=lambda item: item[0].index,
        )
    )
    _assert(
        problem_edges,
        "Cylinder.019 no longer exercises a >2-face non-manifold source edge",
    )

    repaired, report = split_non_manifold_edges(snapshot)
    _assert(report.changed, "non-manifold repair reported no topology change")
    _assert(
        tuple(edge_id for edge_id, _ in problem_edges) == report.split_edge_ids,
        "repair report does not match actual >2-face source edges",
    )
    _assert(
        report.created_edge_count > 0,
        "repair did not create per-face boundary edge identities",
    )

    after_edge_to_faces = build_edge_to_faces(repaired)
    remaining = tuple(
        (edge_id.index, tuple(face.id.index for face in face_ids))
        for edge_id, face_ids in after_edge_to_faces.items()
        if len(face_ids) > 2
    )
    _assert(
        not remaining,
        f"repair left >2-face edge incidence: {remaining!r}",
    )

    _assert(
        repaired.vertices == snapshot.vertices,
        "non-manifold repair changed vertex geometry",
    )
    _assert(
        repaired.faces == snapshot.faces,
        "non-manifold repair changed faces/source-face lineage",
    )
    _assert(
        _lineage_counter(loop.source_id for loop in repaired.loops)
        == _lineage_counter(loop.source_id for loop in snapshot.loops),
        "non-manifold repair changed SourceLoopId multiplicity",
    )
    _assert(
        tuple((loop.id, loop.source_id, loop.vertex_id, loop.uvs) for loop in repaired.loops)
        == tuple((loop.id, loop.source_id, loop.vertex_id, loop.uvs) for loop in snapshot.loops),
        "non-manifold repair changed loop geometry/UV/source lineage",
    )
    return repaired, report, problem_edges


def _run(expected_blend: str) -> None:
    loaded = _require_loaded_blend(expected_blend)
    scene = bpy.context.scene
    source = _require_source_object()
    _assert(scene.camera is not None, "grenade fixture must provide an active camera")

    source_before = _source_fingerprint(source, scene)
    snapshot = read_source_mesh_snapshot(
        source,
        source_object_id=source.name_full,
        snapshot_id=f"{source.name_full}:grenade-non-manifold",
    )
    repaired, report, problem_edges = _assert_repair(snapshot)

    with tempfile.TemporaryDirectory(prefix="spine2d_grenade_nonmanifold_") as temp_root:
        output_directory = Path(temp_root).resolve(strict=False)
        settings = replace(
            _base_settings(output_directory),
            prefix="Grenade Cylinder 019",
            output_stem="Grenade_Cylinder_019_Normal_Z",
            json_output_stem="Grenade_Cylinder_019_Normal_Z",
        )
        prepared = prepare_a1_object(
            source,
            settings,
            context=bpy.context,
            scene=scene,
        )
        _assert(prepared is not None, "Cylinder.019 preparation returned None")

    _assert(
        _source_fingerprint(source, scene) == source_before,
        "Cylinder.019 preparation changed source Blender state",
    )

    print(
        "[GRENADE-NON-MANIFOLD-REPAIR] PASS "
        f"blend={loaded} object={source.name_full!r} "
        f"source_problem_edges={len(problem_edges)} "
        f"split_edges={len(report.split_edge_ids)} "
        f"created_edges={report.created_edge_count} "
        f"input_edges={report.input_edge_count} "
        f"working_edges={len(repaired.edges)} "
        "prepare=success source=unchanged",
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
