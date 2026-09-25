"""Real grenade regression for repaired-edge texturing seam correspondence.

The original artist fixture that exposed this bug was named ``Plane.007``. Object display
names are not the contract: the regression requires a real mesh containing at least one
source edge shared by more than two faces. Geometry preparation repairs that incidence in
an immutable working snapshot while UV texturing unwraps source topology. Stable
``SourceEdgeId`` lineage must reconcile the two without leaking repair-local edge IDs into
the unwrap seam plan.

The resolver prefers the historical fixture when it still exists, then the current
non-manifold regression fixture, and finally accepts a unique structurally equivalent mesh.
It fails closed instead of silently choosing between ambiguous candidates.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
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
from run_grenade_non_manifold_repair_integration import (  # noqa: E402
    _EXPECTED_OBJECT_NAME as _CURRENT_NON_MANIFOLD_FIXTURE_NAME,
)


_LEGACY_FIXTURE_NAME = "Plane.007"


@dataclass(frozen=True, slots=True)
class _LineageFixtureCandidate:
    source_object: bpy.types.Object
    non_manifold_edge_indices: tuple[int, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.source_object, bpy.types.Object):
            raise TypeError("source_object must be bpy.types.Object")
        if self.source_object.type != "MESH" or self.source_object.data is None:
            raise TypeError("source_object must be a mesh object with data")
        if not self.non_manifold_edge_indices:
            raise ValueError("candidate must contain at least one >2-face source edge")


def _parse_arguments() -> argparse.Namespace:
    arguments = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser(
        description="Run real grenade repaired-edge texturing lineage regression."
    )
    parser.add_argument(
        "--expected-blend",
        required=True,
        help="Exact grenade.blend path Blender must already have loaded.",
    )
    return parser.parse_args(arguments)


def _matrix_fingerprint(matrix) -> tuple[tuple[float, ...], ...]:
    return tuple(tuple(float(value) for value in row) for row in matrix)


def _source_fingerprint(source: bpy.types.Object, scene: bpy.types.Scene) -> tuple:
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


def _candidate_for_object(source_object: bpy.types.Object) -> _LineageFixtureCandidate | None:
    if not isinstance(source_object, bpy.types.Object):
        raise TypeError("source_object must be bpy.types.Object")
    if source_object.type != "MESH" or source_object.data is None:
        return None
    if len(source_object.data.polygons) == 0:
        return None

    snapshot = read_source_mesh_snapshot(
        source_object,
        source_object_id=source_object.name_full,
        snapshot_id=f"{source_object.name_full}:lineage-candidate-scan",
    )
    incidence = build_edge_to_faces(snapshot)
    problem_edges = tuple(
        sorted(
            edge_id.index
            for edge_id, face_ids in incidence.items()
            if len(face_ids) > 2
        )
    )
    if not problem_edges:
        return None
    return _LineageFixtureCandidate(source_object, problem_edges)


def _resolve_source() -> tuple[_LineageFixtureCandidate, tuple[str, ...]]:
    candidates: list[_LineageFixtureCandidate] = []
    for source_object in sorted(bpy.data.objects, key=lambda item: item.name_full):
        candidate = _candidate_for_object(source_object)
        if candidate is not None:
            candidates.append(candidate)

    _assert(candidates, "grenade fixture contains no mesh with a >2-face source edge")
    candidate_names = tuple(candidate.source_object.name_full for candidate in candidates)

    for preferred_name in (_LEGACY_FIXTURE_NAME, _CURRENT_NON_MANIFOLD_FIXTURE_NAME):
        preferred = tuple(
            candidate
            for candidate in candidates
            if candidate.source_object.name_full == preferred_name
        )
        if preferred:
            _assert(
                len(preferred) == 1,
                f"duplicate preferred lineage fixture name {preferred_name!r}: {candidate_names}",
            )
            return preferred[0], candidate_names

    _assert(
        len(candidates) == 1,
        "grenade lineage fixture is ambiguous; no known preferred object is present: "
        f"candidates={candidate_names}",
    )
    return candidates[0], candidate_names


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

    candidate, candidate_names = _resolve_source()
    source_object = candidate.source_object
    before = _source_fingerprint(source_object, scene)
    datablocks_before = _datablock_fingerprint()

    raw_snapshot = read_source_mesh_snapshot(
        source_object,
        source_object_id=source_object.name_full,
        snapshot_id=f"{source_object.name_full}:texturing-lineage-fixture",
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
        raw_non_manifold == candidate.non_manifold_edge_indices,
        "lineage fixture changed between structural scan and regression read: "
        f"scan={candidate.non_manifold_edge_indices}, read={raw_non_manifold}",
    )

    repaired, report = split_non_manifold_edges(
        raw_snapshot,
        snapshot_id=raw_snapshot.snapshot_id,
    )
    _assert(report.changed, "lineage fixture repair report unexpectedly reports no change")
    _assert(
        report.created_edge_count > 0,
        "lineage fixture repair did not append per-face edge copies",
    )
    _assert(
        tuple(edge.id for edge in repaired.edges[: len(raw_snapshot.edges)])
        == tuple(edge.id for edge in raw_snapshot.edges),
        "lineage fixture repair shifted original EdgeId values",
    )
    _assert(
        all(len(face_ids) <= 2 for face_ids in build_edge_to_faces(repaired).values()),
        "lineage fixture repair left a >2-face edge",
    )

    with tempfile.TemporaryDirectory(prefix="spine2d_grenade_lineage_") as temp_root:
        output_directory = Path(temp_root).resolve(strict=False)
        settings = replace(
            _base_settings(output_directory),
            prefix=f"Grenade_{source_object.name_full}",
            output_stem=f"Grenade_{source_object.name_full}_Lineage",
            json_output_stem=f"Grenade_{source_object.name_full}_Lineage",
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
            "texturing topology contains seam edges absent from the source snapshot: "
            f"object={source_object.name_full!r}, "
            f"seams={tuple(edge.id.index for edge in topology.snapshot.edges if edge.seam)!r}",
        )
        _assert(
            all(
                topology.snapshot.edge_by_id()[edge_id].seam
                for edge_id in topology.all_seam_edge_ids
            ),
            "resolved seam IDs are not marked on the unwrap snapshot",
        )
        _assert(
            uv.unwrap_result.snapshot.active_uv_layer == settings.uv.layer_name,
            "lineage fixture unwrap did not activate the generated UV layer",
        )
        _assert(
            len(uv.uv_regions.regions) == len(source.geometry.regions),
            "lineage fixture UV propagation region count differs from geometry",
        )

    _assert(
        _source_fingerprint(source_object, scene) == before,
        "lineage preparation changed source object or scene state",
    )
    _assert(
        _datablock_fingerprint() == datablocks_before,
        "lineage preparation leaked or removed Blender datablocks",
    )

    print(
        "[GRENADE-TEXTURING-LINEAGE] PASS "
        f"blend={loaded} object={source_object.name_full!r} "
        f"candidates={candidate_names!r} "
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
