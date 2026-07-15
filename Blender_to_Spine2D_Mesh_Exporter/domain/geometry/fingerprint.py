"""Deterministic fingerprints for geometry golden fixtures."""

from __future__ import annotations

import hashlib
import json

from .model import MeshSnapshot
from .validator import MeshSnapshotValidator


def build_mesh_fingerprint(snapshot: MeshSnapshot) -> str:
    """Hash topology and lineage while excluding mutable UV coordinates."""

    MeshSnapshotValidator().validate_or_raise(snapshot)
    payload = {
        "source_object_id": snapshot.source_object_id,
        "vertices": [
            {
                "source": [vertex.source_id.object_id, vertex.source_id.vertex_index],
                "position": list(vertex.position),
            }
            for vertex in snapshot.vertices
        ],
        "edges": [
            {
                "source": None
                if edge.source_id is None
                else [edge.source_id.object_id, edge.source_id.edge_index],
                "vertices": [edge.vertex_ids[0].index, edge.vertex_ids[1].index],
                "seam": edge.seam,
                "sharp": edge.sharp,
            }
            for edge in snapshot.edges
        ],
        "loops": [
            {
                "source": [
                    loop.source_id.object_id,
                    loop.source_id.face_index,
                    loop.source_id.corner_index,
                ],
                "vertex": loop.vertex_id.index,
                "edge": loop.edge_id.index,
            }
            for loop in snapshot.loops
        ],
        "faces": [
            {
                "source": [face.source_id.object_id, face.source_id.face_index],
                "loops": [loop_id.index for loop_id in face.loop_ids],
                "material": face.material_index,
            }
            for face in snapshot.faces
        ],
    }
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
