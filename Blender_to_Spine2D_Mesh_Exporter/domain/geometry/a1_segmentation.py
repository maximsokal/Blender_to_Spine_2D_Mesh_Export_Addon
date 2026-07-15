"""A1-compatible seed-normal segmentation without legacy overlap bugs.

The legacy angular pass compares every candidate face with the normal and material
of the seed face, not with the immediately adjacent face.  Reproducing that rule
is important for compatibility on smoothly curving strips.  The old implementation
could add a face to more than one segment because it did not exclude faces already
assigned by an earlier seed.  This implementation preserves the seed-normal rule
while guaranteeing a disjoint, complete partition.
"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import replace
from math import acos, degrees, sqrt
from typing import Iterable, Tuple

from .ids import EdgeId, FaceId
from .model import MeshSnapshot, Vector3
from .segmentation import (
    MeshSegment,
    SegmentBoundaryEdge,
    SegmentBoundaryReason,
    SegmentationPlan,
    SegmentationSettings,
    segment_mesh,
)
from .topology import analyse_face_region, build_edge_to_faces, face_edge_ids
from .validator import MeshSnapshotValidator


class A1SegmentationError(ValueError):
    """Raised when the A1 segmentation contract cannot be satisfied."""


def _vector_length(value: Vector3) -> float:
    return sqrt(sum(component * component for component in value))


def _normal_angle_degrees(first: Vector3, second: Vector3) -> float | None:
    first_length = _vector_length(first)
    second_length = _vector_length(second)
    if first_length < 1e-6 or second_length < 1e-6:
        return None
    dot = sum(a * b for a, b in zip(first, second)) / (
        first_length * second_length
    )
    dot = max(-1.0, min(1.0, dot))
    return degrees(acos(dot))


def _seed_normal_groups(
    snapshot: MeshSnapshot,
    face_ids: Iterable[FaceId],
    *,
    blocked_edge_ids: set[EdgeId],
    edge_to_faces: dict[EdgeId, Tuple[FaceId, ...]],
    settings: SegmentationSettings,
) -> Tuple[Tuple[FaceId, ...], ...]:
    face_map = snapshot.face_by_id()
    remaining = set(face_ids)
    groups: list[Tuple[FaceId, ...]] = []

    while remaining:
        seed_id = min(remaining, key=lambda item: item.index)
        seed_face = face_map[seed_id]
        seed_normal = seed_face.normal
        seed_material = seed_face.material_index

        # The legacy code skipped zero-length normals entirely, which could drop
        # faces from export.  A valid rewrite must never lose geometry, so an
        # invalid-normal face becomes its own deterministic segment.
        if _normal_angle_degrees(seed_normal, seed_normal) is None:
            remaining.remove(seed_id)
            groups.append((seed_id,))
            continue

        queue = deque([seed_id])
        queued = {seed_id}
        accepted: list[FaceId] = []

        while queue:
            current_id = queue.popleft()
            if current_id not in remaining:
                continue
            current_face = face_map[current_id]
            if settings.split_materials and current_face.material_index != seed_material:
                continue
            current_angle = _normal_angle_degrees(seed_normal, current_face.normal)
            if current_angle is None or current_angle >= settings.angle_limit_degrees:
                continue

            remaining.remove(current_id)
            accepted.append(current_id)

            for edge_id in face_edge_ids(snapshot, current_id):
                if edge_id in blocked_edge_ids:
                    continue
                for neighbour_id in edge_to_faces.get(edge_id, ()):
                    if (
                        neighbour_id in remaining
                        and neighbour_id not in queued
                    ):
                        queued.add(neighbour_id)
                        neighbour = face_map[neighbour_id]
                        if (
                            settings.split_materials
                            and neighbour.material_index != seed_material
                        ):
                            continue
                        angle = _normal_angle_degrees(seed_normal, neighbour.normal)
                        if angle is not None and angle < settings.angle_limit_degrees:
                            queue.append(neighbour_id)

        if not accepted:
            # Defensive fallback.  The seed has already been validated above, so
            # this should not occur, but it prevents an infinite loop if model
            # validation rules change later.
            remaining.remove(seed_id)
            accepted.append(seed_id)
        groups.append(tuple(sorted(accepted, key=lambda item: item.index)))

    return tuple(groups)


def segment_mesh_a1(
    snapshot: MeshSnapshot,
    settings: SegmentationSettings | None = None,
) -> SegmentationPlan:
    """Segment a snapshot using deterministic legacy seed-normal semantics.

    Non-angular boundaries are calculated by :func:`segment_mesh`.  Each resulting
    primary component is then partitioned by comparing every candidate normal with
    the component seed normal.  Every face is assigned exactly once.
    """

    MeshSnapshotValidator().validate_or_raise(snapshot)
    resolved_settings = settings or SegmentationSettings()
    if not resolved_settings.split_by_angle:
        return segment_mesh(snapshot, resolved_settings)

    base_settings = replace(resolved_settings, split_by_angle=False)
    base_plan = segment_mesh(snapshot, base_settings)
    edge_to_faces = build_edge_to_faces(snapshot)
    edge_map = snapshot.edge_by_id()
    face_map = snapshot.face_by_id()
    base_reasons: dict[EdgeId, set[SegmentBoundaryReason]] = {
        boundary.edge_id: set(boundary.reasons)
        for boundary in base_plan.boundary_edges
    }
    blocked_edge_ids = set(base_reasons)

    grouped_faces: list[Tuple[FaceId, ...]] = []
    for primary_segment in base_plan.segments:
        grouped_faces.extend(
            _seed_normal_groups(
                snapshot,
                primary_segment.face_ids,
                blocked_edge_ids=blocked_edge_ids,
                edge_to_faces=edge_to_faces,
                settings=resolved_settings,
            )
        )

    covered = [face_id for group in grouped_faces for face_id in group]
    if len(covered) != len(set(covered)):
        raise A1SegmentationError("A1 segmentation produced overlapping face groups")
    expected = {face.id for face in snapshot.faces}
    if set(covered) != expected:
        missing = sorted((face_id.index for face_id in expected - set(covered)))
        raise A1SegmentationError(f"A1 segmentation lost faces: {missing}")

    ordered_groups = tuple(
        sorted(
            grouped_faces,
            key=lambda group: (min(face_id.index for face_id in group), len(group)),
        )
    )
    segments = tuple(
        MeshSegment(
            segment_id=segment_id,
            face_ids=face_ids,
            source_face_ids=tuple(face_map[face_id].source_id for face_id in face_ids),
            topology=analyse_face_region(
                snapshot,
                face_ids,
                edge_to_faces=edge_to_faces,
            ),
        )
        for segment_id, face_ids in enumerate(ordered_groups)
    )
    face_to_segment = {
        face_id: segment.segment_id
        for segment in segments
        for face_id in segment.face_ids
    }

    final_reasons: dict[EdgeId, set[SegmentBoundaryReason]] = defaultdict(set)
    for edge_id, reasons in base_reasons.items():
        final_reasons[edge_id].update(reasons)

    for edge_id, linked_faces in edge_to_faces.items():
        if len(linked_faces) != 2:
            continue
        first_id, second_id = linked_faces
        if face_to_segment[first_id] == face_to_segment[second_id]:
            continue
        if edge_id in blocked_edge_ids:
            continue
        first_angle = _normal_angle_degrees(face_map[first_id].normal, face_map[first_id].normal)
        second_angle = _normal_angle_degrees(face_map[second_id].normal, face_map[second_id].normal)
        if first_angle is None or second_angle is None:
            final_reasons[edge_id].add(SegmentBoundaryReason.INVALID_NORMAL)
        else:
            final_reasons[edge_id].add(SegmentBoundaryReason.ANGLE)

    boundaries = tuple(
        SegmentBoundaryEdge(
            edge_id=edge_id,
            source_edge_id=edge_map[edge_id].source_id,
            linked_face_ids=edge_to_faces.get(edge_id, ()),
            segment_ids=tuple(
                sorted(
                    {
                        face_to_segment[face_id]
                        for face_id in edge_to_faces.get(edge_id, ())
                    }
                )
            ),
            reasons=tuple(sorted(reasons, key=lambda item: item.value)),
        )
        for edge_id, reasons in sorted(
            final_reasons.items(),
            key=lambda item: item[0].index,
        )
        if reasons
    )

    return SegmentationPlan(
        snapshot_id=snapshot.snapshot_id,
        settings=resolved_settings,
        segments=segments,
        boundary_edges=boundaries,
    )
