"""Rebase canonical projected geometry onto a different export-only pivot.

Signed-axis projection stores canonical U/V/depth vertex coordinates relative to the
projected Blender Object Origin and stores that origin as the translation-only
``MeshSnapshot.world_matrix``. Multi-object shared-pivot export must change both pieces
as one coordinate-system operation::

    old_origin + old_local == new_origin + new_local

Only immutable snapshot data is replaced. Blender Object origins, transforms, Mesh data,
UV layers, normals, topology, and source lineage are never mutated.
"""

from __future__ import annotations

from dataclasses import replace
from math import isfinite

from ..projection import A1ProjectedPoint
from .model import Matrix4x4, MeshSnapshot
from .validator import MeshSnapshotValidator


class A1SharedPivotRebaseError(ValueError):
    """Raised when projected geometry cannot be safely rebased."""


def _normalized_zero(value: float) -> float:
    resolved = float(value)
    return 0.0 if resolved == 0.0 else resolved


def _translation_origin(matrix: Matrix4x4) -> A1ProjectedPoint:
    """Read a translation-only canonical matrix and reject any linear transform."""

    if not isinstance(matrix, tuple) or len(matrix) != 16:
        raise TypeError("snapshot.world_matrix must be a 16-value tuple")
    values = tuple(float(value) for value in matrix)
    if not all(isfinite(value) for value in values):
        raise A1SharedPivotRebaseError(
            "snapshot.world_matrix contains non-finite values"
        )

    tolerance = 1.0e-10
    expected = (
        1.0,
        0.0,
        0.0,
        values[3],
        0.0,
        1.0,
        0.0,
        values[7],
        0.0,
        0.0,
        1.0,
        values[11],
        0.0,
        0.0,
        0.0,
        1.0,
    )
    mismatches = tuple(
        (index, actual, required)
        for index, (actual, required) in enumerate(zip(values, expected, strict=True))
        if abs(actual - required) > tolerance
    )
    if mismatches:
        raise A1SharedPivotRebaseError(
            "projected snapshot.world_matrix must contain translation only; "
            f"mismatches={mismatches}"
        )

    return A1ProjectedPoint(
        u=_normalized_zero(values[3]),
        v=_normalized_zero(values[7]),
        depth=_normalized_zero(values[11]),
    )


def _translation_matrix(origin: A1ProjectedPoint) -> Matrix4x4:
    if not isinstance(origin, A1ProjectedPoint):
        raise TypeError("origin must be A1ProjectedPoint")
    return (
        1.0,
        0.0,
        0.0,
        origin.u,
        0.0,
        1.0,
        0.0,
        origin.v,
        0.0,
        0.0,
        1.0,
        origin.depth,
        0.0,
        0.0,
        0.0,
        1.0,
    )


def _origins_match(
    first: A1ProjectedPoint,
    second: A1ProjectedPoint,
    *,
    tolerance: float = 1.0e-10,
) -> bool:
    return bool(
        abs(float(first.u) - float(second.u)) <= tolerance
        and abs(float(first.v) - float(second.v)) <= tolerance
        and abs(float(first.depth) - float(second.depth)) <= tolerance
    )


def rebase_a1_projected_snapshot_origin(
    snapshot: MeshSnapshot,
    current_origin: A1ProjectedPoint,
    target_origin: A1ProjectedPoint,
) -> MeshSnapshot:
    """Move the canonical origin while preserving every world-space vertex exactly.

    ``snapshot`` must already be world-transform normalized and signed-axis projected.
    ``current_origin`` is checked against its translation-only matrix so stale or mixed
    coordinate spaces fail closed. The returned local vertex coordinates are translated
    by ``current_origin - target_origin`` in U/V/depth while the matrix translation moves
    to ``target_origin``. Normals and all non-position mesh channels remain unchanged.
    """

    if not isinstance(snapshot, MeshSnapshot):
        raise TypeError("snapshot must be MeshSnapshot")
    if not isinstance(current_origin, A1ProjectedPoint):
        raise TypeError("current_origin must be A1ProjectedPoint")
    if not isinstance(target_origin, A1ProjectedPoint):
        raise TypeError("target_origin must be A1ProjectedPoint")

    MeshSnapshotValidator().validate_or_raise(snapshot)
    matrix_origin = _translation_origin(snapshot.world_matrix)
    if not _origins_match(matrix_origin, current_origin):
        raise A1SharedPivotRebaseError(
            "current_origin does not match projected snapshot translation; "
            f"matrix=({matrix_origin.u}, {matrix_origin.v}, {matrix_origin.depth}), "
            f"current=({current_origin.u}, {current_origin.v}, {current_origin.depth})"
        )

    if _origins_match(current_origin, target_origin, tolerance=0.0):
        return snapshot

    delta_u = _normalized_zero(float(current_origin.u) - float(target_origin.u))
    delta_v = _normalized_zero(float(current_origin.v) - float(target_origin.v))
    delta_depth = _normalized_zero(
        float(current_origin.depth) - float(target_origin.depth)
    )
    if not all(isfinite(value) for value in (delta_u, delta_v, delta_depth)):
        raise A1SharedPivotRebaseError("shared-pivot rebase delta is non-finite")

    vertices = tuple(
        replace(
            vertex,
            position=(
                _normalized_zero(float(vertex.position[0]) + delta_u),
                _normalized_zero(float(vertex.position[1]) + delta_v),
                _normalized_zero(float(vertex.position[2]) + delta_depth),
            ),
        )
        for vertex in snapshot.vertices
    )
    rebased = replace(
        snapshot,
        vertices=vertices,
        world_matrix=_translation_matrix(target_origin),
    )
    MeshSnapshotValidator().validate_or_raise(rebased)

    # Fail closed if any future MeshSnapshot representation change breaks the defining
    # coordinate-system invariant. A small absolute tolerance covers ordinary floating
    # addition order without relaxing the geometry contract.
    tolerance = 1.0e-9
    for before, after in zip(snapshot.vertices, rebased.vertices, strict=True):
        old_world = (
            float(current_origin.u) + float(before.position[0]),
            float(current_origin.v) + float(before.position[1]),
            float(current_origin.depth) + float(before.position[2]),
        )
        new_world = (
            float(target_origin.u) + float(after.position[0]),
            float(target_origin.v) + float(after.position[1]),
            float(target_origin.depth) + float(after.position[2]),
        )
        if any(
            abs(old_value - new_value) > tolerance
            for old_value, new_value in zip(old_world, new_world, strict=True)
        ):
            raise A1SharedPivotRebaseError(
                "shared-pivot rebase changed world-space geometry at "
                f"vertex {before.id.index}: before={old_world}, after={new_world}"
            )

    return rebased


__all__ = [
    "A1SharedPivotRebaseError",
    "rebase_a1_projected_snapshot_origin",
]
