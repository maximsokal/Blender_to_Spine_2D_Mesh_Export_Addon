"""Spine 4.2 mesh-edge index-space conversion and validation.

The Rewrite domain stores mesh edges as logical attachment vertex-index pairs because
triangles, UV-split projection, and topology validation all operate in that space.
Spine JSON serializes the same endpoints as offsets into the interleaved ``x, y``
vertex-coordinate domain, therefore every exported endpoint is ``vertex_index * 2``.

Keeping this conversion at one explicit output boundary prevents triangles and edges
from accidentally sharing an index space even though Spine assigns them different
serialized representations.
"""

from __future__ import annotations

from typing import Tuple


class SpineMeshEdgeContractError(ValueError):
    """Raised when mesh edges cannot be represented by the Spine 4.2 contract."""


def _require_vertex_count(value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError("vertex_count must be int")
    if value < 0:
        raise ValueError("vertex_count must be non-negative")
    return value


def validate_logical_mesh_edges(
    edges: Tuple[int, ...],
    *,
    vertex_count: int,
) -> Tuple[int, ...]:
    """Validate logical attachment-vertex pairs used inside the Rewrite domain."""

    resolved_vertex_count = _require_vertex_count(vertex_count)
    if not isinstance(edges, tuple):
        raise TypeError("edges must be tuple")
    if len(edges) % 2 != 0:
        raise SpineMeshEdgeContractError("edges must contain complete endpoint pairs")
    if edges and resolved_vertex_count == 0:
        raise SpineMeshEdgeContractError("non-empty edges require at least one vertex")

    maximum_index = resolved_vertex_count - 1
    seen: set[tuple[int, int]] = set()
    for pair_index in range(0, len(edges), 2):
        first = edges[pair_index]
        second = edges[pair_index + 1]
        for endpoint_index, endpoint in enumerate((first, second)):
            if isinstance(endpoint, bool) or not isinstance(endpoint, int):
                raise TypeError(
                    f"edges[{pair_index + endpoint_index}] must be int"
                )
            if endpoint < 0 or endpoint > maximum_index:
                raise SpineMeshEdgeContractError(
                    f"edges[{pair_index + endpoint_index}]={endpoint} is outside "
                    f"attachment vertex range [0, {maximum_index}]"
                )
        if first == second:
            raise SpineMeshEdgeContractError(
                f"edge pair {pair_index // 2} is a self-edge for vertex {first}"
            )
        key = (first, second) if first < second else (second, first)
        if key in seen:
            raise SpineMeshEdgeContractError(
                f"edge pair {pair_index // 2} duplicates undirected edge {key}"
            )
        seen.add(key)
    return edges


def encode_spine_mesh_edge_offsets(
    edges: Tuple[int, ...],
    *,
    vertex_count: int,
) -> Tuple[int, ...]:
    """Convert logical vertex indices to Spine JSON interleaved-coordinate offsets."""

    validated = validate_logical_mesh_edges(edges, vertex_count=vertex_count)
    return tuple(endpoint * 2 for endpoint in validated)


def validate_spine_mesh_edge_offsets(
    offsets: Tuple[int, ...],
    *,
    vertex_count: int,
) -> Tuple[int, ...]:
    """Validate already serialized Spine 4.2 mesh-edge coordinate offsets."""

    resolved_vertex_count = _require_vertex_count(vertex_count)
    if not isinstance(offsets, tuple):
        raise TypeError("offsets must be tuple")
    if len(offsets) % 2 != 0:
        raise SpineMeshEdgeContractError(
            "Spine mesh edge offsets must contain complete endpoint pairs"
        )
    if offsets and resolved_vertex_count == 0:
        raise SpineMeshEdgeContractError(
            "non-empty Spine mesh edge offsets require at least one vertex"
        )

    maximum_offset = (resolved_vertex_count - 1) * 2
    seen: set[tuple[int, int]] = set()
    for pair_index in range(0, len(offsets), 2):
        first = offsets[pair_index]
        second = offsets[pair_index + 1]
        for endpoint_index, endpoint in enumerate((first, second)):
            if isinstance(endpoint, bool) or not isinstance(endpoint, int):
                raise TypeError(
                    f"offsets[{pair_index + endpoint_index}] must be int"
                )
            if endpoint % 2 != 0:
                raise SpineMeshEdgeContractError(
                    f"offsets[{pair_index + endpoint_index}]={endpoint} is not an "
                    "interleaved x/y vertex offset"
                )
            if endpoint < 0 or endpoint > maximum_offset:
                raise SpineMeshEdgeContractError(
                    f"offsets[{pair_index + endpoint_index}]={endpoint} is outside "
                    f"Spine coordinate-offset range [0, {maximum_offset}]"
                )
        if first == second:
            raise SpineMeshEdgeContractError(
                f"Spine edge pair {pair_index // 2} is a self-edge at offset {first}"
            )
        key = (first, second) if first < second else (second, first)
        if key in seen:
            raise SpineMeshEdgeContractError(
                f"Spine edge pair {pair_index // 2} duplicates undirected edge {key}"
            )
        seen.add(key)
    return offsets


__all__ = [
    "SpineMeshEdgeContractError",
    "encode_spine_mesh_edge_offsets",
    "validate_logical_mesh_edges",
    "validate_spine_mesh_edge_offsets",
]
