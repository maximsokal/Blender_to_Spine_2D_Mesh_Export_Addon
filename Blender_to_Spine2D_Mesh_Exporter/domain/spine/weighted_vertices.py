"""Encoder and decoder for Spine weighted mesh vertex streams."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence, Tuple

from .spine_scalar_contract import is_finite_number as _is_finite_number


def _require_stream_number(value: object, field_name: str) -> int | float:
    """Return one raw finite stream number without coercing booleans or strings."""

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{field_name} is not numeric")
    if not _is_finite_number(value):
        raise ValueError(f"{field_name} must be finite")
    return value


@dataclass(frozen=True, slots=True)
class WeightedVertexInfluence:
    bone_index: int
    x: float
    y: float
    weight: float

    def __post_init__(self) -> None:
        if (
            isinstance(self.bone_index, bool)
            or not isinstance(self.bone_index, int)
            or self.bone_index < 0
        ):
            raise ValueError("bone_index must be a non-negative integer")
        for field_name, value in (("x", self.x), ("y", self.y), ("weight", self.weight)):
            if not _is_finite_number(value):
                raise ValueError(f"{field_name} must be a finite number")
        if self.weight < 0.0:
            raise ValueError("weight cannot be negative")


@dataclass(frozen=True, slots=True)
class WeightedVertex:
    influences: Tuple[WeightedVertexInfluence, ...]

    def __post_init__(self) -> None:
        if not self.influences:
            raise ValueError("WeightedVertex must contain at least one influence")


def encode_weighted_vertices(vertices: Iterable[WeightedVertex]) -> Tuple[float | int, ...]:
    """Encode typed weighted vertices into Spine's compact numeric stream."""

    encoded: list[float | int] = []
    for vertex_index, vertex in enumerate(vertices):
        if not isinstance(vertex, WeightedVertex):
            raise TypeError(f"vertices[{vertex_index}] must be WeightedVertex")
        encoded.append(len(vertex.influences))
        for influence in vertex.influences:
            encoded.extend(
                (
                    influence.bone_index,
                    float(influence.x),
                    float(influence.y),
                    float(influence.weight),
                )
            )
    return tuple(encoded)


def decode_weighted_vertices(
    stream: Sequence[float | int],
    *,
    expected_vertex_count: int | None = None,
) -> Tuple[WeightedVertex, ...]:
    """Decode and structurally validate a Spine weighted vertex stream.

    Raises:
        TypeError: when the input is not a numeric sequence.
        ValueError: when the stream is truncated, malformed, or has an invalid
            expected vertex count.
    """

    if isinstance(stream, (str, bytes)) or not isinstance(stream, Sequence):
        raise TypeError("stream must be a numeric sequence")
    if expected_vertex_count is not None:
        if isinstance(expected_vertex_count, bool) or not isinstance(
            expected_vertex_count,
            int,
        ):
            raise TypeError("expected_vertex_count must be int or None")
        if expected_vertex_count < 0:
            raise ValueError("expected_vertex_count cannot be negative")

    result: list[WeightedVertex] = []
    index = 0
    stream_length = len(stream)

    while index < stream_length:
        raw_count = _require_stream_number(
            stream[index],
            f"Influence count at stream index {index}",
        )
        influence_count = int(raw_count)
        if raw_count != influence_count or influence_count <= 0:
            raise ValueError(
                f"Influence count at stream index {index} must be a positive integer"
            )
        index += 1

        required_values = influence_count * 4
        if index + required_values > stream_length:
            raise ValueError(
                f"Weighted vertex {len(result)} is truncated: expected "
                f"{required_values} influence values"
            )

        influences: list[WeightedVertexInfluence] = []
        for influence_index in range(influence_count):
            bone_raw, x_raw, y_raw, weight_raw = stream[index : index + 4]
            location = f"vertex {len(result)}, influence {influence_index}"

            resolved_bone = _require_stream_number(
                bone_raw,
                f"Bone index for {location}",
            )
            bone_index = int(resolved_bone)
            if resolved_bone != bone_index:
                raise ValueError("Bone index must be an integer")

            resolved_x = _require_stream_number(
                x_raw,
                f"X coordinate for {location}",
            )
            resolved_y = _require_stream_number(
                y_raw,
                f"Y coordinate for {location}",
            )
            resolved_weight = _require_stream_number(
                weight_raw,
                f"Weight for {location}",
            )
            influences.append(
                WeightedVertexInfluence(
                    bone_index=bone_index,
                    x=float(resolved_x),
                    y=float(resolved_y),
                    weight=float(resolved_weight),
                )
            )
            index += 4
        result.append(WeightedVertex(tuple(influences)))

    if expected_vertex_count is not None and len(result) != expected_vertex_count:
        raise ValueError(
            f"Decoded {len(result)} weighted vertices, expected {expected_vertex_count}"
        )
    return tuple(result)
