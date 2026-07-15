"""Encoder and decoder for Spine weighted mesh vertex streams."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Iterable, Sequence, Tuple


@dataclass(frozen=True, slots=True)
class WeightedVertexInfluence:
    bone_index: int
    x: float
    y: float
    weight: float

    def __post_init__(self) -> None:
        if not isinstance(self.bone_index, int) or self.bone_index < 0:
            raise ValueError("bone_index must be a non-negative integer")
        for field_name, value in (("x", self.x), ("y", self.y), ("weight", self.weight)):
            if not isinstance(value, (int, float)) or not isfinite(float(value)):
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
    if expected_vertex_count is not None and expected_vertex_count < 0:
        raise ValueError("expected_vertex_count cannot be negative")

    result: list[WeightedVertex] = []
    index = 0
    stream_length = len(stream)

    while index < stream_length:
        raw_count = stream[index]
        if isinstance(raw_count, bool) or not isinstance(raw_count, (int, float)):
            raise TypeError(f"Influence count at stream index {index} is not numeric")
        influence_count = int(raw_count)
        if float(raw_count) != float(influence_count) or influence_count <= 0:
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
            if isinstance(bone_raw, bool) or not isinstance(bone_raw, (int, float)):
                raise TypeError(
                    f"Bone index for vertex {len(result)}, influence {influence_index} "
                    "is not numeric"
                )
            bone_index = int(bone_raw)
            if float(bone_raw) != float(bone_index):
                raise ValueError("Bone index must be an integer")
            influences.append(
                WeightedVertexInfluence(
                    bone_index=bone_index,
                    x=float(x_raw),
                    y=float(y_raw),
                    weight=float(weight_raw),
                )
            )
            index += 4
        result.append(WeightedVertex(tuple(influences)))

    if expected_vertex_count is not None and len(result) != expected_vertex_count:
        raise ValueError(
            f"Decoded {len(result)} weighted vertices, expected {expected_vertex_count}"
        )
    return tuple(result)
