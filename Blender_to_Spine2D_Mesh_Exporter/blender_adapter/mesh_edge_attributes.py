"""Blender 5.2 generic edge-attribute access for seams and sharpness."""

from __future__ import annotations

from typing import Any, Iterable


UV_SEAM_ATTRIBUTE = "uv_seam"
SHARP_EDGE_ATTRIBUTE = "sharp_edge"


class MeshEdgeAttributeError(RuntimeError):
    """Raised when a Blender edge boolean attribute is malformed or unwritable."""


def _edge_count(mesh: Any) -> int:
    if mesh is None:
        raise MeshEdgeAttributeError("mesh cannot be None")
    try:
        return len(mesh.edges)
    except Exception as exc:
        raise MeshEdgeAttributeError("Unable to inspect mesh edge count") from exc


def _attribute_collection(mesh: Any) -> Any:
    attributes = getattr(mesh, "attributes", None)
    if attributes is None:
        raise MeshEdgeAttributeError(
            "Blender 5.2 Mesh.attributes API is unavailable"
        )
    return attributes


def _attribute_by_name(mesh: Any, name: str) -> Any | None:
    attributes = _attribute_collection(mesh)
    getter = getattr(attributes, "get", None)
    if not callable(getter):
        raise MeshEdgeAttributeError(
            "Blender 5.2 Mesh.attributes.get() is unavailable"
        )
    try:
        return getter(name)
    except Exception as exc:
        raise MeshEdgeAttributeError(
            f"Unable to read mesh attribute '{name}'"
        ) from exc


def _validate_boolean_edge_attribute(
    attribute: Any,
    *,
    name: str,
    edge_count: int,
) -> None:
    if attribute is None:
        raise MeshEdgeAttributeError(f"Mesh attribute '{name}' is missing")
    domain = str(getattr(attribute, "domain", "") or "")
    data_type = str(getattr(attribute, "data_type", "") or "")
    if domain != "EDGE":
        raise MeshEdgeAttributeError(
            f"Mesh attribute '{name}' must use EDGE domain, got {domain!r}"
        )
    if data_type != "BOOLEAN":
        raise MeshEdgeAttributeError(
            f"Mesh attribute '{name}' must use BOOLEAN data type, got {data_type!r}"
        )
    try:
        data_length = len(attribute.data)
    except Exception as exc:
        raise MeshEdgeAttributeError(
            f"Unable to inspect mesh attribute '{name}' data"
        ) from exc
    if data_length != edge_count:
        raise MeshEdgeAttributeError(
            f"Mesh attribute '{name}' contains {data_length} values for "
            f"{edge_count} edges"
        )


def read_boolean_edge_attribute(
    mesh: Any,
    name: str,
    *,
    missing_value: bool = False,
) -> tuple[bool, ...]:
    """Read one Blender 5.2 BOOLEAN/EDGE attribute in mesh-edge index order."""

    if not isinstance(name, str) or not name.strip():
        raise ValueError("name must be a non-empty string")
    if not isinstance(missing_value, bool):
        raise TypeError("missing_value must be bool")

    edge_count = _edge_count(mesh)
    attribute = _attribute_by_name(mesh, name)
    if attribute is None:
        return tuple(missing_value for _ in range(edge_count))
    _validate_boolean_edge_attribute(
        attribute,
        name=name,
        edge_count=edge_count,
    )

    try:
        return tuple(bool(attribute.data[index].value) for index in range(edge_count))
    except Exception as exc:
        raise MeshEdgeAttributeError(
            f"Unable to read BOOLEAN/EDGE values from '{name}'"
        ) from exc


def write_boolean_edge_attribute(
    mesh: Any,
    name: str,
    values: Iterable[bool],
    *,
    omit_when_all_false: bool = False,
) -> Any | None:
    """Write one Blender 5.2 BOOLEAN/EDGE attribute in mesh-edge index order."""

    if not isinstance(name, str) or not name.strip():
        raise ValueError("name must be a non-empty string")
    if not isinstance(omit_when_all_false, bool):
        raise TypeError("omit_when_all_false must be bool")
    try:
        resolved = tuple(values)
    except Exception as exc:
        raise TypeError("values must be iterable") from exc
    if any(not isinstance(value, bool) for value in resolved):
        raise TypeError("values must contain bool values only")

    edge_count = _edge_count(mesh)
    if len(resolved) != edge_count:
        raise MeshEdgeAttributeError(
            f"Received {len(resolved)} values for '{name}', expected {edge_count}"
        )

    attributes = _attribute_collection(mesh)
    attribute = _attribute_by_name(mesh, name)
    if attribute is None and omit_when_all_false and not any(resolved):
        return None
    if attribute is None:
        creator = getattr(attributes, "new", None)
        if not callable(creator):
            raise MeshEdgeAttributeError(
                "Blender 5.2 Mesh.attributes.new() is unavailable"
            )
        try:
            attribute = creator(name=name, type="BOOLEAN", domain="EDGE")
        except Exception as exc:
            raise MeshEdgeAttributeError(
                f"Unable to create BOOLEAN/EDGE mesh attribute '{name}'"
            ) from exc

    _validate_boolean_edge_attribute(
        attribute,
        name=name,
        edge_count=edge_count,
    )
    try:
        for index, value in enumerate(resolved):
            attribute.data[index].value = value
    except Exception as exc:
        raise MeshEdgeAttributeError(
            f"Unable to write BOOLEAN/EDGE values to '{name}'"
        ) from exc
    return attribute


__all__ = [
    "MeshEdgeAttributeError",
    "SHARP_EDGE_ATTRIBUTE",
    "UV_SEAM_ATTRIBUTE",
    "read_boolean_edge_attribute",
    "write_boolean_edge_attribute",
]
