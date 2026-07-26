"""Classify Blender source UV layers without mutating the user's Mesh."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Any, Iterable, Iterator


class SourceUvIntegrityError(RuntimeError):
    """Raised when a required source UV layer is missing or malformed."""


class SourceUvMutationError(RuntimeError):
    """Raised when Rewrite changed the user's source UV datablock state."""


class ObjectModeRequiredError(RuntimeError):
    """Raised when source geometry is requested from an unsafe Blender mode."""


@dataclass(frozen=True, slots=True)
class SourceUvLayerStatus:
    name: str
    value_count: int
    loop_count: int
    active: bool
    active_render: bool
    required: bool

    @property
    def valid(self) -> bool:
        return self.value_count == self.loop_count


@dataclass(frozen=True, slots=True)
class SourceUvIntegrityReport:
    loop_count: int
    required_layer_names: tuple[str, ...]
    layers: tuple[SourceUvLayerStatus, ...]
    missing_required_layer_names: tuple[str, ...]

    @property
    def readable_layer_names(self) -> tuple[str, ...]:
        return tuple(layer.name for layer in self.layers if layer.valid)

    @property
    def ignored_malformed_layer_names(self) -> tuple[str, ...]:
        return tuple(
            layer.name for layer in self.layers if not layer.valid and not layer.required
        )

    @property
    def malformed_required_layer_names(self) -> tuple[str, ...]:
        return tuple(
            layer.name for layer in self.layers if not layer.valid and layer.required
        )

    def require_usable(self) -> tuple[str, ...]:
        failures: list[str] = []
        if self.missing_required_layer_names:
            failures.append(
                "missing required UV layers=" + repr(self.missing_required_layer_names)
            )
        if self.malformed_required_layer_names:
            details = tuple(
                (layer.name, layer.value_count, layer.loop_count)
                for layer in self.layers
                if layer.name in self.malformed_required_layer_names
            )
            failures.append("malformed required UV layers=" + repr(details))
        if failures:
            raise SourceUvIntegrityError(
                "Source UV integrity preflight failed: "
                + "; ".join(failures)
                + ". Repair the UV layer, or disconnect it from the material/source "
                "boundary setting before export."
            )
        return self.readable_layer_names


@dataclass(frozen=True, slots=True)
class SourceUvLayerFingerprint:
    name: str
    value_count: int
    active: bool
    active_render: bool
    coordinates: tuple[tuple[float, float], ...]


@dataclass(frozen=True, slots=True)
class SourceUvFingerprint:
    mesh_identity: int
    loop_count: int
    layers: tuple[SourceUvLayerFingerprint, ...]


def _rna_identity(value: Any) -> int:
    pointer = getattr(value, "as_pointer", None)
    if callable(pointer):
        try:
            return int(pointer())
        except Exception:
            pass
    return id(value)


def _uv_layers(mesh: Any) -> tuple[Any, ...]:
    layers = getattr(mesh, "uv_layers", None)
    if layers is None:
        return ()
    try:
        return tuple(layers)
    except Exception as exc:
        raise SourceUvIntegrityError("Unable to inspect source Mesh UV layers") from exc


def _uv_layer_names(mesh: Any) -> frozenset[str]:
    return frozenset(
        name
        for layer in _uv_layers(mesh)
        if (name := str(getattr(layer, "name", "") or "").strip())
    )


def _active_uv_name(mesh: Any) -> str | None:
    layers = getattr(mesh, "uv_layers", None)
    active = None if layers is None else getattr(layers, "active", None)
    value = str(getattr(active, "name", "") or "").strip()
    return value or None


def _active_render_uv_name(mesh: Any) -> str | None:
    for layer in _uv_layers(mesh):
        if bool(getattr(layer, "active_render", False)):
            value = str(getattr(layer, "name", "") or "").strip()
            if value:
                return value
    return _active_uv_name(mesh)


def _material_slots(obj: Any) -> tuple[Any, ...]:
    try:
        return tuple(getattr(obj, "material_slots", ()) or ())
    except Exception:
        return ()


def _iter_material_node_trees(obj: Any) -> Iterator[Any]:
    pending: list[Any] = []
    for slot in _material_slots(obj):
        material = getattr(slot, "material", None)
        tree = getattr(material, "node_tree", None)
        if tree is not None:
            pending.append(tree)

    visited: set[int] = set()
    while pending:
        tree = pending.pop()
        identity = _rna_identity(tree)
        if identity in visited:
            continue
        visited.add(identity)
        yield tree
        try:
            nodes = tuple(getattr(tree, "nodes", ()) or ())
        except Exception:
            nodes = ()
        for node in nodes:
            nested = getattr(node, "node_tree", None)
            if nested is not None:
                pending.append(nested)


def _node_type(node: Any) -> str:
    return str(
        getattr(node, "type", None)
        or getattr(node, "bl_idname", None)
        or ""
    ).strip().upper()


def _socket_by_name(sockets: Any, name: str) -> Any | None:
    getter = getattr(sockets, "get", None)
    if callable(getter):
        try:
            return getter(name)
        except Exception:
            return None
    try:
        for socket in sockets:
            if str(getattr(socket, "name", "")) == name:
                return socket
    except Exception:
        return None
    return None


def material_required_uv_layer_names(obj: Any) -> tuple[str, ...]:
    """Return source UV layers explicitly referenced by material node trees.

    Every nested material node tree is inspected. Attribute nodes only become UV
    requirements when their name exactly matches a real UV layer; color and custom
    geometry attributes therefore cannot create false missing-UV blockers.
    """

    mesh = getattr(obj, "data", None)
    if mesh is None:
        return ()
    available_uv_names = _uv_layer_names(mesh)
    active_name = _active_render_uv_name(mesh)
    required: set[str] = set()

    for tree in _iter_material_node_trees(obj):
        try:
            nodes = tuple(getattr(tree, "nodes", ()) or ())
        except Exception:
            nodes = ()
        for node in nodes:
            node_type = _node_type(node)
            if node_type in {"UVMAP", "SHADERNODEUVMAP"}:
                value = str(getattr(node, "uv_map", "") or "").strip()
                if value:
                    required.add(value)
                continue
            if node_type in {"TEX_COORD", "SHADERNODETEXCOORD"}:
                uv_socket = _socket_by_name(getattr(node, "outputs", ()), "UV")
                if uv_socket is not None and bool(getattr(uv_socket, "is_linked", False)):
                    if active_name:
                        required.add(active_name)
                continue
            if node_type in {"ATTRIBUTE", "SHADERNODEATTRIBUTE"}:
                value = str(getattr(node, "attribute_name", "") or "").strip()
                if value in available_uv_names:
                    required.add(value)

    return tuple(sorted(required))


def required_source_uv_layer_names(obj: Any, settings: Any) -> tuple[str, ...]:
    """Resolve material and source-boundary UV requirements for one request."""

    required = set(material_required_uv_layer_names(obj))
    mesh = getattr(obj, "data", None)
    mode = str(
        getattr(getattr(settings, "source_uv_boundary_mode", None), "value", None)
        or getattr(settings, "source_uv_boundary_mode", "DISABLED")
        or "DISABLED"
    ).strip().upper()
    if mode == "EXPLICIT_LAYER":
        value = str(getattr(settings, "source_uv_boundary_layer_name", "") or "").strip()
        if value:
            required.add(value)
    elif mode == "ACTIVE_LAYER_LEGACY" and mesh is not None:
        active_name = _active_uv_name(mesh)
        if active_name:
            required.add(active_name)
    return tuple(sorted(required))


def inspect_source_uv_integrity(
    obj: Any,
    *,
    required_layer_names: Iterable[str] = (),
) -> SourceUvIntegrityReport:
    """Classify every source UV layer without reading malformed coordinates."""

    if obj is None or getattr(obj, "type", None) != "MESH":
        raise SourceUvIntegrityError("obj must be a Blender MESH object")
    mesh = getattr(obj, "data", None)
    if mesh is None:
        raise SourceUvIntegrityError("obj.data is missing")
    try:
        loop_count = len(mesh.loops)
    except Exception as exc:
        raise SourceUvIntegrityError("Unable to inspect source Mesh loops") from exc

    required = tuple(
        sorted(
            {
                str(name).strip()
                for name in required_layer_names
                if str(name).strip()
            }
        )
    )
    required_set = set(required)
    active_name = _active_uv_name(mesh)
    layers: list[SourceUvLayerStatus] = []
    available_names: set[str] = set()
    for layer in _uv_layers(mesh):
        name = str(getattr(layer, "name", "") or "").strip()
        if not name:
            raise SourceUvIntegrityError("Source Mesh contains an unnamed UV layer")
        available_names.add(name)
        collection = getattr(layer, "uv", None)
        try:
            value_count = 0 if collection is None else len(collection)
        except Exception as exc:
            raise SourceUvIntegrityError(
                f"Unable to inspect UV layer '{name}' value count"
            ) from exc
        layers.append(
            SourceUvLayerStatus(
                name=name,
                value_count=value_count,
                loop_count=loop_count,
                active=name == active_name,
                active_render=bool(getattr(layer, "active_render", False)),
                required=name in required_set,
            )
        )

    return SourceUvIntegrityReport(
        loop_count=loop_count,
        required_layer_names=required,
        layers=tuple(layers),
        missing_required_layer_names=tuple(sorted(required_set - available_names)),
    )


def resolve_readable_source_uv_layer_names(
    obj: Any,
    settings: Any,
) -> SourceUvIntegrityReport:
    """Return a validated report whose readable layers are safe for MeshSnapshot."""

    required = required_source_uv_layer_names(obj, settings)
    report = inspect_source_uv_integrity(obj, required_layer_names=required)
    report.require_usable()
    return report


def _read_coordinate(value: Any, *, layer_name: str, index: int) -> tuple[float, float]:
    try:
        coordinate = (float(value[0]), float(value[1]))
    except Exception as exc:
        raise SourceUvIntegrityError(
            f"Unable to read UV layer '{layer_name}' value {index}"
        ) from exc
    if not all(isfinite(component) for component in coordinate):
        raise SourceUvIntegrityError(
            f"UV layer '{layer_name}' value {index} contains a non-finite component"
        )
    return coordinate


def capture_source_uv_fingerprint(obj: Any) -> SourceUvFingerprint:
    """Capture names, roles, counts, and valid coordinates of every source UV layer."""

    if obj is None or getattr(obj, "type", None) != "MESH":
        raise SourceUvIntegrityError("obj must be a Blender MESH object")
    mesh = getattr(obj, "data", None)
    if mesh is None:
        raise SourceUvIntegrityError("obj.data is missing")
    loop_count = len(mesh.loops)
    active_name = _active_uv_name(mesh)
    fingerprints: list[SourceUvLayerFingerprint] = []
    for layer in _uv_layers(mesh):
        name = str(getattr(layer, "name", "") or "").strip()
        collection = getattr(layer, "uv", None)
        value_count = 0 if collection is None else len(collection)
        coordinates: tuple[tuple[float, float], ...] = ()
        if collection is not None and value_count == loop_count:
            coordinates = tuple(
                _read_coordinate(
                    collection[index].vector,
                    layer_name=name,
                    index=index,
                )
                for index in range(value_count)
            )
        fingerprints.append(
            SourceUvLayerFingerprint(
                name=name,
                value_count=value_count,
                active=name == active_name,
                active_render=bool(getattr(layer, "active_render", False)),
                coordinates=coordinates,
            )
        )
    return SourceUvFingerprint(
        mesh_identity=_rna_identity(mesh),
        loop_count=loop_count,
        layers=tuple(fingerprints),
    )


def require_source_uv_unchanged(before: SourceUvFingerprint, obj: Any) -> None:
    """Fail when Analyze/export replaced or modified source UV state."""

    if not isinstance(before, SourceUvFingerprint):
        raise TypeError("before must be SourceUvFingerprint")
    after = capture_source_uv_fingerprint(obj)
    if after != before:
        raise SourceUvMutationError(
            "Rewrite changed the source Mesh UV state. The operation was aborted to "
            f"protect user data; before={before!r}, after={after!r}"
        )


def require_object_mode(context: Any) -> None:
    """Reject stale Mesh RNA reads while Blender is in Edit or another data mode."""

    if context is None:
        return
    mode = str(getattr(context, "mode", "OBJECT") or "OBJECT").strip().upper()
    if mode == "OBJECT":
        return
    mode_label = "Edit Mode" if mode.startswith("EDIT") else mode.replace("_", " ").title()
    raise ObjectModeRequiredError(
        f"Finish or cancel {mode_label} before exporting; Spine2D Rewrite reads "
        "immutable source Mesh data in Object Mode only."
    )


__all__ = [
    "ObjectModeRequiredError",
    "SourceUvFingerprint",
    "SourceUvIntegrityError",
    "SourceUvIntegrityReport",
    "SourceUvLayerFingerprint",
    "SourceUvLayerStatus",
    "SourceUvMutationError",
    "capture_source_uv_fingerprint",
    "inspect_source_uv_integrity",
    "material_required_uv_layer_names",
    "required_source_uv_layer_names",
    "require_object_mode",
    "require_source_uv_unchanged",
    "resolve_readable_source_uv_layer_names",
]
