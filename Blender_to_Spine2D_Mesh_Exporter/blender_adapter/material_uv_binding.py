"""Bind implicit material UV sampling to one explicit source UV layer.

Only temporary material copies owned by the semantic bake pipeline are mutated.
The user's source materials and shared node-group datablocks are never changed.
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Any, Iterable, Tuple
from uuid import uuid4


logger = logging.getLogger(__name__)


class MaterialUvBindingError(RuntimeError):
    """Raised when implicit source-UV consumers cannot be rebound safely."""


@dataclass(frozen=True, slots=True)
class MaterialUvBindingReport:
    """Summary for one temporary material whose implicit UV inputs were rebound."""

    material_name: str
    uv_layer_name: str
    texture_coordinate_link_count: int
    unlinked_image_texture_count: int

    def __post_init__(self) -> None:
        if not isinstance(self.material_name, str) or not self.material_name.strip():
            raise ValueError("material_name must be a non-empty string")
        if not isinstance(self.uv_layer_name, str) or not self.uv_layer_name.strip():
            raise ValueError("uv_layer_name must be a non-empty string")
        for field_name in (
            "texture_coordinate_link_count",
            "unlinked_image_texture_count",
        ):
            value = getattr(self, field_name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError(f"{field_name} must be a non-negative integer")

    @property
    def consumer_count(self) -> int:
        return self.texture_coordinate_link_count + self.unlinked_image_texture_count


def _rna_identity(value: Any) -> int:
    pointer = getattr(value, "as_pointer", None)
    if callable(pointer):
        try:
            return int(pointer())
        except (TypeError, ValueError, OverflowError, RuntimeError, ReferenceError):
            pass
    return id(value)


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
            resolved = getter(name)
            if resolved is not None:
                return resolved
        except Exception:
            logger.debug("Socket lookup by name failed", exc_info=True)
    try:
        for socket in sockets:
            if str(getattr(socket, "name", "") or "") == name:
                return socket
    except Exception:
        return None
    return None


def _incoming_links(socket: Any | None) -> Tuple[Any, ...]:
    if socket is None:
        return ()
    try:
        return tuple(getattr(socket, "links", ()) or ())
    except Exception:
        return ()


def _outgoing_links(socket: Any | None) -> Tuple[Any, ...]:
    if socket is None:
        return ()
    try:
        return tuple(getattr(socket, "links", ()) or ())
    except Exception:
        return ()


def _material_node_tree(material: Any) -> Any:
    if material is None:
        raise MaterialUvBindingError("temporary material cannot be None")
    node_tree = getattr(material, "node_tree", None)
    if node_tree is None:
        raise MaterialUvBindingError("temporary material has no node tree")
    if getattr(node_tree, "nodes", None) is None or getattr(node_tree, "links", None) is None:
        raise MaterialUvBindingError("temporary material node tree is incomplete")
    return node_tree


def _material_name(material: Any) -> str:
    value = str(
        getattr(material, "name_full", None)
        or getattr(material, "name", None)
        or ""
    ).strip()
    if not value:
        raise MaterialUvBindingError("temporary material has an empty name")
    return value


def _restore_original_links(
    node_tree: Any,
    original_links: Tuple[Tuple[Any, Any], ...],
) -> list[str]:
    failures: list[str] = []
    for from_socket, to_socket in original_links:
        try:
            node_tree.links.new(from_socket, to_socket)
        except Exception as exc:
            failures.append(f"restore original UV link: {exc}")
    return failures


def bind_material_implicit_uv_sampling(
    material: Any,
    uv_layer_name: str,
    *,
    excluded_nodes: Iterable[Any] = (),
) -> MaterialUvBindingReport:
    """Replace root-graph implicit UV sampling with one explicit UV Map node.

    The copied graph is changed only where Blender would otherwise choose a UV layer
    implicitly:

    * outgoing links from ``Texture Coordinate: UV``;
    * Image Texture ``Vector`` inputs with no incoming link.

    Existing explicit UV Map nodes and all already-linked Image Texture vectors are
    preserved. Nested group datablocks are intentionally not mutated because they may
    still be shared with the user's source material.
    """

    if not isinstance(uv_layer_name, str) or not uv_layer_name.strip():
        raise ValueError("uv_layer_name must be a non-empty string")
    resolved_uv_name = uv_layer_name.strip()
    try:
        excluded = frozenset(_rna_identity(node) for node in tuple(excluded_nodes))
    except TypeError as exc:
        raise TypeError("excluded_nodes must be iterable") from exc

    node_tree = _material_node_tree(material)
    material_name = _material_name(material)
    try:
        nodes = tuple(node_tree.nodes)
    except Exception as exc:
        raise MaterialUvBindingError(
            f"Unable to inspect nodes in temporary material '{material_name}'"
        ) from exc

    links_to_replace: list[Tuple[Any, Any, Any]] = []
    unlinked_image_inputs: list[Any] = []
    target_socket_ids: set[int] = set()

    for node in nodes:
        if _rna_identity(node) in excluded:
            continue
        node_type = _node_type(node)
        if node_type in {"TEX_COORD", "SHADERNODETEXCOORD"}:
            uv_output = _socket_by_name(getattr(node, "outputs", ()), "UV")
            if uv_output is None:
                raise MaterialUvBindingError(
                    f"Texture Coordinate node '{getattr(node, 'name', '<unnamed>')}' "
                    f"in '{material_name}' has no UV output"
                )
            for link in _outgoing_links(uv_output):
                to_socket = getattr(link, "to_socket", None)
                if to_socket is None:
                    raise MaterialUvBindingError(
                        f"Texture Coordinate UV link in '{material_name}' has no target socket"
                    )
                target_id = _rna_identity(to_socket)
                if target_id in target_socket_ids:
                    continue
                target_socket_ids.add(target_id)
                links_to_replace.append((link, uv_output, to_socket))
            continue

        if node_type in {"TEX_IMAGE", "SHADERNODETEXIMAGE"}:
            vector_input = _socket_by_name(getattr(node, "inputs", ()), "Vector")
            if vector_input is None or _incoming_links(vector_input):
                continue
            target_id = _rna_identity(vector_input)
            if target_id in target_socket_ids:
                continue
            target_socket_ids.add(target_id)
            unlinked_image_inputs.append(vector_input)

    consumer_count = len(links_to_replace) + len(unlinked_image_inputs)
    if consumer_count == 0:
        return MaterialUvBindingReport(
            material_name=material_name,
            uv_layer_name=resolved_uv_name,
            texture_coordinate_link_count=0,
            unlinked_image_texture_count=0,
        )

    uv_node = None
    removed_original_links: list[Tuple[Any, Any]] = []
    added_links: list[Any] = []
    try:
        uv_node = node_tree.nodes.new(type="ShaderNodeUVMap")
        uv_node.name = f"__Spine2D_SourceUV_{uuid4().hex}"
        uv_node.label = f"Spine2D source UV: {resolved_uv_name}"
        uv_node.uv_map = resolved_uv_name
        if str(getattr(uv_node, "uv_map", "") or "") != resolved_uv_name:
            raise MaterialUvBindingError(
                f"Blender did not keep UV Map node binding '{resolved_uv_name}' "
                f"in temporary material '{material_name}'"
            )
        uv_output = _socket_by_name(getattr(uv_node, "outputs", ()), "UV")
        if uv_output is None:
            raise MaterialUvBindingError(
                f"Temporary UV Map node in '{material_name}' has no UV output"
            )

        for link, from_socket, to_socket in links_to_replace:
            node_tree.links.remove(link)
            removed_original_links.append((from_socket, to_socket))
            added_links.append(node_tree.links.new(uv_output, to_socket))
        for to_socket in unlinked_image_inputs:
            added_links.append(node_tree.links.new(uv_output, to_socket))

        for to_socket in tuple(item[2] for item in links_to_replace) + tuple(
            unlinked_image_inputs
        ):
            incoming = _incoming_links(to_socket)
            if len(incoming) != 1:
                raise MaterialUvBindingError(
                    f"Implicit UV target in '{material_name}' has {len(incoming)} links "
                    "after explicit source-UV binding"
                )
            actual_source = getattr(incoming[0], "from_socket", None)
            if _rna_identity(actual_source) != _rna_identity(uv_output):
                raise MaterialUvBindingError(
                    f"Implicit UV target in '{material_name}' is not connected to "
                    f"the explicit '{resolved_uv_name}' UV Map node"
                )
    except Exception as exc:
        rollback_failures: list[str] = []
        for link in reversed(added_links):
            try:
                node_tree.links.remove(link)
            except Exception as rollback_exc:
                rollback_failures.append(f"remove explicit UV link: {rollback_exc}")
        if uv_node is not None:
            try:
                node_tree.nodes.remove(uv_node)
            except Exception as rollback_exc:
                rollback_failures.append(f"remove explicit UV Map node: {rollback_exc}")
        rollback_failures.extend(
            _restore_original_links(node_tree, tuple(removed_original_links))
        )
        if rollback_failures:
            raise MaterialUvBindingError(
                f"Unable to bind source UV '{resolved_uv_name}' in temporary material "
                f"'{material_name}', and rollback was incomplete: "
                + "; ".join(rollback_failures)
            ) from exc
        if isinstance(exc, MaterialUvBindingError):
            raise
        raise MaterialUvBindingError(
            f"Unable to bind source UV '{resolved_uv_name}' in temporary material "
            f"'{material_name}': {exc}"
        ) from exc

    report = MaterialUvBindingReport(
        material_name=material_name,
        uv_layer_name=resolved_uv_name,
        texture_coordinate_link_count=len(links_to_replace),
        unlinked_image_texture_count=len(unlinked_image_inputs),
    )
    logger.debug(
        "Bound %d implicit UV consumers in temporary material '%s' to '%s' "
        "(Texture Coordinate links=%d, unlinked Image Texture inputs=%d)",
        report.consumer_count,
        report.material_name,
        report.uv_layer_name,
        report.texture_coordinate_link_count,
        report.unlinked_image_texture_count,
    )
    return report


def bind_materials_implicit_uv_sampling(
    materials: Tuple[Any, ...],
    uv_layer_name: str,
    *,
    used_material_indices: Tuple[int, ...],
    excluded_nodes: Iterable[Any] = (),
) -> Tuple[MaterialUvBindingReport, ...]:
    """Bind implicit source-UV consumers for used temporary material slots only."""

    if not isinstance(materials, tuple):
        raise TypeError("materials must be tuple")
    if not isinstance(used_material_indices, tuple):
        raise TypeError("used_material_indices must be tuple")
    if any(
        not isinstance(index, int) or isinstance(index, bool) or index < 0
        for index in used_material_indices
    ):
        raise ValueError("used_material_indices must contain non-negative integers")
    if len(used_material_indices) != len(set(used_material_indices)):
        raise ValueError("used_material_indices cannot contain duplicates")
    if used_material_indices and max(used_material_indices) >= len(materials):
        raise MaterialUvBindingError(
            f"Used material slot {max(used_material_indices)} is outside "
            f"temporary material range [0, {len(materials)})"
        )

    excluded = tuple(excluded_nodes)
    return tuple(
        bind_material_implicit_uv_sampling(
            materials[index],
            uv_layer_name,
            excluded_nodes=excluded,
        )
        for index in used_material_indices
    )


__all__ = [
    "MaterialUvBindingError",
    "MaterialUvBindingReport",
    "bind_material_implicit_uv_sampling",
    "bind_materials_implicit_uv_sampling",
]
