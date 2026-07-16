"""Temporarily expose semantic channels on copied Blender materials.

Only material copies owned by ``temporary_bake_materials`` are mutated. Original
Material Output links are restored and every temporary node is removed in ``finally``.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import logging
from typing import Any, Iterable, Iterator, Tuple
from uuid import uuid4

from ..domain.baking import (
    BakePassPlan,
    MaterialPreparationMode,
    MaterialSlotPreparation,
)

logger = logging.getLogger(__name__)


class BakeMaterialPreparationError(RuntimeError):
    """Raised when a semantic channel cannot be exposed on a copied material."""


@dataclass(frozen=True, slots=True)
class _ValueExpression:
    socket: Any | None = None
    constant: float = 0.0


@dataclass(slots=True)
class _PreparedMutation:
    node_tree: Any
    output_surface_socket: Any
    original_surface_sources: Tuple[Any, ...]
    temporary_nodes: list[Any]


def _node_type(node: Any) -> str:
    return str(getattr(node, "type", "") or "")


def _node_name(node: Any) -> str:
    return str(getattr(node, "name", "") or _node_type(node) or "Node")


def _socket_name(socket: Any) -> str:
    return str(getattr(socket, "name", "") or "Socket")


def _input_socket(node: Any, name: str, *, index: int | None = None) -> Any | None:
    inputs = getattr(node, "inputs", None)
    if inputs is None:
        return None
    getter = getattr(inputs, "get", None)
    if callable(getter):
        try:
            result = getter(name)
            if result is not None:
                return result
        except Exception:
            logger.debug("Input socket lookup by name failed", exc_info=True)
    if index is not None:
        try:
            return inputs[index]
        except Exception:
            pass
    try:
        return next(socket for socket in inputs if _socket_name(socket) == name)
    except Exception:
        return None


def _active_material_output(node_tree: Any) -> Any:
    try:
        outputs = tuple(
            node for node in node_tree.nodes if _node_type(node) == "OUTPUT_MATERIAL"
        )
    except Exception as exc:
        raise BakeMaterialPreparationError("Unable to inspect copied material nodes") from exc
    if not outputs:
        raise BakeMaterialPreparationError("Copied material has no Material Output node")
    active = tuple(node for node in outputs if bool(getattr(node, "is_active_output", False)))
    return active[0] if active else outputs[0]


def _incoming_links(socket: Any | None) -> Tuple[Any, ...]:
    if socket is None:
        return ()
    try:
        return tuple(getattr(socket, "links", ()))
    except Exception:
        return ()


def _linked_source_node(socket: Any | None) -> Any | None:
    links = _incoming_links(socket)
    return None if not links else getattr(links[0], "from_node", None)


def _linked_source_socket(socket: Any | None) -> Any | None:
    links = _incoming_links(socket)
    return None if not links else getattr(links[0], "from_socket", None)


def _numeric_default(socket: Any | None, default: float) -> float:
    if socket is None:
        return float(default)
    value = getattr(socket, "default_value", default)
    try:
        if hasattr(value, "__len__") and not isinstance(value, (str, bytes)):
            return float(value[0])
        return float(value)
    except Exception:
        return float(default)


def _value_from_input(socket: Any | None, *, default: float) -> _ValueExpression:
    source = _linked_source_socket(socket)
    return (
        _ValueExpression(socket=source)
        if source is not None
        else _ValueExpression(constant=_numeric_default(socket, default))
    )


def _assign_socket_constant(target_socket: Any, value: float) -> None:
    """Assign one scalar to scalar/vector/color RNA socket defaults generically."""

    current = getattr(target_socket, "default_value", None)
    try:
        if hasattr(current, "__len__") and not isinstance(current, (str, bytes)):
            length = len(current)
            if length <= 0:
                raise ValueError("target socket has an empty sequence default")
            resolved = [float(value)] * length
            if length == 4:
                resolved[3] = 1.0
            target_socket.default_value = tuple(resolved)
        else:
            target_socket.default_value = float(value)
    except Exception as exc:
        raise BakeMaterialPreparationError(
            f"Unable to assign opacity constant to socket '{_socket_name(target_socket)}'"
        ) from exc


def _connect_value(node_tree: Any, expression: _ValueExpression, target_socket: Any) -> None:
    if expression.socket is not None:
        try:
            node_tree.links.new(expression.socket, target_socket)
            return
        except Exception as exc:
            raise BakeMaterialPreparationError(
                "Unable to connect opacity expression to temporary node"
            ) from exc
    _assign_socket_constant(target_socket, expression.constant)


def _new_math(
    node_tree: Any,
    temporary_nodes: list[Any],
    *,
    operation: str,
    token: str,
    label: str,
    clamp: bool = False,
) -> Any:
    try:
        node = node_tree.nodes.new(type="ShaderNodeMath")
        node.name = f"__Spine2D_Alpha_{label}_{token}_{len(temporary_nodes)}"
        node.label = f"Spine2D temporary alpha {label}"
        node.operation = operation
        if hasattr(node, "use_clamp"):
            node.use_clamp = clamp
        temporary_nodes.append(node)
        return node
    except Exception as exc:
        raise BakeMaterialPreparationError(
            f"Unable to create temporary alpha math node '{operation}'"
        ) from exc


def _multiply(
    node_tree: Any,
    left: _ValueExpression,
    right: _ValueExpression,
    temporary_nodes: list[Any],
    token: str,
) -> _ValueExpression:
    node = _new_math(
        node_tree,
        temporary_nodes,
        operation="MULTIPLY",
        token=token,
        label="multiply",
    )
    _connect_value(node_tree, left, node.inputs[0])
    _connect_value(node_tree, right, node.inputs[1])
    return _ValueExpression(socket=node.outputs[0])


def _one_minus(
    node_tree: Any,
    value: _ValueExpression,
    temporary_nodes: list[Any],
    token: str,
) -> _ValueExpression:
    node = _new_math(
        node_tree,
        temporary_nodes,
        operation="SUBTRACT",
        token=token,
        label="one_minus",
    )
    _assign_socket_constant(node.inputs[0], 1.0)
    _connect_value(node_tree, value, node.inputs[1])
    return _ValueExpression(socket=node.outputs[0])


def _add_clamped(
    node_tree: Any,
    left: _ValueExpression,
    right: _ValueExpression,
    temporary_nodes: list[Any],
    token: str,
) -> _ValueExpression:
    node = _new_math(
        node_tree,
        temporary_nodes,
        operation="ADD",
        token=token,
        label="add",
        clamp=True,
    )
    _connect_value(node_tree, left, node.inputs[0])
    _connect_value(node_tree, right, node.inputs[1])
    return _ValueExpression(socket=node.outputs[0])


def _shader_input_node(node: Any, index: int) -> Any | None:
    inputs = getattr(node, "inputs", None)
    if inputs is None:
        return None
    try:
        return _linked_source_node(inputs[index])
    except Exception:
        return None


def _opacity_from_shader(
    node_tree: Any,
    shader_node: Any | None,
    temporary_nodes: list[Any],
    token: str,
    visiting: set[str],
) -> _ValueExpression:
    if shader_node is None:
        return _ValueExpression(constant=0.0)

    name = _node_name(shader_node)
    if name in visiting:
        raise BakeMaterialPreparationError(
            f"Shader graph cycle detected while extracting alpha at '{name}'"
        )
    visiting.add(name)
    try:
        node_type = _node_type(shader_node)
        if node_type in {"BSDF_TRANSPARENT", "HOLDOUT"}:
            return _ValueExpression(constant=0.0)
        if node_type == "BSDF_PRINCIPLED":
            return _value_from_input(_input_socket(shader_node, "Alpha"), default=1.0)
        if node_type == "MIX_SHADER":
            factor = _value_from_input(
                _input_socket(shader_node, "Fac", index=0),
                default=0.5,
            )
            opacity_a = _opacity_from_shader(
                node_tree,
                _shader_input_node(shader_node, 1),
                temporary_nodes,
                token,
                visiting,
            )
            opacity_b = _opacity_from_shader(
                node_tree,
                _shader_input_node(shader_node, 2),
                temporary_nodes,
                token,
                visiting,
            )
            return _add_clamped(
                node_tree,
                _multiply(
                    node_tree,
                    opacity_a,
                    _one_minus(node_tree, factor, temporary_nodes, token),
                    temporary_nodes,
                    token,
                ),
                _multiply(node_tree, opacity_b, factor, temporary_nodes, token),
                temporary_nodes,
                token,
            )
        if node_type == "ADD_SHADER":
            return _add_clamped(
                node_tree,
                _opacity_from_shader(
                    node_tree,
                    _shader_input_node(shader_node, 0),
                    temporary_nodes,
                    token,
                    visiting,
                ),
                _opacity_from_shader(
                    node_tree,
                    _shader_input_node(shader_node, 1),
                    temporary_nodes,
                    token,
                    visiting,
                ),
                temporary_nodes,
                token,
            )
        return _ValueExpression(constant=1.0)
    finally:
        visiting.remove(name)


def _remove_surface_links(node_tree: Any, surface_socket: Any) -> Tuple[Any, ...]:
    sources = tuple(
        getattr(link, "from_socket", None)
        for link in _incoming_links(surface_socket)
        if getattr(link, "from_socket", None) is not None
    )
    try:
        for link in tuple(_incoming_links(surface_socket)):
            node_tree.links.remove(link)
    except Exception as exc:
        raise BakeMaterialPreparationError(
            "Unable to detach copied Material Output surface"
        ) from exc
    return sources


def _prepare_alpha_material(
    material: Any,
    mode: MaterialPreparationMode,
    *,
    token: str,
) -> _PreparedMutation:
    try:
        material.use_nodes = True
    except Exception as exc:
        raise BakeMaterialPreparationError("Unable to enable copied material nodes") from exc
    node_tree = getattr(material, "node_tree", None)
    if node_tree is None:
        raise BakeMaterialPreparationError("Copied material has no node tree")

    output = _active_material_output(node_tree)
    surface_socket = _input_socket(output, "Surface")
    if surface_socket is None:
        raise BakeMaterialPreparationError("Material Output has no Surface input")
    original_nodes = tuple(
        getattr(link, "from_node", None)
        for link in _incoming_links(surface_socket)
        if getattr(link, "from_node", None) is not None
    )
    original_sources = _remove_surface_links(node_tree, surface_socket)
    temporary_nodes: list[Any] = []

    try:
        if mode is MaterialPreparationMode.OPAQUE_ALPHA_TO_EMISSION:
            opacity = _ValueExpression(constant=1.0)
        elif mode is MaterialPreparationMode.EXTRACT_ALPHA_TO_EMISSION:
            opacity = _opacity_from_shader(
                node_tree,
                original_nodes[0] if original_nodes else None,
                temporary_nodes,
                token,
                set(),
            )
        else:
            raise BakeMaterialPreparationError(
                f"Unsupported material preparation mode: {mode.value}"
            )

        emission = node_tree.nodes.new(type="ShaderNodeEmission")
        emission.name = f"__Spine2D_Alpha_Emission_{token}_{len(temporary_nodes)}"
        emission.label = "Spine2D temporary alpha output"
        temporary_nodes.append(emission)
        _connect_value(node_tree, opacity, emission.inputs["Color"])
        _assign_socket_constant(emission.inputs["Strength"], 1.0)
        node_tree.links.new(emission.outputs["Emission"], surface_socket)
    except Exception:
        for node in reversed(temporary_nodes):
            try:
                node_tree.nodes.remove(node)
            except Exception:
                logger.exception("Failed to remove partially created alpha node")
        for source in original_sources:
            try:
                node_tree.links.new(source, surface_socket)
            except Exception:
                logger.exception("Failed to restore copied material after preparation error")
        raise

    return _PreparedMutation(
        node_tree=node_tree,
        output_surface_socket=surface_socket,
        original_surface_sources=original_sources,
        temporary_nodes=temporary_nodes,
    )


def _restore_mutation(mutation: _PreparedMutation) -> None:
    failures: list[str] = []
    try:
        for link in tuple(_incoming_links(mutation.output_surface_socket)):
            mutation.node_tree.links.remove(link)
    except Exception as exc:
        failures.append(f"remove temporary output links: {exc}")

    for node in reversed(mutation.temporary_nodes):
        try:
            mutation.node_tree.nodes.remove(node)
        except Exception as exc:
            failures.append(f"remove node '{_node_name(node)}': {exc}")
    for source in mutation.original_surface_sources:
        try:
            mutation.node_tree.links.new(source, mutation.output_surface_socket)
        except Exception as exc:
            failures.append(f"restore original Surface link: {exc}")
    if failures:
        raise BakeMaterialPreparationError(
            "Unable to restore copied material: " + "; ".join(failures)
        )


def _preparation_map(
    preparations: Iterable[MaterialSlotPreparation],
) -> dict[int, MaterialPreparationMode]:
    result: dict[int, MaterialPreparationMode] = {}
    for item in preparations:
        if not isinstance(item, MaterialSlotPreparation):
            raise TypeError("preparations must contain MaterialSlotPreparation")
        if item.slot_index in result:
            raise BakeMaterialPreparationError(
                f"Duplicate material preparation for slot {item.slot_index}"
            )
        result[item.slot_index] = item.mode
    return result


@contextmanager
def temporary_prepare_material_pass(
    materials: Tuple[Any, ...],
    pass_plan: BakePassPlan,
) -> Iterator[None]:
    """Apply and restore one typed preparation plan on copied materials."""

    if not isinstance(materials, tuple):
        raise TypeError("materials must be tuple")
    if not isinstance(pass_plan, BakePassPlan):
        raise TypeError("pass_plan must be BakePassPlan")
    modes = _preparation_map(pass_plan.material_preparations)
    if not modes or all(mode is MaterialPreparationMode.PRESERVE for mode in modes.values()):
        yield
        return

    token = uuid4().hex
    mutations: list[_PreparedMutation] = []
    primary_error: BaseException | None = None
    try:
        for slot_index, mode in sorted(modes.items()):
            if slot_index >= len(materials):
                raise BakeMaterialPreparationError(
                    f"Preparation references slot {slot_index}, but only "
                    f"{len(materials)} copied materials exist"
                )
            if mode is MaterialPreparationMode.PRESERVE:
                continue
            mutations.append(
                _prepare_alpha_material(materials[slot_index], mode, token=token)
            )
        yield
    except BaseException as exc:
        primary_error = exc
        raise
    finally:
        restore_errors: list[Exception] = []
        for mutation in reversed(mutations):
            try:
                _restore_mutation(mutation)
            except Exception as exc:
                restore_errors.append(exc)
                logger.exception("Failed to restore copied material pass preparation")
        if restore_errors and primary_error is None:
            raise BakeMaterialPreparationError(
                "One or more copied materials could not be restored"
            ) from restore_errors[0]
