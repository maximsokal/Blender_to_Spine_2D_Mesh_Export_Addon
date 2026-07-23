"""Prepare copied Blender 5.2 materials for one typed semantic bake pass.

Only temporary material copies owned by :mod:`bake_materials` are mutated.  The
selected Material Output, its Surface links, and every temporary proxy node are
restored deterministically before control leaves the context manager.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum
import logging
from typing import Any, Iterable, Iterator, Tuple
from uuid import uuid4

from ..domain.baking import (
    BakeMode,
    BakePassPlan,
    BakeStrategyId,
    MaterialPreparationMode,
    MaterialSlotPreparation,
)
from .render_engine_contract import render_engine_contract


logger = logging.getLogger(__name__)


class BakeMaterialPreparationError(RuntimeError):
    """Raised when a copied Blender 5.2 material cannot be prepared safely."""


class _ProxyKind(str, Enum):
    STRAIGHT_SURFACE_COLOR = "STRAIGHT_SURFACE_COLOR"
    ZERO_COLOR = "ZERO_COLOR"
    EXTRACT_ALPHA = "EXTRACT_ALPHA"
    OPAQUE_ALPHA = "OPAQUE_ALPHA"


@dataclass(frozen=True, slots=True)
class _ScalarExpression:
    socket: Any | None = None
    constant: float = 0.0


@dataclass(frozen=True, slots=True)
class _ColorExpression:
    socket: Any | None = None
    constant: Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0)


@dataclass(slots=True)
class _PreparedMutation:
    node_tree: Any
    output_surface_socket: Any
    original_surface_sources: Tuple[Any, ...]
    temporary_nodes: list[Any]
    output_active_states: Tuple[Tuple[Any, bool], ...]


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


def _output_socket(node: Any, name: str, *, index: int | None = None) -> Any | None:
    outputs = getattr(node, "outputs", None)
    if outputs is None:
        return None
    getter = getattr(outputs, "get", None)
    if callable(getter):
        try:
            result = getter(name)
            if result is not None:
                return result
        except Exception:
            logger.debug("Output socket lookup by name failed", exc_info=True)
    if index is not None:
        try:
            return outputs[index]
        except Exception:
            pass
    try:
        return next(socket for socket in outputs if _socket_name(socket) == name)
    except Exception:
        return None


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


def _material_node_tree(material: Any) -> Any:
    if material is None:
        raise BakeMaterialPreparationError("Copied material is missing")
    node_tree = getattr(material, "node_tree", None)
    if node_tree is None:
        raise BakeMaterialPreparationError(
            "Copied material has no node tree; Blender 5.2 materials must expose one"
        )
    if getattr(node_tree, "nodes", None) is None or getattr(node_tree, "links", None) is None:
        raise BakeMaterialPreparationError("Copied material node tree is incomplete")
    return node_tree


def _output_target(node: Any) -> str:
    value = str(getattr(node, "target", "ALL") or "ALL").strip().upper()
    if value not in {"ALL", "CYCLES", "EEVEE"}:
        raise BakeMaterialPreparationError(
            f"Material Output '{_node_name(node)}' has unsupported target {value!r}"
        )
    return value


def _select_material_output(node_tree: Any, render_target: str) -> tuple[Any, Tuple[Any, ...]]:
    target = render_engine_contract(render_target).shader_target
    try:
        outputs = tuple(
            node for node in node_tree.nodes if _node_type(node) == "OUTPUT_MATERIAL"
        )
    except Exception as exc:
        raise BakeMaterialPreparationError(
            "Unable to inspect copied material outputs"
        ) from exc
    if not outputs:
        raise BakeMaterialPreparationError("Copied material has no Material Output node")

    exact = tuple(node for node in outputs if _output_target(node) == target)
    generic = tuple(node for node in outputs if _output_target(node) == "ALL")
    candidates = exact or generic
    if not candidates:
        raise BakeMaterialPreparationError(
            f"Copied material has no Material Output for render target '{target}'"
        )
    active = tuple(
        node for node in candidates if bool(getattr(node, "is_active_output", False))
    )
    return (active[0] if active else candidates[0]), outputs


def _activate_material_output(
    selected: Any,
    outputs: Tuple[Any, ...],
) -> Tuple[Tuple[Any, bool], ...]:
    states = tuple(
        (output, bool(getattr(output, "is_active_output", False))) for output in outputs
    )
    try:
        for output in outputs:
            output.is_active_output = output is selected
    except Exception as exc:
        for output, original in states:
            try:
                output.is_active_output = original
            except Exception:
                logger.exception("Failed to roll back Material Output selection")
        raise BakeMaterialPreparationError(
            f"Unable to activate Material Output '{_node_name(selected)}'"
        ) from exc
    return states


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


def _color_default(
    socket: Any | None,
    default: Tuple[float, float, float, float],
) -> Tuple[float, float, float, float]:
    if socket is None:
        return default
    value = getattr(socket, "default_value", default)
    try:
        if hasattr(value, "__len__") and not isinstance(value, (str, bytes)):
            resolved = tuple(float(item) for item in value)
            if len(resolved) >= 4:
                return resolved[:4]
            if len(resolved) == 3:
                return resolved[0], resolved[1], resolved[2], 1.0
            if len(resolved) == 1:
                return resolved[0], resolved[0], resolved[0], 1.0
        scalar = float(value)
        return scalar, scalar, scalar, 1.0
    except Exception:
        return default


def _scalar_from_input(socket: Any | None, *, default: float) -> _ScalarExpression:
    source = _linked_source_socket(socket)
    return (
        _ScalarExpression(socket=source)
        if source is not None
        else _ScalarExpression(constant=_numeric_default(socket, default))
    )


def _color_from_input(
    socket: Any | None,
    *,
    default: Tuple[float, float, float, float],
) -> _ColorExpression:
    source = _linked_source_socket(socket)
    return (
        _ColorExpression(socket=source)
        if source is not None
        else _ColorExpression(constant=_color_default(socket, default))
    )


def _assign_socket_constant(target_socket: Any, value: Any) -> None:
    current = getattr(target_socket, "default_value", None)
    try:
        is_sequence = hasattr(current, "__len__") and not isinstance(
            current, (str, bytes)
        )
        if is_sequence:
            length = len(current)
            if isinstance(value, (tuple, list)):
                source = tuple(float(item) for item in value)
                if not source:
                    raise ValueError("constant sequence is empty")
                resolved = [source[min(index, len(source) - 1)] for index in range(length)]
            else:
                resolved = [float(value)] * length
            if length == 4 and not isinstance(value, (tuple, list)):
                resolved[3] = 1.0
            target_socket.default_value = tuple(resolved)
        elif isinstance(value, (tuple, list)):
            target_socket.default_value = float(value[0])
        else:
            target_socket.default_value = float(value)
    except Exception as exc:
        raise BakeMaterialPreparationError(
            f"Unable to assign constant to socket '{_socket_name(target_socket)}'"
        ) from exc


def _connect_scalar(node_tree: Any, expression: _ScalarExpression, target_socket: Any) -> None:
    if expression.socket is not None:
        try:
            node_tree.links.new(expression.socket, target_socket)
            return
        except Exception as exc:
            raise BakeMaterialPreparationError(
                "Unable to connect scalar expression to temporary node"
            ) from exc
    _assign_socket_constant(target_socket, expression.constant)


def _connect_color(node_tree: Any, expression: _ColorExpression, target_socket: Any) -> None:
    if expression.socket is not None:
        try:
            node_tree.links.new(expression.socket, target_socket)
            return
        except Exception as exc:
            raise BakeMaterialPreparationError(
                "Unable to connect color expression to temporary node"
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
        node.name = f"__Spine2D_Proxy_{label}_{token}_{len(temporary_nodes)}"
        node.label = f"Spine2D temporary {label}"
        node.operation = operation
        if hasattr(node, "use_clamp"):
            node.use_clamp = clamp
        temporary_nodes.append(node)
        return node
    except Exception as exc:
        raise BakeMaterialPreparationError(
            f"Unable to create temporary Math node '{operation}'"
        ) from exc


def _new_mix_rgb(
    node_tree: Any,
    temporary_nodes: list[Any],
    *,
    blend_type: str,
    token: str,
    label: str,
    clamp: bool = False,
) -> Any:
    try:
        node = node_tree.nodes.new(type="ShaderNodeMixRGB")
        node.name = f"__Spine2D_Proxy_{label}_{token}_{len(temporary_nodes)}"
        node.label = f"Spine2D temporary {label}"
        node.blend_type = blend_type
        if hasattr(node, "use_clamp"):
            node.use_clamp = clamp
        temporary_nodes.append(node)
        return node
    except Exception as exc:
        raise BakeMaterialPreparationError(
            f"Unable to create temporary MixRGB node '{blend_type}'"
        ) from exc


def _multiply_scalar(
    node_tree: Any,
    left: _ScalarExpression,
    right: _ScalarExpression,
    temporary_nodes: list[Any],
    token: str,
) -> _ScalarExpression:
    node = _new_math(
        node_tree,
        temporary_nodes,
        operation="MULTIPLY",
        token=token,
        label="alpha_multiply",
    )
    _connect_scalar(node_tree, left, node.inputs[0])
    _connect_scalar(node_tree, right, node.inputs[1])
    return _ScalarExpression(socket=node.outputs[0])


def _one_minus(
    node_tree: Any,
    value: _ScalarExpression,
    temporary_nodes: list[Any],
    token: str,
) -> _ScalarExpression:
    node = _new_math(
        node_tree,
        temporary_nodes,
        operation="SUBTRACT",
        token=token,
        label="alpha_one_minus",
    )
    _assign_socket_constant(node.inputs[0], 1.0)
    _connect_scalar(node_tree, value, node.inputs[1])
    return _ScalarExpression(socket=node.outputs[0])


def _add_scalar_clamped(
    node_tree: Any,
    left: _ScalarExpression,
    right: _ScalarExpression,
    temporary_nodes: list[Any],
    token: str,
) -> _ScalarExpression:
    node = _new_math(
        node_tree,
        temporary_nodes,
        operation="ADD",
        token=token,
        label="alpha_add",
        clamp=True,
    )
    _connect_scalar(node_tree, left, node.inputs[0])
    _connect_scalar(node_tree, right, node.inputs[1])
    return _ScalarExpression(socket=node.outputs[0])


def _mix_colors(
    node_tree: Any,
    color_a: _ColorExpression,
    color_b: _ColorExpression,
    factor: _ScalarExpression,
    temporary_nodes: list[Any],
    token: str,
) -> _ColorExpression:
    node = _new_mix_rgb(
        node_tree,
        temporary_nodes,
        blend_type="MIX",
        token=token,
        label="surface_mix",
    )
    _connect_scalar(node_tree, factor, node.inputs[0])
    _connect_color(node_tree, color_a, node.inputs[1])
    _connect_color(node_tree, color_b, node.inputs[2])
    return _ColorExpression(socket=node.outputs[0])


def _add_colors(
    node_tree: Any,
    color_a: _ColorExpression,
    color_b: _ColorExpression,
    temporary_nodes: list[Any],
    token: str,
) -> _ColorExpression:
    node = _new_mix_rgb(
        node_tree,
        temporary_nodes,
        blend_type="ADD",
        token=token,
        label="surface_add",
        clamp=True,
    )
    _assign_socket_constant(node.inputs[0], 1.0)
    _connect_color(node_tree, color_a, node.inputs[1])
    _connect_color(node_tree, color_b, node.inputs[2])
    return _ColorExpression(socket=node.outputs[0])


def _shader_input_node(node: Any, index: int) -> Any | None:
    inputs = getattr(node, "inputs", None)
    if inputs is None:
        return None
    try:
        return _linked_source_node(inputs[index])
    except Exception:
        return None


def _reroute_source_node(node: Any) -> Any | None:
    return _linked_source_node(_input_socket(node, "Input", index=0))


def _opacity_from_shader(
    node_tree: Any,
    shader_node: Any | None,
    temporary_nodes: list[Any],
    token: str,
    visiting: set[str],
) -> _ScalarExpression:
    if shader_node is None:
        return _ScalarExpression(constant=0.0)

    name = _node_name(shader_node)
    if name in visiting:
        raise BakeMaterialPreparationError(
            f"Shader graph cycle detected while extracting alpha at '{name}'"
        )
    visiting.add(name)
    try:
        node_type = _node_type(shader_node)
        if node_type == "REROUTE":
            return _opacity_from_shader(
                node_tree,
                _reroute_source_node(shader_node),
                temporary_nodes,
                token,
                visiting,
            )
        if node_type in {"BSDF_TRANSPARENT", "HOLDOUT"}:
            return _ScalarExpression(constant=0.0)
        if node_type == "BSDF_PRINCIPLED":
            return _scalar_from_input(_input_socket(shader_node, "Alpha"), default=1.0)
        if node_type == "MIX_SHADER":
            factor = _scalar_from_input(
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
            return _add_scalar_clamped(
                node_tree,
                _multiply_scalar(
                    node_tree,
                    opacity_a,
                    _one_minus(node_tree, factor, temporary_nodes, token),
                    temporary_nodes,
                    token,
                ),
                _multiply_scalar(
                    node_tree,
                    opacity_b,
                    factor,
                    temporary_nodes,
                    token,
                ),
                temporary_nodes,
                token,
            )
        if node_type == "ADD_SHADER":
            return _add_scalar_clamped(
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
        if node_type == "GROUP":
            raise BakeMaterialPreparationError(
                f"Shader group '{name}' requires camera projection or a flattened graph"
            )
        return _ScalarExpression(constant=1.0)
    finally:
        visiting.remove(name)


def _surface_color_from_shader(
    node_tree: Any,
    shader_node: Any | None,
    temporary_nodes: list[Any],
    token: str,
    visiting: set[str],
) -> _ColorExpression | None:
    """Return straight surface color, excluding shader opacity and emission."""

    if shader_node is None:
        return None
    name = _node_name(shader_node)
    if name in visiting:
        raise BakeMaterialPreparationError(
            f"Shader graph cycle detected while extracting color at '{name}'"
        )
    visiting.add(name)
    try:
        node_type = _node_type(shader_node)
        if node_type == "REROUTE":
            return _surface_color_from_shader(
                node_tree,
                _reroute_source_node(shader_node),
                temporary_nodes,
                token,
                visiting,
            )
        if node_type in {"BSDF_TRANSPARENT", "HOLDOUT", "EMISSION"}:
            return None
        if node_type == "BSDF_PRINCIPLED":
            return _color_from_input(
                _input_socket(shader_node, "Base Color"),
                default=(0.8, 0.8, 0.8, 1.0),
            )
        if node_type == "MIX_SHADER":
            factor = _scalar_from_input(
                _input_socket(shader_node, "Fac", index=0),
                default=0.5,
            )
            color_a = _surface_color_from_shader(
                node_tree,
                _shader_input_node(shader_node, 1),
                temporary_nodes,
                token,
                visiting,
            )
            color_b = _surface_color_from_shader(
                node_tree,
                _shader_input_node(shader_node, 2),
                temporary_nodes,
                token,
                visiting,
            )
            if color_a is None:
                return color_b
            if color_b is None:
                return color_a
            return _mix_colors(
                node_tree,
                color_a,
                color_b,
                factor,
                temporary_nodes,
                token,
            )
        if node_type == "ADD_SHADER":
            color_a = _surface_color_from_shader(
                node_tree,
                _shader_input_node(shader_node, 0),
                temporary_nodes,
                token,
                visiting,
            )
            color_b = _surface_color_from_shader(
                node_tree,
                _shader_input_node(shader_node, 1),
                temporary_nodes,
                token,
                visiting,
            )
            if color_a is None:
                return color_b
            if color_b is None:
                return color_a
            return _add_colors(
                node_tree,
                color_a,
                color_b,
                temporary_nodes,
                token,
            )
        if node_type == "GROUP":
            raise BakeMaterialPreparationError(
                f"Shader group '{name}' requires camera projection or a flattened graph"
            )
        for socket_name in ("Base Color", "Color"):
            socket = _input_socket(shader_node, socket_name)
            if socket is not None:
                return _color_from_input(
                    socket,
                    default=(0.8, 0.8, 0.8, 1.0),
                )
        raise BakeMaterialPreparationError(
            f"Surface shader '{name}' ({node_type}) has no supported color channel"
        )
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


def _prepare_proxy_material(
    material: Any,
    proxy_kind: _ProxyKind,
    *,
    render_target: str,
    token: str,
) -> _PreparedMutation:
    node_tree = _material_node_tree(material)
    output, outputs = _select_material_output(node_tree, render_target)
    active_states = _activate_material_output(output, outputs)
    surface_socket = _input_socket(output, "Surface")
    if surface_socket is None:
        _restore_output_states(active_states)
        raise BakeMaterialPreparationError("Material Output has no Surface input")

    original_nodes = tuple(
        getattr(link, "from_node", None)
        for link in _incoming_links(surface_socket)
        if getattr(link, "from_node", None) is not None
    )
    original_sources = _remove_surface_links(node_tree, surface_socket)
    temporary_nodes: list[Any] = []

    try:
        source_node = original_nodes[0] if original_nodes else None
        emission = node_tree.nodes.new(type="ShaderNodeEmission")
        emission.name = f"__Spine2D_Proxy_Emission_{token}_{len(temporary_nodes)}"
        emission.label = f"Spine2D temporary {proxy_kind.value} output"
        temporary_nodes.append(emission)

        color_input = _input_socket(emission, "Color")
        strength_input = _input_socket(emission, "Strength")
        emission_output = _output_socket(emission, "Emission")
        if color_input is None or strength_input is None or emission_output is None:
            raise BakeMaterialPreparationError(
                "Blender 5.2 Emission node exposes unexpected sockets"
            )

        if proxy_kind is _ProxyKind.EXTRACT_ALPHA:
            opacity = _opacity_from_shader(
                node_tree,
                source_node,
                temporary_nodes,
                token,
                set(),
            )
            if opacity.socket is not None:
                node_tree.links.new(opacity.socket, color_input)
            else:
                _assign_socket_constant(color_input, opacity.constant)
        elif proxy_kind is _ProxyKind.OPAQUE_ALPHA:
            _assign_socket_constant(color_input, 1.0)
        elif proxy_kind is _ProxyKind.ZERO_COLOR:
            _assign_socket_constant(color_input, (0.0, 0.0, 0.0, 1.0))
        elif proxy_kind is _ProxyKind.STRAIGHT_SURFACE_COLOR:
            color = _surface_color_from_shader(
                node_tree,
                source_node,
                temporary_nodes,
                token,
                set(),
            )
            if color is None:
                _assign_socket_constant(color_input, (0.0, 0.0, 0.0, 1.0))
            else:
                _connect_color(node_tree, color, color_input)
        else:
            raise BakeMaterialPreparationError(
                f"Unsupported proxy kind: {proxy_kind.value}"
            )

        _assign_socket_constant(strength_input, 1.0)
        node_tree.links.new(emission_output, surface_socket)
    except Exception:
        partial = _PreparedMutation(
            node_tree=node_tree,
            output_surface_socket=surface_socket,
            original_surface_sources=original_sources,
            temporary_nodes=temporary_nodes,
            output_active_states=active_states,
        )
        try:
            _restore_mutation(partial)
        except Exception:
            logger.exception("Failed to roll back partial material preparation")
        raise

    return _PreparedMutation(
        node_tree=node_tree,
        output_surface_socket=surface_socket,
        original_surface_sources=original_sources,
        temporary_nodes=temporary_nodes,
        output_active_states=active_states,
    )


def _restore_output_states(states: Tuple[Tuple[Any, bool], ...]) -> list[str]:
    failures: list[str] = []
    for output, original in states:
        try:
            output.is_active_output = original
        except Exception as exc:
            failures.append(f"restore output '{_node_name(output)}': {exc}")
    return failures


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
    failures.extend(_restore_output_states(mutation.output_active_states))
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


def _resolve_proxy_kinds(
    pass_plan: BakePassPlan,
    used_material_indices: Tuple[int, ...],
) -> dict[int, _ProxyKind]:
    if (
        pass_plan.strategy_id is BakeStrategyId.SURFACE_COLOR
        and pass_plan.bake_mode is BakeMode.EMIT
    ):
        surface_slots = set(pass_plan.material_slot_indices)
        return {
            slot_index: (
                _ProxyKind.STRAIGHT_SURFACE_COLOR
                if slot_index in surface_slots
                else _ProxyKind.ZERO_COLOR
            )
            for slot_index in used_material_indices
        }

    modes = _preparation_map(pass_plan.material_preparations)
    resolved: dict[int, _ProxyKind] = {}
    for slot_index, mode in modes.items():
        if slot_index not in used_material_indices:
            continue
        if mode is MaterialPreparationMode.ZERO_TO_EMISSION:
            resolved[slot_index] = _ProxyKind.ZERO_COLOR
        elif mode is MaterialPreparationMode.EXTRACT_ALPHA_TO_EMISSION:
            resolved[slot_index] = _ProxyKind.EXTRACT_ALPHA
        elif mode is MaterialPreparationMode.OPAQUE_ALPHA_TO_EMISSION:
            resolved[slot_index] = _ProxyKind.OPAQUE_ALPHA
        elif mode is MaterialPreparationMode.PRESERVE:
            continue
        else:
            raise BakeMaterialPreparationError(
                f"Unsupported material preparation mode: {mode.value}"
            )
    return resolved


@contextmanager
def temporary_prepare_material_pass(
    materials: Tuple[Any, ...],
    pass_plan: BakePassPlan,
    *,
    used_material_indices: Tuple[int, ...],
    render_target: str,
) -> Iterator[None]:
    """Apply and restore one typed strategy preparation on copied materials."""

    if not isinstance(materials, tuple):
        raise TypeError("materials must be tuple")
    if not isinstance(pass_plan, BakePassPlan):
        raise TypeError("pass_plan must be BakePassPlan")
    if not isinstance(used_material_indices, tuple):
        raise TypeError("used_material_indices must be tuple")
    if any(
        not isinstance(index, int) or isinstance(index, bool) or index < 0
        for index in used_material_indices
    ):
        raise ValueError("used_material_indices must contain non-negative integers")
    normalized_target = render_engine_contract(render_target).shader_target

    proxy_kinds = _resolve_proxy_kinds(pass_plan, used_material_indices)
    if not proxy_kinds:
        yield
        return

    token = uuid4().hex
    mutations: list[_PreparedMutation] = []
    primary_error: BaseException | None = None
    try:
        for slot_index, proxy_kind in sorted(proxy_kinds.items()):
            if slot_index >= len(materials):
                raise BakeMaterialPreparationError(
                    f"Preparation references slot {slot_index}, but only "
                    f"{len(materials)} copied materials exist"
                )
            mutations.append(
                _prepare_proxy_material(
                    materials[slot_index],
                    proxy_kind,
                    render_target=normalized_target,
                    token=token,
                )
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


__all__ = [
    "BakeMaterialPreparationError",
    "temporary_prepare_material_pass",
]
