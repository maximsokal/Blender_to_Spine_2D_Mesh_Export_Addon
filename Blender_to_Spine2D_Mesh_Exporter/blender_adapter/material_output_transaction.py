"""Capture and restore copied Blender Material Output state transactionally.

This owner surrounds semantic material preparation. It is deliberately
independent from proxy construction so it can recover even when preparation
fails before its own mutation record has been created, including a partial
Surface-link removal.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import logging
from typing import Any, Iterator, Tuple


logger = logging.getLogger(__name__)


class MaterialOutputTransactionError(RuntimeError):
    """Raised when copied Material Output state cannot be captured or restored."""


@dataclass(frozen=True, slots=True)
class MaterialOutputSurfaceState:
    """Exact state of one Material Output Surface input."""

    node_tree: Any
    output_node: Any
    surface_socket: Any
    source_sockets: Tuple[Any, ...]
    active: bool


@dataclass(frozen=True, slots=True)
class MaterialOutputTransactionState:
    """All Material Output states owned by one copied-material pass."""

    outputs: Tuple[MaterialOutputSurfaceState, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.outputs, tuple):
            raise TypeError("outputs must be tuple")

    @classmethod
    def capture(
        cls,
        materials: Tuple[Any, ...],
    ) -> "MaterialOutputTransactionState":
        if not isinstance(materials, tuple):
            raise TypeError("materials must be tuple")

        captured: list[MaterialOutputSurfaceState] = []
        for material_index, material in enumerate(materials):
            if material is None:
                raise MaterialOutputTransactionError(
                    f"Copied material {material_index} is missing"
                )
            node_tree = getattr(material, "node_tree", None)
            if node_tree is None:
                raise MaterialOutputTransactionError(
                    f"Copied material {material_index} has no node tree"
                )
            nodes = getattr(node_tree, "nodes", None)
            links = getattr(node_tree, "links", None)
            if nodes is None or links is None:
                raise MaterialOutputTransactionError(
                    f"Copied material {material_index} node tree is incomplete"
                )
            try:
                outputs = tuple(
                    node
                    for node in nodes
                    if str(getattr(node, "type", "") or "") == "OUTPUT_MATERIAL"
                )
            except Exception as exc:
                raise MaterialOutputTransactionError(
                    f"Unable to inspect Material Outputs for copied material {material_index}"
                ) from exc
            if not outputs:
                raise MaterialOutputTransactionError(
                    f"Copied material {material_index} has no Material Output"
                )

            for output in outputs:
                surface_socket = _socket_by_name(
                    getattr(output, "inputs", None),
                    "Surface",
                )
                if surface_socket is None:
                    raise MaterialOutputTransactionError(
                        f"Material Output '{_node_name(output)}' has no Surface input"
                    )
                source_sockets = tuple(
                    source
                    for source in (
                        getattr(link, "from_socket", None)
                        for link in _incoming_links(surface_socket)
                    )
                    if source is not None
                )
                captured.append(
                    MaterialOutputSurfaceState(
                        node_tree=node_tree,
                        output_node=output,
                        surface_socket=surface_socket,
                        source_sockets=source_sockets,
                        active=bool(getattr(output, "is_active_output", False)),
                    )
                )

        return cls(outputs=tuple(captured))

    def restore(self) -> None:
        failures: list[str] = []
        for state in self.outputs:
            try:
                _restore_surface_links(state)
            except Exception as exc:
                failures.append(
                    f"restore Surface links for '{_node_name(state.output_node)}': {exc}"
                )

        for state in self.outputs:
            try:
                state.output_node.is_active_output = state.active
            except Exception as exc:
                failures.append(
                    f"restore active flag for '{_node_name(state.output_node)}': {exc}"
                )

        if failures:
            raise MaterialOutputTransactionError(
                "Unable to restore copied Material Output state: " + "; ".join(failures)
            )


def _node_name(node: Any) -> str:
    return str(
        getattr(node, "name", "")
        or getattr(node, "type", "")
        or "Material Output"
    )


def _socket_by_name(collection: Any, name: str) -> Any | None:
    if collection is None:
        return None
    getter = getattr(collection, "get", None)
    if callable(getter):
        try:
            socket = getter(name)
            if socket is not None:
                return socket
        except Exception:
            logger.debug("Socket lookup by name failed", exc_info=True)
    try:
        return next(
            socket
            for socket in collection
            if str(getattr(socket, "name", "") or "") == name
        )
    except Exception:
        return None


def _incoming_links(socket: Any) -> Tuple[Any, ...]:
    try:
        return tuple(getattr(socket, "links", ()))
    except Exception:
        return ()


def _rna_identity(value: Any) -> int:
    pointer = getattr(value, "as_pointer", None)
    if callable(pointer):
        try:
            resolved = int(pointer())
            if resolved:
                return resolved
        except Exception:
            logger.debug("RNA pointer lookup failed", exc_info=True)
    return id(value)


def _same_socket(first: Any, second: Any) -> bool:
    return first is second or _rna_identity(first) == _rna_identity(second)


def _restore_surface_links(state: MaterialOutputSurfaceState) -> None:
    current_links = _incoming_links(state.surface_socket)
    original_sources = state.source_sockets

    # Remove only links introduced after capture. Original links that survived a
    # partial failure are retained, preventing duplicate links during recovery.
    for link in current_links:
        source = getattr(link, "from_socket", None)
        if source is not None and any(
            _same_socket(source, original) for original in original_sources
        ):
            continue
        state.node_tree.links.remove(link)

    current_sources = tuple(
        source
        for source in (
            getattr(link, "from_socket", None)
            for link in _incoming_links(state.surface_socket)
        )
        if source is not None
    )
    for original in original_sources:
        if any(_same_socket(original, current) for current in current_sources):
            continue
        state.node_tree.links.new(original, state.surface_socket)
        current_sources = (*current_sources, original)


@contextmanager
def preserve_material_output_state(
    materials: Tuple[Any, ...],
) -> Iterator[MaterialOutputTransactionState]:
    """Restore copied output links and active flags on every exit path."""

    state = MaterialOutputTransactionState.capture(materials)
    primary_error: BaseException | None = None
    try:
        yield state
    except BaseException as exc:
        primary_error = exc
        raise
    finally:
        try:
            state.restore()
        except Exception:
            if primary_error is None:
                raise
            logger.exception(
                "Failed to restore copied Material Output state while handling "
                "another preparation error"
            )


__all__ = [
    "MaterialOutputSurfaceState",
    "MaterialOutputTransactionError",
    "MaterialOutputTransactionState",
    "preserve_material_output_state",
]
