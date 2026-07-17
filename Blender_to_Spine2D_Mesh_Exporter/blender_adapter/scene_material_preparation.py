"""Extend B2 copied-material preparation with generic per-pass slot masking."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import replace
import logging
from typing import Any, Iterator, Tuple
from uuid import uuid4

from ..domain.baking import BakePassPlan, MaterialPreparationMode
from . import bake_material_preparation as base
from .render_engine_contract import render_engine_contract

logger = logging.getLogger(__name__)


def _node_type(node: Any) -> str:
    return str(getattr(node, "type", "") or "")


def _output_target(node: Any) -> str:
    value = str(getattr(node, "target", "ALL") or "ALL").strip().upper()
    if "CYCLE" in value:
        return "CYCLES"
    if "EEVEE" in value:
        return "EEVEE"
    return "ALL"


def _renderer_output(node_tree: Any, render_target: str) -> Any:
    target = render_engine_contract(render_target).shader_target
    getter = getattr(node_tree, "get_output_node", None)
    if callable(getter):
        for candidate_target in (target, "ALL"):
            try:
                candidate = getter(candidate_target)
            except TypeError:
                try:
                    candidate = getter(target=candidate_target)
                except Exception:
                    logger.debug(
                        "Copied material output lookup failed for target %s",
                        candidate_target,
                        exc_info=True,
                    )
                    continue
            except Exception:
                logger.debug(
                    "Copied material output lookup failed for target %s",
                    candidate_target,
                    exc_info=True,
                )
                continue
            if candidate is not None and _node_type(candidate) == "OUTPUT_MATERIAL":
                return candidate

    try:
        outputs = tuple(
            node for node in node_tree.nodes if _node_type(node) == "OUTPUT_MATERIAL"
        )
    except Exception as exc:
        raise base.BakeMaterialPreparationError(
            "Unable to inspect copied material outputs"
        ) from exc
    exact = tuple(node for node in outputs if _output_target(node) == target)
    generic = tuple(node for node in outputs if _output_target(node) == "ALL")
    candidates = exact or generic
    if not candidates:
        raise base.BakeMaterialPreparationError(
            f"Copied material has no Material Output for render target '{target}'"
        )
    active = tuple(
        node for node in candidates if bool(getattr(node, "is_active_output", False))
    )
    return active[0] if active else candidates[0]


@contextmanager
def _temporary_renderer_output_selection(
    materials: Tuple[Any, ...],
    render_target: str,
) -> Iterator[None]:
    """Make the renderer-effective output unambiguous for legacy proxy primitives."""

    target = render_engine_contract(render_target).shader_target
    states: list[tuple[Any, bool]] = []
    primary_error: BaseException | None = None
    try:
        for material in materials:
            node_tree = getattr(material, "node_tree", None)
            if node_tree is None:
                continue
            try:
                outputs = tuple(
                    node
                    for node in node_tree.nodes
                    if _node_type(node) == "OUTPUT_MATERIAL"
                )
            except Exception as exc:
                raise base.BakeMaterialPreparationError(
                    "Unable to inspect copied material outputs"
                ) from exc
            if not outputs:
                continue
            selected = _renderer_output(node_tree, target)
            for output in outputs:
                states.append(
                    (output, bool(getattr(output, "is_active_output", False)))
                )
            try:
                for output in outputs:
                    output.is_active_output = output is selected
            except Exception as exc:
                raise base.BakeMaterialPreparationError(
                    f"Unable to select copied Material Output for '{target}'"
                ) from exc
        yield
    except BaseException as exc:
        primary_error = exc
        raise
    finally:
        failures: list[str] = []
        for output, original in states:
            try:
                output.is_active_output = original
            except Exception as exc:
                failures.append(
                    f"{getattr(output, 'name', 'Material Output')}: {exc}"
                )
        if failures:
            error = base.BakeMaterialPreparationError(
                "Unable to restore copied renderer outputs: " + "; ".join(failures)
            )
            if primary_error is None:
                raise error
            logger.exception(
                "Failed to restore copied renderer outputs while handling another error",
                exc_info=error,
            )


@contextmanager
def temporary_prepare_scene_material_pass(
    materials: Tuple[Any, ...],
    pass_plan: BakePassPlan,
    *,
    used_material_indices: Tuple[int, ...],
    render_target: str = "CYCLES",
) -> Iterator[None]:
    """Apply B2 extraction plus explicit black masks for non-matching B3 slots."""

    if not isinstance(materials, tuple):
        raise TypeError("materials must be tuple")
    if not isinstance(pass_plan, BakePassPlan):
        raise TypeError("pass_plan must be BakePassPlan")
    if not isinstance(used_material_indices, tuple):
        raise TypeError("used_material_indices must be tuple")
    target = render_engine_contract(render_target).shader_target

    zero_indices = tuple(
        item.slot_index
        for item in pass_plan.material_preparations
        if item.mode is MaterialPreparationMode.ZERO_TO_EMISSION
        and item.slot_index in used_material_indices
    )
    base_preparations = tuple(
        item
        for item in pass_plan.material_preparations
        if item.mode is not MaterialPreparationMode.ZERO_TO_EMISSION
    )
    base_plan = replace(pass_plan, material_preparations=base_preparations)

    token = uuid4().hex
    mutations: list[Any] = []
    primary_error: BaseException | None = None
    try:
        with _temporary_renderer_output_selection(materials, target):
            with base.temporary_prepare_material_pass(
                materials,
                base_plan,
                used_material_indices=used_material_indices,
            ):
                for slot_index in zero_indices:
                    if slot_index >= len(materials):
                        raise base.BakeMaterialPreparationError(
                            f"Scene pass mask references slot {slot_index}, but only "
                            f"{len(materials)} copied materials exist"
                        )
                    mutations.append(
                        base._prepare_proxy_material(
                            materials[slot_index],
                            base._ProxyKind.ZERO_COLOR,
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
                base._restore_mutation(mutation)
            except Exception as exc:
                restore_errors.append(exc)
                logger.exception("Failed to restore a scene material slot mask")
        if restore_errors and primary_error is None:
            raise base.BakeMaterialPreparationError(
                "One or more scene material masks could not be restored"
            ) from restore_errors[0]
