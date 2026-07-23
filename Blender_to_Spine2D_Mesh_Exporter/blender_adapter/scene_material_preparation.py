"""Scene-aware entry point for typed Blender 5.2 material preparation."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Iterator, Tuple

from ..domain.baking import BakePassPlan
from .bake_material_preparation import temporary_prepare_material_pass
from .material_output_transaction import preserve_material_output_state
from .render_engine_contract import render_engine_contract


@contextmanager
def temporary_prepare_scene_material_pass(
    materials: Tuple[Any, ...],
    pass_plan: BakePassPlan,
    *,
    used_material_indices: Tuple[int, ...],
    render_target: str,
) -> Iterator[None]:
    """Prepare copied materials and restore exact output state on every exit path.

    The outer Material Output transaction is captured before proxy preparation.
    It therefore recovers links and active-output flags even if preparation fails
    before its internal mutation record exists, including a partial Surface-link
    removal.
    """

    if not isinstance(materials, tuple):
        raise TypeError("materials must be tuple")
    if not isinstance(pass_plan, BakePassPlan):
        raise TypeError("pass_plan must be BakePassPlan")
    if not isinstance(used_material_indices, tuple):
        raise TypeError("used_material_indices must be tuple")

    target = render_engine_contract(render_target).shader_target
    with preserve_material_output_state(materials):
        with temporary_prepare_material_pass(
            materials,
            pass_plan,
            used_material_indices=used_material_indices,
            render_target=target,
        ):
            yield


__all__ = ["temporary_prepare_scene_material_pass"]
