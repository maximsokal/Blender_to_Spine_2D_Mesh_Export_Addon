"""Scene-aware entry point for typed Blender 5.2 material preparation."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import replace
from typing import Any, Iterator, Tuple

from ..domain.baking import BakeMode, BakePassPlan, BakeStrategyId
from .bake_material_preparation import temporary_prepare_material_pass
from .material_output_transaction import preserve_material_output_state
from .render_engine_contract import render_engine_contract


def _material_preparation_pass(pass_plan: BakePassPlan) -> BakePassPlan:
    """Map camera-context Normal/UV EMIT into the existing surface-color proxy.

    ``NormalUvCameraCombinedBakeStrategy`` keeps CAMERA evaluation scope so Blender
    resolves allowed camera/object-dependent nodes consistently, but its EMIT bake must
    not evaluate the original BSDF surface. The established SURFACE_COLOR+EMIT material
    preparation path already extracts linked Base Color into a temporary Emission node.
    Re-tag only the isolated adapter-side preparation view of the immutable pass; the
    domain plan and exported statistics retain their original typed strategy identity.
    """

    if not isinstance(pass_plan, BakePassPlan):
        raise TypeError("pass_plan must be BakePassPlan")
    if (
        pass_plan.strategy_id is BakeStrategyId.CAMERA_COMBINED
        and pass_plan.bake_mode is BakeMode.EMIT
    ):
        return replace(pass_plan, strategy_id=BakeStrategyId.SURFACE_COLOR)
    return pass_plan


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
    preparation_pass = _material_preparation_pass(pass_plan)
    with preserve_material_output_state(materials):
        with temporary_prepare_material_pass(
            materials,
            preparation_pass,
            used_material_indices=used_material_indices,
            render_target=target,
        ):
            yield


__all__ = [
    "_material_preparation_pass",
    "temporary_prepare_scene_material_pass",
]
