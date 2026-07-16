"""Extend B2 copied-material preparation with generic per-pass slot masking."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import replace
import logging
from typing import Any, Iterator, Tuple
from uuid import uuid4

from ..domain.baking import (
    BakePassPlan,
    MaterialPreparationMode,
)
from . import bake_material_preparation as base

logger = logging.getLogger(__name__)


@contextmanager
def temporary_prepare_scene_material_pass(
    materials: Tuple[Any, ...],
    pass_plan: BakePassPlan,
    *,
    used_material_indices: Tuple[int, ...],
) -> Iterator[None]:
    """Apply B2 extraction plus explicit black masks for non-matching B3 slots.

    The implementation intentionally reuses the proven reversible B2 proxy primitives.
    Only copied materials are touched, and both the base extraction and scene masks are
    restored even when Blender baking fails.
    """

    if not isinstance(materials, tuple):
        raise TypeError("materials must be tuple")
    if not isinstance(pass_plan, BakePassPlan):
        raise TypeError("pass_plan must be BakePassPlan")
    if not isinstance(used_material_indices, tuple):
        raise TypeError("used_material_indices must be tuple")

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
