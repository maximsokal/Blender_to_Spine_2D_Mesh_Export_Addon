"""Analyze Blender object material slots in stable dense order."""

from __future__ import annotations

import logging
from typing import Any

from ..domain.baking import ObjectMaterialAnalysis
from .material_analysis_error import MaterialAnalysisError
from .material_analysis_rna import (
    object_material_slots,
    object_name,
    resolve_render_target,
)
from .material_slot_analysis import analyse_material_slot


logger = logging.getLogger(__name__)


def analyse_object_materials(
    obj: Any,
    *,
    source_object_id: str | None = None,
    render_target: str | None = None,
) -> ObjectMaterialAnalysis:
    """Analyze all material slots of one Blender mesh object in stable order."""

    if obj is None:
        raise MaterialAnalysisError("obj cannot be None")
    if getattr(obj, "type", None) != "MESH":
        raise MaterialAnalysisError("obj must be a Blender MESH object")

    resolved_object_name = object_name(obj)
    resolved_source_object_id = source_object_id or resolved_object_name
    target = resolve_render_target(render_target)

    try:
        material_slots = object_material_slots(obj)
        analyses = tuple(
            analyse_material_slot(
                slot_index,
                getattr(slot, "material", None),
                render_target=target,
            )
            for slot_index, slot in enumerate(material_slots)
        )
        result = ObjectMaterialAnalysis(
            source_object_id=resolved_source_object_id,
            slots=analyses,
        )
        logger.debug(
            "Analyzed %d material slots for '%s' target=%s: kinds=%s channels=%s",
            len(result.slots),
            resolved_object_name,
            target,
            tuple(slot.kind.value for slot in result.slots),
            tuple(
                tuple(channel.value for channel in slot.semantic_channels)
                for slot in result.slots
            ),
        )
        return result
    except MaterialAnalysisError:
        raise
    except Exception as exc:
        logger.exception(
            "Failed to analyze materials for '%s'",
            resolved_object_name,
        )
        raise MaterialAnalysisError(
            f"Failed to analyze materials for '{resolved_object_name}': {exc}"
        ) from exc


__all__ = ["analyse_object_materials"]
