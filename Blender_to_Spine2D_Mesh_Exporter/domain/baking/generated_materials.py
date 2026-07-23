"""Typed contracts for deterministic generated Rewrite materials."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from math import isfinite
from typing import Tuple

from ..geometry import MeshSnapshot
from .model import BakePlan


ColorRGBA = Tuple[float, float, float, float]


class A1MaterialSourcePolicy(str, Enum):
    """Choose whether Rewrite requires source materials or generates replacements."""

    REQUIRE_SOURCE = "REQUIRE_SOURCE"
    GENERATE_IF_MISSING = "GENERATE_IF_MISSING"
    FORCE_GENERATED = "FORCE_GENERATED"


class A1GeneratedMaterialPattern(str, Enum):
    """Coloring pattern used by one temporary generated material."""

    SOLID_GRAY = "SOLID_GRAY"
    REGION_COLORS = "REGION_COLORS"
    EXPORTED_FACE_COLORS = "EXPORTED_FACE_COLORS"


def _validate_color(value: ColorRGBA, field_name: str) -> None:
    if not isinstance(value, tuple) or len(value) != 4:
        raise ValueError(f"{field_name} must contain four values")
    for index, component in enumerate(value):
        if isinstance(component, bool) or not isinstance(component, (int, float)):
            raise TypeError(f"{field_name}[{index}] must be numeric")
        numeric = float(component)
        if not isfinite(numeric) or numeric < 0.0 or numeric > 1.0:
            raise ValueError(f"{field_name}[{index}] must be finite in the range [0, 1]")
    if float(value[3]) != 1.0:
        raise ValueError(f"{field_name}[3] must be 1.0 for opaque generated textures")


@dataclass(frozen=True, slots=True)
class GeneratedMaterialPlan:
    """One immutable generated-color binding for a final triangulated bake snapshot."""

    source_policy: A1MaterialSourcePolicy
    pattern: A1GeneratedMaterialPattern
    target_snapshot: MeshSnapshot
    face_colors: Tuple[ColorRGBA, ...]
    color_attribute_name: str = "Spine2DGeneratedColor"
    material_name: str = "__Spine2D_GeneratedMaterial"

    def __post_init__(self) -> None:
        if not isinstance(self.source_policy, A1MaterialSourcePolicy):
            raise TypeError("source_policy must be A1MaterialSourcePolicy")
        if not isinstance(self.pattern, A1GeneratedMaterialPattern):
            raise TypeError("pattern must be A1GeneratedMaterialPattern")
        if not isinstance(self.target_snapshot, MeshSnapshot):
            raise TypeError("target_snapshot must be MeshSnapshot")
        if not isinstance(self.face_colors, tuple):
            raise TypeError("face_colors must be tuple")
        if len(self.face_colors) != len(self.target_snapshot.faces):
            raise ValueError(
                "face_colors must contain one color for every target snapshot face"
            )
        for index, color in enumerate(self.face_colors):
            _validate_color(color, f"face_colors[{index}]")
        for field_name in ("color_attribute_name", "material_name"):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{field_name} must be a non-empty string")
        if self.target_snapshot.faces and {
            face.material_index for face in self.target_snapshot.faces
        } != {0}:
            raise ValueError(
                "generated target snapshot faces must all reference material slot zero"
            )


@dataclass(frozen=True, slots=True)
class GeneratedBakePlan(BakePlan):
    """BakePlan specialization carrying a generated material and target snapshot."""

    generated_material: GeneratedMaterialPlan | None = None

    def __post_init__(self) -> None:
        BakePlan.__post_init__(self)
        if not isinstance(self.generated_material, GeneratedMaterialPlan):
            raise TypeError("generated_material must be GeneratedMaterialPlan")
        if (
            self.generated_material.target_snapshot.source_object_id
            != self.source_object_id
        ):
            raise ValueError(
                "generated material target snapshot must match BakePlan source_object_id"
            )
        if len(self.material_analysis.slots) != 1:
            raise ValueError("GeneratedBakePlan requires exactly one synthetic material slot")
        if any(
            slot_index != 0
            for pass_plan in self.passes
            for slot_index in pass_plan.material_slot_indices
        ):
            raise ValueError("GeneratedBakePlan passes may reference only slot zero")

    @classmethod
    def from_bake_plan(
        cls,
        plan: BakePlan,
        generated_material: GeneratedMaterialPlan,
    ) -> "GeneratedBakePlan":
        if not isinstance(plan, BakePlan):
            raise TypeError("plan must be BakePlan")
        if not isinstance(generated_material, GeneratedMaterialPlan):
            raise TypeError("generated_material must be GeneratedMaterialPlan")
        return cls(
            source_object_id=plan.source_object_id,
            settings=plan.settings,
            material_analysis=plan.material_analysis,
            bake_mode=plan.bake_mode,
            frame_tasks=plan.frame_tasks,
            representative_task_index=plan.representative_task_index,
            passes=plan.passes,
            composite=plan.composite,
            object_context=plan.object_context,
            scene_context=plan.scene_context,
            generated_material=generated_material,
        )


__all__ = [
    "A1GeneratedMaterialPattern",
    "A1MaterialSourcePolicy",
    "ColorRGBA",
    "GeneratedBakePlan",
    "GeneratedMaterialPlan",
]
