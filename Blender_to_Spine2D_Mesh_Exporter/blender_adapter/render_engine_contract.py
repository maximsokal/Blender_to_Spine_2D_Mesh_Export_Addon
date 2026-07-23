"""Canonical renderer identity shared by analysis, planning, and execution."""

from __future__ import annotations

from dataclasses import dataclass

from ..domain.baking import BakeExecutionSettings, SceneBakeContext


class RenderEngineContractError(ValueError):
    """Raised when one export mixes incompatible Blender render engines."""


@dataclass(frozen=True, slots=True)
class RenderEngineContract:
    """One immutable Blender 5.2+ engine and ShaderNodeTree output target."""

    blender_engine: str
    shader_target: str

    def __post_init__(self) -> None:
        if self.blender_engine not in {"CYCLES", "BLENDER_EEVEE"}:
            raise ValueError("blender_engine must be CYCLES or BLENDER_EEVEE")
        if self.shader_target not in {"CYCLES", "EEVEE"}:
            raise ValueError("shader_target must be CYCLES or EEVEE")
        expected_target = "CYCLES" if self.blender_engine == "CYCLES" else "EEVEE"
        if self.shader_target != expected_target:
            raise ValueError("shader_target does not match blender_engine")

    @property
    def uses_eevee(self) -> bool:
        return self.shader_target == "EEVEE"

    def validate_scene(self, scene_context: SceneBakeContext) -> None:
        if not isinstance(scene_context, SceneBakeContext):
            raise TypeError("scene_context must be SceneBakeContext")
        actual = render_engine_contract(scene_context.render_engine)
        if actual != self:
            raise RenderEngineContractError(
                "render engine changed between export settings and Scene analysis; "
                f"requested={self.blender_engine}, scene={actual.blender_engine}"
            )


def render_engine_contract(value: str) -> RenderEngineContract:
    """Normalize Blender 5.2+ and ShaderNodeTree renderer identifiers."""

    if not isinstance(value, str) or not value.strip():
        raise ValueError("render engine must be a non-empty string")
    normalized = value.strip().upper()
    if "CYCLE" in normalized:
        return RenderEngineContract("CYCLES", "CYCLES")
    if "EEVEE" in normalized:
        return RenderEngineContract("BLENDER_EEVEE", "EEVEE")
    raise RenderEngineContractError(
        f"unsupported Blender 5.2+ render engine for texture export: {value!r}"
    )


def render_engine_contract_from_execution(
    execution_settings: BakeExecutionSettings,
) -> RenderEngineContract:
    if not isinstance(execution_settings, BakeExecutionSettings):
        raise TypeError("execution_settings must be BakeExecutionSettings")
    return render_engine_contract(execution_settings.render_engine)
