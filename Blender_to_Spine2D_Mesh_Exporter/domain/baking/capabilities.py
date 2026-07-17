"""Blender-independent capability diagnostics for one reachable shader graph."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Iterable, Tuple


class ShaderBakeCapability(str, Enum):
    """Strongest execution boundary required to reproduce a material safely."""

    LOCAL_UV_SAFE = "LOCAL_UV_SAFE"
    SCENE_UV_SAFE = "SCENE_UV_SAFE"
    CAMERA_RENDER_REQUIRED = "CAMERA_RENDER_REQUIRED"
    GROUP_RENDER_REQUIRED = "GROUP_RENDER_REQUIRED"
    UNSUPPORTED = "UNSUPPORTED"


_CAPABILITY_PRIORITY = {
    ShaderBakeCapability.LOCAL_UV_SAFE: 0,
    ShaderBakeCapability.SCENE_UV_SAFE: 1,
    ShaderBakeCapability.CAMERA_RENDER_REQUIRED: 2,
    ShaderBakeCapability.GROUP_RENDER_REQUIRED: 3,
    ShaderBakeCapability.UNSUPPORTED: 4,
}


def strongest_shader_capability(
    capabilities: Iterable[ShaderBakeCapability],
) -> ShaderBakeCapability:
    """Return the strongest capability from a non-empty iterable."""

    resolved = tuple(capabilities)
    if not resolved:
        raise ValueError("capabilities must contain at least one value")
    if not all(isinstance(value, ShaderBakeCapability) for value in resolved):
        raise TypeError("capabilities must contain ShaderBakeCapability values")
    return max(resolved, key=_CAPABILITY_PRIORITY.__getitem__)


@dataclass(frozen=True, slots=True)
class ShaderCapabilityFinding:
    """One deterministic reason why a graph needs a specific execution boundary."""

    capability: ShaderBakeCapability
    code: str
    reason: str
    node_id: str | None = None
    node_type: str | None = None
    output_socket: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.capability, ShaderBakeCapability):
            raise TypeError("capability must be ShaderBakeCapability")
        for field_name in ("code", "reason"):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{field_name} must be a non-empty string")
        for field_name in ("node_id", "node_type", "output_socket"):
            value = getattr(self, field_name)
            if value is not None and (
                not isinstance(value, str) or not value.strip()
            ):
                raise ValueError(f"{field_name} must be a non-empty string or None")


@dataclass(frozen=True, slots=True)
class MaterialCapabilityAudit:
    """Capability report for one renderer-specific reachable material graph."""

    material_name: str
    render_target: str
    required_capability: ShaderBakeCapability
    findings: Tuple[ShaderCapabilityFinding, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.material_name, str) or not self.material_name.strip():
            raise ValueError("material_name must be a non-empty string")
        if self.render_target not in {"ALL", "CYCLES", "EEVEE"}:
            raise ValueError("render_target must be ALL, CYCLES, or EEVEE")
        if not isinstance(self.required_capability, ShaderBakeCapability):
            raise TypeError("required_capability must be ShaderBakeCapability")
        if not isinstance(self.findings, tuple) or not self.findings:
            raise ValueError("findings must be a non-empty tuple")
        if not all(
            isinstance(value, ShaderCapabilityFinding) for value in self.findings
        ):
            raise TypeError("findings must contain ShaderCapabilityFinding values")
        strongest = strongest_shader_capability(
            finding.capability for finding in self.findings
        )
        if strongest is not self.required_capability:
            raise ValueError(
                "required_capability must equal the strongest finding capability"
            )
