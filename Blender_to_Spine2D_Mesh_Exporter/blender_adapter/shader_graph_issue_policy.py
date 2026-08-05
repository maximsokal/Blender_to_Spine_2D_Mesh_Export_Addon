"""Classify recursive shader-graph diagnostics by export safety severity."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class ShaderGraphIssueSeverity(str, Enum):
    """Whether one traversal diagnostic blocks capability routing."""

    ADVISORY = "ADVISORY"
    BLOCKING = "BLOCKING"


@dataclass(frozen=True, slots=True)
class ShaderGraphIssueClassification:
    """Typed result for one deterministic traversal diagnostic."""

    issue: str
    severity: ShaderGraphIssueSeverity
    capability_code: str

    def __post_init__(self) -> None:
        if not isinstance(self.issue, str) or not self.issue.strip():
            raise ValueError("issue must be a non-empty string")
        if not isinstance(self.severity, ShaderGraphIssueSeverity):
            raise TypeError("severity must be ShaderGraphIssueSeverity")
        if (
            not isinstance(self.capability_code, str)
            or not self.capability_code.strip()
        ):
            raise ValueError("capability_code must be a non-empty string")

    @property
    def blocks_export(self) -> bool:
        """Return whether the issue means reachable shader behavior was skipped."""

        return self.severity is ShaderGraphIssueSeverity.BLOCKING


_MUTED_CONSERVATIVE_PREFIX = "Muted node '"
_MUTED_CONSERVATIVE_SUFFIX = "; all inputs were analyzed conservatively"


def classify_shader_graph_issue(issue: str) -> ShaderGraphIssueClassification:
    """Classify one issue without weakening genuinely incomplete graph analysis.

    A muted node can expose no unique internal bypass when multiple sockets share the
    same Blender display name. The recursive walker then visits every input, which is
    conservative but complete for capability discovery. That exact diagnostic remains
    visible to users as an advisory instead of becoming a false ``UNSUPPORTED`` result.

    Every other issue remains blocking by default. Missing group trees, unresolved
    interfaces, recursive groups, and depth-limit failures can omit executable shader
    behavior and therefore must continue to fail closed.
    """

    if not isinstance(issue, str):
        raise TypeError("issue must be str")
    normalized = issue.strip()
    if not normalized:
        raise ValueError("issue must be a non-empty string")

    if (
        normalized.startswith(_MUTED_CONSERVATIVE_PREFIX)
        and normalized.endswith(_MUTED_CONSERVATIVE_SUFFIX)
    ):
        return ShaderGraphIssueClassification(
            issue=normalized,
            severity=ShaderGraphIssueSeverity.ADVISORY,
            capability_code="MUTED_NODE_CONSERVATIVE_ANALYSIS",
        )

    return ShaderGraphIssueClassification(
        issue=normalized,
        severity=ShaderGraphIssueSeverity.BLOCKING,
        capability_code="GRAPH_ANALYSIS_INCOMPLETE",
    )


__all__ = [
    "ShaderGraphIssueClassification",
    "ShaderGraphIssueSeverity",
    "classify_shader_graph_issue",
]
