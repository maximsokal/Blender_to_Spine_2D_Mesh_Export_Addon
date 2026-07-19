"""Generic deterministic finding helpers for shader capability auditing."""

from __future__ import annotations

from collections import defaultdict
from typing import Iterable

from ..domain.baking.capabilities import (
    ShaderBakeCapability,
    ShaderCapabilityFinding,
)
from ..domain.baking.graph import MaterialGraphSnapshot, ShaderNodeSnapshot


FindingKey = tuple[str, str, str | None, str | None, str | None]


def used_outputs(graph: MaterialGraphSnapshot) -> dict[str, tuple[str, ...]]:
    """Index the used output sockets of every reachable node deterministically."""

    values: dict[str, set[str]] = defaultdict(set)
    for link in graph.reachable_links:
        values[link.from_node_id].add(link.from_socket)
    return {
        node_id: tuple(sorted(sockets, key=str.casefold))
        for node_id, sockets in values.items()
    }


def build_finding(
    capability: ShaderBakeCapability,
    code: str,
    reason: str,
    *,
    node: ShaderNodeSnapshot | None = None,
    output_socket: str | None = None,
) -> ShaderCapabilityFinding:
    """Build one finding while preserving node metadata when available."""

    return ShaderCapabilityFinding(
        capability=capability,
        code=code,
        reason=reason,
        node_id=None if node is None else node.node_id,
        node_type=None if node is None else node.node_type,
        output_socket=output_socket,
    )


def finding_key(finding: ShaderCapabilityFinding) -> FindingKey:
    """Return the historical deduplication and ordering key for a finding."""

    if not isinstance(finding, ShaderCapabilityFinding):
        raise TypeError("finding must be ShaderCapabilityFinding")
    return (
        finding.capability.value,
        finding.code,
        finding.node_id,
        finding.node_type,
        finding.output_socket,
    )


def order_unique_findings(
    findings: Iterable[ShaderCapabilityFinding],
) -> tuple[ShaderCapabilityFinding, ...]:
    """Deduplicate findings by capability/code/location and return stable ordering."""

    unique: dict[FindingKey, ShaderCapabilityFinding] = {}
    for finding in findings:
        key = finding_key(finding)
        unique.setdefault(key, finding)
    return tuple(
        unique[key]
        for key in sorted(
            unique,
            key=lambda value: tuple(
                "" if item is None else item.casefold() for item in value
            ),
        )
    )


__all__ = [
    "FindingKey",
    "build_finding",
    "finding_key",
    "order_unique_findings",
    "used_outputs",
]
