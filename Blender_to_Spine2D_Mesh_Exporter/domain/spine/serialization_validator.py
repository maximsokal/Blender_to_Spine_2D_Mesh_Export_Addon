"""Serialization-boundary validation for typed and raw Spine mesh attachments.

Typed :class:`MeshAttachment` values use the Rewrite domain's logical vertex-index
space for ``edges``. Raw JSON mappings are already in Spine's serialized coordinate-
offset space, where each endpoint is ``vertex_index * 2``. The base validator owns
all common document, mesh, triangle, weighted-stream, and cross-reference checks;
this specialization normalizes only raw edge offsets before delegating to it.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .mesh_edge_contract import (
    SpineMeshEdgeContractError,
    validate_spine_mesh_edge_offsets,
)
from .validator import (
    SpineValidationIssue,
    SpineValidator,
    ValidationSeverity,
)


class SpineSerializationValidator(SpineValidator):
    """Validate values exactly as they enter :class:`SpineSerializer`."""

    def _validate_raw_attachment(
        self,
        attachment: Mapping[str, Any],
        *,
        path: str,
        bone_count: int,
    ) -> list[SpineValidationIssue]:
        if not isinstance(attachment, Mapping):
            return super()._validate_raw_attachment(
                attachment,
                path=path,
                bone_count=bone_count,
            )

        attachment_type = attachment.get("type", "region")
        parent = attachment.get("parent")
        if attachment_type != "mesh" or parent or "edges" not in attachment:
            return super()._validate_raw_attachment(
                attachment,
                path=path,
                bone_count=bone_count,
            )

        uvs = attachment.get("uvs")
        edges = attachment.get("edges")
        if (
            not isinstance(uvs, (list, tuple))
            or len(uvs) % 2 != 0
            or not isinstance(edges, (list, tuple))
        ):
            # The base validator owns the primary shape/type diagnostics. Avoid
            # inventing a secondary edge-space interpretation when vertex count is
            # unavailable or the edge payload itself is not sequence-like.
            return super()._validate_raw_attachment(
                attachment,
                path=path,
                bone_count=bone_count,
            )

        vertex_count = len(uvs) // 2
        edge_tuple = tuple(edges)
        try:
            validate_spine_mesh_edge_offsets(
                edge_tuple,
                vertex_count=vertex_count,
            )
        except (TypeError, SpineMeshEdgeContractError, ValueError) as exc:
            # Validate the remainder of the attachment without allowing the base
            # logical-index validator to add misleading range errors for the same
            # serialized payload.
            without_edges = dict(attachment)
            without_edges["edges"] = []
            issues = super()._validate_raw_attachment(
                without_edges,
                path=path,
                bone_count=bone_count,
            )
            issues.append(
                SpineValidationIssue(
                    severity=ValidationSeverity.ERROR,
                    code="INVALID_SERIALIZED_MESH_EDGES",
                    path=f"{path}.edges",
                    message=str(exc),
                )
            )
            return issues

        normalized = dict(attachment)
        normalized["edges"] = [offset // 2 for offset in edge_tuple]
        return super()._validate_raw_attachment(
            normalized,
            path=path,
            bone_count=bone_count,
        )


__all__ = ["SpineSerializationValidator"]
