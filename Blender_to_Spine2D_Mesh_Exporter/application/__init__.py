"""Application use-case contracts for the rewrite pipeline."""

from .a1_attachment_projection import (
    A1AttachmentProjectionError,
    A1AttachmentProjectionResult,
    A1AttachmentProjectionSettings,
    A1VertexZBinding,
    project_triangulated_disk_attachment,
)
from .a1_document_assembly import (
    A1DocumentAssemblyError,
    A1DocumentAssemblyResult,
    A1DocumentAssemblySettings,
    assemble_a1_document,
)
from .a1_geometry_preparation import (
    A1GeometryPreparationError,
    A1GeometryPreparationResult,
    A1GeometryPreparationSettings,
    A1PreparedRegion,
    prepare_a1_geometry_regions,
)
from .a1_z_groups import (
    A1SourceVertexZBinding,
    A1ZGroupAssignmentError,
    A1ZGroupAssignmentPlan,
    A1ZGroupHeightOverride,
    build_a1_z_group_assignment,
)
from .contracts import ExportIssue, ExportRequest, ExportResult, ExportSettings

__all__ = [
    "A1AttachmentProjectionError",
    "A1AttachmentProjectionResult",
    "A1AttachmentProjectionSettings",
    "A1DocumentAssemblyError",
    "A1DocumentAssemblyResult",
    "A1DocumentAssemblySettings",
    "A1GeometryPreparationError",
    "A1GeometryPreparationResult",
    "A1GeometryPreparationSettings",
    "A1PreparedRegion",
    "A1SourceVertexZBinding",
    "A1VertexZBinding",
    "A1ZGroupAssignmentError",
    "A1ZGroupAssignmentPlan",
    "A1ZGroupHeightOverride",
    "ExportIssue",
    "ExportRequest",
    "ExportResult",
    "ExportSettings",
    "assemble_a1_document",
    "build_a1_z_group_assignment",
    "prepare_a1_geometry_regions",
    "project_triangulated_disk_attachment",
]
