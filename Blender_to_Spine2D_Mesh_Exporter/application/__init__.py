"""Application use-case contracts for the rewrite pipeline."""

from .a1_attachment_projection import (
    A1AttachmentProjectionError,
    A1AttachmentProjectionResult,
    A1AttachmentProjectionSettings,
    A1VertexZBinding,
    project_triangulated_disk_attachment,
)
from .contracts import ExportIssue, ExportRequest, ExportResult, ExportSettings

__all__ = [
    "A1AttachmentProjectionError",
    "A1AttachmentProjectionResult",
    "A1AttachmentProjectionSettings",
    "A1VertexZBinding",
    "ExportIssue",
    "ExportRequest",
    "ExportResult",
    "ExportSettings",
    "project_triangulated_disk_attachment",
]
