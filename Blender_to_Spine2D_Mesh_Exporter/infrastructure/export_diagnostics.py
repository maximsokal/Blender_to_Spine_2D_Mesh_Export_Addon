"""Process-local diagnostics policy shared by Blender adapters and infrastructure."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ExportDiagnosticsPolicy:
    preserve_failed_work_files: bool = False
    recover_stale_work_files: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.preserve_failed_work_files, bool):
            raise TypeError("preserve_failed_work_files must be bool")
        if not isinstance(self.recover_stale_work_files, bool):
            raise TypeError("recover_stale_work_files must be bool")


_policy = ExportDiagnosticsPolicy()


def configure_export_diagnostics(
    *,
    preserve_failed_work_files: bool,
    recover_stale_work_files: bool = True,
) -> ExportDiagnosticsPolicy:
    """Replace and return the process-local diagnostics policy.

    Blender invokes the add-on lifecycle synchronously on its main Python thread, so
    this process-local policy deliberately avoids Python thread synchronization.
    """

    global _policy
    resolved = ExportDiagnosticsPolicy(
        preserve_failed_work_files=preserve_failed_work_files,
        recover_stale_work_files=recover_stale_work_files,
    )
    _policy = resolved
    return resolved


def get_export_diagnostics_policy() -> ExportDiagnosticsPolicy:
    return _policy


__all__ = [
    "ExportDiagnosticsPolicy",
    "configure_export_diagnostics",
    "get_export_diagnostics_policy",
]
