"""Filesystem, logging, diagnostics, and resource infrastructure for Rewrite."""

from .atomic_files import (
    AtomicCleanupReport,
    AtomicFileCommitError,
    AtomicFileTransaction,
    AtomicOutputReservation,
    atomic_file_transaction,
    recover_stale_atomic_work_files,
)
from .export_diagnostics import (
    ExportDiagnosticsPolicy,
    configure_export_diagnostics,
    get_export_diagnostics_policy,
)
from .export_events import (
    ExportEvent,
    ExportEventDispatcher,
    ExportEventKind,
    GLOBAL_EXPORT_EVENTS,
)
from .logging_registry import (
    ModuleLogLevel,
    discover_python_modules,
    merge_module_levels,
    resolve_logger_name,
)
from .staged_text import StagedTextWriteError, write_staged_utf8_text

__all__ = [
    "AtomicCleanupReport",
    "AtomicFileCommitError",
    "AtomicFileTransaction",
    "AtomicOutputReservation",
    "ExportDiagnosticsPolicy",
    "ExportEvent",
    "ExportEventDispatcher",
    "ExportEventKind",
    "GLOBAL_EXPORT_EVENTS",
    "ModuleLogLevel",
    "StagedTextWriteError",
    "atomic_file_transaction",
    "configure_export_diagnostics",
    "discover_python_modules",
    "get_export_diagnostics_policy",
    "merge_module_levels",
    "recover_stale_atomic_work_files",
    "resolve_logger_name",
    "write_staged_utf8_text",
]
