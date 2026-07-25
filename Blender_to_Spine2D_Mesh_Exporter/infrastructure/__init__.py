"""Filesystem, logging, diagnostics, and resource infrastructure for Rewrite."""

from .atomic_files import (
    AtomicCleanupReport,
    AtomicFileCommitError,
    AtomicFileTransaction,
    AtomicOutputReservation,
    AtomicRecoveryAction,
    AtomicRecoveryRecord,
    atomic_file_transaction,
    recover_stale_atomic_work_files,
)
from .atomic_work_state import (
    AtomicRecoveryReason,
    AtomicWorkFileAssessment,
    AtomicWorkFileState,
    AtomicWorkTokenMetadata,
    DEFAULT_STALE_WORK_FILE_AGE_SECONDS,
    assess_atomic_work_file,
    read_process_start_marker,
)
from .blender_registration import (
    RegistrationCleanupAction,
    RegistrationCleanupError,
    RegistrationCleanupFailure,
    RnaPropertyRegistration,
    class_cleanup_actions,
    register_classes_transactionally,
    register_rna_properties_transactionally,
    rna_property_cleanup_actions,
    unregister_all_best_effort,
)
from .exclusive_operation import (
    A1_EXPORT_OPERATION_KEY,
    ExclusiveOperationLease,
    OperationAlreadyRunningError,
    active_exclusive_operations,
    exclusive_operation,
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
    "A1_EXPORT_OPERATION_KEY",
    "AtomicCleanupReport",
    "AtomicFileCommitError",
    "AtomicFileTransaction",
    "AtomicOutputReservation",
    "AtomicRecoveryAction",
    "AtomicRecoveryReason",
    "AtomicRecoveryRecord",
    "AtomicWorkFileAssessment",
    "AtomicWorkFileState",
    "AtomicWorkTokenMetadata",
    "DEFAULT_STALE_WORK_FILE_AGE_SECONDS",
    "ExclusiveOperationLease",
    "ExportDiagnosticsPolicy",
    "ExportEvent",
    "ExportEventDispatcher",
    "ExportEventKind",
    "GLOBAL_EXPORT_EVENTS",
    "ModuleLogLevel",
    "OperationAlreadyRunningError",
    "RegistrationCleanupAction",
    "RegistrationCleanupError",
    "RegistrationCleanupFailure",
    "RnaPropertyRegistration",
    "StagedTextWriteError",
    "active_exclusive_operations",
    "assess_atomic_work_file",
    "atomic_file_transaction",
    "class_cleanup_actions",
    "configure_export_diagnostics",
    "discover_python_modules",
    "exclusive_operation",
    "get_export_diagnostics_policy",
    "merge_module_levels",
    "read_process_start_marker",
    "recover_stale_atomic_work_files",
    "register_classes_transactionally",
    "register_rna_properties_transactionally",
    "resolve_logger_name",
    "rna_property_cleanup_actions",
    "unregister_all_best_effort",
    "write_staged_utf8_text",
]
