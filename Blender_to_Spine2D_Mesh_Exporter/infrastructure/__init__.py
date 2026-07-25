"""Filesystem, logging, diagnostics, and resource infrastructure for Rewrite."""

from .atomic_files import AtomicCleanupReport, AtomicFileCommitError, AtomicOutputReservation, AtomicRecoveryAction, AtomicRecoveryRecord, recover_stale_atomic_work_files
from .atomic_journal import AtomicJournalRecoveryReport, recover_interrupted_atomic_journals
from .atomic_work_state import AtomicRecoveryReason, AtomicWorkFileAssessment, AtomicWorkFileState, AtomicWorkTokenMetadata, DEFAULT_STALE_WORK_FILE_AGE_SECONDS, assess_atomic_work_file, read_process_start_marker
from .blender_registration import RegistrationCleanupAction, RegistrationCleanupError, RegistrationCleanupFailure, RnaPropertyRegistration, class_cleanup_actions, register_classes_transactionally, register_rna_properties_transactionally, rna_property_cleanup_actions, unregister_all_best_effort
from .durable_atomic_transaction import DurableAtomicFileTransaction, durable_atomic_file_transaction
from .durable_io import DurableIoError, durable_replace, durable_unlink, fsync_directory, fsync_file
from .exclusive_operation import A1_EXPORT_OPERATION_KEY, ExclusiveOperationLease, OperationAlreadyRunningError, active_exclusive_operations, exclusive_operation
from .export_diagnostics import ExportDiagnosticsPolicy, configure_export_diagnostics, get_export_diagnostics_policy
from .export_events import ExportEvent, ExportEventDispatcher, ExportEventKind, GLOBAL_EXPORT_EVENTS
from .interprocess_lock import InterprocessFileLock, InterprocessLockError, interprocess_file_lock
from .logging_registry import ModuleLogLevel, discover_python_modules, merge_module_levels, resolve_logger_name
from .performance_budget import PerformanceSample, RelativePerformanceBudget, measure_median
from .process_identity import process_identity_is_alive
from .staged_text import StagedTextWriteError, write_staged_utf8_text

AtomicFileTransaction = DurableAtomicFileTransaction
atomic_file_transaction = durable_atomic_file_transaction

__all__ = ["A1_EXPORT_OPERATION_KEY", "AtomicCleanupReport", "AtomicFileCommitError", "AtomicFileTransaction", "AtomicJournalRecoveryReport", "AtomicOutputReservation", "AtomicRecoveryAction", "AtomicRecoveryReason", "AtomicRecoveryRecord", "AtomicWorkFileAssessment", "AtomicWorkFileState", "AtomicWorkTokenMetadata", "DEFAULT_STALE_WORK_FILE_AGE_SECONDS", "DurableAtomicFileTransaction", "DurableIoError", "ExclusiveOperationLease", "ExportDiagnosticsPolicy", "ExportEvent", "ExportEventDispatcher", "ExportEventKind", "GLOBAL_EXPORT_EVENTS", "InterprocessFileLock", "InterprocessLockError", "ModuleLogLevel", "OperationAlreadyRunningError", "PerformanceSample", "RegistrationCleanupAction", "RegistrationCleanupError", "RegistrationCleanupFailure", "RelativePerformanceBudget", "RnaPropertyRegistration", "StagedTextWriteError", "active_exclusive_operations", "assess_atomic_work_file", "atomic_file_transaction", "class_cleanup_actions", "configure_export_diagnostics", "discover_python_modules", "durable_atomic_file_transaction", "durable_replace", "durable_unlink", "exclusive_operation", "fsync_directory", "fsync_file", "get_export_diagnostics_policy", "interprocess_file_lock", "measure_median", "merge_module_levels", "process_identity_is_alive", "read_process_start_marker", "recover_interrupted_atomic_journals", "recover_stale_atomic_work_files", "register_classes_transactionally", "register_rna_properties_transactionally", "resolve_logger_name", "rna_property_cleanup_actions", "unregister_all_best_effort", "write_staged_utf8_text"]
