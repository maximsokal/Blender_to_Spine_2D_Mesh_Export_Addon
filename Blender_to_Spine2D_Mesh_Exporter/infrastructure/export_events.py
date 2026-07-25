"""Non-fatal event dispatch for export lifecycle and cleanup diagnostics."""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
import logging
from pathlib import Path
from threading import RLock
from types import MappingProxyType
from typing import Callable, Mapping
from uuid import uuid4

logger = logging.getLogger(__name__)

class ExportEventKind(str, Enum):
    TRANSACTION_STARTED = "TRANSACTION_STARTED"
    OUTPUT_RESERVED = "OUTPUT_RESERVED"
    COMMIT_STARTED = "COMMIT_STARTED"
    COMMIT_SUCCEEDED = "COMMIT_SUCCEEDED"
    ROLLBACK_STARTED = "ROLLBACK_STARTED"
    ROLLBACK_SUCCEEDED = "ROLLBACK_SUCCEEDED"
    WORK_FILE_REMOVED = "WORK_FILE_REMOVED"
    WORK_FILE_PRESERVED = "WORK_FILE_PRESERVED"
    BACKUP_RESTORED = "BACKUP_RESTORED"
    STALE_WORK_RECOVERED = "STALE_WORK_RECOVERED"
    TRANSACTION_FAILED = "TRANSACTION_FAILED"
    CLEANUP_FAILED = "CLEANUP_FAILED"

@dataclass(frozen=True, slots=True)
class ExportEvent:
    kind: ExportEventKind
    operation_id: str
    message: str
    path: Path | None = None
    context: Mapping[str, object] = field(default_factory=dict)
    def __post_init__(self) -> None:
        if not isinstance(self.kind, ExportEventKind): raise TypeError("kind must be ExportEventKind")
        if not isinstance(self.operation_id, str) or not self.operation_id.strip(): raise ValueError("operation_id must be a non-empty string")
        if not isinstance(self.message, str) or not self.message.strip(): raise ValueError("message must be a non-empty string")
        if self.path is not None and not isinstance(self.path, Path): raise TypeError("path must be pathlib.Path or None")
        if not isinstance(self.context, Mapping): raise TypeError("context must be a mapping")
        object.__setattr__(self, "context", MappingProxyType(dict(self.context)))

ExportEventListener = Callable[[ExportEvent], None]

class ExportEventDispatcher:
    def __init__(self) -> None:
        self._lock = RLock(); self._listeners: dict[str, ExportEventListener] = {}
    def subscribe(self, listener: ExportEventListener) -> str:
        if not callable(listener): raise TypeError("listener must be callable")
        token = uuid4().hex
        with self._lock: self._listeners[token] = listener
        return token
    def unsubscribe(self, token: str) -> bool:
        with self._lock: return self._listeners.pop(token, None) is not None
    def emit(self, event: ExportEvent) -> None:
        if not isinstance(event, ExportEvent): raise TypeError("event must be ExportEvent")
        with self._lock: listeners = tuple(self._listeners.values())
        for listener in listeners:
            try: listener(event)
            except Exception: logger.exception("Export event listener failed for %s", event.kind.value)
        failure_kinds = {ExportEventKind.TRANSACTION_FAILED, ExportEventKind.CLEANUP_FAILED}
        log_method = logger.error if event.kind in failure_kinds else logger.debug
        stage = event.context.get("stage")
        log_method("[%s] %s%s%s", event.operation_id, event.message, "" if event.path is None else f" path={event.path}", "" if stage is None else f" stage={stage}")

GLOBAL_EXPORT_EVENTS = ExportEventDispatcher()
__all__ = ["ExportEvent", "ExportEventDispatcher", "ExportEventKind", "ExportEventListener", "GLOBAL_EXPORT_EVENTS"]
