"""Process identity, work-token parsing, and process-local atomic reservations."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
import os
from pathlib import Path
from threading import RLock
import time
from uuid import uuid4


STAGE_MARKER = ".spine2d-stage-"
BACKUP_MARKER = ".spine2d-backup-"
DEFAULT_STALE_WORK_FILE_AGE_SECONDS = 300.0
_TOKEN_VERSION = "v2"
_TOKEN_SEPARATOR = "~"


class AtomicWorkFileState(str, Enum):
    """Classification used before stale-work recovery mutates a file."""

    ACTIVE = "ACTIVE"
    DEFERRED = "DEFERRED"
    STALE = "STALE"


class AtomicRecoveryReason(str, Enum):
    """Explicit reason why one work file was skipped, deferred, or recovered."""

    PROCESS_LOCAL_ACTIVE_TOKEN = "PROCESS_LOCAL_ACTIVE_TOKEN"
    OWNER_PROCESS_ACTIVE = "OWNER_PROCESS_ACTIVE"
    OWNER_PROCESS_IDENTITY_UNAVAILABLE = "OWNER_PROCESS_IDENTITY_UNAVAILABLE"
    LEGACY_OWNER_PROCESS_ACTIVE = "LEGACY_OWNER_PROCESS_ACTIVE"
    MINIMUM_STALE_AGE_NOT_REACHED = "MINIMUM_STALE_AGE_NOT_REACHED"
    UNREGISTERED_CURRENT_PROCESS_TOKEN = "UNREGISTERED_CURRENT_PROCESS_TOKEN"
    OWNER_PROCESS_EXITED = "OWNER_PROCESS_EXITED"
    OWNER_PROCESS_IDENTITY_MISMATCH = "OWNER_PROCESS_IDENTITY_MISMATCH"
    LEGACY_OWNER_PROCESS_EXITED = "LEGACY_OWNER_PROCESS_EXITED"
    MALFORMED_WORK_TOKEN = "MALFORMED_WORK_TOKEN"


@dataclass(frozen=True, slots=True)
class AtomicWorkTokenMetadata:
    """Versioned ownership metadata encoded into every stage/backup filename."""

    process_id: int
    process_start_marker: str
    created_ns: int
    nonce: str

    def __post_init__(self) -> None:
        if isinstance(self.process_id, bool) or not isinstance(self.process_id, int):
            raise TypeError("process_id must be int")
        if self.process_id <= 0:
            raise ValueError("process_id must be positive")
        for field_name, value in (
            ("process_start_marker", self.process_start_marker),
            ("nonce", self.nonce),
        ):
            if not isinstance(value, str) or not value:
                raise ValueError(f"{field_name} must be a non-empty string")
            if value != value.strip() or _TOKEN_SEPARATOR in value or "." in value:
                raise ValueError(
                    f"{field_name} must be filename-safe and contain no separators"
                )
        if isinstance(self.created_ns, bool) or not isinstance(self.created_ns, int):
            raise TypeError("created_ns must be int")
        if self.created_ns <= 0:
            raise ValueError("created_ns must be positive")

    @property
    def token(self) -> str:
        return _TOKEN_SEPARATOR.join(
            (
                _TOKEN_VERSION,
                str(self.process_id),
                self.process_start_marker,
                str(self.created_ns),
                self.nonce,
            )
        )

    @classmethod
    def parse(cls, token: str) -> "AtomicWorkTokenMetadata | None":
        if not isinstance(token, str) or not token:
            return None
        parts = token.split(_TOKEN_SEPARATOR)
        if len(parts) != 5 or parts[0] != _TOKEN_VERSION:
            return None
        try:
            return cls(
                process_id=int(parts[1]),
                process_start_marker=parts[2],
                created_ns=int(parts[3]),
                nonce=parts[4],
            )
        except (TypeError, ValueError):
            return None


@dataclass(frozen=True, slots=True)
class AtomicWorkFileAssessment:
    """Non-mutating classification of one stage or backup work file."""

    path: Path
    token: str | None
    state: AtomicWorkFileState
    reason: AtomicRecoveryReason
    age_seconds: float
    metadata: AtomicWorkTokenMetadata | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.path, Path):
            raise TypeError("path must be pathlib.Path")
        if self.token is not None and not isinstance(self.token, str):
            raise TypeError("token must be str or None")
        if not isinstance(self.state, AtomicWorkFileState):
            raise TypeError("state must be AtomicWorkFileState")
        if not isinstance(self.reason, AtomicRecoveryReason):
            raise TypeError("reason must be AtomicRecoveryReason")
        if isinstance(self.age_seconds, bool) or not isinstance(
            self.age_seconds, (int, float)
        ):
            raise TypeError("age_seconds must be a finite number")
        if not math.isfinite(float(self.age_seconds)) or self.age_seconds < 0.0:
            raise ValueError("age_seconds must be finite and non-negative")
        if self.metadata is not None and not isinstance(
            self.metadata, AtomicWorkTokenMetadata
        ):
            raise TypeError("metadata must be AtomicWorkTokenMetadata or None")

    @property
    def recoverable(self) -> bool:
        return self.state is AtomicWorkFileState.STALE


def _validate_minimum_age(value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError("minimum_stale_age_seconds must be a finite number")
    resolved = float(value)
    if not math.isfinite(resolved) or resolved < 0.0:
        raise ValueError("minimum_stale_age_seconds must be finite and non-negative")
    return resolved


def _windows_process_is_alive(process_id: int) -> bool:
    """Query process state without using ``os.kill(pid, 0)`` on Windows.

    Python documents that non-console signals on Windows call TerminateProcess, so
    signal zero is not a safe liveness probe there. Access-denied probes fail closed
    as active because deleting another live process's work file is worse than
    deferring recovery.
    """

    try:
        import ctypes
        from ctypes import wintypes

        process_query_limited_information = 0x1000
        error_access_denied = 5
        still_active = 259
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        open_process = kernel32.OpenProcess
        open_process.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        open_process.restype = wintypes.HANDLE
        get_exit_code = kernel32.GetExitCodeProcess
        get_exit_code.argtypes = [
            wintypes.HANDLE,
            ctypes.POINTER(wintypes.DWORD),
        ]
        get_exit_code.restype = wintypes.BOOL
        close_handle = kernel32.CloseHandle
        close_handle.argtypes = [wintypes.HANDLE]
        close_handle.restype = wintypes.BOOL

        handle = open_process(
            process_query_limited_information,
            False,
            process_id,
        )
        if not handle:
            return ctypes.get_last_error() == error_access_denied
        try:
            exit_code = wintypes.DWORD()
            if not get_exit_code(handle, ctypes.byref(exit_code)):
                return ctypes.get_last_error() == error_access_denied
            return int(exit_code.value) == still_active
        finally:
            close_handle(handle)
    except (AttributeError, ImportError, OSError, TypeError, ValueError):
        return False


def _process_is_alive(process_id: int) -> bool:
    if process_id <= 0:
        return False
    if process_id == os.getpid():
        return True
    if os.name == "nt":
        return _windows_process_is_alive(process_id)
    try:
        os.kill(process_id, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True


def _linux_process_start_marker(process_id: int) -> str | None:
    try:
        payload = Path(f"/proc/{process_id}/stat").read_text(encoding="utf-8")
        command_end = payload.rfind(")")
        if command_end < 0:
            return None
        fields_after_command = payload[command_end + 2 :].split()
        # /proc/<pid>/stat field 22 is starttime; field 3 begins at index zero here.
        start_ticks = int(fields_after_command[19])
        return f"linux-{start_ticks:x}"
    except (OSError, IndexError, TypeError, ValueError):
        return None


def _windows_process_start_marker(process_id: int) -> str | None:
    try:
        import ctypes
        from ctypes import wintypes

        process_query_limited_information = 0x1000
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        open_process = kernel32.OpenProcess
        open_process.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        open_process.restype = wintypes.HANDLE
        get_process_times = kernel32.GetProcessTimes
        get_process_times.argtypes = [
            wintypes.HANDLE,
            ctypes.POINTER(wintypes.FILETIME),
            ctypes.POINTER(wintypes.FILETIME),
            ctypes.POINTER(wintypes.FILETIME),
            ctypes.POINTER(wintypes.FILETIME),
        ]
        get_process_times.restype = wintypes.BOOL
        close_handle = kernel32.CloseHandle
        close_handle.argtypes = [wintypes.HANDLE]
        close_handle.restype = wintypes.BOOL

        handle = open_process(
            process_query_limited_information,
            False,
            process_id,
        )
        if not handle:
            return None
        try:
            creation = wintypes.FILETIME()
            exit_time = wintypes.FILETIME()
            kernel_time = wintypes.FILETIME()
            user_time = wintypes.FILETIME()
            if not get_process_times(
                handle,
                ctypes.byref(creation),
                ctypes.byref(exit_time),
                ctypes.byref(kernel_time),
                ctypes.byref(user_time),
            ):
                return None
            value = (int(creation.dwHighDateTime) << 32) | int(
                creation.dwLowDateTime
            )
            return f"windows-{value:x}"
        finally:
            close_handle(handle)
    except (AttributeError, ImportError, OSError, TypeError, ValueError):
        return None


def read_process_start_marker(process_id: int) -> str | None:
    """Return an OS process-creation identity, or ``None`` when unavailable."""

    if isinstance(process_id, bool) or not isinstance(process_id, int):
        raise TypeError("process_id must be int")
    if process_id <= 0:
        return None
    if os.name == "nt":
        return _windows_process_start_marker(process_id)
    if os.name == "posix" and Path("/proc").is_dir():
        return _linux_process_start_marker(process_id)
    return None


_CURRENT_PROCESS_ID = os.getpid()
_CURRENT_PROCESS_START_MARKER = read_process_start_marker(_CURRENT_PROCESS_ID)
if _CURRENT_PROCESS_START_MARKER is None:
    # The fallback remains unique for this imported process even on platforms where
    # the OS does not expose a portable creation-time API. Other processes then fail
    # closed as active instead of risking deletion of their files.
    _CURRENT_PROCESS_START_MARKER = f"session-{uuid4().hex}"


def create_atomic_work_token_metadata() -> AtomicWorkTokenMetadata:
    return AtomicWorkTokenMetadata(
        process_id=_CURRENT_PROCESS_ID,
        process_start_marker=_CURRENT_PROCESS_START_MARKER,
        created_ns=time.time_ns(),
        nonce=uuid4().hex,
    )


def work_file_token(path: Path) -> str | None:
    if not isinstance(path, Path):
        raise TypeError("path must be pathlib.Path")
    name = path.name
    if STAGE_MARKER in name:
        return name.split(STAGE_MARKER, 1)[1].split(".", 1)[0]
    if BACKUP_MARKER in name:
        return name.split(BACKUP_MARKER, 1)[1]
    return None


def _legacy_process_id(token: str) -> int | None:
    process_text = token.split("-", 1)[0]
    try:
        process_id = int(process_text)
    except (TypeError, ValueError):
        return None
    return process_id if process_id > 0 else None


def _work_file_age_seconds(
    path: Path,
    metadata: AtomicWorkTokenMetadata | None,
    now_ns: int,
) -> float:
    if metadata is not None and metadata.created_ns <= now_ns:
        return max(
            0.0,
            (now_ns - metadata.created_ns) / 1_000_000_000.0,
        )
    try:
        modified_ns = path.stat().st_mtime_ns
    except OSError:
        return 0.0
    return max(0.0, (now_ns - modified_ns) / 1_000_000_000.0)


class _ProcessLocalAtomicRegistry:
    """Thread-safe ownership of active tokens and final paths in this process."""

    def __init__(self) -> None:
        self._lock = RLock()
        self._tokens: dict[str, AtomicWorkTokenMetadata] = {}
        self._final_path_owners: dict[Path, str] = {}

    def register(self, metadata: AtomicWorkTokenMetadata) -> None:
        if not isinstance(metadata, AtomicWorkTokenMetadata):
            raise TypeError("metadata must be AtomicWorkTokenMetadata")
        with self._lock:
            if metadata.token in self._tokens:
                raise RuntimeError("atomic work token is already active")
            self._tokens[metadata.token] = metadata

    def is_token_active(self, token: str) -> bool:
        with self._lock:
            return token in self._tokens

    def claim_final_path(self, final_path: Path, token: str) -> None:
        if not isinstance(final_path, Path):
            raise TypeError("final_path must be pathlib.Path")
        if (
            not final_path.is_absolute()
            or final_path != final_path.resolve(strict=False)
        ):
            raise ValueError("final_path must be absolute and normalized")
        with self._lock:
            if token not in self._tokens:
                raise RuntimeError("atomic work token is not active")
            owner = self._final_path_owners.get(final_path)
            if owner is not None and owner != token:
                raise RuntimeError(
                    "final output is already reserved by another active "
                    f"transaction: {final_path}"
                )
            self._final_path_owners[final_path] = token

    def unregister(self, token: str) -> None:
        with self._lock:
            self._tokens.pop(token, None)
            owned = tuple(
                path
                for path, owner in self._final_path_owners.items()
                if owner == token
            )
            for path in owned:
                self._final_path_owners.pop(path, None)


_REGISTRY = _ProcessLocalAtomicRegistry()


def register_atomic_transaction(metadata: AtomicWorkTokenMetadata) -> None:
    _REGISTRY.register(metadata)


def unregister_atomic_transaction(token: str) -> None:
    if not isinstance(token, str) or not token:
        raise ValueError("token must be a non-empty string")
    _REGISTRY.unregister(token)


def claim_atomic_final_path(final_path: Path, token: str) -> None:
    _REGISTRY.claim_final_path(final_path, token)


def assess_atomic_work_file(
    path: Path,
    *,
    minimum_stale_age_seconds: float = DEFAULT_STALE_WORK_FILE_AGE_SECONDS,
    now_ns: int | None = None,
) -> AtomicWorkFileAssessment:
    """Classify one work file without mutating it or trusting PID alone."""

    if not isinstance(path, Path):
        raise TypeError("path must be pathlib.Path")
    minimum_age = _validate_minimum_age(minimum_stale_age_seconds)
    resolved_now_ns = time.time_ns() if now_ns is None else now_ns
    if isinstance(resolved_now_ns, bool) or not isinstance(resolved_now_ns, int):
        raise TypeError("now_ns must be int or None")
    if resolved_now_ns <= 0:
        raise ValueError("now_ns must be positive")

    token = work_file_token(path)
    metadata = (
        AtomicWorkTokenMetadata.parse(token)
        if token is not None
        else None
    )
    age_seconds = _work_file_age_seconds(
        path,
        metadata,
        resolved_now_ns,
    )

    if token is not None and _REGISTRY.is_token_active(token):
        return AtomicWorkFileAssessment(
            path,
            token,
            AtomicWorkFileState.ACTIVE,
            AtomicRecoveryReason.PROCESS_LOCAL_ACTIVE_TOKEN,
            age_seconds,
            metadata,
        )

    stale_reason: AtomicRecoveryReason
    if metadata is not None:
        if (
            metadata.process_id == _CURRENT_PROCESS_ID
            and metadata.process_start_marker
            == _CURRENT_PROCESS_START_MARKER
        ):
            stale_reason = (
                AtomicRecoveryReason.UNREGISTERED_CURRENT_PROCESS_TOKEN
            )
        elif not _process_is_alive(metadata.process_id):
            stale_reason = AtomicRecoveryReason.OWNER_PROCESS_EXITED
        else:
            observed_marker = read_process_start_marker(metadata.process_id)
            if observed_marker is None:
                return AtomicWorkFileAssessment(
                    path,
                    token,
                    AtomicWorkFileState.ACTIVE,
                    AtomicRecoveryReason.OWNER_PROCESS_IDENTITY_UNAVAILABLE,
                    age_seconds,
                    metadata,
                )
            if observed_marker == metadata.process_start_marker:
                return AtomicWorkFileAssessment(
                    path,
                    token,
                    AtomicWorkFileState.ACTIVE,
                    AtomicRecoveryReason.OWNER_PROCESS_ACTIVE,
                    age_seconds,
                    metadata,
                )
            stale_reason = (
                AtomicRecoveryReason.OWNER_PROCESS_IDENTITY_MISMATCH
            )
    elif token is not None:
        legacy_pid = _legacy_process_id(token)
        if legacy_pid is not None:
            if _process_is_alive(legacy_pid):
                return AtomicWorkFileAssessment(
                    path,
                    token,
                    AtomicWorkFileState.ACTIVE,
                    AtomicRecoveryReason.LEGACY_OWNER_PROCESS_ACTIVE,
                    age_seconds,
                    None,
                )
            stale_reason = (
                AtomicRecoveryReason.LEGACY_OWNER_PROCESS_EXITED
            )
        else:
            stale_reason = AtomicRecoveryReason.MALFORMED_WORK_TOKEN
    else:
        stale_reason = AtomicRecoveryReason.MALFORMED_WORK_TOKEN

    if age_seconds < minimum_age:
        return AtomicWorkFileAssessment(
            path,
            token,
            AtomicWorkFileState.DEFERRED,
            AtomicRecoveryReason.MINIMUM_STALE_AGE_NOT_REACHED,
            age_seconds,
            metadata,
        )
    return AtomicWorkFileAssessment(
        path,
        token,
        AtomicWorkFileState.STALE,
        stale_reason,
        age_seconds,
        metadata,
    )


__all__ = [
    "AtomicRecoveryReason",
    "AtomicWorkFileAssessment",
    "AtomicWorkFileState",
    "AtomicWorkTokenMetadata",
    "BACKUP_MARKER",
    "DEFAULT_STALE_WORK_FILE_AGE_SECONDS",
    "STAGE_MARKER",
    "assess_atomic_work_file",
    "claim_atomic_final_path",
    "create_atomic_work_token_metadata",
    "read_process_start_marker",
    "register_atomic_transaction",
    "unregister_atomic_transaction",
    "work_file_token",
]
