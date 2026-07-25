"""Cross-process final-output leases backed by exclusive lock files."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import socket
import time
from typing import Iterator, Mapping
from uuid import uuid4

from .atomic_work_state import create_atomic_work_token_metadata
from .process_identity import process_identity_is_alive
from .durable_io import fsync_directory

LOCK_SUFFIX = ".spine2d.lock"
DEFAULT_MALFORMED_LOCK_STALE_AGE_SECONDS = 300.0

class InterprocessLockError(RuntimeError):
    """Raised when another process owns a requested output resource."""

@dataclass(frozen=True, slots=True)
class InterprocessLockMetadata:
    token: str
    process_id: int
    process_start_marker: str
    created_ns: int
    hostname: str
    resource_path: str

    def __post_init__(self) -> None:
        if not isinstance(self.token, str) or not self.token:
            raise ValueError("token must be a non-empty string")
        if isinstance(self.process_id, bool) or not isinstance(self.process_id, int):
            raise TypeError("process_id must be int")
        if self.process_id <= 0:
            raise ValueError("process_id must be positive")
        for field_name in ("process_start_marker", "hostname", "resource_path"):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value:
                raise ValueError(f"{field_name} must be a non-empty string")
        if isinstance(self.created_ns, bool) or not isinstance(self.created_ns, int):
            raise TypeError("created_ns must be int")
        if self.created_ns <= 0:
            raise ValueError("created_ns must be positive")

    def to_mapping(self) -> dict[str, object]:
        return {"schema_version": 1, "token": self.token, "process_id": self.process_id, "process_start_marker": self.process_start_marker, "created_ns": self.created_ns, "hostname": self.hostname, "resource_path": self.resource_path}

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> "InterprocessLockMetadata":
        if not isinstance(value, Mapping):
            raise TypeError("lock metadata must be a mapping")
        if value.get("schema_version") != 1:
            raise ValueError("unsupported lock metadata schema")
        return cls(token=str(value["token"]), process_id=int(value["process_id"]), process_start_marker=str(value["process_start_marker"]), created_ns=int(value["created_ns"]), hostname=str(value["hostname"]), resource_path=str(value["resource_path"]))

def lock_path_for_resource(resource_path: Path) -> Path:
    if not isinstance(resource_path, Path):
        raise TypeError("resource_path must be pathlib.Path")
    resolved = resource_path.expanduser().resolve(strict=False)
    return resolved.with_name(f".{resolved.name}{LOCK_SUFFIX}")

def _age_seconds(path: Path, now_ns: int | None = None) -> float:
    resolved_now = time.time_ns() if now_ns is None else now_ns
    try:
        modified_ns = path.stat().st_mtime_ns
    except OSError:
        return 0.0
    return max(0.0, (resolved_now - modified_ns) / 1_000_000_000.0)

def _read_metadata(path: Path) -> InterprocessLockMetadata:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("lock file root must be an object")
    return InterprocessLockMetadata.from_mapping(payload)

def _validate_stale_age(value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError("minimum_stale_age_seconds must be a finite number")
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise ValueError("minimum_stale_age_seconds must be finite and non-negative")
    return result

class InterprocessFileLock:
    """Exclusive lease for one normalized final output path."""
    def __init__(self, resource_path: Path, *, token: str | None = None, minimum_stale_age_seconds: float = DEFAULT_MALFORMED_LOCK_STALE_AGE_SECONDS) -> None:
        if not isinstance(resource_path, Path):
            raise TypeError("resource_path must be pathlib.Path")
        self.resource_path = resource_path.expanduser().resolve(strict=False)
        self.lock_path = lock_path_for_resource(self.resource_path)
        self.minimum_stale_age_seconds = _validate_stale_age(minimum_stale_age_seconds)
        work = create_atomic_work_token_metadata()
        self.metadata = InterprocessLockMetadata(token=token or f"lock-{uuid4().hex}", process_id=work.process_id, process_start_marker=work.process_start_marker, created_ns=time.time_ns(), hostname=socket.gethostname() or "unknown-host", resource_path=str(self.resource_path))
        self._acquired = False

    @property
    def acquired(self) -> bool:
        return self._acquired

    def _remove_stale_lock(self) -> bool:
        try:
            metadata = _read_metadata(self.lock_path)
        except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
            if _age_seconds(self.lock_path) < self.minimum_stale_age_seconds:
                return False
        else:
            if metadata.resource_path != str(self.resource_path):
                if _age_seconds(self.lock_path) < self.minimum_stale_age_seconds:
                    return False
            elif process_identity_is_alive(metadata.process_id, metadata.process_start_marker):
                return False
        try:
            self.lock_path.unlink()
            fsync_directory(self.lock_path.parent)
            return True
        except FileNotFoundError:
            return True
        except OSError:
            return False

    def acquire(self) -> "InterprocessFileLock":
        if self._acquired:
            return self
        self.lock_path.parent.mkdir(parents=True, exist_ok=True)
        payload = (json.dumps(self.metadata.to_mapping(), ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")
        for _attempt in range(2):
            descriptor: int | None = None
            try:
                descriptor = os.open(self.lock_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
                with os.fdopen(descriptor, "wb", closefd=True) as stream:
                    descriptor = None
                    stream.write(payload)
                    stream.flush()
                    os.fsync(stream.fileno())
                fsync_directory(self.lock_path.parent)
                self._acquired = True
                return self
            except FileExistsError:
                if not self._remove_stale_lock():
                    try:
                        owner = _read_metadata(self.lock_path)
                        owner_text = f"pid={owner.process_id}, host={owner.hostname}, token={owner.token}"
                    except Exception:
                        owner_text = "unreadable fresh lock"
                    raise InterprocessLockError(f"final output is already reserved by another active transaction/process ({owner_text}): {self.resource_path}")
            except Exception:
                if descriptor is not None:
                    os.close(descriptor)
                raise
        raise InterprocessLockError(f"unable to acquire lock: {self.resource_path}")

    def release(self) -> None:
        if not self._acquired:
            return
        try:
            current = _read_metadata(self.lock_path)
        except FileNotFoundError:
            self._acquired = False
            return
        except Exception as exc:
            raise InterprocessLockError(f"cannot verify lock ownership for '{self.resource_path}': {exc}") from exc
        if current.token != self.metadata.token:
            raise InterprocessLockError(f"refusing to release a lock owned by another token: {self.resource_path}")
        try:
            self.lock_path.unlink()
            fsync_directory(self.lock_path.parent)
        except OSError as exc:
            raise InterprocessLockError(f"unable to release lock '{self.lock_path}': {exc}") from exc
        self._acquired = False

    def __enter__(self) -> "InterprocessFileLock":
        return self.acquire()

    def __exit__(self, exc_type, exc, traceback) -> bool:
        self.release()
        return False

@contextmanager
def interprocess_file_lock(resource_path: Path, *, token: str | None = None, minimum_stale_age_seconds: float = DEFAULT_MALFORMED_LOCK_STALE_AGE_SECONDS) -> Iterator[InterprocessFileLock]:
    lock = InterprocessFileLock(resource_path, token=token, minimum_stale_age_seconds=minimum_stale_age_seconds)
    with lock:
        yield lock

__all__ = ["DEFAULT_MALFORMED_LOCK_STALE_AGE_SECONDS", "InterprocessFileLock", "InterprocessLockError", "InterprocessLockMetadata", "LOCK_SUFFIX", "interprocess_file_lock", "lock_path_for_resource"]
