"""Crash-durable filesystem primitives used by atomic output transactions.

The helpers intentionally keep policy out of low-level I/O.  They only guarantee
that bytes and directory-entry mutations have been handed to the operating system
for durable storage where the platform supports ``fsync`` on the relevant object.
"""

from __future__ import annotations

import errno
import json
import os
from pathlib import Path
from typing import Any, Mapping


class DurableIoError(RuntimeError):
    """Raised when a required durable filesystem operation cannot complete."""


def _normalise(path: Path, field_name: str = "path") -> Path:
    if not isinstance(path, Path):
        raise TypeError(f"{field_name} must be pathlib.Path")
    return path.expanduser().resolve(strict=False)


def fsync_file(path: Path) -> None:
    """Flush one existing regular file to durable storage.

    On Windows, ``os.fsync`` delegates to the CRT ``_commit`` function, which
    rejects descriptors opened read-only with ``EBADF``. Atomic outputs are
    owned by this process and are writable, so open an explicit read/write
    descriptor on every platform. This keeps the same code path observable to
    tests while avoiding a Windows-only false commit failure.
    """

    resolved = _normalise(path)
    if not resolved.is_file():
        raise DurableIoError(f"cannot fsync missing regular file: {resolved}")

    flags = os.O_RDWR
    if hasattr(os, "O_BINARY"):
        flags |= os.O_BINARY

    descriptor: int | None = None
    try:
        descriptor = os.open(resolved, flags)
        os.fsync(descriptor)
    except OSError as exc:
        raise DurableIoError(f"unable to fsync file '{resolved}': {exc}") from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)


def fsync_directory(directory: Path) -> None:
    """Flush directory-entry mutations where the host platform supports it.

    Windows does not expose portable directory ``fsync`` semantics through Python.
    The file flushes still provide useful durability there; unsupported directory
    handles are therefore treated as a documented platform limitation rather than
    a false transaction failure.
    """

    resolved = _normalise(directory, "directory")
    if not resolved.is_dir():
        raise DurableIoError(f"cannot fsync missing directory: {resolved}")
    flags = os.O_RDONLY
    if hasattr(os, "O_DIRECTORY"):
        flags |= os.O_DIRECTORY
    descriptor: int | None = None
    try:
        descriptor = os.open(resolved, flags)
        os.fsync(descriptor)
    except OSError as exc:
        unsupported = {
            errno.EACCES,
            errno.EBADF,
            errno.EINVAL,
            errno.ENOTSUP,
            getattr(errno, "EOPNOTSUPP", errno.ENOTSUP),
        }
        if os.name == "nt" and exc.errno in unsupported:
            return
        raise DurableIoError(
            f"unable to fsync directory '{resolved}': {exc}"
        ) from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)


def durable_replace(source: Path, destination: Path) -> None:
    """Replace one path and durably publish the destination directory entry."""

    resolved_source = _normalise(source, "source")
    resolved_destination = _normalise(destination, "destination")
    if resolved_source.parent != resolved_destination.parent:
        raise ValueError("durable_replace requires source and destination siblings")
    try:
        os.replace(resolved_source, resolved_destination)
        if resolved_destination.is_file():
            fsync_file(resolved_destination)
        fsync_directory(resolved_destination.parent)
    except (OSError, DurableIoError) as exc:
        raise DurableIoError(
            f"unable to durably replace '{resolved_source}' with "
            f"'{resolved_destination}': {exc}"
        ) from exc


def durable_unlink(path: Path, *, missing_ok: bool = True) -> bool:
    """Remove one filesystem entry and flush its parent directory."""

    resolved = _normalise(path)
    try:
        resolved.unlink(missing_ok=missing_ok)
        fsync_directory(resolved.parent)
        return True
    except FileNotFoundError:
        if missing_ok:
            return False
        raise
    except (OSError, DurableIoError) as exc:
        raise DurableIoError(f"unable to durably remove '{resolved}': {exc}") from exc


def write_json_durable(path: Path, payload: Mapping[str, Any]) -> None:
    """Atomically write deterministic UTF-8 JSON and fsync file plus directory."""

    resolved = _normalise(path)
    if not isinstance(payload, Mapping):
        raise TypeError("payload must be a mapping")
    resolved.parent.mkdir(parents=True, exist_ok=True)
    temporary = resolved.with_name(f".{resolved.name}.tmp-{os.getpid()}")
    encoded = (
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")
    try:
        descriptor = os.open(
            temporary,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o600,
        )
        try:
            with os.fdopen(descriptor, "wb", closefd=True) as stream:
                stream.write(encoded)
                stream.flush()
                os.fsync(stream.fileno())
        except BaseException:
            # fdopen owns the descriptor once created.
            raise
        durable_replace(temporary, resolved)
    except Exception:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass
        raise


__all__ = [
    "DurableIoError",
    "durable_replace",
    "durable_unlink",
    "fsync_directory",
    "fsync_file",
    "write_json_durable",
]
