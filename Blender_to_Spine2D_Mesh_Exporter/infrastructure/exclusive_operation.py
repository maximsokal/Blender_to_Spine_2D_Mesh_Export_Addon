"""Process-local exclusive-operation leases for user-triggered Rewrite actions.

Blender executes the public operators that use this registry on its main Python thread.
The registry therefore deliberately contains no Python thread primitives: persistent
``threading``/``queue`` based coordination is not supported by the extension runtime.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterator
from uuid import uuid4


A1_EXPORT_OPERATION_KEY = "a1-export"


class OperationAlreadyRunningError(RuntimeError):
    """Raised when a second caller attempts an already active exclusive operation."""

    def __init__(self, *, key: str, active_label: str, requested_label: str) -> None:
        self.key = key
        self.active_label = active_label
        self.requested_label = requested_label
        super().__init__(
            f"Operation '{requested_label}' cannot start because "
            f"'{active_label}' is already running"
        )


@dataclass(frozen=True, slots=True)
class ExclusiveOperationLease:
    """Identity proving ownership of one process-local exclusive operation."""

    key: str
    label: str
    token: str

    def __post_init__(self) -> None:
        for field_name in ("key", "label", "token"):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{field_name} must be a non-empty string")
            if value != value.strip():
                raise ValueError(f"{field_name} must not contain boundary whitespace")


class _ExclusiveOperationRegistry:
    """Main-thread ownership registry shared by Rewrite export entrypoints."""

    def __init__(self) -> None:
        self._active: dict[str, ExclusiveOperationLease] = {}

    @staticmethod
    def _canonical(value: str, field_name: str) -> str:
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{field_name} must be a non-empty string")
        resolved = value.strip()
        if resolved != value:
            raise ValueError(f"{field_name} must not contain boundary whitespace")
        return resolved

    def acquire(self, key: str, *, label: str) -> ExclusiveOperationLease:
        """Acquire ``key`` for the current Blender operator invocation."""

        resolved_key = self._canonical(key, "key")
        resolved_label = self._canonical(label, "label")
        active = self._active.get(resolved_key)
        if active is not None:
            raise OperationAlreadyRunningError(
                key=resolved_key,
                active_label=active.label,
                requested_label=resolved_label,
            )

        lease = ExclusiveOperationLease(
            key=resolved_key,
            label=resolved_label,
            token=uuid4().hex,
        )
        self._active[resolved_key] = lease
        return lease

    def release(self, lease: ExclusiveOperationLease) -> None:
        """Release a lease only when the caller still owns the active token."""

        if not isinstance(lease, ExclusiveOperationLease):
            raise TypeError("lease must be ExclusiveOperationLease")

        active = self._active.get(lease.key)
        if active is None:
            return
        if active.token != lease.token:
            raise RuntimeError(
                f"Operation lease for '{lease.key}' is no longer owned by this caller"
            )
        self._active.pop(lease.key, None)

    def active_leases(self) -> tuple[ExclusiveOperationLease, ...]:
        """Return a deterministic immutable snapshot for diagnostics."""

        return tuple(sorted(self._active.values(), key=lambda item: item.key))


_REGISTRY = _ExclusiveOperationRegistry()


@contextmanager
def exclusive_operation(
    key: str,
    *,
    label: str,
) -> Iterator[ExclusiveOperationLease]:
    """Acquire one operation key and always release it after success or failure."""

    lease = _REGISTRY.acquire(key, label=label)
    try:
        yield lease
    finally:
        _REGISTRY.release(lease)


def active_exclusive_operations() -> tuple[ExclusiveOperationLease, ...]:
    """Return an immutable diagnostic snapshot of active operation leases."""

    return _REGISTRY.active_leases()


__all__ = [
    "A1_EXPORT_OPERATION_KEY",
    "ExclusiveOperationLease",
    "OperationAlreadyRunningError",
    "active_exclusive_operations",
    "exclusive_operation",
]
