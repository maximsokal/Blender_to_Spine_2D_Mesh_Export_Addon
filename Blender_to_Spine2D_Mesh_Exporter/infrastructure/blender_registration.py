"""Transactional helpers for Blender class and RNA registration lifecycles."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable, Tuple


CleanupCallback = Callable[[], None]


@dataclass(frozen=True, slots=True)
class RegistrationCleanupAction:
    """One named cleanup callback executed during rollback or add-on shutdown."""

    label: str
    callback: CleanupCallback

    def __post_init__(self) -> None:
        if not isinstance(self.label, str) or not self.label.strip():
            raise ValueError("label must be a non-empty string")
        if self.label != self.label.strip():
            raise ValueError("label must not contain boundary whitespace")
        if not callable(self.callback):
            raise TypeError("callback must be callable")


@dataclass(frozen=True, slots=True)
class RegistrationCleanupFailure:
    """One failed cleanup action retained for diagnostics."""

    label: str
    error: Exception

    def __post_init__(self) -> None:
        if not isinstance(self.label, str) or not self.label.strip():
            raise ValueError("label must be a non-empty string")
        if not isinstance(self.error, Exception):
            raise TypeError("error must be Exception")


class RegistrationCleanupError(RuntimeError):
    """Raised after all requested cleanup actions ran and at least one failed."""

    def __init__(
        self,
        *,
        operation: str,
        failures: Tuple[RegistrationCleanupFailure, ...],
        primary_error: Exception | None = None,
    ) -> None:
        if not isinstance(operation, str) or not operation.strip():
            raise ValueError("operation must be a non-empty string")
        if operation != operation.strip():
            raise ValueError("operation must not contain boundary whitespace")
        if not isinstance(failures, tuple) or not failures:
            raise ValueError("failures must be a non-empty tuple")
        if not all(isinstance(item, RegistrationCleanupFailure) for item in failures):
            raise TypeError("failures must contain RegistrationCleanupFailure values")
        if primary_error is not None and not isinstance(primary_error, Exception):
            raise TypeError("primary_error must be Exception or None")

        self.operation = operation
        self.failures = failures
        self.primary_error = primary_error
        details = "; ".join(
            f"{failure.label}: {type(failure.error).__name__}: {failure.error}"
            for failure in failures
        )
        primary = (
            ""
            if primary_error is None
            else f" after {type(primary_error).__name__}: {primary_error}"
        )
        super().__init__(
            f"{operation} completed with {len(failures)} cleanup failure(s){primary}: "
            f"{details}"
        )


@dataclass(frozen=True, slots=True)
class RnaPropertyRegistration:
    """One owned RNA property assignment on a Blender type."""

    owner: Any
    name: str
    value: Any

    def __post_init__(self) -> None:
        if self.owner is None:
            raise ValueError("owner cannot be None")
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("name must be a non-empty string")
        if self.name != self.name.strip() or not self.name.isidentifier():
            raise ValueError("name must be a canonical Python identifier")


def unregister_all_best_effort(
    actions: Iterable[RegistrationCleanupAction],
    *,
    operation: str,
    primary_error: Exception | None = None,
) -> None:
    """Run every cleanup action and raise one aggregate error afterwards."""

    if not isinstance(operation, str) or not operation.strip():
        raise ValueError("operation must be a non-empty string")
    if operation != operation.strip():
        raise ValueError("operation must not contain boundary whitespace")
    if primary_error is not None and not isinstance(primary_error, Exception):
        raise TypeError("primary_error must be Exception or None")

    resolved_actions = tuple(actions)
    if not all(isinstance(action, RegistrationCleanupAction) for action in resolved_actions):
        raise TypeError("actions must contain RegistrationCleanupAction values")

    failures: list[RegistrationCleanupFailure] = []
    for action in resolved_actions:
        try:
            action.callback()
        except Exception as exc:
            failures.append(RegistrationCleanupFailure(action.label, exc))

    if failures:
        raise RegistrationCleanupError(
            operation=operation,
            failures=tuple(failures),
            primary_error=primary_error,
        ) from primary_error


def class_cleanup_actions(
    classes: Iterable[type],
    *,
    unregister_class: Callable[[type], None],
) -> Tuple[RegistrationCleanupAction, ...]:
    """Build reverse-order class cleanup actions."""

    if not callable(unregister_class):
        raise TypeError("unregister_class must be callable")
    resolved = tuple(classes)
    if not all(isinstance(cls, type) for cls in resolved):
        raise TypeError("classes must contain types")
    return tuple(
        RegistrationCleanupAction(
            label=f"class {cls.__module__}.{cls.__qualname__}",
            callback=lambda cls=cls: unregister_class(cls),
        )
        for cls in reversed(resolved)
    )


def rna_property_cleanup_actions(
    registrations: Iterable[RnaPropertyRegistration],
) -> Tuple[RegistrationCleanupAction, ...]:
    """Build reverse-order idempotent RNA property cleanup actions."""

    resolved = tuple(registrations)
    if not all(isinstance(item, RnaPropertyRegistration) for item in resolved):
        raise TypeError("registrations must contain RnaPropertyRegistration values")

    def remove(registration: RnaPropertyRegistration) -> None:
        if hasattr(registration.owner, registration.name):
            delattr(registration.owner, registration.name)

    return tuple(
        RegistrationCleanupAction(
            label=(
                f"RNA property {getattr(item.owner, '__name__', type(item.owner).__name__)}"
                f".{item.name}"
            ),
            callback=lambda item=item: remove(item),
        )
        for item in reversed(resolved)
    )


def register_classes_transactionally(
    classes: Iterable[type],
    *,
    register_class: Callable[[type], None],
    unregister_class: Callable[[type], None],
) -> Tuple[type, ...]:
    """Register every class or roll back every class registered by this call."""

    if not callable(register_class):
        raise TypeError("register_class must be callable")
    if not callable(unregister_class):
        raise TypeError("unregister_class must be callable")
    resolved = tuple(classes)
    if not all(isinstance(cls, type) for cls in resolved):
        raise TypeError("classes must contain types")
    if len({id(cls) for cls in resolved}) != len(resolved):
        raise ValueError("classes must not contain duplicate class objects")

    registered: list[type] = []
    try:
        for cls in resolved:
            register_class(cls)
            registered.append(cls)
    except Exception as exc:
        unregister_all_best_effort(
            class_cleanup_actions(registered, unregister_class=unregister_class),
            operation="class registration rollback",
            primary_error=exc,
        )
        raise
    return tuple(registered)


def register_rna_properties_transactionally(
    registrations: Iterable[RnaPropertyRegistration],
) -> Tuple[RnaPropertyRegistration, ...]:
    """Assign every RNA property or remove every property assigned by this call."""

    resolved = tuple(registrations)
    if not all(isinstance(item, RnaPropertyRegistration) for item in resolved):
        raise TypeError("registrations must contain RnaPropertyRegistration values")
    identities = tuple((id(item.owner), item.name) for item in resolved)
    if len(set(identities)) != len(identities):
        raise ValueError("registrations must not contain duplicate owner/name pairs")

    registered: list[RnaPropertyRegistration] = []
    try:
        for item in resolved:
            if hasattr(item.owner, item.name):
                raise ValueError(
                    f"RNA property '{item.name}' is already present on "
                    f"{getattr(item.owner, '__name__', type(item.owner).__name__)}"
                )
            setattr(item.owner, item.name, item.value)
            registered.append(item)
    except Exception as exc:
        unregister_all_best_effort(
            rna_property_cleanup_actions(registered),
            operation="RNA property registration rollback",
            primary_error=exc,
        )
        raise
    return tuple(registered)


__all__ = [
    "RegistrationCleanupAction",
    "RegistrationCleanupError",
    "RegistrationCleanupFailure",
    "RnaPropertyRegistration",
    "class_cleanup_actions",
    "register_classes_transactionally",
    "register_rna_properties_transactionally",
    "rna_property_cleanup_actions",
    "unregister_all_best_effort",
]
