"""Shared error contract for B4 camera projection execution."""


class CameraProjectionExecutionError(RuntimeError):
    """Raised when an active-camera projection cannot be staged safely."""


__all__ = ["CameraProjectionExecutionError"]
