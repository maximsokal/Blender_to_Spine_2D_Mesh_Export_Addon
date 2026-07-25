"""Exact process-incarnation checks shared by cross-process resource owners."""

from __future__ import annotations

from .atomic_work_state import _process_is_alive, read_process_start_marker


def process_identity_is_alive(
    process_id: int,
    process_start_marker: str,
) -> bool:
    """Return whether a PID still identifies the exact process incarnation.

    Comparing only the PID is unsafe because operating systems reuse process IDs.
    When a live process marker cannot be inspected, the function fails closed and
    treats the process as active instead of stealing its filesystem resources.
    """

    if isinstance(process_id, bool) or not isinstance(process_id, int):
        raise TypeError("process_id must be int")
    if not isinstance(process_start_marker, str) or not process_start_marker:
        raise ValueError("process_start_marker must be a non-empty string")
    if process_id <= 0 or not _process_is_alive(process_id):
        return False
    observed = read_process_start_marker(process_id)
    if observed is None:
        return True
    return observed == process_start_marker


__all__ = ["process_identity_is_alive"]
