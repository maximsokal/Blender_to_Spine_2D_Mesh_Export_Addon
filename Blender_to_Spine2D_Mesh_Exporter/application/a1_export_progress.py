"""Blender-independent progress events for long-running A1 export operations.

Progress is advisory UI state. A failing observer must never change the export result,
filesystem transaction, Blender context restoration, or source-data ownership.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import logging
from math import isfinite
from typing import Callable


logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class A1ExportProgressUpdate:
    """One immutable export progress update in the inclusive range ``0..100``."""

    percent: int
    stage: str
    message: str
    object_id: str | None = None
    object_index: int | None = None
    object_count: int | None = None

    def __post_init__(self) -> None:
        if isinstance(self.percent, bool) or not isinstance(self.percent, int):
            raise TypeError("percent must be int")
        if self.percent < 0 or self.percent > 100:
            raise ValueError("percent must be in [0, 100]")
        for field_name in ("stage", "message"):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{field_name} must be a non-empty string")
            if value != value.strip():
                raise ValueError(f"{field_name} must not contain boundary whitespace")
        if self.object_id is not None:
            if not isinstance(self.object_id, str) or not self.object_id.strip():
                raise ValueError("object_id must be None or a non-empty string")
            if self.object_id != self.object_id.strip():
                raise ValueError("object_id must not contain boundary whitespace")
        if (self.object_index is None) != (self.object_count is None):
            raise ValueError("object_index and object_count must be provided together")
        if self.object_index is not None:
            if (
                isinstance(self.object_index, bool)
                or not isinstance(self.object_index, int)
                or isinstance(self.object_count, bool)
                or not isinstance(self.object_count, int)
            ):
                raise TypeError("object_index and object_count must be integers")
            if self.object_count < 1:
                raise ValueError("object_count must be positive")
            if self.object_index < 1 or self.object_index > self.object_count:
                raise ValueError("object_index must be in [1, object_count]")


A1ExportProgressCallback = Callable[[A1ExportProgressUpdate], None]


def _stage_name(stage: str | Enum) -> str:
    value = stage.value if isinstance(stage, Enum) else stage
    normalized = str(value).strip()
    if not normalized:
        raise ValueError("stage must resolve to a non-empty string")
    return normalized


def emit_a1_export_progress(
    callback: A1ExportProgressCallback | None,
    *,
    percent: int,
    stage: str | Enum,
    message: str,
    object_id: str | None = None,
    object_index: int | None = None,
    object_count: int | None = None,
) -> None:
    """Emit one validated update while isolating export correctness from UI failures."""

    if callback is None:
        return
    if not callable(callback):
        raise TypeError("callback must be callable or None")
    update = A1ExportProgressUpdate(
        percent=percent,
        stage=_stage_name(stage),
        message=str(message).strip(),
        object_id=object_id,
        object_index=object_index,
        object_count=object_count,
    )
    try:
        callback(update)
    except Exception:
        logger.exception(
            "A1 export progress observer failed at %d%% (%s)",
            update.percent,
            update.stage,
        )


def scale_a1_export_progress_callback(
    callback: A1ExportProgressCallback | None,
    *,
    start_percent: float,
    end_percent: float,
    object_id: str | None = None,
    object_index: int | None = None,
    object_count: int | None = None,
    message_prefix: str = "",
) -> A1ExportProgressCallback | None:
    """Map child progress ``0..100`` into one monotonic parent percentage range."""

    if callback is None:
        return None
    if not callable(callback):
        raise TypeError("callback must be callable or None")
    for field_name, value in (
        ("start_percent", start_percent),
        ("end_percent", end_percent),
    ):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"{field_name} must be numeric")
        if not isfinite(float(value)):
            raise ValueError(f"{field_name} must be finite")
    start = float(start_percent)
    end = float(end_percent)
    if start < 0.0 or end > 100.0 or end < start:
        raise ValueError("progress range must satisfy 0 <= start <= end <= 100")
    prefix = str(message_prefix)

    def scaled(update: A1ExportProgressUpdate) -> None:
        if not isinstance(update, A1ExportProgressUpdate):
            raise TypeError("child progress update must be A1ExportProgressUpdate")
        mapped = int(round(start + (end - start) * (update.percent / 100.0)))
        emit_a1_export_progress(
            callback,
            percent=max(0, min(100, mapped)),
            stage=update.stage,
            message=f"{prefix}{update.message}".strip(),
            object_id=update.object_id or object_id,
            object_index=(
                update.object_index
                if update.object_index is not None
                else object_index
            ),
            object_count=(
                update.object_count
                if update.object_count is not None
                else object_count
            ),
        )

    return scaled


__all__ = [
    "A1ExportProgressCallback",
    "A1ExportProgressUpdate",
    "emit_a1_export_progress",
    "scale_a1_export_progress_callback",
]
