"""Runtime guards for executing scene-aware bake plans without duplicate geometry."""

from __future__ import annotations

from contextlib import contextmanager
import logging
from typing import Any, Iterator

logger = logging.getLogger(__name__)


class SceneBakeExecutionError(RuntimeError):
    """Raised when scene-aware runtime state cannot be changed or restored."""


@contextmanager
def temporarily_exclude_source_from_render(
    source_obj: Any,
    *,
    enabled: bool,
    context: Any | None = None,
) -> Iterator[None]:
    """Hide the live source while its temporary bake target occupies the same transform.

    Scene and camera passes must see one copy of the source geometry, not the original and
    the temporary UV target on top of each other. The original object remains available as
    a datablock for Object Info and other references; only render visibility is changed.
    """

    if not isinstance(enabled, bool):
        raise TypeError("enabled must be bool")
    if not enabled:
        yield
        return
    if source_obj is None:
        raise SceneBakeExecutionError("source_obj cannot be None")

    try:
        previous = bool(getattr(source_obj, "hide_render", False))
    except Exception as exc:
        raise SceneBakeExecutionError("Unable to capture source hide_render") from exc

    primary_error: BaseException | None = None
    try:
        try:
            source_obj.hide_render = True
            update = getattr(getattr(context, "view_layer", None), "update", None)
            if callable(update):
                update()
        except Exception as exc:
            raise SceneBakeExecutionError(
                "Unable to exclude the source object from scene-aware rendering"
            ) from exc
        yield
    except BaseException as exc:
        primary_error = exc
        raise
    finally:
        try:
            source_obj.hide_render = previous
            update = getattr(getattr(context, "view_layer", None), "update", None)
            if callable(update):
                update()
        except Exception:
            if primary_error is None:
                raise SceneBakeExecutionError(
                    "Unable to restore source render visibility"
                )
            logger.exception(
                "Failed to restore source render visibility while handling another error"
            )
