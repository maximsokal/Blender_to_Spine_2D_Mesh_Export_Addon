# utils.py
"""
This module provides a collection of specialized utility functions that support the main export pipeline, particularly in handling Z-depth information (Z-groups) and post-processing animation data (stretch values).

The key functionalities are:
1.  Closest Z-Value Search: The `find_closest_z` function implements an efficient binary search to find the nearest Z-group value for a given target Z-coordinate. This is crucial for correctly parenting vertices to the appropriate Z-layer bone in the Spine rig. It can handle a tolerance to ensure only reasonably close matches are accepted.
2.  Stretch Value Smoothing: The `smooth_stretch_values` function applies a median filter to a series of "stretch" values. This helps to smooth out any jitter or noise in the calculated stretch animation data, resulting in a more fluid and natural-looking animation in Spine.
3.  Stretch Value Limiting: The `limit_stretch_changes` function clamps the rate of change between consecutive stretch values. This prevents abrupt, jarring jumps in the animation by ensuring that the stretch factor does not change too drastically from one Z-layer to the next.
4.  Efficiency and Type Safety: The functions are designed to be efficient, using sorted lists and optimized search algorithms. They are also written with type hints and clear return types to improve code clarity and reliability.

ATTENTION: - The functions in this module are designed to operate on specific data structures produced by other parts of the addon. For example, `smooth_stretch_values` and `limit_stretch_changes` expect a dictionary mapping Z-coordinates to stretch factors. The `find_closest_z` function's performance relies on the input list being pre-sorted. Incorrect input data will lead to unexpected behavior in the final animation or rig structure.
Author: Maxim Sokolenko
"""

from __future__ import annotations

import logging
from bisect import bisect_left
from statistics import median
from typing import Mapping, Optional, Iterable, Dict, List

logger = logging.getLogger(__name__)


def _closest(sorted_values: List[float], target: float) -> float:
    """Return the value in ``sorted_values`` that is nearest to ``target``.

    If ``sorted_values`` is empty a ``ValueError`` will be raised.  The input
    list **must** be pre‑sorted in ascending order; no additional checks are
    performed for efficiency.

    :param sorted_values: ascending list of floats
    :param target: value to search for
    :returns: the element of ``sorted_values`` nearest to ``target``
    """
    pos = bisect_left(sorted_values, target)
    # Handle edge cases where the target lies outside the provided range
    if pos == 0:
        return sorted_values[0]
    if pos == len(sorted_values):
        return sorted_values[-1]
    before, after = sorted_values[pos - 1], sorted_values[pos]
    # Return the closer of the two neighbouring values
    return before if target - before <= after - target else after


def find_closest_z(
    target: float,
    original_z_groups: Optional[Iterable[float]] = None,
    local_z_groups: Optional[Iterable[float]] = None,
    obj_name: Optional[str] = None,
    tolerance: Optional[float] = None,
) -> Optional[float]:
    """Return the z value from the provided groups nearest to ``target``.

    The function selects the first non‑empty sequence from ``original_z_groups``
    or ``local_z_groups``.  The chosen sequence is converted to a list,
    sorted once and then searched using a binary search via :func:`bisect`.  If
    the closest value differs from ``target`` by more than ``tolerance`` the
    function returns ``None``.  Otherwise the nearest value is returned.

    :param target: the z value to match
    :param original_z_groups: optional sequence of original z group values
    :param local_z_groups: optional fallback sequence of z group values
    :param obj_name: optional name of the object being processed, used for
        debug logging only
    :param tolerance: optional maximum permitted absolute difference between
        the closest z and ``target``.  When ``None`` (the default), any
        distance is accepted and the nearest value is always returned.  If
        a non‑``None`` tolerance is provided and the nearest candidate
        exceeds this value the function returns ``None`` instead of the
        candidate.
    :returns: the nearest z value (or ``None`` if outside tolerance)
    :rtype: Optional[float]
    """
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug("[find_closest_z] start: target=%s, obj_name=%s", target, obj_name)
    # Choose which group list to use: prefer original groups when provided
    chosen_groups: Optional[Iterable[float]] = None
    if original_z_groups:
        chosen_groups = original_z_groups
    elif local_z_groups:
        chosen_groups = local_z_groups

    if not chosen_groups:
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "[find_closest_z] no z groups provided for object %s", obj_name
            )
        return None

    # Convert to list and sort once for efficient searching
    groups_list: List[float] = sorted(float(z) for z in chosen_groups)
    closest = _closest(groups_list, float(target))

    # If a tolerance is provided, enforce it; otherwise always return the closest
    if tolerance is not None and abs(closest - target) > tolerance:
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "[find_closest_z] nearest value %s exceeds tolerance %.6g for target %s",
                closest,
                tolerance,
                target,
            )
        return None
    return closest


def smooth_stretch_values(
    stretch_values: Mapping[float, float],
    window_size: int = 3,
    obj_name: Optional[str] = None,
) -> Dict[float, float]:
    """Apply a median filter to a mapping of z values to stretch values.

    A sliding window of size ``window_size`` (default 3) is centred on each
    z key.  The median of all stretch values in the window is computed and
    stored as the new value for that key.  When the window extends beyond
    the first or last key the existing values are used without wrapping.

    An empty input mapping is returned unchanged.  When only a single
    element is provided the result contains the same single key/value.

    :param stretch_values: mapping of z keys to stretch floats
    :param window_size: length of the sliding window (must be positive)
    :param obj_name: optional name of the object being processed
    :returns: a new ``dict`` with smoothed stretch values
    :rtype: dict[float, float]
    """
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(
            "[smooth_stretch_values] start: %d entries, window=%d, obj_name=%s",
            len(stretch_values),
            window_size,
            obj_name,
        )

    if not stretch_values:
        return {}
    # If window size is less than 2 the original values are returned
    if window_size <= 1 or len(stretch_values) <= 1:
        return dict(stretch_values)
    # Sort keys once up front
    sorted_z: List[float] = sorted(stretch_values)
    half_window = window_size // 2
    smoothed: Dict[float, float] = {}
    for idx, z in enumerate(sorted_z):
        # Determine window bounds within the sorted list
        start = max(0, idx - half_window)
        end = min(len(sorted_z), idx + half_window + 1)
        # Compute the median of the stretch values in the current window
        values = [stretch_values[z2] for z2 in sorted_z[start:end]]
        smoothed[z] = median(values)
    return smoothed


def limit_stretch_changes(
    stretch_values: Mapping[float, float],
    max_change: float = 0.1,
    obj_name: Optional[str] = None,
) -> Dict[float, float]:
    """Clamp the stepwise difference between successive stretch values.

    Given a mapping of z keys to stretch values this function iterates over
    the keys in ascending order and ensures that the difference between the
    current stretch and the previous processed stretch does not exceed
    ``max_change`` in either direction.  If the raw difference is larger
    then the new stretch value is set to the previous stretch plus or minus
    ``max_change`` accordingly.

    :param stretch_values: mapping of z keys to floats
    :param max_change: maximum allowed change between neighbouring values
    :param obj_name: optional name of the object being processed
    :returns: a new ``dict`` with the limited stretch values
    :rtype: dict[float, float]
    """
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(
            "[limit_stretch_changes] start: %d entries, max_change=%.6g, obj_name=%s",
            len(stretch_values),
            max_change,
            obj_name,
        )

    if not stretch_values:
        return {}
    # Sort the z keys once
    sorted_z: List[float] = sorted(stretch_values)
    limited: Dict[float, float] = {}
    # Initialise with the first value without modification
    first_z = sorted_z[0]
    prev_stretch = stretch_values[first_z]
    limited[first_z] = prev_stretch
    # Process remaining keys
    for z in sorted_z[1:]:
        current_raw = stretch_values[z]
        delta = current_raw - prev_stretch
        if abs(delta) > max_change:
            # Clamp the delta to max_change in the direction of the change
            current = prev_stretch + (max_change if delta > 0 else -max_change)
        else:
            current = current_raw
        limited[z] = current
        prev_stretch = current
    return limited


def register() -> None:
    """Optional registration hook for Blender addons.

    This function does nothing by default.  It exists to mirror the common
    pattern used by Blender add‑ons where modules expose a `register()`
    function.  Users of this module are not required to call it.
    """
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug("[utils.register] called")
    return None


def unregister() -> None:
    """Optional unregistration hook for Blender addons.

    This function does nothing by default.  It exists to mirror the common
    pattern used by Blender add‑ons where modules expose an `unregister()`
    function.  Users of this module are not required to call it.
    """
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug("[utils.unregister] called")
    return None
