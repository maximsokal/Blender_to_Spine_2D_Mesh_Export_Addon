"""Validate complete Active Camera Object Root setup chains on real coin geometry.

The full typed/serialized depth-scale, depth-rotation, inverse-setup, and weighted-vertex
assertions remain in ``coin_star_normal_object_root_setup_core``. This entrypoint only
supplies the supported temporary Normal/UV material for standalone real-asset execution.
"""

from __future__ import annotations

import traceback

from coin_star_normal_object_root_setup_core import _parse_arguments, _run
from coin_star_normal_test_support import safe_coin_normal_material
from run_coin_star_real_blend_shader_capability_integration import _require_source_object


def main() -> None:
    arguments = _parse_arguments()
    source = _require_source_object()
    try:
        with safe_coin_normal_material(source):
            _run(arguments.expected_blend)
    except Exception:
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
