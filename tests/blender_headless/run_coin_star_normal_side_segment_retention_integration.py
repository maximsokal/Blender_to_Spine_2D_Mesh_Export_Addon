"""Prove that every real coin Normal region reaches a Spine mesh attachment.

The complete edge-on/collapsed-triangle ownership assertions remain in
``coin_star_normal_side_segment_core``. Standalone execution uses the shared surface-only
Normal/UV material because the current artist material contains true displacement and is
correctly rejected by production Normal/UV routing.
"""

from __future__ import annotations

import traceback

from coin_star_normal_side_segment_core import _parse_arguments, _run
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
