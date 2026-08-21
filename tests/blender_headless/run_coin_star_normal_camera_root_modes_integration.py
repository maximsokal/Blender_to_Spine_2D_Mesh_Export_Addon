"""Validate both Normal/UV Active Camera rig roots on the real coin geometry.

The deep root-mode assertions live in ``coin_star_normal_camera_root_modes_core``. This
entrypoint owns only the real-asset material fixture: the artist ``Gold metal`` contains
true displacement and is intentionally unavailable to Normal/UV, so standalone geometry
acceptance runs under the shared deterministic surface-only material and restores the
artist material afterwards.
"""

from __future__ import annotations

from pathlib import Path
import sys
import traceback


SCRIPT_DIRECTORY = Path(__file__).resolve().parent
REPOSITORY_ROOT = SCRIPT_DIRECTORY.parents[1]
for path in (SCRIPT_DIRECTORY, REPOSITORY_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from coin_star_normal_camera_root_modes_core import (  # noqa: E402
    _parse_arguments,
    _run,
    _two_axis_settings,
)
from coin_star_normal_test_support import safe_coin_normal_material  # noqa: E402
from run_coin_star_real_blend_shader_capability_integration import (  # noqa: E402
    _require_source_object,
)


# _two_axis_settings is intentionally re-exported: the Object Root setup gate imports it
# from this runner as the canonical Active Camera real-coin settings builder.


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
