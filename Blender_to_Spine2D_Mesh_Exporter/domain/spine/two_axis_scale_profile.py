"""Naming profile for the X/Y rotation plus uniform scale rig."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

from .legacy_profile import LegacyRigProfile
from .rig_profiles import A1RigProfile


@dataclass(frozen=True, slots=True)
class TwoAxisScaleRigProfile(LegacyRigProfile):
    """Namespaced Spine 4.2.43 names generalized from the reference box rig."""

    profile_id: str = A1RigProfile.TWO_AXIS_ROTATION_SCALE.value

    def scale_control_bone(self, prefix: str) -> str:
        return f"{self._require_prefix(prefix)}_scale"

    def control_z_bone(self, prefix: str) -> str:
        """Reuse the third metadata slot for the scale control, not Z rotation."""

        return self.scale_control_bone(prefix)

    def control_bones(self, prefix: str) -> Tuple[str, str, str]:
        return (
            self.control_x_bone(prefix),
            self.control_y_bone(prefix),
            self.scale_control_bone(prefix),
        )

    def rotation_x_constraint(self, prefix: str) -> str:
        return f"{self._require_prefix(prefix)}_rotation_X_constraint"

    def rotation_y_constraint(self, prefix: str) -> str:
        return f"{self._require_prefix(prefix)}_rotation_Y"

    def scale_ik_constraint(self, prefix: str) -> str:
        return f"{self._require_prefix(prefix)}_IK"

    def scale_constraint(self, prefix: str) -> str:
        return self.scale_control_bone(prefix)

    def scale_depth_constraint(self, prefix: str) -> str:
        return f"{self._require_prefix(prefix)}_scale_rotate_X_constraint"

    def constraint_names(self, prefix: str) -> Tuple[str, ...]:
        return (
            self.rotation_x_constraint(prefix),
            self.scale_ik_constraint(prefix),
            self.scale_constraint(prefix),
            self.scale_depth_constraint(prefix),
            self.rotation_y_constraint(prefix),
        )


__all__ = ["TwoAxisScaleRigProfile"]
