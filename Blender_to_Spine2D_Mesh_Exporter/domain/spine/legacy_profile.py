"""A1 naming contract for the legacy rotatable-mesh rig."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True, slots=True)
class LegacyRigProfile:
    """Centralize every stable name required for v0.23 output compatibility."""

    spine_version: str = "4.2.43"
    profile_id: str = "LEGACY_ROTATABLE_MESH"
    root_name: str = "root"
    z_index_base: int = 1

    def __post_init__(self) -> None:
        if not isinstance(self.spine_version, str) or not self.spine_version.strip():
            raise ValueError("spine_version must be a non-empty string")
        if not isinstance(self.profile_id, str) or not self.profile_id.strip():
            raise ValueError("profile_id must be a non-empty string")
        if not isinstance(self.root_name, str) or not self.root_name.strip():
            raise ValueError("root_name must be a non-empty string")
        if (
            isinstance(self.z_index_base, bool)
            or not isinstance(self.z_index_base, int)
            or self.z_index_base < 0
        ):
            raise ValueError("z_index_base must be a non-negative integer")

    def _require_prefix(self, prefix: str) -> str:
        if not isinstance(prefix, str):
            raise TypeError("prefix must be str")
        normalized = prefix.strip()
        if not normalized:
            raise ValueError("prefix cannot be empty")
        return normalized

    def root_bone(self) -> str:
        return self.root_name

    def main_bone(self, prefix: str) -> str:
        return f"{self._require_prefix(prefix)}_main"

    def base_bone(self, prefix: str) -> str:
        return self._require_prefix(prefix)

    def scale_rotate_x_bone(self, prefix: str) -> str:
        return f"{self._require_prefix(prefix)}_scale_rotate_X"

    def rotate_x_bone(self, prefix: str) -> str:
        return f"{self._require_prefix(prefix)}_rotate_X"

    def control_x_bone(self, prefix: str) -> str:
        return f"{self._require_prefix(prefix)}_rotation_X"

    def control_y_bone(self, prefix: str) -> str:
        return f"{self._require_prefix(prefix)}_rotation_Y"

    def control_z_bone(self, prefix: str) -> str:
        return f"{self._require_prefix(prefix)}_rotation_Z"

    def control_bones(self, prefix: str) -> Tuple[str, str, str]:
        return (
            self.control_x_bone(prefix),
            self.control_y_bone(prefix),
            self.control_z_bone(prefix),
        )

    def rotate_x_constraint_bone(self, prefix: str) -> str:
        return f"{self._require_prefix(prefix)}_rotate_X_constraint"

    def rotate_x_constraint_scale_ik_bone(self, prefix: str) -> str:
        return f"{self._require_prefix(prefix)}_rotate_X_constraint_scale_IK"

    def rotate_x_constraint_rotate_ik_bone(self, prefix: str) -> str:
        return f"{self._require_prefix(prefix)}_rotate_X_constraint_rotate_IK"

    def rotate_x_constraint_ik_bone(self, prefix: str) -> str:
        return f"{self._require_prefix(prefix)}_rotate_X_constraint_IK"

    def ik_chain_bones(self, prefix: str) -> Tuple[str, str, str, str]:
        return (
            self.rotate_x_constraint_bone(prefix),
            self.rotate_x_constraint_scale_ik_bone(prefix),
            self.rotate_x_constraint_rotate_ik_bone(prefix),
            self.rotate_x_constraint_ik_bone(prefix),
        )

    def rotation_x_constraint(self, prefix: str) -> str:
        return self.control_x_bone(prefix)

    def rotation_y_constraint(self, prefix: str) -> str:
        return self.control_y_bone(prefix)

    def scale_ik_constraint(self, prefix: str) -> str:
        return f"{self._require_prefix(prefix)}_scale_constraint_IK"

    def scale_constraint(self, prefix: str) -> str:
        return f"{self._require_prefix(prefix)}_scale_constraint"

    def rotation_z_constraint(self, prefix: str) -> str:
        return self.control_z_bone(prefix)

    def scale_compensator_constraint(self, prefix: str) -> str:
        return f"{self._require_prefix(prefix)}_scale_compensator"

    def constraint_names(self, prefix: str) -> Tuple[str, ...]:
        """Return the historical semantic name order used by configuration code."""

        return (
            self.rotation_x_constraint(prefix),
            self.rotation_y_constraint(prefix),
            self.scale_ik_constraint(prefix),
            self.scale_constraint(prefix),
            self.rotation_z_constraint(prefix),
            self.scale_compensator_constraint(prefix),
        )

    def z_scale_bone(self, prefix: str, index: int) -> str:
        if (
            isinstance(index, bool)
            or not isinstance(index, int)
            or index < self.z_index_base
        ):
            raise ValueError(
                f"index must be an integer >= z_index_base ({self.z_index_base})"
            )
        return f"{self._require_prefix(prefix)}_{index}_scale"

    def z_bone(self, prefix: str, index: int) -> str:
        if (
            isinstance(index, bool)
            or not isinstance(index, int)
            or index < self.z_index_base
        ):
            raise ValueError(
                f"index must be an integer >= z_index_base ({self.z_index_base})"
            )
        return f"{self._require_prefix(prefix)}_{index}"

    def segment_slot(self, prefix: str, segment_index: int) -> str:
        if (
            isinstance(segment_index, bool)
            or not isinstance(segment_index, int)
            or segment_index < 0
        ):
            raise ValueError("segment_index must be a non-negative integer")
        return f"{self._require_prefix(prefix)}_Segment_{segment_index}"

    def vertex_bone(self, segment_name: str, vertex_index: int) -> str:
        name = self._require_prefix(segment_name)
        if (
            isinstance(vertex_index, bool)
            or not isinstance(vertex_index, int)
            or vertex_index < 0
        ):
            raise ValueError("vertex_index must be a non-negative integer")
        return f"{name}_vertex_{vertex_index}"
