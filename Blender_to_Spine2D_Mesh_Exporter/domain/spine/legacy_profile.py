"""A1 naming contract for the legacy rotatable-mesh rig."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True, slots=True)
class LegacyRigProfile:
    """Centralize every stable name required for v0.23 output compatibility."""

    spine_version: str = "4.2.43"
    profile_id: str = "LEGACY_ROTATABLE_MESH"

    def _require_prefix(self, prefix: str) -> str:
        if not isinstance(prefix, str):
            raise TypeError("prefix must be str")
        normalized = prefix.strip()
        if not normalized:
            raise ValueError("prefix cannot be empty")
        return normalized

    def main_bone(self, prefix: str) -> str:
        return f"{self._require_prefix(prefix)}_main"

    def base_bone(self, prefix: str) -> str:
        return self._require_prefix(prefix)

    def scale_rotate_x_bone(self, prefix: str) -> str:
        return f"{self._require_prefix(prefix)}_scale_rotate_X"

    def rotate_x_bone(self, prefix: str) -> str:
        return f"{self._require_prefix(prefix)}_rotate_X"

    def control_bones(self, prefix: str) -> Tuple[str, str, str]:
        value = self._require_prefix(prefix)
        return (
            f"{value}_rotation_X",
            f"{value}_rotation_Y",
            f"{value}_rotation_Z",
        )

    def ik_chain_bones(self, prefix: str) -> Tuple[str, str, str, str]:
        value = self._require_prefix(prefix)
        return (
            f"{value}_rotate_X_constraint",
            f"{value}_rotate_X_constraint_scale_IK",
            f"{value}_rotate_X_constraint_rotate_IK",
            f"{value}_rotate_X_constraint_IK",
        )

    def constraint_names(self, prefix: str) -> Tuple[str, ...]:
        value = self._require_prefix(prefix)
        return (
            f"{value}_rotation_X",
            f"{value}_rotation_Y",
            f"{value}_scale_constraint_IK",
            f"{value}_scale_constraint",
            f"{value}_rotation_Z",
            f"{value}_scale_compensator",
        )

    def z_scale_bone(self, prefix: str, index: int) -> str:
        if not isinstance(index, int) or index < 0:
            raise ValueError("index must be a non-negative integer")
        return f"{self._require_prefix(prefix)}_{index}_scale"

    def z_bone(self, prefix: str, index: int) -> str:
        if not isinstance(index, int) or index < 0:
            raise ValueError("index must be a non-negative integer")
        return f"{self._require_prefix(prefix)}_{index}"
