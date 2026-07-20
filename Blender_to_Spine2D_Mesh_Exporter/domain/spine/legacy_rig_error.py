"""Errors raised while constructing the immutable legacy A1 rig."""


class LegacyRigBuildError(ValueError):
    """Raised when the A1 hierarchy cannot be constructed consistently."""


__all__ = ["LegacyRigBuildError"]
