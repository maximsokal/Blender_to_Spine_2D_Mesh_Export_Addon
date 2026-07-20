"""Errors raised while composing connected A1 object documents."""


class ConnectedGroupBuildError(ValueError):
    """Raised when A1 object documents cannot form one connected global rig."""


__all__ = ["ConnectedGroupBuildError"]
