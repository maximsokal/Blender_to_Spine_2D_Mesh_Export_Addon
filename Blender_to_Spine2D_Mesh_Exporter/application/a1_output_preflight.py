"""Pure Windows output-namespace preflight for A1 rewrite exports."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Tuple

from ..domain.baking import predict_bake_output_paths, windows_path_identity
from .a1_numeric_contracts import require_identity, require_non_empty_string
from .a1_single_object import (
    A1SingleObjectExportSettings,
    build_a1_bake_settings,
)


class A1OutputPathKind(str, Enum):
    JSON = "JSON"
    TEXTURE = "TEXTURE"


@dataclass(frozen=True, slots=True)
class A1OutputPathClaim:
    path: Path
    owner: str
    kind: A1OutputPathKind

    def __post_init__(self) -> None:
        if not isinstance(self.path, Path):
            raise TypeError("path must be pathlib.Path")
        require_non_empty_string(self.owner, "owner")
        if not isinstance(self.kind, A1OutputPathKind):
            raise TypeError("kind must be A1OutputPathKind")


@dataclass(frozen=True, slots=True)
class A1OutputPreflightSource:
    owner: str
    object_name: str
    settings: A1SingleObjectExportSettings

    def __post_init__(self) -> None:
        require_identity(self.owner, "owner")
        require_non_empty_string(self.object_name, "object_name")
        if not isinstance(self.settings, A1SingleObjectExportSettings):
            raise TypeError("settings must be A1SingleObjectExportSettings")


@dataclass(frozen=True, slots=True)
class A1OutputNamespacePreflight:
    output_root: Path
    json_path: Path
    texture_paths: Tuple[Path, ...]

    def __post_init__(self) -> None:
        for field_name in ("output_root", "json_path"):
            if not isinstance(getattr(self, field_name), Path):
                raise TypeError(f"{field_name} must be pathlib.Path")
        if not isinstance(self.texture_paths, tuple) or not self.texture_paths:
            raise ValueError("texture_paths must be a non-empty tuple")
        if not all(isinstance(path, Path) for path in self.texture_paths):
            raise TypeError("texture_paths must contain pathlib.Path values")


def validate_a1_output_claims(
    output_root: Path,
    claims: Tuple[A1OutputPathClaim, ...],
) -> Tuple[Path, ...]:
    """Validate path ownership using Windows semantics on every host OS."""

    if not isinstance(output_root, Path):
        raise TypeError("output_root must be pathlib.Path")
    if not isinstance(claims, tuple) or not claims:
        raise ValueError("claims must be a non-empty tuple")
    if not all(isinstance(claim, A1OutputPathClaim) for claim in claims):
        raise TypeError("claims must contain A1OutputPathClaim values")

    root = output_root.expanduser().resolve(strict=False)
    owner_by_identity: dict[Tuple[str, ...], A1OutputPathClaim] = {}
    normalized_paths: list[Path] = []

    for claim in claims:
        resolved = claim.path.expanduser().resolve(strict=False)
        try:
            resolved.relative_to(root)
        except ValueError as exc:
            raise ValueError(
                f"{claim.kind.value} output for '{claim.owner}' escapes output root: "
                f"{resolved}"
            ) from exc

        identity = windows_path_identity(resolved)
        previous = owner_by_identity.get(identity)
        if previous is not None:
            previous_path = previous.path.expanduser().resolve(strict=False)
            raise ValueError(
                "Windows output path collision between "
                f"{previous.kind.value} '{previous.owner}' ({previous_path}) and "
                f"{claim.kind.value} '{claim.owner}' ({resolved})"
            )
        owner_by_identity[identity] = claim
        normalized_paths.append(resolved)

    return tuple(normalized_paths)


def validate_a1_realized_output_namespace(
    *,
    output_root: Path,
    json_path: Path,
    texture_paths: Tuple[Path, ...],
    additional_texture_paths: Tuple[Path, ...] = (),
) -> Tuple[Path, ...]:
    """Validate prepared and conditionally derived outputs before any reservation."""

    if not isinstance(json_path, Path):
        raise TypeError("json_path must be pathlib.Path")
    for field_name, paths in (
        ("texture_paths", texture_paths),
        ("additional_texture_paths", additional_texture_paths),
    ):
        if not isinstance(paths, tuple):
            raise TypeError(f"{field_name} must be tuple")
        if not all(isinstance(path, Path) for path in paths):
            raise TypeError(f"{field_name} must contain pathlib.Path values")
    if not texture_paths:
        raise ValueError("texture_paths must be a non-empty tuple")

    claims: list[A1OutputPathClaim] = [
        A1OutputPathClaim(
            path=json_path,
            owner="final document",
            kind=A1OutputPathKind.JSON,
        )
    ]
    claims.extend(
        A1OutputPathClaim(
            path=path,
            owner=f"prepared texture[{index}]",
            kind=A1OutputPathKind.TEXTURE,
        )
        for index, path in enumerate(texture_paths)
    )
    claims.extend(
        A1OutputPathClaim(
            path=path,
            owner=f"additional texture[{index}]",
            kind=A1OutputPathKind.TEXTURE,
        )
        for index, path in enumerate(additional_texture_paths)
    )
    return validate_a1_output_claims(output_root, tuple(claims))


def preflight_a1_output_namespace(
    *,
    output_root: Path,
    json_path: Path,
    sources: Tuple[A1OutputPreflightSource, ...],
) -> A1OutputNamespacePreflight:
    """Predict and validate all A1 outputs before geometry or bake preparation."""

    if not isinstance(json_path, Path):
        raise TypeError("json_path must be pathlib.Path")
    if not isinstance(sources, tuple) or not sources:
        raise ValueError("sources must be a non-empty tuple")
    if not all(isinstance(source, A1OutputPreflightSource) for source in sources):
        raise TypeError("sources must contain A1OutputPreflightSource values")

    claims: list[A1OutputPathClaim] = [
        A1OutputPathClaim(
            path=json_path,
            owner="final document",
            kind=A1OutputPathKind.JSON,
        )
    ]
    for source in sources:
        bake_settings = build_a1_bake_settings(source.object_name, source.settings)
        for path in predict_bake_output_paths(bake_settings):
            claims.append(
                A1OutputPathClaim(
                    path=path,
                    owner=source.owner,
                    kind=A1OutputPathKind.TEXTURE,
                )
            )

    normalized = validate_a1_output_claims(output_root, tuple(claims))
    return A1OutputNamespacePreflight(
        output_root=output_root.expanduser().resolve(strict=False),
        json_path=normalized[0],
        texture_paths=normalized[1:],
    )


__all__ = [
    "A1OutputNamespacePreflight",
    "A1OutputPathClaim",
    "A1OutputPathKind",
    "A1OutputPreflightSource",
    "preflight_a1_output_namespace",
    "validate_a1_output_claims",
    "validate_a1_realized_output_namespace",
]
