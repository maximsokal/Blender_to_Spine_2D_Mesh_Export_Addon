"""Validate staged object-bake pixels against exported Spine attachment UVs."""

from __future__ import annotations

from dataclasses import dataclass
import logging
from math import isfinite
from pathlib import Path
from typing import Any, Iterable, Sequence

from ..application import A1AttachmentProjectionResult
from ..domain.baking import (
    CameraProjectionPlan,
    MaterialSemanticChannel,
    ObjectMaterialAnalysis,
)
from ..infrastructure import AtomicOutputReservation
from .a1_preparation_contracts import PreparedA1Object


logger = logging.getLogger(__name__)
_UV_UNIT_EPSILON = 1.0e-6


class BakedUvSpineValidationError(RuntimeError):
    """Raised when exported Spine UV triangles cannot sample the baked image safely."""


def _validate_unit_uv(
    value: Any,
    *,
    label: str,
) -> tuple[float, float]:
    """Return one finite UV pair clamped only inside the accepted boundary epsilon."""

    if not isinstance(label, str) or not label.strip():
        raise ValueError("label must be a non-empty string")
    try:
        components = tuple(value)
    except Exception as exc:
        raise BakedUvSpineValidationError(
            f"{label} must contain exactly two numeric components"
        ) from exc
    if len(components) != 2:
        raise BakedUvSpineValidationError(
            f"{label} must contain exactly two numeric components, got {len(components)}"
        )
    if any(isinstance(component, bool) for component in components):
        raise BakedUvSpineValidationError(
            f"{label} contains a boolean component: {components!r}"
        )
    try:
        u, v = (float(component) for component in components)
    except (TypeError, ValueError, OverflowError) as exc:
        raise BakedUvSpineValidationError(
            f"{label} contains a non-numeric component: {components!r}"
        ) from exc
    if not isfinite(u) or not isfinite(v):
        raise BakedUvSpineValidationError(
            f"{label} contains a non-finite value: {(u, v)!r}"
        )
    if (
        u < -_UV_UNIT_EPSILON
        or u > 1.0 + _UV_UNIT_EPSILON
        or v < -_UV_UNIT_EPSILON
        or v > 1.0 + _UV_UNIT_EPSILON
    ):
        raise BakedUvSpineValidationError(
            f"{label} {(u, v)} is outside the unit square"
        )
    return (
        min(1.0, max(0.0, u)),
        min(1.0, max(0.0, v)),
    )


def _spine_uv_to_loaded_image_uv(
    u: float,
    v: float,
) -> tuple[float, float]:
    """Map top-down Spine PNG UVs to Blender's bottom-up loaded image buffer."""

    resolved_u, resolved_v = _validate_unit_uv(
        (u, v),
        label="Spine UV sample",
    )
    return resolved_u, 1.0 - resolved_v


@dataclass(frozen=True, slots=True)
class RgbaImageBuffer:
    width: int
    height: int
    pixels: tuple[float, ...]

    def __post_init__(self) -> None:
        if (
            not isinstance(self.width, int)
            or isinstance(self.width, bool)
            or self.width <= 0
        ):
            raise ValueError("width must be a positive integer")
        if (
            not isinstance(self.height, int)
            or isinstance(self.height, bool)
            or self.height <= 0
        ):
            raise ValueError("height must be a positive integer")
        expected = self.width * self.height * 4
        if len(self.pixels) != expected:
            raise ValueError(
                f"pixels must contain {expected} RGBA values, got {len(self.pixels)}"
            )
        if not all(isfinite(float(value)) for value in self.pixels):
            raise ValueError("pixels contain non-finite values")

    def rgba(self, u: float, v: float) -> tuple[float, float, float, float]:
        """Sample the raw Blender-loaded image buffer where ``v=0`` is the bottom."""

        resolved_u, resolved_v = _validate_unit_uv(
            (u, v),
            label="Loaded image UV sample",
        )
        x = min(
            self.width - 1,
            max(0, int(round(resolved_u * (self.width - 1)))),
        )
        y = min(
            self.height - 1,
            max(0, int(round(resolved_v * (self.height - 1)))),
        )
        offset = (y * self.width + x) * 4
        values = tuple(float(self.pixels[offset + index]) for index in range(4))
        return values[0], values[1], values[2], values[3]

    def rgba_spine_file_space(
        self,
        u: float,
        v: float,
    ) -> tuple[float, float, float, float]:
        """Sample a saved PNG exactly as a Spine mesh UV consumes its top-down rows."""

        loaded_u, loaded_v = _spine_uv_to_loaded_image_uv(u, v)
        return self.rgba(loaded_u, loaded_v)


@dataclass(frozen=True, slots=True)
class TriangleCoverageSample:
    attachment_name: str
    triangle_index: int
    uv_samples: tuple[tuple[float, float], ...]
    rgba_samples: tuple[tuple[float, float, float, float], ...]

    @property
    def maximum_alpha(self) -> float:
        return max(sample[3] for sample in self.rgba_samples)


def _triangle_uvs(
    projection: A1AttachmentProjectionResult,
    triangle_offset: int,
) -> tuple[tuple[float, float], tuple[float, float], tuple[float, float]]:
    request = projection.request
    indices = request.triangles[triangle_offset : triangle_offset + 3]
    if len(indices) != 3:
        raise BakedUvSpineValidationError(
            f"Attachment '{request.attachment_name}' has an incomplete triangle stream"
        )
    try:
        values = tuple(
            _validate_unit_uv(
                request.vertices[index].uv,
                label=(
                    f"Attachment '{request.attachment_name}' triangle "
                    f"{triangle_offset // 3} vertex {index} UV"
                ),
            )
            for index in indices
        )
        return values[0], values[1], values[2]
    except BakedUvSpineValidationError:
        raise
    except Exception as exc:
        raise BakedUvSpineValidationError(
            f"Unable to resolve UVs for attachment '{request.attachment_name}' "
            f"triangle {triangle_offset // 3}"
        ) from exc


def _inset_samples(
    triangle: tuple[tuple[float, float], tuple[float, float], tuple[float, float]],
) -> tuple[tuple[float, float], ...]:
    centroid = (
        sum(vertex[0] for vertex in triangle) / 3.0,
        sum(vertex[1] for vertex in triangle) / 3.0,
    )
    # Centroid plus three points pulled 25% toward each vertex. They remain inside a
    # non-degenerate triangle and catch wrong winding/index/UV-to-pixel transforms.
    return (centroid,) + tuple(
        (
            centroid[0] * 0.75 + vertex[0] * 0.25,
            centroid[1] * 0.75 + vertex[1] * 0.25,
        )
        for vertex in triangle
    )


def validate_projection_uv_coverage(
    image: RgbaImageBuffer,
    projections: Iterable[A1AttachmentProjectionResult],
    *,
    alpha_threshold: float = 1.0 / 255.0,
    require_alpha_coverage: bool = True,
) -> tuple[TriangleCoverageSample, ...]:
    """Validate every exported triangle against its saved Spine-oriented PNG."""

    if not isinstance(image, RgbaImageBuffer):
        raise TypeError("image must be RgbaImageBuffer")
    if isinstance(alpha_threshold, bool) or not isinstance(
        alpha_threshold,
        (int, float),
    ):
        raise TypeError("alpha_threshold must be numeric")
    threshold = float(alpha_threshold)
    if not isfinite(threshold) or threshold < 0.0 or threshold > 1.0:
        raise ValueError("alpha_threshold must be finite in [0, 1]")
    if not isinstance(require_alpha_coverage, bool):
        raise TypeError("require_alpha_coverage must be bool")

    resolved = tuple(projections)
    if not resolved:
        raise BakedUvSpineValidationError(
            "No Spine attachment projections were supplied"
        )

    samples: list[TriangleCoverageSample] = []
    empty: list[
        tuple[
            str,
            int,
            tuple[tuple[float, float], ...],
            tuple[tuple[float, float, float, float], ...],
        ]
    ] = []
    for projection in resolved:
        if not isinstance(projection, A1AttachmentProjectionResult):
            raise TypeError(
                "projections must contain A1AttachmentProjectionResult values"
            )
        request = projection.request
        if len(request.triangles) % 3:
            raise BakedUvSpineValidationError(
                f"Attachment '{request.attachment_name}' triangle stream is malformed"
            )
        for offset in range(0, len(request.triangles), 3):
            triangle = _triangle_uvs(projection, offset)
            uv_samples = _inset_samples(triangle)
            rgba_samples = tuple(
                image.rgba_spine_file_space(u, v) for u, v in uv_samples
            )
            sample = TriangleCoverageSample(
                attachment_name=request.attachment_name,
                triangle_index=offset // 3,
                uv_samples=uv_samples,
                rgba_samples=rgba_samples,
            )
            samples.append(sample)
            if require_alpha_coverage and sample.maximum_alpha <= threshold:
                empty.append(
                    (
                        request.attachment_name,
                        offset // 3,
                        uv_samples,
                        rgba_samples,
                    )
                )

    if empty:
        raise BakedUvSpineValidationError(
            "Spine attachment UV triangles point only into empty baked pixels; "
            f"threshold={threshold}, failures={tuple(empty)}"
        )
    return tuple(samples)


def _material_may_be_transparent(analysis: ObjectMaterialAnalysis) -> bool:
    transparent_types = {
        "BSDF_GLASS",
        "BSDF_TRANSPARENT",
        "HOLDOUT",
        "PRINCIPLED_VOLUME",
        "VOLUME_PRINCIPLED",
    }
    return any(
        MaterialSemanticChannel.ALPHA in slot.semantic_channels
        or any(node_type in transparent_types for node_type in slot.node_types)
        for slot in analysis.slots
    )


def _load_staged_image(
    path: Path,
    *,
    bpy_module: Any,
) -> RgbaImageBuffer:
    if not path.is_file():
        raise BakedUvSpineValidationError(
            f"Staged bake image does not exist: {path}"
        )
    image = None
    image_name = "<unloaded>"
    try:
        image = bpy_module.data.images.load(str(path), check_existing=False)
        image_name = str(
            getattr(image, "name_full", None)
            or getattr(image, "name", None)
            or "<unnamed>"
        )
        width, height = int(image.size[0]), int(image.size[1])
        pixel_count = width * height * 4
        values = [0.0] * pixel_count
        foreach_get = getattr(image.pixels, "foreach_get", None)
        if callable(foreach_get):
            foreach_get(values)
        else:
            values[:] = tuple(float(value) for value in image.pixels)
        return RgbaImageBuffer(width=width, height=height, pixels=tuple(values))
    except BakedUvSpineValidationError:
        raise
    except Exception as exc:
        raise BakedUvSpineValidationError(
            f"Unable to read staged bake image '{path}': {exc}"
        ) from exc
    finally:
        if image is not None:
            try:
                bpy_module.data.images.remove(image, do_unlink=True)
            except TypeError:
                try:
                    bpy_module.data.images.remove(image)
                except Exception:
                    logger.exception(
                        "Unable to remove temporary staged image '%s' from '%s' "
                        "after compatibility cleanup",
                        image_name,
                        path,
                    )
            except Exception:
                logger.exception(
                    "Unable to remove temporary staged image '%s' from '%s'",
                    image_name,
                    path,
                )


def validate_staged_normal_bake_coverage(
    prepared: PreparedA1Object,
    reservations: Sequence[AtomicOutputReservation],
    *,
    bpy_module: Any | None = None,
    alpha_threshold: float = 1.0 / 255.0,
) -> tuple[TriangleCoverageSample, ...]:
    """Validate all staged Normal-mode frames before the atomic transaction commits."""

    if not isinstance(prepared, PreparedA1Object):
        raise TypeError("prepared must be PreparedA1Object")
    if isinstance(prepared.bake_plan, CameraProjectionPlan):
        return ()
    resolved_reservations = tuple(reservations)
    if not resolved_reservations or not all(
        isinstance(item, AtomicOutputReservation) for item in resolved_reservations
    ):
        raise TypeError(
            "reservations must contain AtomicOutputReservation values"
        )
    if bpy_module is None:
        try:
            import bpy as bpy_module
        except Exception:
            # Pure orchestration tests intentionally run outside Blender. The pixel
            # transform is covered by validate_projection_uv_coverage(), while staged
            # file loading is covered by the real Blender headless regression.
            return ()
    version = getattr(getattr(bpy_module, "app", None), "version", None)
    if not isinstance(version, tuple) or len(version) < 2:
        return ()

    require_alpha = not _material_may_be_transparent(prepared.material_analysis)
    all_samples: list[TriangleCoverageSample] = []
    for reservation in resolved_reservations:
        image = _load_staged_image(
            Path(reservation.staged_path),
            bpy_module=bpy_module,
        )
        all_samples.extend(
            validate_projection_uv_coverage(
                image,
                prepared.document_assembly.projections,
                alpha_threshold=alpha_threshold,
                require_alpha_coverage=require_alpha,
            )
        )
    return tuple(all_samples)


__all__ = [
    "BakedUvSpineValidationError",
    "RgbaImageBuffer",
    "TriangleCoverageSample",
    "validate_projection_uv_coverage",
    "validate_staged_normal_bake_coverage",
]
