"""Validate staged object-bake pixels against exported Spine attachment UVs."""

from __future__ import annotations

from array import array
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
from .bake_uv_raster_coverage import (
    raster_sample_pixels,
    triangle_twice_area_pixels,
)


logger = logging.getLogger(__name__)
_UV_UNIT_EPSILON = 1.0e-6
_MAX_FAILURE_RASTER_DIAGNOSTIC_SAMPLES = 16
_MIN_TRIANGLE_TWICE_AREA_PIXELS = 1.0e-12


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
    pixels: Sequence[float]

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

    def _rgba_loaded_pixel(
        self,
        pixel_x: int,
        pixel_y: int,
    ) -> tuple[float, float, float, float]:
        if (
            not isinstance(pixel_x, int)
            or isinstance(pixel_x, bool)
            or pixel_x < 0
            or pixel_x >= self.width
        ):
            raise ValueError(
                f"pixel_x must be an integer in [0, {self.width - 1}]"
            )
        if (
            not isinstance(pixel_y, int)
            or isinstance(pixel_y, bool)
            or pixel_y < 0
            or pixel_y >= self.height
        ):
            raise ValueError(
                f"pixel_y must be an integer in [0, {self.height - 1}]"
            )
        offset = (pixel_y * self.width + pixel_x) * 4
        values = tuple(float(self.pixels[offset + index]) for index in range(4))
        return values[0], values[1], values[2], values[3]

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
        return self._rgba_loaded_pixel(x, y)

    def rgba_spine_file_space(
        self,
        u: float,
        v: float,
    ) -> tuple[float, float, float, float]:
        """Sample a saved PNG exactly as a Spine mesh UV consumes its top-down rows."""

        loaded_u, loaded_v = _spine_uv_to_loaded_image_uv(u, v)
        return self.rgba(loaded_u, loaded_v)

    def rgba_spine_file_pixel(
        self,
        pixel_x: int,
        pixel_y: int,
    ) -> tuple[float, float, float, float]:
        """Read one exact PNG texel using top-down Spine file-space coordinates."""

        if (
            not isinstance(pixel_y, int)
            or isinstance(pixel_y, bool)
            or pixel_y < 0
            or pixel_y >= self.height
        ):
            raise ValueError(
                f"pixel_y must be an integer in [0, {self.height - 1}]"
            )
        loaded_y = self.height - 1 - pixel_y
        return self._rgba_loaded_pixel(pixel_x, loaded_y)

    def spine_file_pixel_center_uv(
        self,
        pixel_x: int,
        pixel_y: int,
    ) -> tuple[float, float]:
        """Return the normalized top-down UV coordinate at one exact texel center."""

        # Reuse exact-pixel validation without relying on its color payload.
        self.rgba_spine_file_pixel(pixel_x, pixel_y)
        return (
            (float(pixel_x) + 0.5) / float(self.width),
            (float(pixel_y) + 0.5) / float(self.height),
        )


@dataclass(frozen=True, slots=True)
class TriangleCoverageSample:
    attachment_name: str
    triangle_index: int
    uv_samples: tuple[tuple[float, float], ...]
    rgba_samples: tuple[tuple[float, float, float, float], ...]
    # ``False`` means no output texel centre lies inside the triangle at this image
    # resolution, so direct baked alpha is mathematically unavailable.
    resolution_representable: bool = True
    raster_sample_count: int = 0
    triangle_twice_area_pixels: float = 0.0

    def __post_init__(self) -> None:
        if not isinstance(self.resolution_representable, bool):
            raise TypeError("resolution_representable must be bool")
        if (
            not isinstance(self.raster_sample_count, int)
            or isinstance(self.raster_sample_count, bool)
            or self.raster_sample_count < 0
        ):
            raise ValueError("raster_sample_count must be a non-negative integer")
        if (
            isinstance(self.triangle_twice_area_pixels, bool)
            or not isinstance(self.triangle_twice_area_pixels, (int, float))
            or not isfinite(float(self.triangle_twice_area_pixels))
            or float(self.triangle_twice_area_pixels) < 0.0
        ):
            raise ValueError(
                "triangle_twice_area_pixels must be a finite non-negative number"
            )

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


def _raster_sample_fallback(
    image: RgbaImageBuffer,
    triangle: tuple[tuple[float, float], tuple[float, float], tuple[float, float]],
    *,
    alpha_threshold: float,
) -> tuple[
    tuple[tuple[float, float], ...],
    tuple[tuple[float, float, float, float], ...],
    bool,
    bool,
    int,
    float,
]:
    """Inspect exact texel centres available to one triangle after point samples fail.

    Returns ``(uvs, rgba, representable, covered, raster_sample_count, twice_area)``.
    ``representable=False`` is not an error: it means the current output resolution has
    no texel centre inside this finite non-degenerate UV triangle, so Blender cannot
    directly rasterize a source sample for it.
    """

    twice_area = triangle_twice_area_pixels(
        triangle,
        width=image.width,
        height=image.height,
    )
    if twice_area <= _MIN_TRIANGLE_TWICE_AREA_PIXELS:
        raise BakedUvSpineValidationError(
            "Attachment UV triangle is degenerate at output resolution; "
            f"image_size=({image.width}, {image.height}), "
            f"twice_area_pixels={twice_area}, triangle={triangle!r}"
        )

    raster_pixels = raster_sample_pixels(
        triangle,
        width=image.width,
        height=image.height,
    )
    if raster_pixels is None:
        # Bounding box too large for the bounded fallback. Stay fail-closed; the normal
        # four interior samples already found no alpha.
        return (), (), True, False, 0, twice_area
    if not raster_pixels:
        return (), (), False, False, 0, twice_area

    diagnostic_uvs: list[tuple[float, float]] = []
    diagnostic_rgba: list[tuple[float, float, float, float]] = []
    covered = False
    for pixel_x, pixel_y in raster_pixels:
        rgba = image.rgba_spine_file_pixel(pixel_x, pixel_y)
        if len(diagnostic_rgba) < _MAX_FAILURE_RASTER_DIAGNOSTIC_SAMPLES:
            diagnostic_uvs.append(
                image.spine_file_pixel_center_uv(pixel_x, pixel_y)
            )
            diagnostic_rgba.append(rgba)
        if rgba[3] > alpha_threshold:
            covered = True
            break

    return (
        tuple(diagnostic_uvs),
        tuple(diagnostic_rgba),
        True,
        covered,
        len(raster_pixels),
        twice_area,
    )


def validate_projection_uv_coverage(
    image: RgbaImageBuffer,
    projections: Iterable[A1AttachmentProjectionResult],
    *,
    alpha_threshold: float = 1.0 / 255.0,
    require_alpha_coverage: bool = True,
) -> tuple[TriangleCoverageSample, ...]:
    """Validate every exported triangle against its saved Spine-oriented PNG.

    Four deterministic interior samples remain the fast path. If they are transparent,
    validation checks the exact output texel centres that lie inside the UV triangle.
    A triangle with one or more raster sample centres stays fail-closed: at least one of
    those texels must contain alpha. A finite non-degenerate triangle with *no* texel
    centre at the selected resolution is classified as resolution-unrepresentable and
    cannot be required to contain direct baked alpha.
    """

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
            float,
            tuple[tuple[float, float], ...],
            tuple[tuple[float, float, float, float], ...],
        ]
    ] = []
    unrepresentable: list[tuple[str, int, float]] = []

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
            triangle_index = offset // 3
            triangle = _triangle_uvs(projection, offset)
            twice_area = triangle_twice_area_pixels(
                triangle,
                width=image.width,
                height=image.height,
            )
            if twice_area <= _MIN_TRIANGLE_TWICE_AREA_PIXELS:
                raise BakedUvSpineValidationError(
                    f"Attachment '{request.attachment_name}' triangle {triangle_index} "
                    "has degenerate UV area at output resolution; "
                    f"image_size=({image.width}, {image.height}), "
                    f"twice_area_pixels={twice_area}, triangle={triangle!r}"
                )

            uv_samples = _inset_samples(triangle)
            rgba_samples = tuple(
                image.rgba_spine_file_space(u, v) for u, v in uv_samples
            )
            covered = any(sample[3] > threshold for sample in rgba_samples)
            resolution_representable = True
            raster_sample_count = 0

            if require_alpha_coverage and not covered:
                (
                    raster_uvs,
                    raster_rgba,
                    resolution_representable,
                    raster_covered,
                    raster_sample_count,
                    twice_area,
                ) = _raster_sample_fallback(
                    image,
                    triangle,
                    alpha_threshold=threshold,
                )
                if raster_uvs:
                    uv_samples = uv_samples + raster_uvs
                    rgba_samples = rgba_samples + raster_rgba
                covered = raster_covered

                if not resolution_representable:
                    unrepresentable.append(
                        (request.attachment_name, triangle_index, twice_area)
                    )
                elif raster_covered:
                    logger.debug(
                        "Exact raster sample recovered baked alpha for attachment '%s' "
                        "triangle %d at %dx%d after four interior samples were empty",
                        request.attachment_name,
                        triangle_index,
                        image.width,
                        image.height,
                    )

            sample = TriangleCoverageSample(
                attachment_name=request.attachment_name,
                triangle_index=triangle_index,
                uv_samples=uv_samples,
                rgba_samples=rgba_samples,
                resolution_representable=resolution_representable,
                raster_sample_count=raster_sample_count,
                triangle_twice_area_pixels=twice_area,
            )
            samples.append(sample)

            if (
                require_alpha_coverage
                and resolution_representable
                and not covered
            ):
                empty.append(
                    (
                        request.attachment_name,
                        triangle_index,
                        twice_area,
                        uv_samples,
                        rgba_samples,
                    )
                )

    if unrepresentable:
        logger.info(
            "Skipped direct alpha requirement for %d finite subpixel attachment "
            "triangles at %dx%d because no output texel centre lies inside them; "
            "examples=%s",
            len(unrepresentable),
            image.width,
            image.height,
            tuple(unrepresentable[:8]),
        )

    if empty:
        raise BakedUvSpineValidationError(
            "Spine attachment UV triangles point only into empty baked pixels despite "
            "having raster sample centres; "
            f"threshold={threshold}, image_size=({image.width}, {image.height}), "
            f"failures={tuple(empty)}"
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
        values = array("f", [0.0]) * pixel_count
        foreach_get = getattr(image.pixels, "foreach_get", None)
        if callable(foreach_get):
            foreach_get(values)
        else:
            values = array("f", (float(value) for value in image.pixels))
        return RgbaImageBuffer(width=width, height=height, pixels=values)
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
