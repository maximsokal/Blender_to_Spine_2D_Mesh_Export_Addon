#!/usr/bin/env python3
"""Compare Legacy and Rewrite image directories using Blender's image decoder."""

from __future__ import annotations

import argparse
from array import array
import hashlib
import json
from math import isfinite
from pathlib import Path
import sys
import traceback
from typing import Any, Mapping, Sequence

import bpy


_SUPPORTED_EXTENSIONS = {
    ".bmp",
    ".exr",
    ".hdr",
    ".jpeg",
    ".jpg",
    ".png",
    ".tga",
    ".tif",
    ".tiff",
    ".webp",
}


class ImageParityError(RuntimeError):
    """Raised when image comparison input is invalid or incompatible."""


def _arguments_after_separator(argv: Sequence[str]) -> list[str]:
    try:
        separator = argv.index("--")
    except ValueError:
        return []
    return list(argv[separator + 1 :])


def _non_negative_float(value: str) -> float:
    try:
        resolved = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Expected a number, got {value!r}") from exc
    if not isfinite(resolved) or resolved < 0.0:
        raise argparse.ArgumentTypeError("Value must be finite and non-negative")
    return resolved


def _ratio(value: str) -> float:
    resolved = _non_negative_float(value)
    if resolved > 1.0:
        raise argparse.ArgumentTypeError("Ratio must be in [0, 1]")
    return resolved


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-dir", type=Path, required=True)
    parser.add_argument("--actual-dir", type=Path, required=True)
    parser.add_argument("--report-json", type=Path, required=True)
    parser.add_argument(
        "--absolute-tolerance",
        type=_non_negative_float,
        default=1e-6,
    )
    parser.add_argument(
        "--max-differing-pixel-ratio",
        type=_ratio,
        default=0.0,
    )
    parser.add_argument(
        "--max-mean-absolute-delta",
        type=_non_negative_float,
        default=0.0,
    )
    return parser


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    resolved = path.expanduser().resolve(strict=False)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    temporary = resolved.with_name(f".{resolved.name}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(resolved)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while True:
            chunk = stream.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _image_files(root: Path) -> dict[str, Path]:
    resolved = root.expanduser().resolve(strict=False)
    if not resolved.is_dir():
        raise ImageParityError(f"Image directory does not exist: {resolved}")
    result: dict[str, Path] = {}
    for path in resolved.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in _SUPPORTED_EXTENSIONS:
            continue
        relative = path.relative_to(resolved).as_posix()
        if relative in result:
            raise ImageParityError(f"Duplicate relative image path: {relative}")
        result[relative] = path
    return result


def _load_pixels(path: Path) -> tuple[Any, array]:
    image = bpy.data.images.load(str(path), check_existing=False)
    try:
        image.update()
        values = array("f", [0.0]) * len(image.pixels)
        image.pixels.foreach_get(values)
        return image, values
    except Exception:
        bpy.data.images.remove(image)
        raise


def _image_metadata(image: Any, path: Path) -> dict[str, Any]:
    channels = int(image.channels)
    return {
        "width": int(image.size[0]),
        "height": int(image.size[1]),
        "channels": channels,
        "has_alpha": channels in {2, 4},
        "depth": int(image.depth),
        "is_float": bool(image.is_float),
        "alpha_mode": str(image.alpha_mode),
        "colorspace": str(image.colorspace_settings.name),
        "size_bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _compare_pair(
    relative_path: str,
    expected_path: Path,
    actual_path: Path,
    *,
    absolute_tolerance: float,
    max_differing_pixel_ratio: float,
    max_mean_absolute_delta: float,
) -> dict[str, Any]:
    expected_image = actual_image = None
    try:
        expected_image, expected_pixels = _load_pixels(expected_path)
        actual_image, actual_pixels = _load_pixels(actual_path)
        expected_metadata = _image_metadata(expected_image, expected_path)
        actual_metadata = _image_metadata(actual_image, actual_path)

        issues: list[dict[str, Any]] = []
        for field_name in ("width", "height", "channels", "has_alpha"):
            if expected_metadata[field_name] != actual_metadata[field_name]:
                issues.append(
                    {
                        "code": f"IMAGE_{field_name.upper()}_MISMATCH",
                        "message": f"{field_name} differs",
                        "expected": expected_metadata[field_name],
                        "actual": actual_metadata[field_name],
                    }
                )

        pixel_statistics: dict[str, Any] | None = None
        if not issues:
            channel_count = expected_metadata["channels"]
            pixel_count = expected_metadata["width"] * expected_metadata["height"]
            if len(expected_pixels) != len(actual_pixels):
                issues.append(
                    {
                        "code": "IMAGE_PIXEL_BUFFER_LENGTH_MISMATCH",
                        "message": "Decoded pixel buffer lengths differ",
                        "expected": len(expected_pixels),
                        "actual": len(actual_pixels),
                    }
                )
            elif len(expected_pixels) != pixel_count * channel_count:
                issues.append(
                    {
                        "code": "IMAGE_PIXEL_BUFFER_INVALID",
                        "message": "Decoded pixel length does not match dimensions",
                        "expected": pixel_count * channel_count,
                        "actual": len(expected_pixels),
                    }
                )
            else:
                maximum_delta = 0.0
                delta_sum = 0.0
                differing_pixels = 0
                for pixel_index in range(pixel_count):
                    start = pixel_index * channel_count
                    pixel_maximum = 0.0
                    for channel_offset in range(channel_count):
                        delta = abs(
                            float(expected_pixels[start + channel_offset])
                            - float(actual_pixels[start + channel_offset])
                        )
                        maximum_delta = max(maximum_delta, delta)
                        pixel_maximum = max(pixel_maximum, delta)
                        delta_sum += delta
                    if pixel_maximum > absolute_tolerance:
                        differing_pixels += 1
                differing_ratio = (
                    0.0 if pixel_count == 0 else differing_pixels / pixel_count
                )
                mean_delta = (
                    0.0 if not expected_pixels else delta_sum / len(expected_pixels)
                )
                pixel_statistics = {
                    "pixel_count": pixel_count,
                    "differing_pixel_count": differing_pixels,
                    "differing_pixel_ratio": differing_ratio,
                    "maximum_absolute_delta": maximum_delta,
                    "mean_absolute_delta": mean_delta,
                }
                if differing_ratio > max_differing_pixel_ratio:
                    issues.append(
                        {
                            "code": "IMAGE_DIFFERING_PIXEL_RATIO_EXCEEDED",
                            "message": "Differing pixel ratio exceeds the fixture policy",
                            "expected": max_differing_pixel_ratio,
                            "actual": differing_ratio,
                        }
                    )
                if mean_delta > max_mean_absolute_delta:
                    issues.append(
                        {
                            "code": "IMAGE_MEAN_DELTA_EXCEEDED",
                            "message": "Mean absolute channel delta exceeds policy",
                            "expected": max_mean_absolute_delta,
                            "actual": mean_delta,
                        }
                    )

        return {
            "relative_path": relative_path,
            "compatible": not issues,
            "byte_identical": expected_metadata["sha256"]
            == actual_metadata["sha256"],
            "expected": expected_metadata,
            "actual": actual_metadata,
            "pixel_statistics": pixel_statistics,
            "issues": issues,
        }
    finally:
        if actual_image is not None:
            bpy.data.images.remove(actual_image)
        if expected_image is not None:
            bpy.data.images.remove(expected_image)


def compare_image_directories(
    expected_directory: Path,
    actual_directory: Path,
    *,
    absolute_tolerance: float,
    max_differing_pixel_ratio: float,
    max_mean_absolute_delta: float,
) -> dict[str, Any]:
    expected = _image_files(expected_directory)
    actual = _image_files(actual_directory)
    expected_names = set(expected)
    actual_names = set(actual)
    missing = sorted(expected_names - actual_names)
    additional = sorted(actual_names - expected_names)
    shared = sorted(expected_names & actual_names)
    comparisons = [
        _compare_pair(
            relative,
            expected[relative],
            actual[relative],
            absolute_tolerance=absolute_tolerance,
            max_differing_pixel_ratio=max_differing_pixel_ratio,
            max_mean_absolute_delta=max_mean_absolute_delta,
        )
        for relative in shared
    ]
    compatible = not missing and not additional and all(
        comparison["compatible"] for comparison in comparisons
    )
    return {
        "compatible": compatible,
        "expected_directory": str(expected_directory.resolve(strict=False)),
        "actual_directory": str(actual_directory.resolve(strict=False)),
        "settings": {
            "absolute_tolerance": absolute_tolerance,
            "max_differing_pixel_ratio": max_differing_pixel_ratio,
            "max_mean_absolute_delta": max_mean_absolute_delta,
        },
        "expected_image_count": len(expected),
        "actual_image_count": len(actual),
        "missing_images": missing,
        "additional_images": additional,
        "comparisons": comparisons,
    }


def main() -> None:
    namespace = _build_parser().parse_args(_arguments_after_separator(sys.argv))
    report_path = namespace.report_json.expanduser().resolve(strict=False)
    try:
        report = compare_image_directories(
            namespace.expected_dir,
            namespace.actual_dir,
            absolute_tolerance=namespace.absolute_tolerance,
            max_differing_pixel_ratio=namespace.max_differing_pixel_ratio,
            max_mean_absolute_delta=namespace.max_mean_absolute_delta,
        )
        _write_json_atomic(report_path, report)
        if not report["compatible"]:
            raise ImageParityError("Image parity comparison found incompatibilities")
    except Exception as exc:
        if not report_path.exists():
            try:
                _write_json_atomic(
                    report_path,
                    {
                        "compatible": False,
                        "error_type": type(exc).__name__,
                        "error": str(exc) or type(exc).__name__,
                        "traceback": traceback.format_exc(),
                    },
                )
            except Exception:
                traceback.print_exc()
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
