#!/usr/bin/env python3
"""Measure repeated production-export RSS inside one full Blender process."""

from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
import sys
import time
import traceback
from typing import Any, Mapping, Sequence

import bpy


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

import Blender_to_Spine2D_Mesh_Exporter as addon  # noqa: E402


class MemoryStressError(RuntimeError):
    """Raised when memory stress cannot execute a production export safely."""


def _arguments(argv: Sequence[str]) -> list[str]:
    try:
        return list(argv[argv.index("--") + 1 :])
    except ValueError:
        return list(argv[1:])


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--report-json", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--sample-every", type=int, default=5)
    return parser


def _rss_bytes() -> int:
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes

        class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
            _fields_ = [
                ("cb", wintypes.DWORD),
                ("PageFaultCount", wintypes.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            ]

        counters = PROCESS_MEMORY_COUNTERS()
        counters.cb = ctypes.sizeof(counters)
        psapi = ctypes.WinDLL("psapi", use_last_error=True)
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)

        get_current_process = kernel32.GetCurrentProcess
        get_current_process.argtypes = ()
        get_current_process.restype = wintypes.HANDLE

        get_process_memory_info = psapi.GetProcessMemoryInfo
        get_process_memory_info.argtypes = (
            wintypes.HANDLE,
            ctypes.POINTER(PROCESS_MEMORY_COUNTERS),
            wintypes.DWORD,
        )
        get_process_memory_info.restype = wintypes.BOOL

        process_handle = get_current_process()
        ctypes.set_last_error(0)
        if not get_process_memory_info(
            process_handle,
            ctypes.byref(counters),
            wintypes.DWORD(counters.cb),
        ):
            error_code = ctypes.get_last_error()
            error_text = (
                ctypes.FormatError(error_code).strip()
                if error_code
                else "unknown Win32 error"
            )
            raise MemoryStressError(
                "GetProcessMemoryInfo failed: "
                f"WinError {error_code} ({error_text})"
            )
        return int(counters.WorkingSetSize)

    statm = Path("/proc/self/statm")
    if statm.is_file():
        resident_pages = int(statm.read_text(encoding="ascii").split()[1])
        return resident_pages * int(os.sysconf("SC_PAGE_SIZE"))

    import resource

    maximum = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return maximum if sys.platform == "darwin" else maximum * 1024


def _write_report(path: Path, payload: Mapping[str, Any]) -> None:
    resolved = path.expanduser().resolve(strict=False)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    temporary = resolved.with_name(f".{resolved.name}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, resolved)


def _configure(output_root: Path) -> Any:
    source = bpy.data.objects.get("Hero")
    if source is None or source.type != "MESH" or source.data is None:
        raise MemoryStressError("fixture must contain active Mesh object 'Hero'")
    if bpy.context.mode != "OBJECT":
        bpy.ops.object.mode_set(mode="OBJECT")
    for obj in bpy.context.view_layer.objects:
        obj.select_set(False)
    source.select_set(True)
    bpy.context.view_layer.objects.active = source

    scene = bpy.context.scene
    scene.spine2d_json_path = str(output_root)
    scene.spine2d_images_path = "images"
    scene.spine2d_texture_size = 64
    scene.spine2d_angle_limit = 30
    scene.spine2d_seam_maker_mode = "AUTO"
    scene.spine2d_control_icons = False
    scene.spine2d_export_preview_animation = False
    scene.spine2d_bake_frame_start = 0
    scene.spine2d_frames_for_render = 0
    return source


def _remove_outputs(output_root: Path) -> None:
    if not output_root.exists():
        return
    for path in sorted(output_root.rglob("*"), reverse=True):
        if path.is_file() or path.is_symlink():
            path.unlink(missing_ok=True)
        elif path.is_dir():
            try:
                path.rmdir()
            except OSError:
                pass


def _run_export(output_root: Path) -> float:
    started = time.perf_counter()
    result = set(bpy.ops.object.save_uv_as_json())
    if "FINISHED" not in result:
        raise MemoryStressError(f"export returned {sorted(result)}")
    expected = output_root / "Hero_merged.json"
    if not expected.is_file():
        raise MemoryStressError(f"export did not create {expected}")
    elapsed = time.perf_counter() - started
    _remove_outputs(output_root)
    render_result = bpy.data.images.get("Render Result")
    if render_result is not None:
        bpy.data.images.remove(render_result)
    gc.collect()
    return elapsed


def main() -> None:
    args = _parser().parse_args(_arguments(sys.argv))
    if args.warmup < 0 or args.iterations <= 0 or args.sample_every <= 0:
        raise MemoryStressError("warmup/iterations/sample-every values are invalid")

    output_root = args.output_root.expanduser().resolve(strict=False)
    output_root.mkdir(parents=True, exist_ok=True)
    addon.register()
    samples = []
    durations = []
    try:
        _configure(output_root)
        for _index in range(args.warmup):
            _run_export(output_root)
        baseline = _rss_bytes()
        for index in range(1, args.iterations + 1):
            durations.append(_run_export(output_root))
            if index % args.sample_every == 0 or index == args.iterations:
                samples.append({"iteration": index, "rss_bytes": _rss_bytes()})
        report = {
            "success": True,
            "blender_version": bpy.app.version_string,
            "warmup": args.warmup,
            "iterations": args.iterations,
            "sample_every": args.sample_every,
            "baseline_rss_bytes": baseline,
            "samples": samples,
            "duration_seconds": {
                "minimum": min(durations),
                "maximum": max(durations),
                "mean": sum(durations) / len(durations),
            },
        }
        _write_report(args.report_json, report)
    except Exception as exc:
        _write_report(
            args.report_json,
            {
                "success": False,
                "error_type": type(exc).__name__,
                "error": str(exc) or type(exc).__name__,
                "traceback": traceback.format_exc(),
            },
        )
        raise
    finally:
        try:
            addon.unregister()
        except Exception as exc:
            raise MemoryStressError(
                "extension unregister failed after memory stress: "
                f"{type(exc).__name__}: {exc}"
            ) from exc


if __name__ == "__main__":
    main()
