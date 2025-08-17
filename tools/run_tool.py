#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.metadata
import logging
import pathlib
import re
import runpy
import subprocess
import sys
from typing import List

from packaging.specifiers import SpecifierSet
from packaging.version import Version

MIN_PY = (3, 8)
if sys.version_info < MIN_PY:
    sys.exit(f"Python {MIN_PY[0]}.{MIN_PY[1]}+ required, got {sys.version.split()[0]}")

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("run_tool")

_RE_REQ = re.compile(r"\s*#\s*REQUIRES\s*:\s*(.+)", re.I)


def _parse_requires(lines: List[str]) -> List[str]:
    for line in lines:
        m = _RE_REQ.match(line)
        if m:
            return [s.strip() for s in m.group(1).split(",")]
    return []


def _ensure(pkgs: List[str]) -> None:
    needs_install = False
    to_install: List[str] = []
    for spec_str in pkgs:
        req = re.split(r"([<>=!~]+)", spec_str, maxsplit=1)
        pkg_name = req[0]
        module_name = (
            "bpy_stubs" if pkg_name.startswith("fake-bpy-module") else pkg_name
        )
        try:
            version_check_name = (
                pkg_name if pkg_name.startswith("fake-bpy-module") else module_name
            )
            installed_version = Version(importlib.metadata.version(version_check_name))
            if len(req) > 1:
                specifiers = SpecifierSet("".join(req[1:]))
                if installed_version not in specifiers:
                    log.info(
                        f"Updating {pkg_name}: installed {installed_version}, required {specifiers}"
                    )
                    to_install.append(spec_str)
                    needs_install = True
        except importlib.metadata.PackageNotFoundError:
            log.info(f"Missing required package: {pkg_name}")
            to_install.append(spec_str)
            needs_install = True
    if needs_install:
        log.info("Installing/updating dependencies: %s", ", ".join(to_install))
        subprocess.check_call(
            [
                sys.executable,
                "-m",
                "pip",
                "--disable-pip-version-check",
                "install",
                "--quiet",
                "--upgrade",
                "--no-cache-dir",
                *to_install,
            ]
        )
        log.info("\nDependencies have been updated. Please run the task again.")
        sys.exit(0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--script", required=True, help="*.py to run")
    ap.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
    )
    ap.add_argument("extra", nargs=argparse.REMAINDER)
    ns = ap.parse_args()

    logging.basicConfig(level=getattr(logging, ns.log_level), format="%(message)s")
    log = logging.getLogger("run_tool")

    try:
        log.info("Ensuring pip is up-to-date...")
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "--quiet", "--upgrade", "pip"]
        )
        # --- ДОБАВЛЕНО: Очистка кэша pip ---
        log.info("Clearing pip cache...")
        subprocess.check_call([sys.executable, "-m", "pip", "cache", "purge"])
    except subprocess.CalledProcessError as e:
        log.error(f"Failed during pip maintenance: {e}")
        sys.exit(1)

    script = pathlib.Path(ns.script).resolve()
    if not script.is_file():
        ap.error(f"{script} not found")

    with script.open(encoding="utf8") as fh:
        head_lines = [next(fh) for _ in range(10)]
    _ensure(_parse_requires(head_lines))

    argv = ns.extra[1:] if ns.extra and ns.extra[0] == "--" else ns.extra
    sys.argv = [str(script), *argv]
    log.info("▶ %s %s", script.name, " ".join(argv))
    runpy.run_path(str(script), run_name="__main__")


if __name__ == "__main__":
    main()
