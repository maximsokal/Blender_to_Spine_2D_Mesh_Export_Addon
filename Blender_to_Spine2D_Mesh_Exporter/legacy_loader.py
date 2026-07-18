"""Validated lazy loading for explicit Legacy export backends."""

from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module, util as importlib_util
import logging
from pathlib import Path
import sys
from threading import RLock
from types import ModuleType
from typing import Any, Callable


logger = logging.getLogger(__name__)
_PACKAGE = __package__ or __name__.rpartition(".")[0]
_MULTI_PUBLIC_NAME = f"{_PACKAGE}.multi_object_export"
_MULTI_IMPLEMENTATION_NAME = f"{_PACKAGE}._legacy_multi_object_export_impl"
_MULTI_SOURCE_PATH = Path(__file__).with_name("multi_object_export.py")
_MULTI_LOAD_LOCK = RLock()


@dataclass(frozen=True, slots=True)
class LegacySingleBackend:
    main: ModuleType
    json_export: ModuleType

    def __post_init__(self) -> None:
        if not isinstance(self.main, ModuleType):
            raise TypeError("main must be ModuleType")
        if not isinstance(self.json_export, ModuleType):
            raise TypeError("json_export must be ModuleType")
        if not callable(getattr(self.main, "save_uv_as_json", None)):
            raise RuntimeError("Legacy main module does not expose save_uv_as_json")


@dataclass(frozen=True, slots=True)
class LegacyMultiBackend:
    module: ModuleType
    export_selected_objects: Callable[..., Any]

    def __post_init__(self) -> None:
        if not isinstance(self.module, ModuleType):
            raise TypeError("module must be ModuleType")
        if not callable(self.export_selected_objects):
            raise TypeError("export_selected_objects must be callable")


def load_legacy_single_backend() -> LegacySingleBackend:
    """Import and validate single-object Legacy modules only on explicit use."""

    logger.info("Loading explicit Legacy single-object backend")
    main = import_module(".main", _PACKAGE)
    json_export = import_module(".json_export", _PACKAGE)
    return LegacySingleBackend(main=main, json_export=json_export)


def _load_legacy_multi_module() -> ModuleType:
    """Load the existing Legacy source file under a private module name exactly once."""

    with _MULTI_LOAD_LOCK:
        existing = sys.modules.get(_MULTI_IMPLEMENTATION_NAME)
        if isinstance(existing, ModuleType):
            return existing
        if not _MULTI_SOURCE_PATH.is_file():
            raise RuntimeError(
                f"Legacy multi-object source file is missing: {_MULTI_SOURCE_PATH}"
            )

        spec = importlib_util.spec_from_file_location(
            _MULTI_IMPLEMENTATION_NAME,
            _MULTI_SOURCE_PATH,
        )
        if spec is None or spec.loader is None:
            raise RuntimeError(
                f"Unable to create import specification for {_MULTI_SOURCE_PATH}"
            )
        module = importlib_util.module_from_spec(spec)
        sys.modules[_MULTI_IMPLEMENTATION_NAME] = module
        try:
            spec.loader.exec_module(module)
        except Exception:
            sys.modules.pop(_MULTI_IMPLEMENTATION_NAME, None)
            logger.exception("Unable to load Legacy multi-object implementation")
            raise
        return module


def load_legacy_multi_backend() -> LegacyMultiBackend:
    """Load and validate the existing Legacy multi-object source only on explicit use."""

    logger.info("Loading explicit Legacy multi-object backend")
    module = _load_legacy_multi_module()
    export = getattr(module, "export_selected_objects", None)
    return LegacyMultiBackend(module=module, export_selected_objects=export)


def install_legacy_multi_facade() -> ModuleType:
    """Install a lightweight canonical module consumed by ``ui`` during startup.

    The original ``multi_object_export.py`` stays untouched on disk. The facade is placed in
    ``sys.modules`` before UI import. Its first Legacy call loads that file under a private alias,
    preserving all existing relative imports while keeping Rewrite startup free of Legacy imports.
    """

    existing = sys.modules.get(_MULTI_PUBLIC_NAME)
    if isinstance(existing, ModuleType):
        return existing

    facade = ModuleType(
        _MULTI_PUBLIC_NAME,
        "Lazy compatibility facade for the explicit Legacy multi-object exporter.",
    )
    facade.__package__ = _PACKAGE
    facade.__file__ = str(_MULTI_SOURCE_PATH)
    facade.__spine2d_lazy_legacy__ = True

    def export_selected_objects(*args: Any, **kwargs: Any):
        backend = load_legacy_multi_backend()
        return backend.export_selected_objects(*args, **kwargs)

    def facade_getattr(name: str) -> Any:
        if not isinstance(name, str) or not name or name.startswith("__"):
            raise AttributeError(name)
        backend = load_legacy_multi_backend()
        try:
            return getattr(backend.module, name)
        except AttributeError as exc:
            raise AttributeError(
                f"module {_MULTI_PUBLIC_NAME!r} has no attribute {name!r}"
            ) from exc

    facade.export_selected_objects = export_selected_objects
    facade.register = lambda: None
    facade.unregister = lambda: None
    facade.__getattr__ = facade_getattr
    facade.__all__ = ["export_selected_objects"]
    sys.modules[_MULTI_PUBLIC_NAME] = facade
    return facade


__all__ = [
    "LegacyMultiBackend",
    "LegacySingleBackend",
    "install_legacy_multi_facade",
    "load_legacy_multi_backend",
    "load_legacy_single_backend",
]
