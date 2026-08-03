"""Temporary setup-pose render proxies for Depth Camera Projection sequences.

Depth sequences own animated material/texture evaluation, not Blender object or camera
motion. The scene timeline still advances so material F-curves, drivers, and image nodes
can evaluate, while an evaluated source-mesh proxy and an active-camera proxy keep the
captured setup geometry and camera transform immutable for every rendered frame.
"""

from __future__ import annotations

from contextlib import contextmanager
import logging
from typing import Any, Iterator

from ..domain.baking import A1TextureExportMode
from .camera_projection_error import CameraProjectionExecutionError
from .camera_projection_validation import CameraProjectionRuntime


logger = logging.getLogger(__name__)


def _copy_matrix(value: Any, field_name: str) -> Any:
    copy = getattr(value, "copy", None)
    if not callable(copy):
        raise CameraProjectionExecutionError(f"{field_name}.copy() is unavailable")
    try:
        return copy()
    except Exception as exc:
        raise CameraProjectionExecutionError(
            f"Unable to copy {field_name}"
        ) from exc


def _clear_animation_data(owner: Any, field_name: str) -> None:
    clear = getattr(owner, "animation_data_clear", None)
    if not callable(clear):
        return
    try:
        clear()
    except Exception as exc:
        raise CameraProjectionExecutionError(
            f"Unable to clear {field_name} animation data"
        ) from exc


def _clear_collection(owner: Any, field_name: str) -> None:
    collection = getattr(owner, field_name, None)
    if collection is None:
        return
    clear = getattr(collection, "clear", None)
    if not callable(clear):
        raise CameraProjectionExecutionError(
            f"{field_name}.clear() is unavailable on render proxy"
        )
    try:
        clear()
    except Exception as exc:
        raise CameraProjectionExecutionError(
            f"Unable to clear render proxy {field_name}"
        ) from exc


def _evaluated_dependency_graph(runtime: CameraProjectionRuntime) -> Any:
    getter = getattr(runtime.context, "evaluated_depsgraph_get", None)
    if not callable(getter):
        raise CameraProjectionExecutionError(
            "Context.evaluated_depsgraph_get() is unavailable"
        )
    try:
        depsgraph = getter()
    except Exception as exc:
        raise CameraProjectionExecutionError(
            "Unable to acquire dependency graph for Depth render proxy"
        ) from exc
    if depsgraph is None:
        raise CameraProjectionExecutionError(
            "Blender returned no dependency graph for Depth render proxy"
        )
    return depsgraph


def _create_source_proxy(
    runtime: CameraProjectionRuntime,
    depsgraph: Any,
) -> tuple[Any, Any]:
    bpy_module = runtime.bpy_module
    source = runtime.source_object
    evaluated_get = getattr(source, "evaluated_get", None)
    if not callable(evaluated_get):
        raise CameraProjectionExecutionError(
            "Depth source object has no evaluated_get()"
        )
    try:
        evaluated = evaluated_get(depsgraph)
    except Exception as exc:
        raise CameraProjectionExecutionError(
            "Unable to evaluate Depth source setup geometry"
        ) from exc
    if evaluated is None:
        raise CameraProjectionExecutionError(
            "Depth source evaluated_get() returned None"
        )

    try:
        mesh = bpy_module.data.meshes.new_from_object(
            evaluated,
            preserve_all_data_layers=True,
            depsgraph=depsgraph,
        )
    except Exception as exc:
        raise CameraProjectionExecutionError(
            "Unable to create evaluated Depth source mesh proxy"
        ) from exc
    if mesh is None:
        raise CameraProjectionExecutionError(
            "Blender returned no evaluated Depth source mesh proxy"
        )

    proxy = None
    try:
        proxy = source.copy()
        proxy.name = f"__Spine2D_DepthSubject_{source.name}"
        proxy.data = mesh
        proxy.parent = None
        proxy.matrix_world = _copy_matrix(
            evaluated.matrix_world,
            "evaluated source matrix_world",
        )
        _clear_animation_data(proxy, "source proxy")
        _clear_collection(proxy, "constraints")
        _clear_collection(proxy, "modifiers")
        runtime.scene.collection.objects.link(proxy)
        return proxy, mesh
    except Exception:
        if proxy is not None:
            try:
                bpy_module.data.objects.remove(proxy, do_unlink=True)
            except Exception:
                logger.exception("Unable to remove incomplete Depth source proxy")
        try:
            bpy_module.data.meshes.remove(mesh)
        except Exception:
            logger.exception("Unable to remove incomplete Depth proxy mesh")
        raise


def _create_camera_proxy(
    runtime: CameraProjectionRuntime,
    depsgraph: Any,
) -> tuple[Any, Any, Any]:
    bpy_module = runtime.bpy_module
    scene = runtime.scene
    camera = getattr(scene, "camera", None)
    if camera is None or getattr(camera, "type", None) != "CAMERA":
        raise CameraProjectionExecutionError(
            "Depth Camera Projection requires an active CAMERA object"
        )
    evaluated_get = getattr(camera, "evaluated_get", None)
    if not callable(evaluated_get):
        raise CameraProjectionExecutionError(
            "Active camera has no evaluated_get()"
        )
    try:
        evaluated = evaluated_get(depsgraph)
    except Exception as exc:
        raise CameraProjectionExecutionError(
            "Unable to evaluate active camera setup pose"
        ) from exc

    camera_data = None
    proxy = None
    try:
        proxy = camera.copy()
        camera_data = evaluated.data.copy()
        proxy.name = f"__Spine2D_DepthCamera_{camera.name}"
        camera_data.name = f"__Spine2D_DepthCameraData_{camera.data.name}"
        proxy.data = camera_data
        proxy.parent = None
        proxy.matrix_world = _copy_matrix(
            evaluated.matrix_world,
            "evaluated camera matrix_world",
        )
        _clear_animation_data(proxy, "camera proxy")
        _clear_animation_data(camera_data, "camera data proxy")
        _clear_collection(proxy, "constraints")
        runtime.scene.collection.objects.link(proxy)
        scene.camera = proxy
        return camera, proxy, camera_data
    except Exception:
        try:
            scene.camera = camera
        except Exception:
            logger.exception("Unable to restore active camera after proxy failure")
        if proxy is not None:
            try:
                bpy_module.data.objects.remove(proxy, do_unlink=True)
            except Exception:
                logger.exception("Unable to remove incomplete camera proxy")
        if camera_data is not None:
            try:
                bpy_module.data.cameras.remove(camera_data)
            except Exception:
                logger.exception("Unable to remove incomplete camera data proxy")
        raise


def _remove_proxy_resources(
    runtime: CameraProjectionRuntime,
    *,
    original_camera: Any,
    source_proxy: Any,
    source_mesh: Any,
    camera_proxy: Any,
    camera_data: Any,
    source_hide_render: bool,
    source_visible_camera: bool,
) -> None:
    failures: list[str] = []
    try:
        runtime.scene.camera = original_camera
    except Exception as exc:
        failures.append(f"scene.camera: {exc}")
    try:
        runtime.source_object.hide_render = source_hide_render
    except Exception as exc:
        failures.append(f"source.hide_render: {exc}")
    try:
        runtime.source_object.visible_camera = source_visible_camera
    except Exception as exc:
        failures.append(f"source.visible_camera: {exc}")

    for label, collection, datablock in (
        ("camera proxy", runtime.bpy_module.data.objects, camera_proxy),
        ("source proxy", runtime.bpy_module.data.objects, source_proxy),
        ("camera data proxy", runtime.bpy_module.data.cameras, camera_data),
        ("source mesh proxy", runtime.bpy_module.data.meshes, source_mesh),
    ):
        if datablock is None:
            continue
        try:
            if label.endswith("proxy") and collection is runtime.bpy_module.data.objects:
                collection.remove(datablock, do_unlink=True)
            else:
                collection.remove(datablock)
        except Exception as exc:
            failures.append(f"{label}: {exc}")

    if failures:
        raise CameraProjectionExecutionError(
            "Unable to clean Depth render proxies: " + "; ".join(failures)
        )


@contextmanager
def frozen_depth_camera_projection_subject(
    runtime: CameraProjectionRuntime,
) -> Iterator[Any]:
    """Yield a fixed setup-pose source proxy and restore every Blender owner."""

    if not isinstance(runtime, CameraProjectionRuntime):
        raise TypeError("runtime must be CameraProjectionRuntime")
    if (
        runtime.execution_settings.texture_export_mode
        is not A1TextureExportMode.DEPTH_CAMERA_PROJECTION
    ):
        raise ValueError(
            "frozen Depth render proxy requires DEPTH_CAMERA_PROJECTION mode"
        )

    depsgraph = _evaluated_dependency_graph(runtime)
    source_hide_render = bool(runtime.source_object.hide_render)
    source_visible_camera = bool(runtime.source_object.visible_camera)
    source_proxy = None
    source_mesh = None
    original_camera = None
    camera_proxy = None
    camera_data = None
    primary_error: BaseException | None = None

    try:
        source_proxy, source_mesh = _create_source_proxy(runtime, depsgraph)
        original_camera, camera_proxy, camera_data = _create_camera_proxy(
            runtime,
            depsgraph,
        )
        runtime.source_object.hide_render = True
        runtime.source_object.visible_camera = False
        logger.info(
            "Created frozen Depth render proxies at setup frame %s: source='%s' "
            "camera='%s'",
            runtime.scene.frame_current,
            runtime.source_object.name,
            original_camera.name,
        )
        yield source_proxy
    except BaseException as exc:
        primary_error = exc
        raise
    finally:
        if source_proxy is None:
            return
        try:
            _remove_proxy_resources(
                runtime,
                original_camera=original_camera,
                source_proxy=source_proxy,
                source_mesh=source_mesh,
                camera_proxy=camera_proxy,
                camera_data=camera_data,
                source_hide_render=source_hide_render,
                source_visible_camera=source_visible_camera,
            )
        except Exception:
            if primary_error is None:
                raise
            logger.exception(
                "Failed to clean Depth render proxies while handling another error"
            )


__all__ = ["frozen_depth_camera_projection_subject"]
