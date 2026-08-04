"""Temporary setup-pose render proxies for Depth Camera Projection views.

Depth sequences own animated material/texture evaluation, not Blender object or camera
motion. The scene timeline still advances so material F-curves, drivers, and image nodes
can evaluate, while an evaluated source-mesh proxy and a camera proxy keep the captured
setup geometry immutable. Parallax reserve plans additionally override only the temporary
camera transform and fitted lens/ortho scale.
"""

from __future__ import annotations

from contextlib import contextmanager
import logging
from math import isfinite
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


def _matrix_from_tuple(value: tuple[float, ...], field_name: str) -> Any:
    if not isinstance(value, tuple) or len(value) != 16:
        raise CameraProjectionExecutionError(
            f"{field_name} must contain sixteen values"
        )
    if not all(isfinite(float(component)) for component in value):
        raise CameraProjectionExecutionError(
            f"{field_name} contains non-finite values"
        )
    try:
        from mathutils import Matrix

        return Matrix(
            tuple(
                tuple(float(value[row * 4 + column]) for column in range(4))
                for row in range(4)
            )
        )
    except Exception as exc:
        raise CameraProjectionExecutionError(
            f"Unable to construct {field_name}"
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


def _apply_projection_fit(camera_data: Any, runtime: CameraProjectionRuntime) -> None:
    scale = float(runtime.plan.lens_scale)
    if scale == 1.0:
        return
    camera_type = str(getattr(camera_data, "type", "") or "")
    if camera_type == "PERSP":
        lens = float(getattr(camera_data, "lens"))
        fitted = lens * scale
        if not isfinite(fitted) or fitted <= 0.0:
            raise CameraProjectionExecutionError(
                f"Invalid fitted Perspective lens {fitted} for view {runtime.plan.view_id}"
            )
        camera_data.lens = fitted
        return
    if camera_type == "ORTHO":
        ortho_scale = float(getattr(camera_data, "ortho_scale"))
        fitted = ortho_scale / scale
        if not isfinite(fitted) or fitted <= 0.0:
            raise CameraProjectionExecutionError(
                f"Invalid fitted Orthographic scale {fitted} for view {runtime.plan.view_id}"
            )
        camera_data.ortho_scale = fitted
        return
    raise CameraProjectionExecutionError(
        "Depth parallax reserve supports Perspective and Orthographic cameras; "
        f"received {camera_type or '<empty>'!r}"
    )


def _create_camera_proxy(
    runtime: CameraProjectionRuntime,
    depsgraph: Any,
    original_camera: Any,
) -> tuple[Any, Any]:
    bpy_module = runtime.bpy_module
    scene = runtime.scene
    if original_camera is None or getattr(original_camera, "type", None) != "CAMERA":
        raise CameraProjectionExecutionError(
            "Depth Camera Projection requires an active CAMERA object"
        )
    evaluated_get = getattr(original_camera, "evaluated_get", None)
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
        proxy = original_camera.copy()
        # Copy the original ID datablock at the current evaluated frame. Blender may
        # reject ``evaluated.data.copy()`` because evaluated IDs are temporary owners.
        camera_data = original_camera.data.copy()
        proxy.name = (
            f"__Spine2D_DepthCamera_{original_camera.name}_{runtime.plan.view_id}"
        )
        camera_data.name = (
            "__Spine2D_DepthCameraData_"
            f"{original_camera.data.name}_{runtime.plan.view_id}"
        )
        proxy.data = camera_data
        proxy.parent = None
        override = runtime.plan.camera_world_matrix_override
        proxy.matrix_world = (
            _matrix_from_tuple(
                override,
                f"parallax {runtime.plan.view_id} camera_world_matrix_override",
            )
            if override is not None
            else _copy_matrix(
                evaluated.matrix_world,
                "evaluated camera matrix_world",
            )
        )
        _apply_projection_fit(camera_data, runtime)
        _clear_animation_data(proxy, "camera proxy")
        _clear_animation_data(camera_data, "camera data proxy")
        _clear_collection(proxy, "constraints")
        runtime.scene.collection.objects.link(proxy)
        scene.camera = proxy
        return proxy, camera_data
    except Exception:
        try:
            scene.camera = original_camera
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
    if original_camera is not None:
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

    for label, collection, datablock, object_datablock in (
        ("camera proxy", runtime.bpy_module.data.objects, camera_proxy, True),
        ("source proxy", runtime.bpy_module.data.objects, source_proxy, True),
        ("camera data proxy", runtime.bpy_module.data.cameras, camera_data, False),
        ("source mesh proxy", runtime.bpy_module.data.meshes, source_mesh, False),
    ):
        if datablock is None:
            continue
        try:
            if object_datablock:
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
    original_camera = getattr(runtime.scene, "camera", None)
    source_proxy = None
    source_mesh = None
    camera_proxy = None
    camera_data = None
    primary_error: BaseException | None = None

    try:
        source_proxy, source_mesh = _create_source_proxy(runtime, depsgraph)
        camera_proxy, camera_data = _create_camera_proxy(
            runtime,
            depsgraph,
            original_camera,
        )
        runtime.source_object.hide_render = True
        runtime.source_object.visible_camera = False
        logger.info(
            "Created frozen Depth render proxies at setup frame %s: source='%s' "
            "camera='%s' view=%s lens_scale=%s",
            runtime.scene.frame_current,
            runtime.source_object.name,
            original_camera.name,
            runtime.plan.view_id,
            runtime.plan.lens_scale,
        )
        yield source_proxy
    except BaseException as exc:
        primary_error = exc
        raise
    finally:
        if source_proxy is not None:
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
