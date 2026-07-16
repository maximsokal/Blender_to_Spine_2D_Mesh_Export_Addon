"""Execute B4 active-camera renders with stable union crop and alpha hull analysis."""

from __future__ import annotations

from array import array
from contextlib import contextmanager
from dataclasses import dataclass
import logging
from pathlib import Path
from typing import Any, Iterable, Iterator, Tuple

from ..domain.baking import (
    BakeArtifact,
    BakeExecutionResult,
    BakeExecutionSettings,
    CameraProjectionPlan,
    TextureFormat,
)
from ..domain.baking.projection_layout import (
    CameraProjectionLayout,
    CameraProjectionLayoutError,
    build_sequence_union_layout,
)
from ..infrastructure import (
    AtomicFileTransaction,
    AtomicOutputReservation,
    atomic_file_transaction,
)
from .scene_bake_analyzer import validate_runtime_scene_context

logger = logging.getLogger(__name__)

_ALPHA_THRESHOLD = 1.0 / 255.0


class CameraProjectionExecutionError(RuntimeError):
    """Raised when an active-camera projection render cannot be staged safely."""


@dataclass(frozen=True, slots=True)
class CameraProjectionStageResult:
    reservations: Tuple[AtomicOutputReservation, ...]
    layout: CameraProjectionLayout

    def __post_init__(self) -> None:
        if not isinstance(self.reservations, tuple) or not self.reservations:
            raise ValueError("reservations must be a non-empty tuple")
        if not all(isinstance(item, AtomicOutputReservation) for item in self.reservations):
            raise TypeError("reservations must contain AtomicOutputReservation values")
        if not isinstance(self.layout, CameraProjectionLayout):
            raise TypeError("layout must be CameraProjectionLayout")
        if len(self.reservations) != self.layout.frame_count:
            raise ValueError("reservation count must match projection layout frame_count")


@dataclass(frozen=True, slots=True)
class _SceneValue:
    path: str
    value: Any


@dataclass(frozen=True, slots=True)
class _ObjectVisibility:
    obj: Any
    hide_render: bool
    visible_camera: bool | None


@dataclass(frozen=True, slots=True)
class _ProjectionRuntimeState:
    scene_values: Tuple[_SceneValue, ...]
    frame_current: int
    visibility: Tuple[_ObjectVisibility, ...]

    @classmethod
    def capture(cls, scene: Any) -> "_ProjectionRuntimeState":
        if scene is None:
            raise CameraProjectionExecutionError("scene cannot be None")
        scene_values = []
        for path in _SCENE_PATHS:
            try:
                scene_values.append(_SceneValue(path=path, value=_get_path(scene, path)))
            except Exception:
                logger.debug("Optional render property '%s' is unavailable", path)
        visibility = []
        try:
            objects = tuple(scene.objects)
        except Exception as exc:
            raise CameraProjectionExecutionError("Unable to inspect scene objects") from exc
        for obj in objects:
            try:
                hide_render = bool(getattr(obj, "hide_render", False))
            except Exception:
                hide_render = False
            try:
                visible_camera = (
                    bool(getattr(obj, "visible_camera"))
                    if hasattr(obj, "visible_camera")
                    else None
                )
            except Exception:
                visible_camera = None
            visibility.append(
                _ObjectVisibility(
                    obj=obj,
                    hide_render=hide_render,
                    visible_camera=visible_camera,
                )
            )
        return cls(
            scene_values=tuple(scene_values),
            frame_current=int(getattr(scene, "frame_current", 0) or 0),
            visibility=tuple(visibility),
        )

    def restore(self, scene: Any) -> None:
        failures: list[str] = []
        for entry in reversed(self.scene_values):
            try:
                _set_path(scene, entry.path, entry.value)
            except Exception as exc:
                failures.append(f"{entry.path}: {exc}")
        for entry in self.visibility:
            try:
                entry.obj.hide_render = entry.hide_render
            except Exception as exc:
                failures.append(
                    f"{getattr(entry.obj, 'name', 'Object')}.hide_render: {exc}"
                )
            if entry.visible_camera is not None:
                try:
                    entry.obj.visible_camera = entry.visible_camera
                except Exception as exc:
                    failures.append(
                        f"{getattr(entry.obj, 'name', 'Object')}.visible_camera: {exc}"
                    )
        try:
            frame_set = getattr(scene, "frame_set", None)
            if callable(frame_set):
                frame_set(self.frame_current)
            else:
                scene.frame_current = self.frame_current
        except Exception as exc:
            failures.append(f"frame_current: {exc}")
        if failures:
            raise CameraProjectionExecutionError(
                "Unable to restore camera projection state: " + "; ".join(failures)
            )


_SCENE_PATHS = (
    "render.engine",
    "render.resolution_x",
    "render.resolution_y",
    "render.resolution_percentage",
    "render.filepath",
    "render.film_transparent",
    "render.use_file_extension",
    "render.image_settings.file_format",
    "render.image_settings.color_mode",
    "render.image_settings.color_depth",
    "cycles.samples",
    "cycles.film_transparent_glass",
)
_RENDERABLE_TYPES = frozenset({"MESH", "CURVE", "SURFACE", "META", "FONT", "VOLUME"})


def _get_path(root: Any, path: str) -> Any:
    current = root
    for part in path.split("."):
        current = getattr(current, part)
    return current


def _set_path(root: Any, path: str, value: Any) -> None:
    parts = path.split(".")
    current = root
    for part in parts[:-1]:
        current = getattr(current, part)
    setattr(current, parts[-1], value)


def _load_bpy() -> Any:
    try:
        import bpy
    except Exception as exc:
        raise CameraProjectionExecutionError("Blender bpy module is unavailable") from exc
    return bpy


def _set_if_available(root: Any, path: str, value: Any) -> None:
    try:
        _set_path(root, path, value)
    except Exception:
        logger.debug("Optional render property '%s' is not writable", path, exc_info=True)


@contextmanager
def preserve_camera_projection_state(scene: Any) -> Iterator[_ProjectionRuntimeState]:
    state = _ProjectionRuntimeState.capture(scene)
    primary_error: BaseException | None = None
    try:
        yield state
    except BaseException as exc:
        primary_error = exc
        raise
    finally:
        try:
            state.restore(scene)
        except Exception:
            if primary_error is None:
                raise
            logger.exception(
                "Failed to restore camera projection state while handling another error"
            )


def _configure_camera_visibility(source_obj: Any, scene: Any, *, isolate: bool) -> None:
    try:
        objects = tuple(scene.objects)
    except Exception as exc:
        raise CameraProjectionExecutionError("Unable to iterate scene objects") from exc
    if source_obj not in objects:
        raise CameraProjectionExecutionError("source object is not linked to the render scene")

    for obj in objects:
        object_type = str(getattr(obj, "type", "") or "")
        if obj is source_obj:
            try:
                obj.hide_render = False
            except Exception as exc:
                raise CameraProjectionExecutionError(
                    "Unable to make source object render-visible"
                ) from exc
            if hasattr(obj, "visible_camera"):
                try:
                    obj.visible_camera = True
                except Exception as exc:
                    raise CameraProjectionExecutionError(
                        "Unable to enable source camera-ray visibility"
                    ) from exc
            continue

        if isolate and object_type in _RENDERABLE_TYPES and hasattr(obj, "visible_camera"):
            try:
                obj.visible_camera = False
            except Exception as exc:
                raise CameraProjectionExecutionError(
                    f"Unable to isolate camera visibility for '{getattr(obj, 'name', obj)}'"
                ) from exc


def configure_scene_for_camera_projection(
    scene: Any,
    plan: CameraProjectionPlan,
    execution_settings: BakeExecutionSettings,
    staged_path: Path,
) -> None:
    if scene is None:
        raise CameraProjectionExecutionError("scene cannot be None")
    if not isinstance(plan, CameraProjectionPlan):
        raise TypeError("plan must be CameraProjectionPlan")
    if not isinstance(execution_settings, BakeExecutionSettings):
        raise TypeError("execution_settings must be BakeExecutionSettings")
    if not isinstance(staged_path, Path):
        raise TypeError("staged_path must be pathlib.Path")

    scene.render.engine = execution_settings.render_engine
    scene.render.resolution_x = plan.settings.width
    scene.render.resolution_y = plan.settings.height
    scene.render.resolution_percentage = 100
    scene.render.filepath = str(staged_path)
    scene.render.film_transparent = plan.transparent_background
    scene.render.use_file_extension = True
    scene.render.image_settings.file_format = plan.settings.texture_format.value
    scene.render.image_settings.color_mode = "RGBA"
    scene.render.image_settings.color_depth = (
        "32" if plan.settings.texture_format is TextureFormat.OPEN_EXR else "8"
    )
    scene.cycles.samples = execution_settings.samples
    _set_if_available(scene, "cycles.film_transparent_glass", False)


def _set_timeline_frame(scene: Any, context: Any, frame: int | None) -> None:
    if frame is None:
        return
    try:
        scene.frame_set(frame)
        update = getattr(getattr(context, "view_layer", None), "update", None)
        if callable(update):
            update()
    except Exception as exc:
        raise CameraProjectionExecutionError(
            f"Unable to evaluate camera projection frame {frame}"
        ) from exc


def _call_public_render_operator(bpy_module: Any) -> None:
    from . import bake_executor as public_executor

    public_executor._call_render_operator(bpy_module)


def _require_reservations(
    plan: CameraProjectionPlan,
    reservations: Iterable[AtomicOutputReservation],
) -> Tuple[AtomicOutputReservation, ...]:
    resolved = tuple(reservations)
    if len(resolved) != len(plan.frame_tasks):
        raise CameraProjectionExecutionError(
            f"Expected {len(plan.frame_tasks)} projection reservations, got {len(resolved)}"
        )
    for task, reservation in zip(plan.frame_tasks, resolved):
        if not isinstance(reservation, AtomicOutputReservation):
            raise TypeError("reservations must contain AtomicOutputReservation")
        expected = task.output_path.expanduser().resolve(strict=False)
        if reservation.final_path != expected:
            raise CameraProjectionExecutionError(
                f"Projection task {task.task_index} expected '{expected}', got "
                f"'{reservation.final_path}'"
            )
    return resolved


def _render_result_alpha_mask(
    bpy_module: Any,
    *,
    width: int,
    height: int,
    threshold: float,
) -> bytes:
    image = bpy_module.data.images.get("Render Result")
    if image is None:
        raise CameraProjectionExecutionError("Blender Render Result image is unavailable")
    actual_size = tuple(int(value) for value in image.size[:2])
    if actual_size != (width, height):
        raise CameraProjectionExecutionError(
            f"Render Result size {actual_size} does not match planned {(width, height)}"
        )
    pixels = array("f", [0.0]) * (width * height * 4)
    try:
        image.pixels.foreach_get(pixels)
    except Exception as exc:
        raise CameraProjectionExecutionError("Unable to read Render Result pixels") from exc
    mask = bytearray(width * height)
    for pixel_index in range(width * height):
        if float(pixels[pixel_index * 4 + 3]) >= threshold:
            mask[pixel_index] = 1
    return bytes(mask)


def _read_image_pixels(image: Any, width: int, height: int) -> array:
    actual_size = tuple(int(value) for value in image.size[:2])
    if actual_size != (width, height):
        raise CameraProjectionExecutionError(
            f"Rendered image size {actual_size} does not match planned {(width, height)}"
        )
    pixels = array("f", [0.0]) * (width * height * 4)
    try:
        image.pixels.foreach_get(pixels)
    except Exception as exc:
        raise CameraProjectionExecutionError("Unable to read rendered image pixels") from exc
    return pixels


def _crop_pixel_buffer(
    pixels: array,
    *,
    full_width: int,
    full_height: int,
    layout: CameraProjectionLayout,
) -> array:
    if len(pixels) != full_width * full_height * 4:
        raise CameraProjectionExecutionError("rendered pixel buffer has invalid length")
    crop = layout.crop
    result = array("f", [0.0]) * (layout.cropped_width * layout.cropped_height * 4)
    row_components = layout.cropped_width * 4
    for target_y, source_y in enumerate(range(crop.minimum_y, crop.maximum_y)):
        source_start = (source_y * full_width + crop.minimum_x) * 4
        target_start = target_y * row_components
        result[target_start : target_start + row_components] = pixels[
            source_start : source_start + row_components
        ]
    return result


def _remove_image(bpy_module: Any, image: Any | None) -> None:
    if image is None:
        return
    try:
        bpy_module.data.images.remove(image)
    except Exception:
        logger.exception("Failed to remove temporary projection image")


def _rewrite_staged_image_with_crop(
    bpy_module: Any,
    plan: CameraProjectionPlan,
    reservation: AtomicOutputReservation,
    layout: CameraProjectionLayout,
) -> None:
    loaded = None
    cropped = None
    try:
        loaded = bpy_module.data.images.load(
            str(reservation.staged_path),
            check_existing=False,
        )
        pixels = _read_image_pixels(loaded, plan.settings.width, plan.settings.height)
        cropped_pixels = _crop_pixel_buffer(
            pixels,
            full_width=plan.settings.width,
            full_height=plan.settings.height,
            layout=layout,
        )
        color_space_name = None
        try:
            color_space_name = str(loaded.colorspace_settings.name)
        except Exception:
            logger.debug("Unable to read source image color space", exc_info=True)
        _remove_image(bpy_module, loaded)
        loaded = None

        cropped = bpy_module.data.images.new(
            name=f"__Spine2D_ProjectionCrop_{reservation.final_path.stem}",
            width=layout.cropped_width,
            height=layout.cropped_height,
            alpha=True,
            float_buffer=plan.settings.texture_format is TextureFormat.OPEN_EXR,
        )
        if color_space_name:
            try:
                cropped.colorspace_settings.name = color_space_name
            except Exception:
                logger.debug("Unable to restore cropped image color space", exc_info=True)
        cropped.pixels.foreach_set(cropped_pixels)
        cropped.update()
        cropped.file_format = plan.settings.texture_format.value
        cropped.filepath_raw = str(reservation.staged_path)
        cropped.save()
        if not reservation.staged_path.is_file() or reservation.staged_path.stat().st_size <= 0:
            raise CameraProjectionExecutionError(
                f"Cropped projection output is missing or empty: {reservation.staged_path}"
            )
    except CameraProjectionExecutionError:
        raise
    except Exception as exc:
        raise CameraProjectionExecutionError(
            f"Unable to crop staged projection image '{reservation.staged_path}': {exc}"
        ) from exc
    finally:
        _remove_image(bpy_module, loaded)
        _remove_image(bpy_module, cropped)


def _render_to_reservations(
    source_obj: Any,
    plan: CameraProjectionPlan,
    execution_settings: BakeExecutionSettings,
    reservations: Tuple[AtomicOutputReservation, ...],
    *,
    context: Any | None,
    scene: Any | None,
) -> CameraProjectionLayout:
    if source_obj is None or getattr(source_obj, "type", None) != "MESH":
        raise CameraProjectionExecutionError("source_obj must be a Blender MESH object")
    if not isinstance(plan, CameraProjectionPlan):
        raise TypeError("plan must be CameraProjectionPlan")
    if str(getattr(source_obj, "name", "")) != plan.source_object_id:
        raise CameraProjectionExecutionError(
            "source object identity does not match CameraProjectionPlan"
        )

    bpy_module = _load_bpy()
    resolved_context = context or bpy_module.context
    resolved_scene = scene or getattr(resolved_context, "scene", None)
    if resolved_scene is None:
        raise CameraProjectionExecutionError("A Blender Scene is required")
    if getattr(resolved_scene, "camera", None) is None:
        raise CameraProjectionExecutionError("Scene has no active camera")

    validate_runtime_scene_context(
        source_obj,
        plan.object_context,
        plan.scene_context,
        scene=resolved_scene,
        context=resolved_context,
    )
    resolved_reservations = _require_reservations(plan, reservations)
    masks: list[bytes] = []

    with preserve_camera_projection_state(resolved_scene):
        _configure_camera_visibility(
            source_obj,
            resolved_scene,
            isolate=plan.isolate_source_to_camera,
        )
        for task, reservation in zip(plan.frame_tasks, resolved_reservations):
            _set_timeline_frame(resolved_scene, resolved_context, task.timeline_frame)
            configure_scene_for_camera_projection(
                resolved_scene,
                plan,
                execution_settings,
                reservation.staged_path,
            )
            logger.info(
                "Rendering B4 camera projection '%s' frame %d/%d camera='%s'",
                plan.source_object_id,
                task.task_index + 1,
                len(plan.frame_tasks),
                plan.camera_object_id,
            )
            _call_public_render_operator(bpy_module)
            if not reservation.staged_path.is_file():
                raise CameraProjectionExecutionError(
                    "Blender reported a finished render but the staged file is missing: "
                    f"{reservation.staged_path}"
                )
            if reservation.staged_path.stat().st_size <= 0:
                raise CameraProjectionExecutionError(
                    f"Camera projection output is empty: {reservation.staged_path}"
                )
            masks.append(
                _render_result_alpha_mask(
                    bpy_module,
                    width=plan.settings.width,
                    height=plan.settings.height,
                    threshold=_ALPHA_THRESHOLD,
                )
            )

        try:
            layout = build_sequence_union_layout(
                tuple(masks),
                width=plan.settings.width,
                height=plan.settings.height,
                alpha_threshold=_ALPHA_THRESHOLD,
                padding_pixels=plan.settings.margin_pixels,
            )
        except CameraProjectionLayoutError as exc:
            raise CameraProjectionExecutionError(str(exc)) from exc

        for reservation in resolved_reservations:
            _rewrite_staged_image_with_crop(
                bpy_module,
                plan,
                reservation,
                layout,
            )
        logger.info(
            "B4 union layout for '%s': full=%dx%d crop=(%d,%d)-(%d,%d) size=%dx%d hull=%d",
            plan.source_object_id,
            layout.full_width,
            layout.full_height,
            layout.crop.minimum_x,
            layout.crop.minimum_y,
            layout.crop.maximum_x,
            layout.crop.maximum_y,
            layout.cropped_width,
            layout.cropped_height,
            len(layout.hull),
        )
        return layout


def stage_camera_projection_outputs_detailed(
    source_obj: Any,
    plan: CameraProjectionPlan,
    output_transaction: AtomicFileTransaction,
    execution_settings: BakeExecutionSettings | None = None,
    *,
    context: Any | None = None,
    scene: Any | None = None,
) -> CameraProjectionStageResult:
    if not isinstance(output_transaction, AtomicFileTransaction):
        raise TypeError("output_transaction must be an AtomicFileTransaction")
    resolved_settings = execution_settings or BakeExecutionSettings()
    if not isinstance(resolved_settings, BakeExecutionSettings):
        raise TypeError("execution_settings must be BakeExecutionSettings or None")
    try:
        reservations = tuple(
            output_transaction.reserve(task.output_path) for task in plan.frame_tasks
        )
        layout = _render_to_reservations(
            source_obj,
            plan,
            resolved_settings,
            reservations,
            context=context,
            scene=scene,
        )
        return CameraProjectionStageResult(reservations=reservations, layout=layout)
    except CameraProjectionExecutionError:
        raise
    except Exception as exc:
        logger.exception("Unexpected B4 projection failure for '%s'", plan.source_object_id)
        raise CameraProjectionExecutionError(
            f"Camera projection failed for '{plan.source_object_id}': {exc}"
        ) from exc


def stage_camera_projection_outputs(
    source_obj: Any,
    plan: CameraProjectionPlan,
    output_transaction: AtomicFileTransaction,
    execution_settings: BakeExecutionSettings | None = None,
    *,
    context: Any | None = None,
    scene: Any | None = None,
) -> Tuple[AtomicOutputReservation, ...]:
    """Compatibility wrapper returning only caller-owned reservations."""

    return stage_camera_projection_outputs_detailed(
        source_obj,
        plan,
        output_transaction,
        execution_settings,
        context=context,
        scene=scene,
    ).reservations


def _build_execution_result(
    plan: CameraProjectionPlan,
    committed_paths: Iterable[Path],
    layout: CameraProjectionLayout,
) -> BakeExecutionResult:
    resolved = tuple(Path(path).expanduser().resolve(strict=False) for path in committed_paths)
    expected = tuple(task.output_path.expanduser().resolve(strict=False) for task in plan.frame_tasks)
    if resolved != expected:
        raise CameraProjectionExecutionError(
            f"Committed projection paths do not match plan; expected={expected}, got={resolved}"
        )
    artifacts = tuple(
        BakeArtifact(
            task_index=task.task_index,
            timeline_frame=task.timeline_frame,
            image_name=task.image_name,
            output_path=path,
            width=layout.cropped_width,
            height=layout.cropped_height,
        )
        for task, path in zip(plan.frame_tasks, resolved)
    )
    return BakeExecutionResult(plan=plan, artifacts=artifacts)


def execute_camera_projection_plan(
    source_obj: Any,
    plan: CameraProjectionPlan,
    execution_settings: BakeExecutionSettings | None = None,
    *,
    context: Any | None = None,
    scene: Any | None = None,
) -> BakeExecutionResult:
    with atomic_file_transaction() as transaction:
        staged = stage_camera_projection_outputs_detailed(
            source_obj,
            plan,
            transaction,
            execution_settings,
            context=context,
            scene=scene,
        )
        committed = transaction.commit()
    return _build_execution_result(plan, committed, staged.layout)


__all__ = [
    "CameraProjectionExecutionError",
    "CameraProjectionStageResult",
    "configure_scene_for_camera_projection",
    "execute_camera_projection_plan",
    "preserve_camera_projection_state",
    "stage_camera_projection_outputs",
    "stage_camera_projection_outputs_detailed",
]
