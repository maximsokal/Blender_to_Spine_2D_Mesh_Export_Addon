# texture_baker_integration.py
# pylint: disable=import-error
"""
Blender to Spine2D Mesh Exporter
Copyright (c) 2025 Maxim Sokolenko

This file is part of Blender to Spine2D Mesh Exporter.

Blender to Spine2D Mesh Exporter is free software: you can redistribute it
and/or modify it under the terms of the GNU General Public License as
published by the Free Software Foundation, either version 3 of the License,
or (at your option) any later version.

Blender to Spine2D Mesh Exporter is distributed in the hope that it will
be useful, but WITHOUT ANY WARRANTY; without even the implied warranty
of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License along
with Blender to Spine2D Mesh Exporter. If not, see <https://www.gnu.org/licenses/>.
This module serves as a high-level interface for the texture baking process, integrating the core `TextureBaker` class with the main export pipeline. It simplifies the baking workflow by handling the setup, execution, and finalization of the bake for a given object.

The key functionalities are:
1.  Simplified Baking Interface: The main function, `bake_textures_for_object`, provides a single entry point to perform a complete texture bake. It is designed to work on an object that has already been prepared (e.g., duplicated and given a new UV map), abstracting away the lower-level details.
2.  Configuration Management: It pulls all necessary baking parameters (texture size, margin, output directory, etc.) from the central `config.py` module. This allows for easy customization of the bake settings without modifying the core baking logic.
3.  Orchestration of Baking Steps: It initializes the `TextureBaker` class and calls its methods in the correct sequence: `setup_object_for_baking`, `execute_baking` (or `bake_textures` for sequences), `cleanup_materials`, `create_combined_material`, and `apply_material_to_object`.
4.  Single-Frame and Sequence Handling: It manages both single-frame and animated sequence baking. For sequences, it iterates through the required frame range, calling the `_bake_single_frame` helper for each frame. After the sequence is complete, it creates the final material using the first baked frame.
5.  Resource Management: It is responsible for cleaning up temporary resources created during the bake, such as intermediate bake images and duplicate objects, to prevent memory leaks and keep the user's scene clean.

ATTENTION: - This module is designed to be called *after* an object has been prepared for baking (i.e., it has the correct seams and a UV map ready for the bake). The `skip_duplicate_uv=True` parameter is crucial, as it tells the `TextureBaker` not to perform these preparation steps again. The success of the bake depends heavily on the correct `source_obj` and `source_uv_map_name` being passed in, as this is required for "selected-to-active" baking to work.
Author: Maxim Sokolenko
"""
import bpy
import os
import logging
from typing import Optional, List

from .config import (
    get_default_output_dir,
    get_texture_size,
    BAKE_MARGIN,
    BAKE_TEXTURE_FORMAT,
    BAKE_ACTIVE_UV_NAME,
    SEQUENCE_FRAME_DIGITS,
)

# --- FIX: Import the module directly to access its variables dynamically ---
# This prevents importing a static copy of the variables at startup.
from . import texture_baker
from .texture_baker import (
    TextureBaker,
    sync_bake_sequence_params,
)

logger = logging.getLogger(__name__)
logger.debug("texture_baker_integration.py loaded")


def _bake_single_frame(
    frame: Optional[int],
    baker: TextureBaker,
    dup_obj: bpy.types.Object,
    bake_nodes: List[tuple[bpy.types.Material, bpy.types.Node]],
    bake_image: Optional[bpy.types.Image],
) -> Optional[str]:
    """Perform a single bake for the given frame and return the saved file path.

    When ``frame`` is not ``None``, this helper will set the scene to the
    requested frame, create a fresh bake image, assign it to all bake
    nodes, configure the bake settings and call :meth:`TextureBaker.bake_textures`.
    The temporary image is removed after saving.

    When ``frame`` is ``None`` the existing ``bake_image`` is used and
    :meth:`TextureBaker.execute_baking` is invoked instead.  The image is
    preserved for later use in creating the final material.

    :param frame: frame index to bake or ``None`` for a single bake
    :param baker: initialized :class:`TextureBaker` instance
    :param dup_obj: duplicate object to bake onto
    :param bake_nodes: list of (material, image node) tuples to assign images
    :param bake_image: base bake image for the single bake case
    :returns: path to the saved image or ``None`` on failure
    """
    scene = bpy.context.scene
    try:
        if frame is not None:
            try:
                scene.frame_set(frame)
                bpy.context.view_layer.update()
            except Exception as exc:
                logger.exception(
                    "[BakeIntegration] Failed to set frame %s: %s", frame, exc
                )
                return None
            image_name = f"{dup_obj.name}_Baked_{frame:0{SEQUENCE_FRAME_DIGITS}d}"
            bake_img = baker.create_bake_image(image_name)
            if not bake_img:
                logger.error(
                    "[BakeIntegration] Could not create bake image for frame %s", frame
                )
                return None
        else:
            bake_img = bake_image
            image_name = f"{dup_obj.name}_Baked"

        for mat, node in bake_nodes:
            node.image = bake_img

        has_procedural = any(not baker.is_image_based(mat) for mat, _ in bake_nodes)
        if frame is not None:
            bake_type = (
                baker.procedural_bake_mode
                if has_procedural
                else baker.diffuse_bake_mode
            )
            baker.configure_bake_settings(bake_type)
            success = baker.bake_textures(dup_obj, bake_img)
        else:
            success = baker.execute_baking(dup_obj, bake_img, has_procedural)

        if not success:
            logger.error(
                "[BakeIntegration] Baking failed for frame %s",
                frame if frame is not None else "single",
            )
            if frame is not None and bake_img:
                bpy.data.images.remove(bake_img)
            return None

        saved_path = baker.save_baked_image(bake_img, image_name)
        if saved_path is None:
            logger.error(
                "[BakeIntegration] Failed to save baked image for frame %s", frame
            )
        if frame is not None and bake_img:
            bpy.data.images.remove(bake_img)
        return saved_path
    except Exception as exc:
        logger.exception(
            "[BakeIntegration] Exception during bake for frame %s: %s",
            frame,
            exc,
        )
        if frame is not None and bake_img:
            try:
                bpy.data.images.remove(bake_img)
            except Exception:
                pass
        return None


def bake_textures_for_object(
    prepared_obj: bpy.types.Object,
    uv_map_name: str,
    source_obj: Optional[bpy.types.Object] = None,
    source_uv_map_name: Optional[str] = None,
) -> bool:
    """
    Performs texture baking for an already prepared object.
    It is assumed that the object has already been copied with seams set and has a correct UV unwrap (e.g., 'BakeUV').

    Steps:
      1. Get settings (texture size and output path) from config.
      2. Initialize TextureBaker with parameters from config.
      3. Prepare materials for baking (without duplicating and creating UVs, as the object is already prepared).
      4. Perform baking (single-frame or sequential if BAKE_SEQUENCE_FRAME_COUNT > 0).
      5. Save the baked image.
      6. Create a combined material that connects the resulting texture.
      7. Apply the final material to the object.

    Returns True on success, False in case of an error.
    """
    try:
        scene = bpy.context.scene
        # --- FIX: This function now correctly updates the variables inside the texture_baker module ---
        sync_bake_sequence_params(scene)

        json_dir_prop = scene.spine2d_json_path
        if json_dir_prop and os.path.isabs(bpy.path.abspath(json_dir_prop)):
            base_dir = bpy.path.abspath(json_dir_prop)
        else:
            base_dir = get_default_output_dir()
        images_relative_path = scene.spine2d_images_path
        images_output_dir = os.path.abspath(
            os.path.join(base_dir, images_relative_path)
        )
        os.makedirs(images_output_dir, exist_ok=True)
        texture_size = get_texture_size(scene)
        logger.debug(
            "[BakeIntegration] Output directory: %s, Texture size: %s",
            images_output_dir,
            texture_size,
        )

        baker = TextureBaker(
            texture_size=texture_size,
            margin=BAKE_MARGIN,
            uv_name=uv_map_name,
            source_obj=source_obj,
            source_uv_map_name=source_uv_map_name,
            output_dir=str(images_output_dir),
            texture_format=BAKE_TEXTURE_FORMAT,
            diffuse_bake_mode="DIFFUSE",
            procedural_bake_mode="COMBINED",
        )
        logger.info("[BakeIntegration] uv_map_name=%s", uv_map_name)

        dup_obj, bake_image, bake_nodes = baker.setup_object_for_baking(
            prepared_obj,
            skip_duplicate_uv=True,
            active_uv=BAKE_ACTIVE_UV_NAME,
        )
        if not dup_obj or not bake_image or not bake_nodes:
            logger.error("[BakeIntegration] Failed to prepare object for baking")
            return False

        logger.info("[BakeIntegration] Baking textures for object %s", dup_obj.name)

        orig_frame = scene.frame_current

        # --- FIX: Check the variable directly from the texture_baker module ---
        # This ensures we are using the updated value, not the initial '0'.
        if texture_baker.BAKE_SEQUENCE_FRAME_COUNT > 0:
            logger.info(
                f"[BakeIntegration] Starting sequence baking for {texture_baker.BAKE_SEQUENCE_FRAME_COUNT} frames."
            )
            sequence_paths: List[str] = []
            frames_to_bake = range(
                texture_baker.BAKE_SEQUENCE_FRAME_START,
                texture_baker.BAKE_SEQUENCE_FRAME_START
                + texture_baker.BAKE_SEQUENCE_FRAME_COUNT,
            )
            total = len(list(frames_to_bake))
            for idx, frame in enumerate(frames_to_bake, start=1):
                logger.info(
                    "[BakeIntegration] Baking frame %d/%d (timeline frame %d)",
                    idx,
                    total,
                    frame,
                )
                saved = _bake_single_frame(frame, baker, dup_obj, bake_nodes, None)
                if saved:
                    sequence_paths.append(saved)

            try:
                scene.frame_set(orig_frame)
                bpy.context.view_layer.update()
            except Exception as exc:
                logger.exception("[BakeIntegration] Failed to restore frame: %s", exc)

            baker.cleanup_materials(dup_obj, bake_nodes)

            if sequence_paths:
                first_frame_path = sequence_paths[0]
                try:
                    final_image = bpy.data.images.load(first_frame_path)
                except Exception as exc:
                    logger.exception(
                        "[BakeIntegration] Could not load first baked image %s: %s",
                        first_frame_path,
                        exc,
                    )
                    return False
                combined_material = baker.create_combined_material(dup_obj, final_image)
                if not combined_material:
                    logger.error(
                        "[BakeIntegration] Failed to create combined material from first frame"
                    )
                    return False
                if not baker.apply_material_to_object(dup_obj, combined_material):
                    logger.error(
                        "[BakeIntegration] Failed to apply combined material to object"
                    )
                    return False

                try:
                    final_image.pack()
                except Exception:
                    pass
                try:
                    bpy.data.images.remove(final_image)
                except Exception:
                    pass
            else:
                logger.error(
                    "[BakeIntegration] Sequence baking did not produce any frames"
                )
                return False
        else:
            logger.info("[BakeIntegration] Starting single-frame bake.")
            saved = _bake_single_frame(None, baker, dup_obj, bake_nodes, bake_image)
            if not saved:
                logger.error(
                    "[BakeIntegration] Single-frame baking failed for object %s",
                    dup_obj.name,
                )
                return False

            baker.cleanup_materials(dup_obj, bake_nodes)
            combined_material = baker.create_combined_material(dup_obj, bake_image)
            if not combined_material:
                logger.error("[BakeIntegration] Failed to create combined material")
                return False
            if not baker.apply_material_to_object(dup_obj, combined_material):
                logger.error(
                    "[BakeIntegration] Failed to apply combined material to object"
                )
                return False

            try:
                bake_image.pack()
            except Exception:
                pass
            try:
                bpy.data.images.remove(bake_image)
            except Exception:
                pass

        try:
            if dup_obj != prepared_obj:
                bpy.data.objects.remove(dup_obj, do_unlink=True)
        except Exception as exc:
            logger.warning(
                "[BakeIntegration] Could not remove duplicate object %s: %s",
                dup_obj.name,
                exc,
            )

        logger.info(
            "[BakeIntegration] Baking completed successfully for object %s",
            prepared_obj.name,
        )
        return True

    except Exception as exc:
        logger.exception(
            "[BakeIntegration] Unexpected error in bake_textures_for_object: %s", exc
        )
        return False


def register():
    logger.debug("[LOG] Registration texture_baker_integration.py")
    pass


def unregister():
    logger.debug("[LOG] Unregistration texture_baker_integration.py")
    pass
