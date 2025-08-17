# texture_baker.py
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
This module provides a comprehensive `TextureBaker` class that encapsulates all the logic required to perform texture baking in Blender. It is designed to handle various material types (image-based, procedural, solid color) and supports both single-frame and animated sequence baking.

The key functionalities are:
1.  Baking Preparation: It handles all the necessary setup steps before baking. This includes creating a duplicate of the object, generating a new UV map for baking (`create_bake_uv_map`), and creating a blank image to bake onto (`create_bake_image`).
2.  Material Setup: It contains specialized functions to temporarily modify an object's materials for baking. It can set up nodes for image-based materials (`setup_material_for_image_baking`), procedural materials (`setup_material_for_procedural_baking`), and simple color materials, ensuring the bake process captures the correct visual information.
3.  Bake Execution: It configures Blender's Cycles bake settings (e.g., bake type, margin, selected-to-active, cage extrusion) and executes the `bpy.ops.object.bake()` operator.
4.  Post-Bake Processing: After baking, it saves the resulting image to a file, cleans up the temporary nodes it added to the materials, and creates a new, simplified "combined material" that uses the baked texture. This new material is then applied to the object.
5.  Sequence Baking: It includes logic to handle animated sequences by iterating through a specified frame range, baking each frame to a separate image file.
6.  Utility and Validation: It includes helper functions for validating objects before baking, checking the quality of the baked image (`check_bake_image`), and cleaning up temporary data.

ATTENTION: - This module heavily relies on manipulating Blender's internal state, including changing the active object, switching modes (Object/Edit), and modifying material node trees. The `cleanup_materials` function is critical for restoring the original materials to their previous state after baking; if it fails, the user's materials could be left in a broken state. The "selected-to-active" baking mode requires a `source_obj` to be correctly specified during initialization. The module's performance can be affected by the complexity of the materials and the specified texture size.
Author: Maxim Sokolenko
"""
import bpy
import logging

logger = logging.getLogger(__name__)
import os
import time
from typing import Optional, Tuple, List, TYPE_CHECKING
from .config import get_default_output_dir
from .uv_operations import flip_image_vertically, log_uv_bounds

if TYPE_CHECKING:
    UV_NAME: str
    source_obj: bpy.types.Object
    source_uv_map_name: str
# === Global Configuration Settings ===
TEXTURE_SIZE = 256
MARGIN = 4
# UV_NAME = "BakeUV"
# OUTPUT_DIR = bpy.path.abspath('//') if bpy.path.abspath('//') else os.path.expanduser('~')
TEXTURE_FORMAT = "PNG"
DIFFUSE_BAKE_MODE = "DIFFUSE"
PROCEDURAL_BAKE_MODE = "COMBINED"  # "COMBINED"

# Sequential baking parameters:
# If BAKE_SEQUENCE_FRAME_COUNT > 0, a sequence of exactly this number of frames will be baked,
# starting from (BAKE_SEQUENCE_FRAME_START + 1).
# === Global sequence parameters (default — 0) =========================
BAKE_SEQUENCE_FRAME_COUNT = 0
BAKE_SEQUENCE_FRAME_START = 0
BAKE_SEQUENCE_FRAME_END = 0


# ─────────────────────────────────────────────────────────────────────────
def sync_bake_sequence_params(scene: bpy.types.Scene) -> None:
    """
    Takes values from the Scene and puts them into the global BAKE_SEQUENCE_... variables.
    Called every time before baking.
    """
    global BAKE_SEQUENCE_FRAME_COUNT, BAKE_SEQUENCE_FRAME_START, BAKE_SEQUENCE_FRAME_END

    BAKE_SEQUENCE_FRAME_COUNT = max(
        0, int(getattr(scene, "spine2d_frames_for_render", 0))
    )
    BAKE_SEQUENCE_FRAME_START = max(
        0, int(getattr(scene, "spine2d_bake_frame_start", 0))
    )

    if BAKE_SEQUENCE_FRAME_COUNT > 0:
        BAKE_SEQUENCE_FRAME_END = (
            BAKE_SEQUENCE_FRAME_START + BAKE_SEQUENCE_FRAME_COUNT - 1
        )
    else:
        BAKE_SEQUENCE_FRAME_END = scene.frame_end


class TextureBaker:
    """
    A class for baking object textures in Blender with support for regular and sequential baking.
    In sequential baking, the first baked frame is used instead of creating an animated material.
    """

    def __init__(
        self,
        texture_size: int = TEXTURE_SIZE,
        margin: int = MARGIN,
        uv_name: str = "BakeUV",
        source_obj: Optional[bpy.types.Object] = None,
        source_uv_map_name: Optional[str] = None,
        output_dir: Optional[str] = None,
        texture_format: str = TEXTURE_FORMAT,
        diffuse_bake_mode: str = DIFFUSE_BAKE_MODE,
        procedural_bake_mode: str = PROCEDURAL_BAKE_MODE,
        cage_extrusion: float = 0.1,  # New parameter for cage
    ) -> None:
        self.texture_size = texture_size
        self.margin = margin
        self.uv_name = uv_name
        self.source_obj = source_obj
        self.source_uv_map_name = source_uv_map_name
        self.texture_format = texture_format
        self.diffuse_bake_mode = diffuse_bake_mode.upper()
        self.procedural_bake_mode = procedural_bake_mode.upper()
        if output_dir is None:
            self.output_dir = get_default_output_dir()
        else:
            self.output_dir = output_dir
        self.cage_extrusion = cage_extrusion  # Save the new value
        self.logger = logger
        self.logger.debug(
            f"TextureBaker initialized with: texture_size={texture_size}, margin={margin}, "
            f"uv_name={uv_name}, output_dir={self.output_dir}, diffuse_bake_mode={self.diffuse_bake_mode}, "
            f"procedural_bake_mode={self.procedural_bake_mode}, "
            f"source_obj={'None' if self.source_obj is None else self.source_obj.name}, "
            f"source_uv_map_name={self.source_uv_map_name}, cage_extrusion={self.cage_extrusion}"
        )

    def validate_object(self, obj: bpy.types.Object) -> bool:
        try:
            self.logger.debug(f"Validating object: {obj.name}")
            if obj is None or obj.type != "MESH" or len(obj.material_slots) == 0:
                self.logger.error(
                    f"Object {obj.name if obj else 'None'} is not valid for baking"
                )
                return False
            has_valid = any(
                slot.material and slot.material.node_tree for slot in obj.material_slots
            )
            if not has_valid:
                self.logger.error(
                    f"Object {obj.name} has no valid materials with node trees"
                )
                return False
            self.logger.debug(f"Object {obj.name} is valid for baking")
            return True
        except Exception as e:
            self.logger.exception(f"Error validating object: {e}")
            return False

    def create_duplicate(self, obj: bpy.types.Object) -> Optional[bpy.types.Object]:
        try:
            self.logger.debug(f"Creating duplicate of object: {obj.name}")
            bpy.ops.object.select_all(action="DESELECT")
            bpy.context.view_layer.objects.active = obj
            obj.select_set(True)
            bpy.ops.object.duplicate()
            dup_obj = bpy.context.active_object
            dup_obj.name = f"{obj.name}_BAKE_{int(time.time())}"
            # Explicitly copy the Mesh to ensure independence:
            dup_obj.data = dup_obj.data.copy()
            self.logger.debug(f"Created duplicate: {dup_obj.name}")
            for slot in dup_obj.material_slots:
                if slot.material:
                    orig_name = slot.material.name
                    slot.material = slot.material.copy()
                    self.logger.debug(
                        f"Copied material '{orig_name}' for duplicate object."
                    )
            return dup_obj
        except Exception as e:
            self.logger.exception(f"Error duplicating object: {e}")
            return None

    def create_bake_uv_map(self, obj: bpy.types.Object) -> bool:
        self.logger.debug("Starting create_bake_uv_map")
        try:
            self.logger.debug(f"Creating bake UV map on object: {obj.name}")
            mesh = obj.data
            if self.uv_name in mesh.uv_layers:
                self.logger.debug(f"UV map '{self.uv_name}' exists, setting active")
                mesh.uv_layers.active = mesh.uv_layers[self.uv_name]
            else:
                new_uv = mesh.uv_layers.new(name=self.uv_name)
                if not new_uv:
                    self.logger.error(f"Failed to create UV map '{self.uv_name}'")
                    return False
                mesh.uv_layers.active = new_uv
                self.logger.debug(f"Created new UV map: {self.uv_name}")
            current_mode = obj.mode
            try:
                bpy.ops.object.mode_set(mode="EDIT")
                if bpy.context.object.mode != "EDIT":
                    self.logger.warning("Failed to enter EDIT mode for unwrap")
                    return False
            except Exception as e:
                self.logger.exception(f"Error when switching to EDIT mode: {e}")
                return False
            bpy.ops.mesh.select_all(action="SELECT")
            bpy.ops.uv.select_all(action="SELECT")
            self.logger.debug("Performing Smart UV Project for unwrapping")
            bpy.ops.uv.smart_project(
                angle_limit=66.0, island_margin=0.03, correct_aspect=True
            )
            self.logger.debug("Packing UV islands")
            bpy.ops.uv.pack_islands(margin=0.001)
            try:
                bpy.ops.object.mode_set(mode=current_mode)
                if bpy.context.object.mode != current_mode:
                    self.logger.warning(
                        "Failed to return to original mode after unwrap"
                    )
            except Exception as e:
                self.logger.exception(f"Error returning to original mode: {e}")
            self.logger.debug(
                f"UV map '{self.uv_name}' created and configured successfully"
            )
            return True
        except Exception as e:
            self.logger.exception(f"Error creating bake UV map: {e}")
            try:
                bpy.ops.object.mode_set(mode="OBJECT")
            except Exception:
                pass
            return False

    def create_bake_image(self, name: str) -> Optional[bpy.types.Image]:
        try:
            self.logger.debug(f"Creating bake image: {name}")
            if name in bpy.data.images:
                bpy.data.images.remove(bpy.data.images[name])
                self.logger.debug(f"Removed existing image with name: {name}")
            image = bpy.data.images.new(
                name=name, width=self.texture_size, height=self.texture_size, alpha=True
            )
            image.generated_color = (0.5, 0.5, 0.5, 1.0)
            self.logger.debug(
                f"Created image: {image.name} with dimensions {image.size[0]}x{image.size[1]}"
            )
            return image
        except Exception as e:
            self.logger.exception(f"Error creating bake image: {e}")
            return None

    def prepare_procedural_material(self, material: bpy.types.Material) -> None:
        try:
            self.logger.debug(f"Preparing procedural material: {material.name}")
            if not material or not material.node_tree:
                return
            nodes = material.node_tree.nodes
            links = material.node_tree.links
            principled = None
            for node in nodes:
                if node.type == "BSDF_PRINCIPLED":
                    principled = node
                    break
            if not principled:
                self.logger.warning(f"No Principled BSDF found in {material.name}")
                return
            base_color_connected = any(
                link.to_node == principled and link.to_socket.name == "Base Color"
                for link in links
            )
            if not base_color_connected:
                self.logger.debug("Base Color not connected; adding default RGB node")
                rgb = nodes.new(type="ShaderNodeRGB")
                rgb.outputs[0].default_value = (0.8, 0.8, 0.8, 1.0)
                rgb.location = (principled.location.x - 300, principled.location.y)
                links.new(rgb.outputs["Color"], principled.inputs["Base Color"])
            if self.procedural_bake_mode.upper() == "EMIT":
                self.logger.debug(
                    "Procedural bake mode is EMIT; redirecting material output via Emission"
                )
                emission = None
                for node in nodes:
                    if node.type == "EMISSION":
                        emission = node
                        break
                if not emission:
                    emission = nodes.new(type="ShaderNodeEmission")
                    emission.location = (
                        principled.location.x + 200,
                        principled.location.y,
                    )
                mix = nodes.new(type="ShaderNodeMixShader")
                mix.location = (principled.location.x + 400, principled.location.y)
                output = None
                for node in nodes:
                    if node.type == "OUTPUT_MATERIAL":
                        output = node
                        break
                if output:
                    links.new(principled.outputs["BSDF"], mix.inputs[1])
                    links.new(emission.outputs["Emission"], mix.inputs[2])
                    links.new(mix.outputs["Shader"], output.inputs["Surface"])
            self.logger.debug(f"Material {material.name} prepared for baking")
        except Exception as e:
            self.logger.exception(
                f"Error preparing procedural material {material.name}: {e}"
            )

    def is_image_based(self, material: bpy.types.Material) -> bool:
        """
        Returns True if the material is image-based
        (including multi_image) and NOT purely procedural.
        """
        try:
            # 1) Explicit tag set by the bake setup
            if "_bake_mode" in material:
                mode = str(material["_bake_mode"]).lower()
                # image, image_temporary, multi_image → True
                if mode.startswith(("image", "multi")):
                    return True
                # procedural → False
                if mode.startswith("procedural"):
                    return False
            # 2) Heuristic: presence of at least one TEX_IMAGE node
            if material and material.node_tree:
                for n in material.node_tree.nodes:
                    if n.type == "TEX_IMAGE" and not n.name.startswith("TEMP_BAKE_"):
                        return True
            return False
        except Exception as e:
            self.logger.exception(
                f"Error in is_image_based for material {material.name}: {e}"
            )
            return False

    def setup_material_for_image_baking(
        self, material: bpy.types.Material, bake_image: bpy.types.Image, uv_name: str
    ) -> Tuple[bool, Optional[bpy.types.ShaderNodeTexImage]]:
        self.logger.debug(
            "=== Running setup_material_for_image_baking (new version) ==="
        )
        try:
            if not material or not material.node_tree:
                self.logger.error(
                    "[setup_material_for_image_baking]The material is missing or does not have a node_tree."
                )
                return (False, None)

            nodes = material.node_tree.nodes
            links = material.node_tree.links

            # Create a new Image Texture node for baking
            temp_tex_node = nodes.new(type="ShaderNodeTexImage")
            temp_tex_node.name = "TEMP_BAKE_" + bake_image.name
            temp_tex_node.image = bake_image
            temp_tex_node.location = (-400, 300)
            self.logger.debug(
                f"[setup_material_for_image_baking] Created TEMP Image Texture node: {temp_tex_node.name}"
            )

            # Create a new UV Map node with the specified uv_name
            temp_uv_node = nodes.new(type="ShaderNodeUVMap")
            temp_uv_node.name = "TEMP_UV_" + uv_name
            temp_uv_node.uv_map = uv_name
            temp_uv_node.location = (
                temp_tex_node.location.x - 300,
                temp_tex_node.location.y,
            )
            self.logger.debug(
                f"[setup_material_for_image_baking] TEMP UV Map node created:{temp_uv_node.name} (uv_map={uv_name})"
            )

            # Connect the UV node's output to the Vector input of our TEMP Image Texture node
            links.new(temp_uv_node.outputs["UV"], temp_tex_node.inputs["Vector"])
            self.logger.debug(
                f"[setup_material_for_image_baking] Connected output {temp_uv_node.name} to input Vector node {temp_tex_node.name}"
            )

            # Make the TEMP Image Texture node active
            temp_tex_node.select = True
            material.node_tree.nodes.active = temp_tex_node
            self.logger.debug(
                f"[setup_material_for_image_baking] The node is set active: {temp_tex_node.name}"
            )

            # Write the flag; set the value to "image" so the check in is_image_based passes
            material["_bake_mode"] = "image"

            bpy.context.view_layer.update()
            self.logger.debug(
                "[setup_material_for_image_baking] The material has been successfully configured for image-based baking."
            )
            return (True, temp_tex_node)
        except Exception as e:
            self.logger.exception(f"[setup_material_for_image_baking] Exception: {e}")
            return (False, None)

    def setup_material_for_procedural_baking(
        self, material: bpy.types.Material, bake_image: bpy.types.Image, uv_name: str
    ) -> Tuple[bool, Optional[bpy.types.ShaderNodeTexImage]]:
        """
        [IMPROVED VERSION]
        Prepares a procedural or simple color material for baking.
        If the Principled BSDF node is missing, it is created, and its color
        is taken from material.diffuse_color.
        """
        self.logger.debug(
            f"Run setup_material_for_procedural_baking for the material: {material.name}"
        )
        try:
            if not material:
                return False, None

            # [CHANGED] Ensure the material has a node tree
            if not material.use_nodes:
                material.use_nodes = True

            nodes = material.node_tree.nodes
            links = material.node_tree.links

            # [CHANGED] Find the Principled BSDF or create it if not found
            principled_node = None
            for node in nodes:
                if node.type == "BSDF_PRINCIPLED":
                    principled_node = node
                    break

            if not principled_node:
                self.logger.debug(
                    f"Principled BSDF not found in '{material.name}', creating new one."
                )
                # Clear potential default nodes (e.g., just a Material Output)
                nodes.clear()

                output_node = nodes.new(type="ShaderNodeOutputMaterial")
                output_node.location = (200, 0)

                principled_node = nodes.new(type="ShaderNodeBsdfPrincipled")
                principled_node.location = (0, 0)

                links.new(
                    principled_node.outputs["BSDF"], output_node.inputs["Surface"]
                )

                # Set the color from diffuse_color, as this is a simple material
                principled_node.inputs[
                    "Base Color"
                ].default_value = material.diffuse_color

            # --- The rest of the logic remains the same ---

            # Create the bake node and UV map
            bake_node = nodes.new(type="ShaderNodeTexImage")
            bake_node.name = "TEMP_BAKE_" + bake_image.name
            bake_node.image = bake_image
            bake_node.location = (
                principled_node.location.x - 200,
                principled_node.location.y + 200,
            )

            uv_map_node = nodes.new(type="ShaderNodeUVMap")
            uv_map_node.name = "TEMP_UV_" + uv_name
            uv_map_node.uv_map = uv_name
            uv_map_node.location = (bake_node.location.x - 200, bake_node.location.y)

            links.new(uv_map_node.outputs["UV"], bake_node.inputs["Vector"])

            # Make the bake node active
            bake_node.select = True
            material.node_tree.nodes.active = bake_node

            self.logger.debug(
                f"Active baking unit: {material.node_tree.nodes.active.name}"
            )
            material["_bake_mode"] = "procedural"

            try:
                bpy.context.view_layer.update()
            except Exception as e:
                self.logger.warning(f"Error updating view_layer: {e}")

            self.logger.debug(
                f"Material '{material.name}' has been successfully configured for baking."
            )
            return True, bake_node

        except Exception as e:
            self.logger.exception(
                f"Error in setup_material_for_procedural_baking for {material.name}: {e}"
            )
            return False, None

    def configure_bake_settings(self, bake_type: Optional[str] = None) -> None:
        try:
            if bake_type is None:
                bake_type = self.diffuse_bake_mode
            self.logger.debug(f"Configuring bake settings for type: {bake_type}")
            scene = bpy.context.scene
            scene.render.engine = "CYCLES"
            scene.cycles.bake_type = bake_type
            if bake_type.upper() in ("DIFFUSE", "EMIT"):
                # Disable shadows for flat baking
                scene.render.bake.use_pass_direct = False
                scene.render.bake.use_pass_indirect = False
                self.logger.debug(
                    "Selected bake mode: DIFFUSE OR EMIT. Direct and Indirect passes DISABLED."
                )
            else:
                scene.render.bake.use_pass_direct = True
                scene.render.bake.use_pass_indirect = True
                self.logger.debug(
                    f"Selected bake mode: {bake_type.upper()}. Direct and Indirect passes ENABLED."
                )
            scene.render.bake.use_pass_color = True
            scene.cycles.samples = 256  # Set the number of samples for rendering
            scene.render.bake.margin = self.margin
            # If image-based baking is being performed (there is a source object and it's not procedural),
            # enable selected-to-active mode and use a cage with the specified offset.
            if (
                self.source_obj is not None
                and bake_type.upper() == self.diffuse_bake_mode
            ):
                scene.render.bake.use_selected_to_active = True
                scene.render.bake.use_cage = True
                scene.render.bake.cage_extrusion = self.cage_extrusion
                self.logger.debug(
                    f"Cage baking enabled with cage_extrusion={self.cage_extrusion}"
                )
            else:
                scene.render.bake.use_selected_to_active = False
                scene.render.bake.use_cage = False

            self.logger.debug("Bake settings configured successfully")
        except Exception as e:
            self.logger.exception(f"Error configuring bake settings: {e}")

    def bake_textures(self, obj: bpy.types.Object, bake_image: bpy.types.Image) -> bool:
        """
        Performs the actual texture baking.
        Ensures: the active object is in OBJECT mode, and the correct set of
        objects is selected for selected-to-active mode.
        """
        try:
            # ── 0. Activate the low-poly target ───────────────────
            bpy.context.view_layer.objects.active = obj
            if bpy.context.object and bpy.context.object.mode != "OBJECT":
                bpy.ops.object.mode_set(mode="OBJECT")

            # ── 1. Check visibility/render settings ────────────────
            for o in (obj, self.source_obj):
                if o and not o.hide_render:
                    continue
                if o:
                    self.logger.debug(f"{o.name}: hide_render=True → disabling it")
                    o.hide_render = False

            # ── 2. Form the correct selection ───────────────
            use_sta = bpy.context.scene.render.bake.use_selected_to_active
            bpy.ops.object.select_all(action="DESELECT")
            obj.select_set(True)  # target always
            if use_sta:
                if self.source_obj is None:
                    self.logger.error(
                        "selected-to-active noted, " "along with source_obj = None"
                    )
                    return False
                self.source_obj.select_set(True)  # hi-poly

            # ── 3. Active UV map ───────────────────────────
            if self.uv_name in obj.data.uv_layers:
                obj.data.uv_layers.active = obj.data.uv_layers[self.uv_name]

            # ── 4. Diagnostic log before bake ───────────────
            sel_names = [o.name for o in bpy.context.selected_objects]
            act_name = (
                bpy.context.view_layer.objects.active.name
                if bpy.context.view_layer.objects.active
                else "None"
            )
            act_mode = bpy.context.object.mode if bpy.context.object else "None"
            self.logger.debug(f"Selected objects: {sel_names}")
            self.logger.debug(f"Active: {act_name}, mode: {act_mode}")
            self.logger.debug("Calling bpy.ops.object.bake()…")

            # ── 5. Start the bake ────────────────────────────
            bpy.ops.object.bake(type=bpy.context.scene.cycles.bake_type)

            self.logger.debug("Bake finished OK")

            # ── 6. Quick check of the image and UV-bounds ───
            log_uv_bounds(obj, self.uv_name, self.logger)
            check_bake_image(bake_image, self.logger)
            return True

        except Exception as e:
            self.logger.exception(f"bake_textures: exception → {e}")
            return False

    def save_baked_image(self, image: bpy.types.Image, name: str) -> Optional[str]:
        try:
            self.logger.debug(f"Saving baked image {image.name} as {name}")
            # Flip the image vertically before saving:
            self.logger.debug("Calling flip_image_vertically to flip the image")
            flip_image_vertically(image)
            image.update()
            if not os.path.exists(self.output_dir):
                os.makedirs(self.output_dir)
                self.logger.debug(f"Created output directory: {self.output_dir}")
            file_path = os.path.join(
                self.output_dir, f"{name}.{self.texture_format.lower()}"
            )
            image.filepath_raw = file_path
            image.file_format = self.texture_format
            image.save()
            self.logger.debug(f"Image saved successfully to: {file_path}")
            return file_path
        except Exception as e:
            self.logger.exception(f"Error saving image {image.name}: {e}")
            return None

    def cleanup_materials(
        self,
        obj: bpy.types.Object,
        bake_nodes: List[Tuple[bpy.types.Material, bpy.types.Node]],
    ) -> None:
        """
        Restores:
        • original Base Color → BSDF connections;
        • initial Vector links of TEX_IMAGE nodes
            (saved in node["__orig_vector_link"]);
        • deletes all temporary nodes whose names start with 'TEMP_'.
        """
        try:
            for material, _ in bake_nodes:
                # ── 1. Base Color ────────────────────────────────────────────
                if material.get("_orig_base_color"):
                    try:
                        orig_node_name, orig_socket = material["_orig_base_color"]
                        bsdf = next(
                            (
                                n
                                for n in material.node_tree.nodes
                                if n.type == "BSDF_PRINCIPLED"
                            ),
                            None,
                        )
                        if bsdf:
                            # remove current links
                            for link in list(bsdf.inputs["Base Color"].links):
                                material.node_tree.links.remove(link)
                            # restore the old link
                            orig_node = material.node_tree.nodes.get(orig_node_name)
                            if orig_node:
                                material.node_tree.links.new(
                                    orig_node.outputs[orig_socket],
                                    bsdf.inputs["Base Color"],
                                )
                                self.logger.debug(
                                    f"[cleanup] {material.name}: "
                                    f"Base Color ← {orig_node_name}.{orig_socket}"
                                )
                        del material["_orig_base_color"]
                    except Exception as e:
                        self.logger.exception(
                            f"Error restoring Base Color in {material.name}: {e}"
                        )

                # ── 2. Vector links of TEX_IMAGE nodes ───────────────────────
                for node in material.node_tree.nodes:
                    if node.type == "TEX_IMAGE" and "__orig_vector_link" in node:
                        try:
                            from_node_name, from_socket_id = node["__orig_vector_link"]
                            from_node = material.node_tree.nodes.get(from_node_name)
                            from_socket = (
                                from_node.outputs.get(from_socket_id)
                                if from_node
                                else None
                            )

                            # remove "new" links
                            for link in list(node.inputs["Vector"].links):
                                material.node_tree.links.remove(link)

                            if from_node and from_socket:
                                material.node_tree.links.new(
                                    from_socket, node.inputs["Vector"]
                                )
                                self.logger.debug(
                                    f"[cleanup] {material.name}: "
                                    f"{node.name}.Vector ← {from_node_name}.{from_socket_id}"
                                )
                            else:
                                self.logger.warning(
                                    f"[cleanup] {material.name}: "
                                    f"failed to restore Vector link for {node.name}"
                                )
                        except Exception as e:
                            self.logger.exception(
                                f"Error restoring Vector link on {node.name}: {e}"
                            )
                        finally:
                            # remove the service tag in any case
                            if "__orig_vector_link" in node:
                                del node["__orig_vector_link"]

                # ── 3. Deleting temporary nodes ──────────────────────────
                for tmp in list(material.node_tree.nodes):
                    if tmp.name.startswith("TEMP_"):
                        try:
                            self.logger.debug(
                                f"[cleanup] {material.name}: deleting {tmp.name}"
                            )
                            material.node_tree.nodes.remove(tmp)
                        except Exception as e:
                            self.logger.exception(
                                f"Error removing node {tmp.name} from {material.name}: {e}"
                            )

            self.logger.debug(
                "cleanup_materials: all time units have been removed, "
                "original links updated"
            )
        except Exception as e:
            self.logger.exception(f"Error in cleanup_materials: {e}")

    def create_combined_material(
        self, obj: bpy.types.Object, baked_image: bpy.types.Image
    ) -> Optional[bpy.types.Material]:
        try:
            self.logger.debug(f"Creating combined material for object: {obj.name}")
            material_name = f"{obj.name}_Baked"
            if material_name in bpy.data.materials:
                material = bpy.data.materials[material_name]
                material.node_tree.nodes.clear()
            else:
                material = bpy.data.materials.new(name=material_name)
                material.use_nodes = True
            nodes = material.node_tree.nodes
            links = material.node_tree.links
            nodes.clear()
            # Create nodes:
            output_node = nodes.new(type="ShaderNodeOutputMaterial")
            output_node.location = (300, 0)
            bsdf_node = nodes.new(type="ShaderNodeBsdfPrincipled")
            bsdf_node.location = (0, 0)
            tex_node = nodes.new(type="ShaderNodeTexImage")
            tex_node.image = baked_image
            tex_node.location = (-300, 0)
            uv_node = nodes.new(type="ShaderNodeUVMap")
            uv_node.uv_map = self.uv_name
            uv_node.location = (-600, 0)
            # Connections:
            links.new(uv_node.outputs["UV"], tex_node.inputs["Vector"])
            links.new(
                tex_node.outputs.get("Color", tex_node.outputs[0]),
                bsdf_node.inputs["Base Color"],
            )
            links.new(bsdf_node.outputs["BSDF"], output_node.inputs["Surface"])
            # Clear node selection – this is critical:
            for node in nodes:
                node.select = False
            material.node_tree.nodes.active = None
            self.logger.debug(
                f"Created combined material: {material.name} (nodes cleared)"
            )
            return material
        except Exception as e:
            self.logger.exception(f"Error creating combined material: {e}")
            return None

    def apply_material_to_object(
        self, obj: bpy.types.Object, material: bpy.types.Material
    ) -> bool:
        self.logger.debug("Starting apply_material_to_object")
        try:
            self.logger.debug(
                f"Applying material '{material.name}' to object '{obj.name}'"
            )
            bpy.ops.object.select_all(action="DESELECT")
            obj.select_set(True)
            bpy.context.view_layer.objects.active = obj
            while len(obj.material_slots) > 1:
                try:
                    bpy.ops.object.material_slot_remove()
                    self.logger.debug("Removed extra material slot")
                except Exception as e:
                    self.logger.exception(f"Error deleting material slot: {e}")
            if len(obj.material_slots) == 0:
                try:
                    bpy.ops.object.material_slot_add()
                    self.logger.debug("Added new material slot")
                except Exception as e:
                    self.logger.exception(f"Error adding material slot: {e}")
            obj.material_slots[0].material = material
            self.logger.debug("Material assigned to the first slot successfully")
            return True
        except Exception as e:
            self.logger.exception(f"Error applying material to object: {e}")
            return False

    def setup_material_for_color_baking(
        self, material: bpy.types.Material, bake_image: bpy.types.Image, uv_name: str
    ) -> Tuple[bool, Optional[bpy.types.ShaderNodeTexImage]]:
        """
        Preparing a solid color material for baking.
        Enables use_nodes, sets up a Principled BSDF with color, then adds UV and Image nodes.
        """
        self.logger.debug(f"Run setup_material_for_color_baking: {material.name}")
        try:
            if not material.use_nodes:
                material.use_nodes = True

            if not material.node_tree:
                # If there is no tree even after enabling use_nodes, create it
                material.node_tree = bpy.data.node_groups.new(
                    name=f"{material.name}_nodes", type="ShaderNodeTree"
                )

            nodes = material.node_tree.nodes
            links = material.node_tree.links
            nodes.clear()

            output_node = nodes.new(type="ShaderNodeOutputMaterial")
            output_node.location = (200, 0)
            principled = nodes.new(type="ShaderNodeBsdfPrincipled")
            principled.location = (0, 0)
            principled.inputs["Base Color"].default_value = material.diffuse_color
            links.new(principled.outputs["BSDF"], output_node.inputs["Surface"])

            uv_map_node = nodes.new(type="ShaderNodeUVMap")
            uv_map_node.name = "TEMP_UV_" + uv_name
            uv_map_node.uv_map = uv_name
            uv_map_node.location = (-400, 200)

            bake_node = nodes.new(type="ShaderNodeTexImage")
            bake_node.name = "TEMP_BAKE_" + bake_image.name
            bake_node.image = bake_image
            bake_node.location = (-200, 200)

            links.new(uv_map_node.outputs["UV"], bake_node.inputs["Vector"])

            bake_node.select = True
            material.node_tree.nodes.active = bake_node
            material["_bake_mode"] = "color"

            bpy.context.view_layer.update()
            self.logger.debug(
                f"Material {material.name} is configured for color baking."
            )
            return True, bake_node
        except Exception as e:
            self.logger.exception(
                f"Error in setup_material_for_color_baking for {material.name}: {e}"
            )
            return False, None

    def setup_object_for_baking(
        self,
        obj: bpy.types.Object,
        skip_duplicate_uv: bool = True,
        active_uv: str = None,
    ) -> Tuple[
        Optional[bpy.types.Object],
        Optional[bpy.types.Image],
        List[Tuple[bpy.types.Material, bpy.types.Node]],
    ]:
        if not self.validate_object(obj):
            self.logger.error(f"Object {obj.name} failed validation")
            return None, None, []

        dup_obj = obj if skip_duplicate_uv else self.create_duplicate(obj)
        if not dup_obj:
            return None, None, []
        if not skip_duplicate_uv and not self.create_bake_uv_map(dup_obj):
            return None, None, []

        if active_uv and active_uv in dup_obj.data.uv_layers:
            dup_obj.data.uv_layers.active = dup_obj.data.uv_layers[active_uv]

        bake_image = self.create_bake_image(f"{dup_obj.name}_Baked")
        if not bake_image:
            return None, None, []

        bake_nodes: List[Tuple[bpy.types.Material, bpy.types.Node]] = []
        for slot in dup_obj.material_slots:
            if not slot.material:
                continue

            # [MODIFIED LOOP]
            # Determine which setup function to use
            if not slot.material.node_tree:
                # Case 1: Material without a node tree (solid color)
                success, node = self.setup_material_for_color_baking(
                    slot.material, bake_image, self.uv_name
                )
            elif self.is_image_based(slot.material):
                # Case 2: Texture-based material
                success, node = self.setup_material_for_image_baking(
                    slot.material, bake_image, self.uv_name
                )
            else:
                # Case 3: Procedural material (already has nodes, but no textures)
                success, node = self.setup_material_for_procedural_baking(
                    slot.material, bake_image, self.uv_name
                )

            if success and node:
                bake_nodes.append((slot.material, node))

        if not bake_nodes:
            self.logger.error("No valid materials set up for baking")
            return None, None, []

        return dup_obj, bake_image, bake_nodes

    def execute_baking(
        self, obj: bpy.types.Object, bake_image: bpy.types.Image, has_procedural: bool
    ) -> bool:
        try:
            scene = bpy.context.scene
            # If a source object is provided for image-based baking and there are no procedural materials,
            # enable selected-to-active mode with a cage (settings will be configured in configure_bake_settings).
            if not has_procedural and self.source_obj is not None:
                self.logger.debug("Image-based baking: enable selected-to-active mode")
                # Switch the active UV map on the source object
                if (
                    self.source_uv_map_name
                    and self.source_uv_map_name in self.source_obj.data.uv_layers
                ):
                    self.source_obj.data.uv_layers.active = (
                        self.source_obj.data.uv_layers[self.source_uv_map_name]
                    )
                    self.logger.debug(
                        f"The object {self.source_obj.name} has a UV map installed'{self.source_uv_map_name}'"
                    )
                else:
                    self.logger.warning(
                        f"UV map '{self.source_uv_map_name}' not found on object {self.source_obj.name}"
                    )
                bpy.ops.object.select_all(action="DESELECT")
                self.source_obj.select_set(True)
                obj.select_set(True)
                bpy.context.view_layer.objects.active = obj
            else:
                scene.render.bake.use_selected_to_active = False
                bpy.ops.object.select_all(action="DESELECT")
                obj.select_set(True)
                bpy.context.view_layer.objects.active = obj
            bake_type = (
                self.procedural_bake_mode if has_procedural else self.diffuse_bake_mode
            )
            self.logger.debug(
                f"The baking mode has been selected for the object {obj.name}: {bake_type}"
            )
            self.configure_bake_settings(bake_type)
            return self.bake_textures(obj, bake_image)
        except Exception as e:
            self.logger.exception(f"Error in execute_baking: {e}")
            return False

    def finalize_object(
        self,
        obj: bpy.types.Object,
        bake_image: bpy.types.Image,
        bake_nodes: List[Tuple[bpy.types.Material, bpy.types.Node]],
    ) -> Optional[bpy.types.Object]:
        try:
            file_path = self.save_baked_image(bake_image, f"{obj.name}_Baked")
            if not file_path:
                self.logger.warning("Failed to save image to disk")
            self.cleanup_materials(obj, bake_nodes)

            # Update bake_image before creating the material
            bake_image.update()
            self.logger.debug("Updated bake_image via image.update()")

            combined_material = self.create_combined_material(obj, bake_image)
            if not combined_material:
                self.logger.error("Failed to create combined material")
                return None
            if not self.apply_material_to_object(obj, combined_material):
                self.logger.error("Failed to apply combined material to object")
                return None
            self.logger.info(f"Object {obj.name} successfully finalized")
            return obj
        except Exception as e:
            self.logger.exception(f"Error in finalize_object: {e}")
            return None

    def process_object(self, obj: bpy.types.Object) -> Optional[bpy.types.Object]:
        self.logger.debug("Starting process_object")
        try:
            self.logger.info(f"Start baking for object: {obj.name}")
            dup_obj, bake_image, bake_nodes = self.setup_object_for_baking(obj)
            if not dup_obj or not bake_image or not bake_nodes:
                return None
            has_procedural = any(not self.is_image_based(mat) for mat, _ in bake_nodes)
            if not self.execute_baking(dup_obj, bake_image, has_procedural):
                self.logger.error(f"Baking failed for {dup_obj.name}")
                return None
            return self.finalize_object(dup_obj, bake_image, bake_nodes)
        except Exception as e:
            self.logger.exception(f"Error in process_object for {obj.name}: {e}")
            return None

    def process_selected_objects(self) -> List[bpy.types.Object]:
        try:
            selected_objects = [
                obj for obj in bpy.context.selected_objects if obj.type == "MESH"
            ]
            if not selected_objects:
                self.logger.error("No mesh objects selected")
                return []
            self.logger.info(f"Processing {len(selected_objects)} selected objects")
            processed: List[bpy.types.Object] = []
            for obj in selected_objects:
                result = self.process_object(obj)
                if result:
                    processed.append(result)
            self.logger.info(f"Successfully processed {len(processed)} objects")
            return processed
        except Exception as e:
            self.logger.exception(f"Error processing selected objects: {e}")
            return []

    def process_object_sequence(
        self, obj: bpy.types.Object, start_frame: int, end_frame: int
    ) -> bool:
        """
        During sequential baking, we bake BAKE_SEQUENCE_FRAME_COUNT frames,
        and the final material is created using the first frame.
        """
        # ⬇️ update globals right before work
        sync_bake_sequence_params(bpy.context.scene)
        self.logger.info(f"Starting sequence baking for object: {obj.name}")
        prep = self.setup_object_for_baking(obj)
        if not prep[0] or not prep[1] or not prep[2]:
            self.logger.error(f"Object {obj.name} failed sequence setup.")
            return False
        dup_obj, original_bake_image, bake_nodes = prep
        sequence_paths = []
        # Bake BAKE_SEQUENCE_FRAME_COUNT frames, starting from start_frame + 1
        frames_to_bake = [start_frame + i + 1 for i in range(BAKE_SEQUENCE_FRAME_COUNT)]
        for frame in frames_to_bake:
            try:
                bpy.context.scene.frame_set(frame)
                bpy.context.view_layer.update()
            except Exception as e:
                self.logger.exception(f"Error setting frame {frame}: {e}")
                continue
            frame_image_name = f"{dup_obj.name}_Baked_{frame:04d}"
            new_bake_image = self.create_bake_image(frame_image_name)
            if not new_bake_image:
                self.logger.error(f"Failed to create bake image for frame {frame}")
                continue
            for material, node in bake_nodes:
                node.image = new_bake_image
            has_procedural = any(not self.is_image_based(mat) for mat, _ in bake_nodes)
            bake_type = (
                self.procedural_bake_mode if has_procedural else self.diffuse_bake_mode
            )
            self.configure_bake_settings(bake_type)
            if not self.bake_textures(dup_obj, new_bake_image):
                self.logger.error(f"Baking failed for frame {frame}")
                bpy.data.images.remove(new_bake_image)
                continue
            saved_path = self.save_baked_image(new_bake_image, frame_image_name)
            if saved_path:
                self.logger.info(f"Frame {frame} saved to {saved_path}")
                sequence_paths.append(saved_path)
            else:
                self.logger.error(f"Failed to save frame {frame}")
            try:
                bpy.data.images.remove(new_bake_image)
            except Exception as e:
                self.logger.exception(
                    f"Error deleting temporary frame image {frame}: {e}"
                )
        self.cleanup_materials(dup_obj, bake_nodes)
        if sequence_paths:
            # Instead of creating a material with a sequence, use the first frame
            first_frame_path = sequence_paths[0]
            final_image = bpy.data.images.load(first_frame_path)
            combined_material = self.create_combined_material(dup_obj, final_image)
            if not combined_material:
                self.logger.error("Failed to create combined material from first frame")
                return False
            if not self.apply_material_to_object(dup_obj, combined_material):
                self.logger.error("Failed to apply combined material to object")
                return False
        else:
            self.logger.error("No frames were successfully baked in sequence.")
            return False
        return True

    def process_selected_objects_sequence(
        self, start_frame: int, end_frame: int
    ) -> None:
        selected_objects = [
            obj for obj in bpy.context.selected_objects if obj.type == "MESH"
        ]
        if not selected_objects:
            self.logger.error("No mesh objects selected for sequence baking")
            return
        self.logger.info(
            f"Sequence baking for {len(selected_objects)} selected objects, frames {start_frame} to {end_frame}"
        )
        for obj in selected_objects:
            self.process_object_sequence(obj, start_frame, end_frame)


def run_texture_baker(
    texture_size: int = TEXTURE_SIZE, margin: int = MARGIN
) -> List[bpy.types.Object]:
    try:
        scene = bpy.context.scene
        sync_bake_sequence_params(scene)
        logger.debug(
            f"Starting Texture Baker with size={texture_size}, margin={margin}"
        )
        baker = TextureBaker(
            texture_size=texture_size,
            margin=margin,
            uv_name=UV_NAME,  # type: ignore[name-defined]
            source_obj=source_obj,  # type: ignore[name-defined]
            source_uv_map_name=source_uv_map_name,  # type: ignore[name-defined]
            texture_format=TEXTURE_FORMAT,
            diffuse_bake_mode=DIFFUSE_BAKE_MODE,
            procedural_bake_mode=PROCEDURAL_BAKE_MODE,
        )
        processed = baker.process_selected_objects()
        if processed:
            logger.debug(f"Successfully processed {len(processed)} objects")
            return processed
        else:
            logger.warning("No objects were successfully processed")
            return []
    except Exception as e:
        logger.warning(f"Error running texture baker: {e}")
        return []


# ──────────────────────────────────────────────────────────
def check_bake_image(
    img: bpy.types.Image,
    log: logging.Logger,
    alpha_thr: float = 0.01,
    color_thr: float = 0.01,
) -> bool:
    """
    Checks the baking result:
      • opaque_cnt — pixels with α > alpha_thr
      • color_cnt  — opaque pixels where any of RGB > color_thr
    Returns True if "colored" pixels are found; otherwise False.
    """
    try:
        px = img.pixels[:]  # → tuple
        opaque_cnt = 0
        color_cnt = 0
        it = iter(px)
        for r, g, b, a in zip(it, it, it, it):
            if a > alpha_thr:
                opaque_cnt += 1
                if r > color_thr or g > color_thr or b > color_thr:
                    color_cnt += 1
        total = len(px) // 4
        log.debug(
            f"[bake‑check] opaque={opaque_cnt}  color={color_cnt}  " f"total={total}"
        )
        if opaque_cnt == 0 or color_cnt == 0:
            kind = "transparent" if opaque_cnt == 0 else "black"
            log.warning(
                f"[bake‑check] Image is {kind}: "
                "The UV islands might have been lost during the exchange or the passes were removed."
            )
            return False
        return True
    except Exception as e:
        log.exception(f"[bake‑check] exception during image analysis — {e}")
        return False


def is_multi_image_material(material: bpy.types.Material) -> bool:
    """Check how many TEX_IMAGE nodes are in the material (excluding temporary ones)."""
    count = sum(
        1
        for node in material.node_tree.nodes
        if node.type == "TEX_IMAGE" and not node.name.startswith("TEMP_BAKE_")
    )
    return count > 1


def combine_textures_for_bake(
    material: bpy.types.Material, bake_image: bpy.types.Image, uv_name: str
) -> Optional[bpy.types.Node]:
    """
    Creates a node chain for a multi-image material:
      TEMP_UV -> TEMP_BAKE_* for each map -> TEMP_MIX -> TEMP_EMIT
    and makes Emission active for baking.
    """
    nodes = material.node_tree.nodes
    links = material.node_tree.links

    # 1) UVMap
    uv_node = nodes.new(type="ShaderNodeUVMap")
    uv_node.name = f"TEMP_UV_{uv_name}"
    uv_node.uv_map = uv_name

    # 2) For each original map (TEX_IMAGE without TEMP_), create a TEMP_BAKE_...
    tex_nodes = [
        n for n in nodes if n.type == "TEX_IMAGE" and not n.name.startswith("TEMP_")
    ]
    bake_tex_nodes = []
    for orig in tex_nodes:
        tn = nodes.new(type="ShaderNodeTexImage")
        tn.name = f"TEMP_BAKE_{orig.name}"
        tn.image = bake_image
        links.new(uv_node.outputs["UV"], tn.inputs["Vector"])
        bake_tex_nodes.append(tn)

    if not bake_tex_nodes:
        return None

    # 3) MixRGB (multiply the first map by the second, if it exists)
    mix = nodes.new(type="ShaderNodeMixRGB")
    mix.name = f"TEMP_MIX_{uv_name}"
    mix.blend_type = "MULTIPLY"
    mix.inputs["Fac"].default_value = 1.0
    mix.location = (uv_node.location.x + 300, uv_node.location.y)
    links.new(bake_tex_nodes[0].outputs["Color"], mix.inputs["Color1"])
    if len(bake_tex_nodes) > 1:
        links.new(bake_tex_nodes[1].outputs["Color"], mix.inputs["Color2"])
    else:
        links.new(bake_tex_nodes[0].outputs["Color"], mix.inputs["Color2"])

    # 4) Emission
    emission = nodes.new(type="ShaderNodeEmission")
    emission.name = f"TEMP_EMIT_{uv_name}"
    emission.location = (mix.location.x + 300, mix.location.y)
    links.new(mix.outputs["Color"], emission.inputs["Color"])

    # 5) Make Emission active
    emission.select = True
    material.node_tree.nodes.active = emission

    # mark the bake mode
    material["_bake_mode"] = "multi_image"
    return emission


def _redirect_tex_images_to_uv(
    material: bpy.types.Material, uv_node: bpy.types.Node
) -> int:
    """
    Reconnects ALL standard ShaderNodeTexImage nodes (without the TEMP_ prefix)
    to the output of the `uv_node` UV node.

    Returns the number of reconfigured nodes.
    """
    counter = 0
    links = material.node_tree.links
    for node in material.node_tree.nodes:
        if node.type != "TEX_IMAGE" or node.name.startswith("TEMP_"):
            continue

        # Remove ALL existing links to the Vector input
        for link in list(node.inputs["Vector"].links):
            # Save data for restoration in custom-props
            node["__orig_vector_link"] = (
                link.from_node.name,
                link.from_socket.identifier,
            )
            links.remove(link)

        # Connect UV
        links.new(uv_node.outputs["UV"], node.inputs["Vector"])
        logger.debug(f"[multi-img] {material.name}: {node.name} → {uv_node.name}")
        counter += 1
    return counter


def setup_material_for_multi_image_bake(
    material: bpy.types.Material, bake_image: bpy.types.Image, uv_name: str
) -> Tuple[bool, Optional[bpy.types.ShaderNodeTexImage]]:
    """
    Minimal set of nodes for bake:
      TEMP_UV  →  TEMP_BAKE
    The original shader chain is not touched —
    colors are taken from the source object via selected-to-active.
    """
    try:
        if not (material and material.node_tree):
            logger.error("[multi-img] Material is None or has no node_tree.")
            return False, None

        nodes = material.node_tree.nodes
        links = material.node_tree.links

        uv_node = nodes.new(type="ShaderNodeUVMap")
        uv_node.name = f"TEMP_UV_{uv_name}"
        uv_node.uv_map = uv_name
        uv_node.location = (-600, 200)

        img_node = nodes.new(type="ShaderNodeTexImage")
        img_node.name = f"TEMP_BAKE_{bake_image.name}"
        img_node.image = bake_image
        img_node.location = (-300, 200)
        links.new(uv_node.outputs["UV"], img_node.inputs["Vector"])

        img_node.select = True
        nodes.active = img_node

        material["_bake_mode"] = "image_multi"  # now is_image_based() will pass
        bpy.context.view_layer.update()
        return True, img_node
    except Exception as e:
        logger.exception(f"[setup_multi_image] {material.name}: {e}")
        return False, None


if False and __name__ == "__main__":
    # If BAKE_SEQUENCE_FRAME_COUNT > 0, start sequential baking,
    # where the final material uses the first frame.
    if BAKE_SEQUENCE_FRAME_COUNT > 0:
        baker = TextureBaker(
            texture_size=TEXTURE_SIZE,
            margin=MARGIN,
            uv_name=UV_NAME,  # type: ignore[name-defined]
            source_obj=source_obj,  # type: ignore[name-defined]
            source_uv_map_name=source_uv_map_name,  # type: ignore[name-defined]
            texture_format=TEXTURE_FORMAT,
            diffuse_bake_mode=DIFFUSE_BAKE_MODE,
            procedural_bake_mode=PROCEDURAL_BAKE_MODE,
        )
        baker.process_selected_objects_sequence(
            BAKE_SEQUENCE_FRAME_START, BAKE_SEQUENCE_FRAME_END
        )
    else:
        run_texture_baker()


def register():
    logger.debug("[LOG] Registration texture_baker.py")
    pass


def unregister():
    logger.debug("[LOG] Unregistration texture_baker.py")
    pass
