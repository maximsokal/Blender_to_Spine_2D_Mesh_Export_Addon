# Architecture Overview

This document describes how the addon loads and how each module contributes to the export pipeline.

## Addon Registration
- [`__init__.py`](../Blender_to_Spine_2D_Mesh_Export_Addon/__init__.py#L123-L147)  
  Holds Blender's `bl_info` and aggregates all submodules in a `MODULES` tuple.  
  The `register()` function sets up logging via `config.setup_logging()` and registers each module in order,  
  while `unregister()` reverses the process.

## Configuration and Logging
- [`config.py`](../Blender_to_Spine_2D_Mesh_Export_Addon/config.py#L30-L69)  
  Defines global settings and custom properties.  
  The `setup_logging()` function configures the logging system used throughout the addon.  
  See also [`config.py`](../Blender_to_Spine_2D_Mesh_Export_Addon/config.py#L287-L298).

## User Interface
- [`ui.py`](../Blender_to_Spine_2D_Mesh_Export_Addon/ui.py)  
  Builds the panel in the 3D View and exposes operators for export.  
  The `OBJECT_OT_SaveUVAsJSON` button triggers the export pipeline defined in [`main.py`](../Blender_to_Spine_2D_Mesh_Export_Addon/main.py#L1487-L1569).

## Export Pipeline (`main.py`)
- [`main.py`](../Blender_to_Spine_2D_Mesh_Export_Addon/main.py#L1164-L1179)  
  The `save_uv_as_json` function orchestrates the workflow: preprocessing, UV creation, segmentation, baking, JSON export and merging.  
  It ends by saving the merged JSON file and logging completion.  
  See also [`main.py`](../Blender_to_Spine_2D_Mesh_Export_Addon/main.py#L1460-L1480).

## Mesh Segmentation
- [`plane_cut.py`](../Blender_to_Spine_2D_Mesh_Export_Addon/plane_cut.py#L256-L289)  
  Performs hybrid cutting of the mesh using UV islands and angle limits.  
  `execute_smart_cut()` returns boundary edges for later seam marking.

## UV Operations
- [`uv_operations.py`](../Blender_to_Spine_2D_Mesh_Export_Addon/uv_operations.py#L1-L35)  
  Handles UV creation and manipulation, including texel density equalisation via `rescale_uv_islands`.  
  See also [`uv_operations.py`](../Blender_to_Spine_2D_Mesh_Export_Addon/uv_operations.py#L206-L359) for baked UV transfer.

## Texture Baking
- [`texture_baker.py`](../Blender_to_Spine_2D_Mesh_Export_Addon/texture_baker.py#L1-L33)  
  Defines the `TextureBaker` class used to bake materials to textures.  
- [`texture_baker_integration.py`](../Blender_to_Spine_2D_Mesh_Export_Addon/texture_baker_integration.py#L1-L38)  
  Exposes `bake_textures_for_object()`, wrapping the baking process with configuration from `config.py`.  
  See also [`texture_baker_integration.py`](../Blender_to_Spine_2D_Mesh_Export_Addon/texture_baker_integration.py#L152-L205).

## JSON Export and Merging
- [`json_export.py`](../Blender_to_Spine_2D_Mesh_Export_Addon/json_export.py#L1-L39)  
  Converts mesh data to Spine JSON.  
- [`json_merger.py`](../Blender_to_Spine_2D_Mesh_Export_Addon/json_merger.py#L1-L32)  
  Combines the main object JSON with segment JSONs into one file and handles sequence data.

## Multi-Object Export
- [`multi_object_export.py`](../Blender_to_Spine_2D_Mesh_Export_Addon/multi_object_export.py#L1-L32)  
  Allows exporting several objects at once.  
  It groups connected meshes, calls `save_uv_as_json()` for each, then merges results.  
  See also [`multi_object_export.py`](../Blender_to_Spine_2D_Mesh_Export_Addon/multi_object_export.py#L502-L535).

## Directory Overview
- **Blender_to_Spine2D_Mesh_Export_Addon/** – addon source code.  
- **tests/** – unit tests covering config, main pipeline and exporters  
  (see [`tests/test_main_misc.py`](../tests/test_main_misc.py#L1-L5)).  
- **docs/** – user documentation; see [installation](installation.md) and [usage](usage.md).  
- **tools/** – helper scripts.
