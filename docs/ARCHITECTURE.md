# Architecture Overview

This document describes how the addon loads and how each module contributes to the export pipeline.

## Addon Registration
- `__init__.py` holds Blender's `bl_info` and aggregates all submodules in a `MODULES` tuple. The `register()` function sets up logging via `config.setup_logging()` and registers each module in order, while `unregister()` reverses the process【F:spine2d_mesh_exporter/__init__.py†L123-L147】.

## Configuration and Logging
- `config.py` defines global settings and custom properties. Its `setup_logging()` function configures the logging system used throughout the addon【F:spine2d_mesh_exporter/config.py†L30-L69】【F:spine2d_mesh_exporter/config.py†L287-L298】.

## User Interface
- `ui.py` builds the panel in the 3D View and exposes operators for export. The `OBJECT_OT_SaveUVAsJSON` button triggers the export pipeline defined in `main.py`【F:spine2d_mesh_exporter/main.py†L1487-L1569】.

## Export Pipeline (`main.py`)
- The `save_uv_as_json` function orchestrates the workflow: preprocessing, UV creation, segmentation, baking, JSON export and merging. It ends by saving the merged JSON file and logging completion【F:spine2d_mesh_exporter/main.py†L1164-L1179】【F:spine2d_mesh_exporter/main.py†L1460-L1480】.

## Mesh Segmentation
- `plane_cut.py` performs hybrid cutting of the mesh using UV islands and angle limits. `execute_smart_cut()` returns boundary edges for later seam marking【F:spine2d_mesh_exporter/plane_cut.py†L256-L289】.

## UV Operations
- `uv_operations.py` handles UV creation and manipulation, including texel density equalisation via `rescale_uv_islands` and transfer of baked UVs between objects【F:spine2d_mesh_exporter/uv_operations.py†L1-L35】【F:spine2d_mesh_exporter/uv_operations.py†L206-L359】.

## Texture Baking
- `texture_baker.py` defines the `TextureBaker` class used to bake materials to textures. `texture_baker_integration.py` exposes `bake_textures_for_object()` which wraps the baking process with configuration from `config.py`【F:spine2d_mesh_exporter/texture_baker.py†L1-L33】【F:spine2d_mesh_exporter/texture_baker_integration.py†L1-L38】【F:spine2d_mesh_exporter/texture_baker_integration.py†L152-L205】.

## JSON Export and Merging
- `json_export.py` converts mesh data to Spine JSON, while `json_merger.py` combines the main object JSON with any segment JSONs into one file and handles sequence data【F:spine2d_mesh_exporter/json_export.py†L1-L39】【F:spine2d_mesh_exporter/json_merger.py†L1-L32】.

## Multi-Object Export
- `multi_object_export.py` allows exporting several objects at once. It groups connected meshes, calls `save_uv_as_json()` for each, then merges results so all animations remain intact【F:spine2d_mesh_exporter/multi_object_export.py†L1-L32】【F:spine2d_mesh_exporter/multi_object_export.py†L502-L535】.

## Directory Overview
- **spine2d_mesh_exporter/** – addon source code.
- **tests/** – unit tests covering config, main pipeline and exporters【F:tests/test_main_misc.py†L1-L5】.
- **docs/** – user documentation; see [installation](docs/installation.md) and [usage](docs/usage.md).
- **tools/** – helper scripts.
