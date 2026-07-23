# Blender to Spine2D Mesh Exporter

Blender extension for converting 3D mesh objects into Spine-compatible JSON,
meshes, UV data, textures, and generated rig structures.

## Requirements

- **Blender 5.2 or newer**
- Spine 4.2-oriented JSON output
- Windows is the currently tested desktop platform

Blender 4.x and Blender 5.0/5.1 are intentionally not supported by the current
Rewrite branch. The extension manifest and runtime registration gate both enforce
Blender 5.2+.

## Rewrite pipeline

The current pipeline separates Blender state from immutable application and
domain contracts:

1. source mesh capture and evaluated-geometry preparation;
2. topology validation, segmentation, and disk-region decomposition;
3. transactional UV unwrap on isolated temporary meshes;
4. material graph analysis and renderer capability routing;
5. Cycles object baking or camera-render projection;
6. Spine document assembly and atomic JSON/texture output;
7. deterministic cleanup and diagnostics on success or failure.

Source Blender objects, material graphs, UV layers, and color attributes are not
mutated by the Rewrite export path.

## Generated materials

When source materials are missing or explicitly ignored, the exporter can create
temporary opaque diagnostic materials:

- one solid gray color;
- one deterministic color per final segment;
- one deterministic color per final triangulated polygon.

Policies:

- `REQUIRE_SOURCE` — require usable source materials;
- `GENERATE_IF_MISSING` — generate only when required source slots are absent;
- `FORCE_GENERATED` — ignore source shading and use the selected generated pattern.

Generated materials use temporary Blender 5.2 node trees and a temporary CORNER
color attribute. All temporary materials, images, meshes, objects, and attributes
are removed after export, including failure paths.

## Render engines

- **Cycles** is used for semantic object baking.
- **Blender 5.2 EEVEE** uses the runtime identifier `BLENDER_EEVEE`.
- Material graphs that cannot be object-baked are routed through validated camera
  projection when that strategy is supported.
- Generated-material requests originating from an EEVEE scene use an internal
  transactional Cycles bake and restore the original EEVEE engine afterwards.

## Color management

The exporter records Blender 5.2's stable working-space interoperability ID from
`bpy.data.colorspace.working_space_interop_id` together with View Transform,
Look, Exposure, and Gamma. It does not overwrite the user's OCIO configuration.

## Installation

Use **Edit > Preferences > Extensions > Install from Disk** and select a release
ZIP. The archive root contains `blender_manifest.toml` and `__init__.py`.

See [docs/installation.md](docs/installation.md) for complete installation,
validation, and local build instructions.

## Building

The old selective legacy packager has been replaced. Build with Blender's
official extension CLI through:

```text
python tools/prepare_package.py --blender <path-to-Blender-5.2-executable>
```

The script validates the manifest and source before calling
`blender --command extension build`.

## Development and validation

The repository contains Blender-independent unit tests and Blender 5.2 headless
integration scripts. Important regression areas include:

- transactional register/unregister behavior;
- evaluated mesh lineage and modifier handling;
- UV unwrap and seam propagation;
- Cycles and EEVEE renderer routing;
- generated materials and cleanup;
- single, connected, standalone, and mixed multi-object exports;
- atomic output staging and rollback;
- camera projection and color-management capture.

CI/CD is not required to work on the Rewrite branch; Blender tests can be run
manually with a local Blender 5.2 installation.

## Project structure

```text
Blender_to_Spine2D_Mesh_Exporter/
  application/        Use-case orchestration
  blender_adapter/    bpy-facing state and execution boundary
  domain/             Blender-independent geometry, UV, baking, and Spine models
  infrastructure/     Registration, atomic output, diagnostics, and tracing
  blender_manifest.toml
  __init__.py

tests/
  blender_headless/   Real Blender integration regressions

tools/
  prepare_package.py  Official Blender 5.2 extension build wrapper
```

## License

GNU GPL v3.0 or later. Spine Editor and Spine runtimes remain subject to their
respective Esoteric Software licenses.
