# Release checkpoint 0.47.10

## Scope

Version `0.47.10` is the limited Spine 4.1 compatibility candidate.

Accepted Spine 4.1 scope:

- exact Spine target `4.1.24`;
- `2-Axis Rotation + Scale` rig profile;
- single-object output;
- standalone multi-object output with independent object rigs under the shared `root`.

Still blocked for Spine 4.1:

- `3-Axis Rotation`;
- connected multi-object output;
- mixed standalone/connected output.

Spine 4.2.43 remains the primary target and keeps both rig profiles and existing composition modes.

## Scale fix

The accepted Spine 4.1 topology preserves the canonical Spine 4.2 scale roles:

- the uniform Scale constraint remains relative and world-space;
- only the unsafe `*_rotate_X` constrained driver is replaced with `*_scale_rotate_X`;
- the depth-scale constraint remains on the original `*_scale` wrapper bones;
- the original constrained-bone order is preserved;
- internal `onlyTranslation` bridge bones provide invertible parents without replacing authored zero scales;
- weighted attachment bone indices are remapped by bone identity after bridge insertion.

The rejected implementation added `local=true` to Scale and retargeted depth scale to final layer bones. It must not be restored.

## Evidence

The production Blender 5.2 standalone pipeline generated a Spine 4.1.24 JSON containing three independent object rigs. The exact Spine 4.1 runtime accepted the document with complete constraint scheduling, finite bone matrices, renderable attachments, and positive bounds.

Manual Spine Editor 4.1.24 testing confirmed that the corrected Scale control now scales the object correctly.

## Required release gates

The package build commit must pass:

1. focused Spine 4.1 pure-Python tests;
2. complete pure-Python suite;
3. complete real-`bpy` suite;
4. Blender-headless standalone acceptance;
5. `spine41_runtime_oracle.mjs`;
6. `spine41_scale_response_probe.mjs`;
7. Blender extension validation and build;
8. isolated install/enable/export/disable/uninstall smoke gate.

The external Spine runtime repository is read-only and is not included in the extension package.
