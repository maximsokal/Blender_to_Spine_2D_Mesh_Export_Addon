# Private production `.blend` parity and release gate

## Purpose

Public synthetic fixtures cannot prove parity for real production scenes, customer material graphs,
modifier stacks, animation data, connected-object layouts or render configuration. The final release
gate therefore runs only on a protected self-hosted Blender 4.4 runner with a private manifest and
private fixture directory.

The public repository contains the runner and schema, but no customer `.blend`, texture, legacy
output or private report.

## Fail-closed behavior

The release workflow cannot pass when:

- no private manifest is supplied;
- the manifest lives in the public checkout;
- the protected runner labels are unavailable;
- the Blender version differs from the manifest;
- the checked-out SHA differs from the candidate SHA;
- fixture count or required capability coverage is incomplete;
- any real export, JSON parity, image parity or state-integrity check fails.

The gate is manual-only and is not part of ordinary development CI.

## Manifest

Use `docs/private-release-manifest.example.json` as the schema example. Private paths are relative
to the manifest directory. Generated rewrite paths are relative to a fresh temporary output directory.

Each fixture records:

- source `.blend`;
- frozen v0.23 legacy JSON;
- production operator (`object.save_uv_as_json` or `object.spine2d_multi_export`);
- active and selected objects;
- Scene RNA attributes and custom properties;
- object RNA attributes and custom properties;
- operator keyword arguments;
- expected rewrite JSON path;
- explicit legacy/rewrite image pairs and pixel tolerances;
- JSON tolerances and ignored paths;
- accepted warning codes;
- strict-edge and animation comparison policy;
- capability labels.

Supported placeholders in settings are:

```text
${OUTPUT_DIR}
${FIXTURE_DIR}
${MANIFEST_DIR}
```

The manifest-level gate declares minimum fixture count and required capabilities. Example capability
labels include:

```text
single
standalone_multi
connected
mixed
b4_cycles
b4_eevee
grouped_b4
custom_compositor
hdr_exr
animated
modifiers
sequences
```

Labels are deliberate release evidence, not inferred guesses.

## Per-fixture execution

`tests/blender_headless/run_private_production_release_gate.py` performs this pipeline:

1. verify manifest and exact Blender version;
2. verify the checkout SHA;
3. hash the source `.blend` on disk;
4. open the source with `load_ui=False`;
5. apply manifest settings and selection;
6. fingerprint selected objects, geometry, UV layers, modifiers, render state and context;
7. invoke the real production operator;
8. fingerprint runtime state again;
9. verify the source `.blend` SHA is unchanged;
10. detect new `__Spine2D_` temporary datablocks;
11. compare legacy and rewrite JSON using the existing semantic parity comparator;
12. reject every warning not explicitly accepted by code;
13. reject stale accepted warning codes that no longer occur;
14. decode legacy and rewrite images through Blender;
15. compare dimensions, RGB maximum/mean error and alpha maximum error;
16. retain file SHA-256 values in the report;
17. write one aggregate report bound to the candidate SHA.

The source file is never saved by the runner.

## Runtime mutation fingerprint

The in-memory fingerprint detects changes that a disk hash alone cannot see:

- active/selected objects and mode;
- current frame;
- render engine, dimensions, output path, transparency and postprocess flags;
- selected object transforms and visibility;
- modifier order/type/render state;
- mesh vertices, edges and polygons;
- material indices and smoothing;
- UV layer names, active layer and loop UV values;
- material slot names.

A production export that silently adds UV layers, changes geometry, selection, frame or render state
fails the gate even when the `.blend` file was not saved.

## JSON parity

The runner uses the existing `A1ParitySettings` and `compare_a1_exports()` implementation.
Release manifests normally require:

- strict mesh edges;
- animation comparison for animated fixtures;
- zero compatibility errors;
- no unaccepted warnings.

Ignored paths and accepted warning codes must be fixture-specific. Unused accepted warning codes fail
the gate, preventing permanent broad suppressions.

## Image parity

Images are decoded by the same Blender 4.4 runtime used for export, including PNG, WEBP and OPEN_EXR.
Every pair defines:

- maximum absolute RGB error;
- mean absolute RGB error;
- maximum absolute alpha error.

Dimensions must match exactly. The report also records expected/actual SHA-256 values and Blender
alpha modes.

Exact byte digests can be enforced with zero pixel tolerances for deterministic fixtures. Renderer
noise should use documented nonzero thresholds based on accepted production evidence.

## Protected workflow

`.github/workflows/private-production-release-gate.yml` runs only through `workflow_dispatch` on:

```text
[self-hosted, blender-4.4, a1-private-fixtures]
```

Inputs are:

- absolute private manifest path;
- absolute protected report path;
- Blender executable.

Private files and the report are intentionally not uploaded as public GitHub artifacts. The job
status and a sanitized summary are the only public workflow evidence.

## Release decision

A version bump, Legacy removal, merge to `main` or release package is prohibited until:

- all public pure tests pass;
- all public manual Blender matrices pass on the same candidate SHA;
- the private production gate passes on the same SHA;
- the protected report is reviewed and retained;
- every accepted warning has a documented fixture-specific reason.

## Current status

The schema, parser, pure validation tests, Blender runner and protected workflow are implemented.
No private production manifest or customer fixture is present in the public repository, so the final
release gate has not been executed and the branch must remain draft.
