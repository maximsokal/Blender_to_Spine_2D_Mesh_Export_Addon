# Rewrite per-file pipeline audit and runtime trace

## Purpose

The Rewrite pipeline contains many small modules across `application`, `domain`,
`blender_adapter`, and `infrastructure`. Reviewing files manually does not prove that a
specific `.blend` actually reaches them or that data contracts remain stable between files.

This tooling provides two independent views:

1. a static AST audit of every production Python file;
2. a runtime trace of one real fixture export inside Blender.

Neither tool is imported by normal production exports. Diagnostics therefore add no runtime
overhead unless explicitly launched.

## Static audit

```bash
python tools/audit_a1_pipeline.py \
  --output-json test-results/pipeline-static-audit.json \
  --fail-on never
```

To inspect one file or module:

```bash
python tools/audit_a1_pipeline.py \
  --focus-module blender_adapter.a1_object_preparation \
  --output-json test-results/a1-object-preparation-static.json \
  --fail-on never
```

The audit checks every production file for:

- invalid dependency direction between layers;
- `bpy` or `bmesh` imported outside `blender_adapter`;
- `bpy.ops` calls inside loops;
- `bmesh.new()` without guaranteed `free()` in `finally`;
- likely double-free paths;
- bare or silent exception handlers;
- broad exceptions without logging or re-raise;
- direct filesystem mutation outside `infrastructure`;
- `print()` in production modules;
- oversized functions and files;
- mutable module-level state.

Findings are review signals, not automatic proof of a defect. Runtime trace data should be
used to prioritize hot-path findings.

## Runtime trace for one `.blend`

The runner consumes the existing fixture manifest schema. It makes a sibling copy of the
source `.blend` by default and never saves the source file.

```bash
python tools/run_a1_pipeline_probe.py \
  path/to/fixture-manifest.json \
  --case my-case \
  --blender "C:/Program Files/Blender Foundation/Blender 4.4/blender.exe" \
  --output-root test-results/pipeline-probe \
  --focus-module blender_adapter.a1_object_preparation \
  --overwrite
```

A concrete source file can be selected instead of a module name:

```bash
python tools/run_a1_pipeline_probe.py \
  path/to/fixture-manifest.json \
  --case my-case \
  --output-root test-results/pipeline-probe \
  --focus-file Blender_to_Spine2D_Mesh_Exporter/blender_adapter/a1_object_preparation.py \
  --overwrite
```

Add `--capture-values` only when scalar values and file paths are needed. Without it, the
report records types and shapes but not user data.

## Runtime report

`pipeline-trace-report.json` contains:

- every discovered production file;
- `executed`, `imported_not_called`, or `not_imported` status;
- function call and return counts;
- inclusive and self time;
- maximum call duration;
- bounded input/output type and shape signatures;
- exception events and exception types;
- cross-file call edges;
- hottest functions;
- mandatory route calls that were not observed;
- a chronological call/exception/return timeline for the selected focus file.

A file that was not executed is not automatically broken. It means the selected scenario did
not cover it. Use separate single, standalone multi, connected multi, mixed, local bake,
scene bake, and camera projection cases to build scenario coverage.

## Mandatory route contracts

The Blender probe derives a route from the fixture payload:

- `single`;
- `multi_standalone`;
- `multi_connected`;
- `mixed`.

For Rewrite, each route must reach its expected UI bridge, output service, preparation,
texture staging, projection finalization, composition where applicable, and atomic commit.
Missing mandatory calls make the probe fail even when an operator incorrectly reports
success.

## Output layout

```text
<output-root>/<case-id>/
  worker-payload.json
  blender.log
  pipeline-static-audit.json
  pipeline-trace-report.json
  pipeline-probe-summary.json
  exports/
```

## Reading one focused file

For a selected file, inspect:

1. `focus.matched_modules` — aggregate function metrics;
2. `focus.timeline` — exact call/exception/return order;
3. `focus.call_edges` — callers and callees across file boundaries;
4. `input_signatures` — data types/shapes entering the file;
5. `output_signatures` — data types/shapes leaving the file;
6. `exception_hotspots` — handled and propagated exception activity;
7. the matching module in `pipeline-static-audit.json`.

This makes it possible to isolate a single file without enabling DEBUG for the complete addon.
Per-file logging remains useful for semantic messages, while the trace supplies structural
and timing evidence.

## Performance policy

The tracer is diagnostics-only because Python tracing adds overhead. It disables line and
opcode events and records function boundaries only. Reports have bounded event and signature
limits. Normal exports do not import or activate the tracer.
