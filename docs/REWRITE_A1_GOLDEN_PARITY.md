# A1 golden parity workflow

## Purpose

The A1 rewrite may replace the legacy v0.23 production exporter only after the same
representative Blender scenes have been exported by both engines and the resulting
Spine JSON files satisfy the explicit compatibility policy in this document.

This gate is intentionally stricter than visual inspection and less brittle than
byte-for-byte JSON equality.

## Why raw JSON equality is not the gate

The following differences may be harmless and are therefore handled by policy:

- volatile skeleton metadata such as `hash`, `images`, and `audio`;
- floating-point representation inside a configured tolerance;
- nonessential mesh `edges`, which are warnings by default;
- animations while the first A1 setup-pose gate is being evaluated.

The following differences are compatibility errors:

- ordered bone names or parents;
- slot names or slot-to-bone references;
- IK and transform constraint names, order, bones, or targets;
- skin/slot/attachment paths;
- attachment type or texture path;
- UV vertex count or UV values outside tolerance;
- triangle indices, hull size, image dimensions, or sequence metadata;
- malformed weighted vertex streams;
- weighted influence count or bone-index changes;
- setup-pose transforms outside tolerance.

## Comparator

Run from the repository root:

```bash
python tools/compare_a1_exports.py \
  path/to/legacy-v0.23.json \
  path/to/rewrite-a1.json \
  --report-json path/to/parity-report.json
```

Exit codes:

- `0`: compatible under the selected policy;
- `1`: comparison completed and found compatibility errors;
- `2`: invalid arguments, unreadable files, or invalid JSON input.

Useful strict modes:

```bash
python tools/compare_a1_exports.py legacy.json rewrite.json \
  --strict-edges \
  --compare-animations \
  --absolute-tolerance 0.0001 \
  --relative-tolerance 0.000001
```

Additional volatile fields can be excluded deliberately:

```bash
python tools/compare_a1_exports.py legacy.json rewrite.json \
  --ignore-path 'skeleton.fps' \
  --ignore-path 'animations.preview*'
```

Every additional ignore pattern must be documented with the fixture. An ignore may
never be added only to make a failing case green.

## Required fixture record

Each real fixture should keep the following information together:

```text
fixture-name/
├── README.md
├── source.blend
├── legacy-v0.23.json
├── rewrite-a1.json
├── parity-report.json
├── legacy-images/
└── rewrite-images/
```

Do not commit private customer scenes or textures to a public repository without
explicit authorization. Private fixtures can run in a protected local or private CI
suite while sanitized derived fixtures are committed publicly.

`README.md` must record:

- Blender version;
- operating system when relevant;
- legacy addon commit/version;
- rewrite commit SHA;
- object names selected for export;
- every Scene/addon export property;
- material/image dependencies;
- expected number of segments and attachments;
- whether modifiers are read as original or evaluated geometry;
- the modifier lineage policy;
- whether animation and nonessential edges are strict;
- every accepted warning and its reason.

## Reproducible export procedure

1. Copy the source `.blend` before running either engine.
2. Open the first copy in the frozen Blender version with legacy v0.23 only.
3. Record all export properties and export the legacy JSON and images.
4. Open the second untouched copy in the same Blender version with the rewrite only.
5. Apply exactly the same semantic settings.
6. Export the rewritten JSON and images.
7. Run `tools/compare_a1_exports.py` and save `parity-report.json`.
8. Review every warning manually.
9. Compare generated image dimensions, frame counts, filenames, and alpha behavior.
10. Mark the fixture accepted only when the JSON report has zero errors and all image
    checks pass.

The two engines must never run sequentially on the same already-mutated `.blend`
copy. A legacy operator may leave selection, mode, UV layers, temporary objects, or
other state that changes the rewrite input.

## Minimum single-object fixture matrix

Before migrating the production operator, the suite should include at least:

1. a planar single-material mesh with one attachment;
2. a mesh whose global unwrap creates UV seam duplicates;
3. multiple source Z groups;
4. automatic angle segmentation;
5. custom seam segmentation;
6. a segment requiring deterministic manifold-disk decomposition;
7. a topology-preserving evaluated modifier stack;
8. a Mirror case under `ALLOW_SOURCE_DUPLICATION`;
9. an intentionally unsupported topology-generating modifier with the expected
   structured failure;
10. multiple material slots;
11. an image material;
12. a procedural material;
13. an image sequence or movie-backed sequence;
14. non-default object world translation;
15. non-square texture dimensions.

## Acceptance policy

A fixture passes the default setup-pose gate when:

- the comparator exits with code `0`;
- the report contains zero errors;
- every warning has an explicit fixture-specific explanation;
- the output image set has the expected filenames, dimensions, frame count, and
  usable alpha;
- the source `.blend` remains unchanged after the rewrite export;
- no temporary Blender datablocks remain.

For final production migration, rerun accepted fixtures with `--strict-edges`. Enable
`--compare-animations` for every fixture whose legacy export contains production
animation data.

## What the JSON comparator does not prove

The comparator does not establish pixel equality of baked images. Two JSON files can
be structurally identical while the bake differs because of color management,
material evaluation, frame selection, render engine, alpha handling, or image
packing.

Image parity therefore remains a separate gate. The first implementation should
record at least:

- image format;
- width and height;
- alpha presence;
- frame count and filename order;
- file digest for deterministic cases;
- an explicit visual or pixel-difference threshold for cases where exact bytes are
  not stable.

## Current status

The comparison infrastructure, machine-readable report, exit codes, weighted-stream
validation, and real Blender UV-seam export fixture are implemented. Representative
project `.blend` files and their actual v0.23 exports are still required before the
legacy production operator can be replaced.
