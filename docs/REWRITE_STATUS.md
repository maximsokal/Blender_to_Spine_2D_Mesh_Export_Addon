# Rewrite status

The active branch is `rewrite/a1-domain-foundation`; A1 targets Spine 4.2.43.
Rewrite remains the default backend. Legacy remains explicitly selectable and is never an
automatic fallback. The add-on version is unchanged and no release package has been produced.

## Production operators

```text
object.save_uv_as_json
object.spine2d_multi_export
```

Single, standalone multi, connected multi and mixed exports use one atomic JSON plus texture
transaction.

## Geometry and topology

Implemented:

- deterministic geometry and loop-level UV lineage;
- `LEGACY_SEED_CONE` compatibility behavior;
- `SEED_CONE_AND_LOCAL_DIHEDRAL` shared-edge guard;
- one immutable `DiskTopologyIndex` per mesh snapshot;
- incremental `DiskRegionState` growth;
- incremental frontier and merge adjacency updates;
- complete topology analysis only as independent input/final invariants.

## Automatic material pipeline

```text
reachable renderer-effective shader graph
    -> recursive Shader Node Group expansion
    -> semantic channels/dependencies
    -> ObjectBakeContext + SceneBakeContext
    -> automatic strategy selection
    |
    +-- LOCAL / AUXILIARY / SCENE
    |     -> DIFFUSE / EMIT / COMBINED object bake
    |     -> straight-RGBA composition
    |
    +-- CAMERA / VOLUME / render displacement
          -> B4 camera projection
```

Recursive group analysis supports renderer-specific outputs, muted bypasses, nested groups,
instance-qualified IDs, cycles, bounded depth and no source-node mutation.

## B4 production pipeline

Every B4 static or sequence export:

1. validates immutable object, Scene, World, camera, light and renderer context;
2. captures frame, render and camera-visibility state;
3. isolates only direct camera visibility while preserving dependency rays;
4. disables Scene Compositor and Sequencer execution without mutating their data;
5. renders a transparent staged frame;
6. decodes deterministic 8-bit alpha coverage;
7. max-unions coverage across the complete sequence;
8. applies hysteresis and conservative morphology;
9. derives one stable padded crop;
10. traces a simplified concave contour or safe convex fallback;
11. triangulates the simple contour exactly;
12. applies the resolved HDR/tone-mapping/alpha policy during crop rewrite;
13. rebuilds typed Spine attachments/documents after the final layout exists;
14. commits JSON and every texture together;
15. restores Blender state in `finally`.

## Simplified concave screen-space contour

The production default is `ProjectionContourMode.SIMPLIFIED_CONCAVE`.

- one outer component becomes a simple concave contour;
- internal holes remain texture alpha;
- disconnected outer components use a deterministic convex fallback;
- exact collinear vertices are removed;
- only shallow reflex notches may be filled;
- convex corners are never removed;
- convex contours retain the historical fan;
- concave contours use deterministic ear clipping;
- triangle count, orientation and exact total area are validated.

See `docs/REWRITE_B4_CONCAVE_CONTOUR.md`.

## Coverage-weighted antialias and morphology

Production uses `HYSTERESIS_MORPHOLOGY`:

- weak threshold defaults to `1 / 255`;
- strong threshold defaults to `0.5`;
- weak antialias coverage is retained only when connected to a strong core;
- translucent-only objects use an explicit weak-only fallback;
- foreground components use 8-connectivity;
- detached components smaller than two pixels are removed while the largest is always retained;
- only one-pixel enclosed pinholes are filled by default;
- no generic closing operation can bridge separate objects.

Pure binary callers retain an explicit compatibility mode.

See `docs/REWRITE_B4_COVERAGE_MORPHOLOGY.md`.

## Grouped connected B4 depth policy

`ConnectedB4RenderPolicy` supports:

- `INDIVIDUAL_LAYERS`;
- `AUTO_GROUPED_CAMERA` (default);
- `GROUPED_CAMERA_REQUIRED`.

A complete compatible connected B4 set is rendered together so Blender resolves real per-pixel
depth. Individual source visual slots remain in the typed document but are transparent; their
slot color/attachment timelines are removed so they cannot reappear. One root-bound grouped mesh
becomes the visible layer.

AUTO falls back for mixed B1-B3/B4 connected sets. REQUIRED fails explicitly. Mixed export applies
the grouped overlay inside the connected subgroup before composing standalone components.

The intentional tradeoff is flattening: source relative motion and depth changes must be baked in
the B4 sequence rather than expected from runtime source-bone movement.

See `docs/REWRITE_B4_GROUPED_CONNECTED.md`.

## Eevee and custom Compositor matrix

A separate manual-only Blender 4.4 matrix contains:

- real `BLENDER_EEVEE_NEXT` B4 export, renderer-specific Material Output selection, cropped PNG and
  attachment parity;
- a real destructive custom Compositor node tree proving that B4 disables Compositor/Sequencer
  execution during render, preserves `scene.use_nodes`, leaves the node tree unchanged and restores
  all flags afterward.

Workflow: `.github/workflows/blender-eevee-compositor-matrix.yml`.

## HDR, tone mapping and alpha representation

`ProjectionOutputPolicy` resolves by format:

```text
PNG / WEBP -> display-referred SDR -> Scene view transform -> straight alpha -> 8-bit
OPEN_EXR   -> scene-linear HDR      -> no tone mapping       -> premultiplied alpha -> 32-bit float
```

Invalid format/dynamic-range/tone-mapping combinations fail before render. Crop rewrite reads the
staged Blender `Image.alpha_mode`, converts straight/premultiplied RGB explicitly, normalizes
zero-alpha RGB and never clamps finite HDR RGB values.

A manual Blender matrix compares the real SDR PNG and scene-linear OPEN_EXR paths.

See `docs/REWRITE_B4_OUTPUT_POLICY.md`.

## Private production parity and release gate

Implemented:

- typed private manifest schema and pure validation tests;
- required capability coverage and minimum fixture count;
- strict-edge and animated-fixture rules;
- protected Blender runner for real `.blend` files;
- exact candidate SHA and Blender version checks;
- production operator invocation from manifest settings;
- source-file SHA and in-memory geometry/UV/state mutation detection;
- temporary datablock leak detection;
- semantic legacy/rewrite JSON comparison;
- accepted-warning and stale-suppression checks;
- Blender-decoded pixel parity for PNG/WEBP/OPEN_EXR;
- protected self-hosted manual workflow with no public fixture/report artifact upload.

See:

- `docs/REWRITE_PRIVATE_PRODUCTION_RELEASE_GATE.md`;
- `docs/private-release-manifest.example.json`;
- `.github/workflows/private-production-release-gate.yml`.

## Validation state

The last complete automatic matrix before workflows became manual-only passed:

- Python 3.10: **484 passed, 4 skipped**;
- Python 3.11: **484 passed, 4 skipped**;
- Blender 4.4 Alpha Bake: success;
- Blender 4.4 Scene Bake: success;
- Blender 4.4 Camera Projection: success;
- full Blender 4.4 Headless: success.

The current HEAD adds focused tests and manual Blender fixtures for all slices above. The complete
pytest suite and Blender matrices have not been rerun on the current HEAD. Automatic workflow
triggers remain disabled; all new Blender workflows use `workflow_dispatch`.

A local clone/compile attempt from this environment could not run because outbound access to
GitHub was unavailable. This does not count as validation.

## Remaining release blockers

Implementation of the requested ordered slices is complete. Release remains blocked until the same
candidate SHA passes:

1. the complete public Python suite;
2. all manual Blender 4.4 matrices;
3. the protected private production `.blend` release gate;
4. review of every retained private parity report and accepted warning;
5. restoration of intended CI triggers for the final candidate;
6. explicit approval for Legacy removal, version bump and packaging.

The branch and PR must remain draft until those gates pass.
