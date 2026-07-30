# Spine JSON target-version export plan

Status: active implementation plan  
Branch: `feature/spine-json-version-targets`  
Base: `rewrite/a1-domain-foundation` at `d51f1a3c1151ae873e3d85ec5365437f7b2b4dcc`

## Goal

Add one `Spine version` selector to the Export foldout and produce setup-pose JSON for these target families:

| UI target | Exact output version |
| --- | --- |
| Spine 3.8 | `3.8.99` |
| Spine 4.0 | `4.0.64` |
| Spine 4.1 | `4.1.24` |
| Spine 4.2 | `4.2.43` |
| Spine 4.3 | `4.3.23` |

The selected version must control the complete JSON schema. Changing only `skeleton.spine` is not accepted.

## Deliberate scope reduction: no preview animation

Preview animation export is disabled for this feature:

- the `Preview animation` label and toggle are hidden from the Export UI;
- persisted legacy values remain readable for `.blend` compatibility, but Scene capture must force `include_preview_animation=False`;
- version codecs do not translate preview timelines;
- non-empty animation timelines must fail closed until animation versioning is implemented explicitly;
- empty animation namespaces should be removed before target serialization.

This reduction covers only the generated preview animation. Texture-sequence export can still create real attachment timelines when frame count is greater than zero. Until a version codec owns those timelines, target-version preflight must reject non-zero sequence frame counts with a structured error instead of emitting a guessed format.

## Canonical pipeline

```text
Blender Scene RNA
  -> immutable Scene/Object export profiles
  -> ExportSettings
  -> canonical typed SpineDocument
  -> target-version serializer/codec
  -> JSON-compatible mapping
  -> atomic UTF-8 output
```

Geometry, UVs, weighted vertices, slots, skins, texture staging, multi-object composition, and connected placement remain version-independent. Version differences belong at the final JSON boundary unless a target schema cannot represent the current canonical constraint model.

## Reference captures

The complete user-provided multi-object captures are stored next to this plan:

- [`reference/Cone_plus_2_objects_spine_3.8.json`](reference/Cone_plus_2_objects_spine_3.8.json)
- [`reference/Cone_plus_2_objects_spine_4.0.json`](reference/Cone_plus_2_objects_spine_4.0.json)
- [`reference/Cone_plus_2_objects_spine_4.1.json`](reference/Cone_plus_2_objects_spine_4.1.json)
- [`reference/Cone_plus_2_objects_spine_4.2.json`](reference/Cone_plus_2_objects_spine_4.2.json)
- [`reference/Cone_plus_2_objects_spine_4.3.json`](reference/Cone_plus_2_objects_spine_4.3.json)

These are complete semantic JSON copies of the supplied captures and are immutable research references, not approved golden outputs. Formatting may be normalized for repository storage, so the inventory below records the byte length and SHA-256 of the original uploaded source files rather than promising repository byte identity.

### Reference inventory

| File | Stored `skeleton.spine` | Source bytes | Source SHA-256 | Bones | Slots | IK | Transform | Constraints | Animations |
| --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 3.8 | `3.8-from-4.0-from-4.1-from-4.2.43` | 21257 | `5a363eb8f8a95adf87de127dfce6e0886d21d8baeb7e902f24b5bcad49d7d41e` | 58 | 24 | 3 | 12 | 0 | 6 |
| 4.0 | `4.0-from-4.1-from-4.2.43` | 21170 | `ad5a7477d634282487ba8421c7912f4faefab60f7bcaac0211c003d470204171` | 58 | 24 | 3 | 12 | 0 | 6 |
| 4.1 | `4.1-from-4.2.43` | 21161 | `97dd56e209c2e8c2f19a28e9e7634094f6f40fdea014692738c07ed94d175e1c` | 58 | 24 | 3 | 12 | 0 | 6 |
| 4.2 | `4.2.43` | 21140 | `01dd688828bdfb1f85dffadfb8b3509d1584f1292451b679b8783da6b287271e` | 58 | 24 | 3 | 12 | 0 | 6 |
| 4.3 | `4.3.23` | 23215 | `7f1f113ab71abc2ad15d17f30d922df2e649c1f404532133a55365634fd2fcd3` | 58 | 24 | 0 | 0 | 0 | 6 |

### Important evidence limits

1. The 3.8, 4.0, and 4.1 captures contain conversion-lineage version strings rather than exact Editor version strings. They are useful schema observations, but exact target metadata must use the registry values above.
2. The 4.3 capture contains no `ik`, `transform`, or unified `constraints` section. It cannot prove 4.3 constraint conversion and must not be used as the sole 4.3 golden oracle.
3. All captures contain preview animations. Those sections are retained only as source evidence and are out of scope for the first target-version implementation.
4. The references show three objects in one JSON, which makes them the primary structural comparison source for standalone multi-object composition. Connected exports require separate generated fixtures because their shared global rig is not represented here.

## Directly observed schema differences

### Bones

- 3.8, 4.0, and 4.1 use `transform: "onlyTranslation"` in the supplied captures.
- 4.2 and 4.3 use `inherit: "onlyTranslation"`.
- The 4.3 capture normalizes zero-like `scaleX` values to `1.0E-5`; this behavior needs an explicit policy and must not be copied blindly into earlier versions.

### Transform constraints

- 3.8 uses legacy channel names such as `translateMix`, `scaleMix`, and `shearMix`.
- 4.0, 4.1, and 4.2 use split names such as `mixX`, `mixScaleX`, and `mixShearY`.
- The supplied 4.3 file has no constraint payload, so 4.3 mapping remains blocked pending a constraint-preserving fixture and official runtime/source verification.

### Rotation timelines

- The supplied 3.8 preview uses `angle`.
- The supplied 4.0-4.3 previews use `value`.
- Preview timelines are excluded from the first implementation.

## Version target model

Create one Blender-independent registry in `domain/spine/version_target.py`:

- `SpineJsonTarget` enum;
- immutable `SpineJsonVersionDescriptor`;
- exact version;
- family;
- UI label and description;
- capabilities such as legacy bone inheritance spelling, legacy constraint mix names, unified constraints, sequence support, and animation support.

No production module may branch on ad hoc `startswith()` or floating-point version comparisons.

## Implementation phases

### Phase 0 - freeze scope and preserve evidence

- [x] Create isolated feature branch.
- [x] Store this implementation plan.
- [x] Store all five complete multi-object JSON captures.
- [x] Hide the Preview animation UI row.
- [x] Force immutable Scene capture to disable preview animation even when an old `.blend` stores `True`.
- [x] Add focused UI/request-boundary tests.

Implementation is committed, but the focused tests and the complete regression suites have not yet been executed on this feature branch.

### Phase 1 - target registry and Blender UI

- [ ] Add `SpineJsonTarget` and descriptor registry.
- [ ] Register `Scene.spine2d_target_spine_version` as an EnumProperty.
- [ ] Default existing and new Scenes to Spine 4.2.
- [ ] Add the selector immediately after Export mode.
- [ ] Show the exact resolved version under the selector.
- [ ] Reset the selector to Spine 4.2.
- [ ] Capture the target exactly once in `_SceneExportProfile`.
- [ ] Propagate the exact version into `ExportSettings`.
- [ ] Reject unknown exact versions at the application contract boundary.

### Phase 2 - serializer facade and unchanged 4.2 output

- [ ] Add a version-codec interface and registry.
- [ ] Route single-, standalone multi-, and connected output through one facade.
- [ ] Keep the connected serialization validator active before target conversion.
- [ ] Implement Spine 4.2 by delegating to the current serializer.
- [ ] Prove existing 4.2 output is byte-identical before enabling any other target.
- [ ] Reject non-empty animation timelines and non-zero sequence frame counts.

### Phase 3 - Spine 4.1 and 4.0

- [ ] Implement only evidence-backed setup, bone, slot, skin, mesh, IK, and transform mappings.
- [ ] Convert bone inheritance spelling where required.
- [ ] Remove unsupported fields rather than guessing defaults.
- [ ] Preserve weighted bone indices after multi-object composition.
- [ ] Normalize target constraint orders when the target loader requires unique contiguous ordinals.
- [ ] Validate against official matching runtimes and Editor imports.

### Phase 4 - Spine 3.8

- [ ] Convert inheritance and legacy transform mix names.
- [ ] Keep array-shaped skins from the supplied capture.
- [ ] Preserve mesh topology, UVs, weighted vertex streams, hulls, and edges.
- [ ] Exclude all animation conversion from this phase.
- [ ] Validate with exact Spine 3.8.99 runtime and Editor.

### Phase 5 - Spine 4.3 research and codec

- [ ] Acquire a 4.3.23 fixture that preserves the same IK and transform controls.
- [ ] Prove how current 4.2 constraints map to the 4.3 unified constraint model.
- [ ] Define deterministic ordering for standalone and connected constraints.
- [ ] Prove setup-pose equivalence for both rig profiles.
- [ ] Fail closed for any constraint pattern without an exact mapping.
- [ ] Enable 4.3 in the UI only after these gates pass.

### Phase 6 - complete multi-object validation

For every enabled version and both rig profiles:

- [ ] single-object export;
- [ ] standalone multi-object export with at least three objects;
- [ ] connected multi-object export;
- [ ] same-Z-layer connected scheduling;
- [ ] weighted bone remapping;
- [ ] atomic rollback;
- [ ] source UV immutability;
- [ ] official runtime load;
- [ ] manual Spine Editor import.

## Fail-closed diagnostics

Use structured errors before texture files are committed:

- `SPINE_VERSION_UNSUPPORTED`
- `SPINE_VERSION_ANIMATION_UNSUPPORTED`
- `SPINE_VERSION_SEQUENCE_UNSUPPORTED`
- `SPINE_VERSION_CONSTRAINT_MAPPING_FAILED`
- `SPINE_VERSION_FIELD_MAPPING_FAILED`

A syntactically valid JSON file is not considered a successful export unless the target runtime accepts it and setup-pose invariants remain correct.

## Definition of done

The feature is complete only when selecting 3.8, 4.0, 4.1, 4.2, or 4.3 produces a target-native setup-pose JSON for single, standalone multi-object, and connected multi-object exports. Spine 4.2 must remain byte-identical to the current production output, and no target may silently drop constraints or emit guessed animation data.
