# Phase 2 checkpoint: version-codec facade

Branch: `feature/spine-json-version-targets`  
Scope: serializer facade and unchanged Spine 4.2 production output  
CI/CD: intentionally not used

## Implemented

- Added a Blender-independent codec contract in `domain/spine/version_codecs/`.
- Added an immutable codec registry whose ready targets must exactly match
  `SpineJsonVersionDescriptor.serializer_ready` capabilities.
- Registered only `SPINE_4_2` as production ready.
- Implemented `Spine42JsonCodec` as a literal delegation to the existing
  `SpineSerializer` with the caller-selected validator and indent.
- Routed single-object, standalone/connected multi-object, and mixed-object JSON output
  through `serialize_spine_document()`.
- Preserved `ConnectedGroupSerializationValidator` for connected composition.
- Added one shared resolver that requires every source in a multi/mixed transaction to
  carry the same immutable target.
- Removed the obsolete `A1SingleObjectExportSettings` string comparison against
  `4.2.43`. Registered target settings can now cross the UI/application boundary, while
  `prepare_a1_object()` still rejects codecs that are not ready before geometry work.
- Legacy modules and CI/CD workflow files were not modified.

## Grill-with-docs corrections applied

### Incorrect checkpoint SHA

The earlier terminal block expected
`33e450ef946ca73d7220c17e98bdba8538b26a9c`, but that commit does not exist in the
repository. The feature branch was actually at
`9ebdc85a71038cfd47c721831eb01cacf4ac5fb1` before this Phase 2 slice.

### Preserve 4.2 animation and sequence behavior

The initial plan wording suggested rejecting all non-empty animation timelines and
sequence frame counts in Phase 2. That would contradict the required byte-identical
Spine 4.2 regression gate because the current production serializer already validates
and emits supported 4.2 animation and sequence payloads.

The implemented policy is therefore:

- Spine 4.2 delegates unchanged and preserves current behavior;
- Spine 3.8, 4.0, 4.1, and 4.3 remain blocked before geometry until their codecs own
  every required setup, constraint, sequence, and animation mapping;
- no codec may silently strip or guess unsupported data.

## Added tests

- `tests/test_spine_json_version_codecs.py`
  - exact string equality between the facade and current 4.2 serializer;
  - multiple indentation values;
  - attachment sequence preservation;
  - input-document immutability;
  - caller-selected validator forwarding;
  - fail-closed behavior for unready targets.
- `tests/test_a1_spine_version_output.py`
  - all registered targets cross the immutable request boundary;
  - one target across standalone and mixed sources;
  - deterministic mismatch diagnostics.
- `tests/test_spine_json_version_output_routing.py`
  - no direct `SpineSerializer` calls remain in A1 output modules;
  - every route uses the codec facade;
  - connected validation remains specialized.
- `tests/test_spine_json_version_preflight.py`
  - unready codecs fail at `VALIDATE_REQUEST` before geometry.

## Validation status

The files are committed to the feature branch. No GitHub Actions workflow was invoked.
Local compile, focused pytest, complete pure-Python pytest, and real-bpy suites must be
run from the developer workstation before the byte-identical 4.2 gate is considered
proven.

## Next implementation slice after a green local checkpoint

Implement the evidence-backed Spine 4.1 setup-pose codec only. Do not enable Spine 4.0,
3.8, or 4.3 in production serialization in the same slice.
