# B4 coverage-weighted antialias and conservative morphology

## Purpose

B4 previously converted every rendered frame directly into a binary alpha mask. That lost
fractional antialias information before the sequence union was complete and made isolated
low-alpha render noise indistinguishable from a real edge fringe.

The production pipeline now decodes every staged frame into deterministic 8-bit alpha coverage:

```text
rendered alpha 0.0 .. 1.0 -> coverage byte 0 .. 255
```

Every sequence frame is max-unioned into one fixed-size coverage buffer. Geometry cleanup runs
once after all frames succeed, so the complete sequence uses one stable policy and one stable
contour.

## Execution contract

The immutable production default is:

```python
BakeExecutionSettings(
    projection_alpha_threshold=1.0 / 255.0,
    projection_coverage_policy=ProjectionCoveragePolicy(
        mode=ProjectionCoverageMode.HYSTERESIS_MORPHOLOGY,
        core_alpha_threshold=0.5,
        minimum_component_pixels=2,
        maximum_hole_pixels=1,
    ),
)
```

`projection_alpha_threshold` is the weak/fringe threshold. The coverage policy adds the strong
core threshold and conservative morphology limits.

## Modes

### `BINARY_THRESHOLD`

Compatibility mode for callers that already provide a binary `0/1` mask. Every non-zero byte is
visible. The numeric alpha threshold is retained as layout metadata but is not applied a second
time.

The pure `ProjectionAlphaUnionAccumulator` and `build_sequence_union_layout()` keep this mode by
default so existing external callers preserve their previous result.

### `COVERAGE_THRESHOLD`

Directly classifies 8-bit coverage using `projection_alpha_threshold` without hysteresis. This is
available when fractional coverage should be thresholded but morphology should be effectively
disabled by setting component minimum to `1` and hole maximum to `0`.

### `HYSTERESIS_MORPHOLOGY`

Production default. Two coverage classes are built:

```text
weak coverage   >= projection_alpha_threshold
strong coverage >= core_alpha_threshold
```

Starting from strong pixels, an 8-connected flood keeps weak antialias pixels connected to the
strong object core. Detached weak noise is not admitted.

If a translucent-only object contains no strong pixels, the weak mask is retained instead of
rejecting the object. The final layout records this through
`coverage_used_weak_only_fallback`.

## Sequence union

For every pixel, the accumulator stores the maximum coverage seen in any rendered frame:

```text
union_coverage[pixel] = max(frame_0, frame_1, ..., frame_n)
```

Memory remains `O(width * height)` regardless of sequence length. Per-frame coverage is released
before the next render.

The same final cleaned mask drives:

- crop bounds;
- visible pixel count;
- component diagnostics;
- concave contour extraction;
- convex fallback;
- Spine mesh triangulation.

## Component cleanup

Foreground components use 8-connectivity so a one-pixel diagonal antialias stroke is not split
into unrelated specks.

Components smaller than `minimum_component_pixels` are removed, with one safety rule: the
largest component is always retained. A legitimate one-pixel object therefore cannot disappear
even when the configured minimum is larger than one.

This cleanup removes only detached components. It does not erode the retained silhouette and it
does not bridge separate objects.

## Hole cleanup

Transparent regions use 4-connectivity. A region is filled only when:

- it does not touch the image border;
- its size is not greater than `maximum_hole_pixels`.

The production default fills only a single-pixel enclosed pinhole. Open background regions and
larger intentional holes remain transparent.

No generic dilation/erosion closing is used. Such closing could connect unrelated limbs or
separate alpha islands and would violate the conservative geometry contract.

## Staged-image decoding

`camera_projection_image.py::read_staged_alpha_coverage()`:

- validates the rendered dimensions;
- reads the complete Blender pixel buffer;
- rejects non-finite alpha;
- clamps finite alpha to `[0, 1]`;
- quantizes with deterministic rounding to `[0, 255]`;
- removes the temporary Blender image in `finally`.

`read_staged_alpha_mask()` remains as a compatibility wrapper.

## Layout diagnostics

`CameraProjectionLayout` records:

- `coverage_mode`;
- `coverage_core_alpha_threshold`;
- raw non-zero coverage pixel count;
- strong pixel count;
- component counts before and after cleanup;
- removed detached-component pixel count;
- filled pinhole pixel count;
- weak-only fallback state;
- final visible pixel count.

The same values are copied into A1 export statistics and B4 executor logs.

## Validation

Focused tests cover:

- policy type/range validation;
- connected weak antialias fringe retention;
- detached weak-noise rejection;
- translucent-only fallback;
- opaque one-pixel speck removal beside a larger component;
- preservation of a standalone one-pixel object;
- diagonal foreground connectivity;
- one-pixel enclosed pinhole filling;
- absence of morphology bridges between valid components;
- binary-mask compatibility;
- direct coverage-threshold behavior;
- max coverage union across sequence frames;
- staged-image coverage quantization, clamping, non-finite rejection and cleanup;
- production executor wiring.

## Remaining boundary

Coverage is quantized to 8-bit before geometry analysis. This matches PNG/WEBP alpha precision and
keeps sequence memory bounded. OPENEXR/HDR alpha precision, tone mapping and premultiplied-alpha
output remain a later explicit output-policy slice.
