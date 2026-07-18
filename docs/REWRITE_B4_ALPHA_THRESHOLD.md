# B4 camera-projection alpha threshold

## Purpose

B4 derives one crop and one screen-space hull from the decoded alpha union of every
rendered frame. The alpha cutoff is an output-layout policy, not a material-strategy
switch.

The policy is exposed by the immutable execution contract:

```python
BakeExecutionSettings(
    projection_alpha_threshold=1.0 / 255.0,
)
```

## Compatibility default

The default remains exactly `1 / 255`. Existing callers that construct
`BakeExecutionSettings()` or omit execution settings therefore retain the previous B4
crop, hull, texture dimensions, and attachment geometry.

## Validation

`projection_alpha_threshold`:

- accepts finite numeric values in the inclusive range `[0, 1]`;
- rejects strings, `None`, booleans, NaN, infinities, and values outside the range;
- is applied identically to every static or sequence frame;
- is used both to decode each frame mask and to populate
  `CameraProjectionLayout.alpha_threshold`.

Using one immutable value for the complete sequence prevents frame-dependent crop drift.

## Output behavior

- `0.0` includes every decoded pixel, including fully transparent pixels, and normally
  produces a full-frame union.
- `1.0 / 255.0` preserves the historical behavior.
- Higher values ignore low-alpha antialias fringes and may produce a tighter crop/hull.
- `1.0` keeps only fully opaque pixels and may reject a translucent-only render as empty.

An all-transparent result after applying the selected threshold still fails before atomic
commit.

## Production bridge

The existing `BakeExecutionSettings` object already flows through single-object,
standalone multi-object, connected multi-object, and mixed exports. The A1 Blender bridge
uses the compatibility default when no additional property exists.

Automation may provide either an RNA attribute or a Blender ID custom property named:

```text
spine2d_projection_alpha_threshold
```

The bridge normalizes numeric values and rejects booleans or non-numeric values before
constructing the immutable execution settings. No per-material UI mode is introduced. A
future global advanced UI field may use the same property without changing B4 execution
or layout contracts.

## Regression coverage

Focused tests verify:

- the exact legacy default;
- an explicit custom threshold;
- rejection of invalid values;
- removal of the former executor constant;
- use of one execution threshold for staged-image mask extraction and union-layout
  metadata;
- single-object and every multi-object source receiving the same scene-level policy;
- missing scene properties preserving the old result.
