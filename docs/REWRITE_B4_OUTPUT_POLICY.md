# B4 HDR, tone-mapping, and alpha output policy

## Purpose

B4 now treats dynamic range, tone mapping and alpha representation as one typed output contract.
The policy is resolved before Blender renders, validated against the planned texture format, used
during crop rewrite and recorded in export statistics.

## Default compatibility policy

```python
BakeExecutionSettings(
    projection_output_policy=ProjectionOutputPolicy(),
)
```

The default uses `AUTO_BY_FORMAT`:

```text
PNG / WEBP
    -> DISPLAY_REFERRED_SDR
    -> SCENE_VIEW_TRANSFORM
    -> STRAIGHT alpha
    -> 8-bit non-float image

OPEN_EXR
    -> SCENE_LINEAR_HDR
    -> no tone mapping
    -> PREMULTIPLIED alpha
    -> 32-bit float image
```

JPEG remains invalid for B4 because camera projection requires alpha.

## Validation

The resolver rejects ambiguous or destructive combinations before render:

- scene-linear HDR in PNG or WEBP;
- display-referred SDR in OPEN_EXR;
- scene-linear HDR with a view transform;
- display-referred SDR with tone mapping disabled;
- any B4 JPEG output;
- unresolved or untyped policy values.

This prevents a render from succeeding and only failing later during crop or serialization.

## Tone mapping

`SCENE_VIEW_TRANSFORM` means the staged SDR render follows the analyzed Blender Scene color
management snapshot. B4 does not silently replace AgX, Standard, exposure, gamma or look.
The existing Scene-context validator detects changes between analysis and execution.

`NONE` is valid only for scene-linear HDR OPEN_EXR output. HDR values are not clipped to `[0, 1]`
during crop rewrite.

## Alpha representation

Blender render/compositor buffers commonly use associated, premultiplied alpha internally, while
runtime PNG/WEBP textures are conventionally consumed as straight RGBA. B4 now makes the target
representation explicit.

The crop rewrite:

1. reads the staged Blender image and its `Image.alpha_mode`;
2. crops the float RGBA buffer;
3. converts straight to premultiplied or premultiplied to straight;
4. clamps alpha only to `[0, 1]`;
5. leaves finite RGB values unbounded for HDR;
6. creates a float or byte image according to the resolved policy;
7. sets Blender `Image.alpha_mode` explicitly;
8. writes the final staged file.

For premultiplication:

```text
rgb_associated = rgb_straight * alpha
```

For unpremultiplication:

```text
rgb_straight = rgb_associated / alpha, when alpha > 0
rgb_straight = 0, when alpha == 0
```

Zero-alpha RGB is normalized to black when unpremultiplying, preventing undefined hidden color
from producing edge artifacts.

## HDR preservation

The alpha conversion never clamps RGB. A scene-linear EXR value such as `(8, 3, 2, 1)` remains
above one after crop. A partially transparent straight pixel is associated without losing HDR:

```text
(4, 2, 1, 0.25) straight -> (1, 0.5, 0.25, 0.25) premultiplied
```

## Coverage analysis

Geometry coverage still uses quantized 8-bit alpha. This is independent of RGB dynamic range and
keeps sequence memory bounded. The final texture may remain 32-bit float HDR while the crop and
contour use the conservative coverage policy.

## Statistics

Final B4 object statistics include:

- texture format;
- resolved dynamic range;
- resolved tone mapping;
- resolved alpha representation;
- color depth;
- float-buffer flag.

Grouped B4 requires identical `BakeExecutionSettings`, so every grouped source uses one output
policy.

## Validation matrix

Pure tests cover:

- automatic PNG/WEBP/EXR resolution;
- invalid format/dynamic-range/tone-mapping combinations;
- JPEG rejection;
- straight/premultiplied round trips;
- HDR values above one;
- zero-alpha normalization;
- non-finite pixel rejection;
- unknown Blender alpha modes.

The manual Blender output-policy workflow runs two real production exports:

1. an HDR-emission B4 material written to PNG, verifying display-referred SDR, straight alpha,
   crop and attachment parity;
2. the same class of material written to OPEN_EXR, verifying a float image, values above one,
   premultiplied alpha, crop and attachment parity.

The workflow remains manual-only and has not been run automatically on the current branch.
