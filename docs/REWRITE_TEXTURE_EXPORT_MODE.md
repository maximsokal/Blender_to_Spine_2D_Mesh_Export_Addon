# Rewrite texture export mode

Version 0.33 removes automatic topology replacement by camera projection.

## User-visible modes

The Rewrite Export panel owns one saved Scene setting:

```text
Export mode
  Normal — UV Segments
  Camera Projection
```

`Normal — UV Segments` is the default.

## Normal — UV Segments

Normal mode preserves the geometry pipeline:

```text
source Mesh
  -> angular/custom-seam segmentation
  -> disk-region decomposition
  -> shared generated SpineBakeUV layout
  -> per-region Spine mesh attachments
  -> Cycles object bake
```

The Scene may remain configured for EEVEE. Object baking temporarily selects Cycles inside the
existing reversible bake-scene transaction and restores the original Scene engine in `finally`.
Materials are analysed against the Cycles Material Output target because that is the actual
object-bake evaluator.

A camera-dependent material does not silently switch modes. Preparation fails with an explicit
instruction to select Camera Projection.

## Camera Projection

Camera Projection is selected only by the user:

```text
active-camera transparent render
  -> sequence coverage union
  -> crop and contour
  -> one screen-space projection attachment
```

A simple local material is still projected when this mode is selected explicitly.

## Capability policy

```text
Normal + LOCAL_UV_SAFE / SCENE_UV_SAFE
  -> object UV bake

Normal + CAMERA_RENDER_REQUIRED
  -> explicit error; no fallback

Camera Projection + supported material
  -> CameraProjectionPlan

GROUP_RENDER_REQUIRED / UNSUPPORTED
  -> explicit error
```

The shader capability audit remains diagnostic. It no longer chooses Camera Projection by
itself, and the active render engine is not a mode switch.

## Readiness

Changing Export Mode clears the cached readiness report and schedules one debounced automatic
analysis. Readiness remains diagnostic and does not disable the production export button.
