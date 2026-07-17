# Shader capability audit

## Purpose

`blender_adapter/shader_capability_audit.py` is a diagnostic boundary between reachable
Blender shader analysis and production texture routing. It does not alter
`build_texture_plan()` yet.

The audit answers a narrower question before any bake is selected:

```text
Can the current reconstructed UV-bake target reproduce every reachable value?
```

Each renderer-specific `MaterialGraphSnapshot` receives one strongest capability:

```text
LOCAL_UV_SAFE
SCENE_UV_SAFE
CAMERA_RENDER_REQUIRED
GROUP_RENDER_REQUIRED
UNSUPPORTED
```

Unknown reachable node types are deliberately `UNSUPPORTED`. New Blender nodes must receive an
explicit policy before a later routing slice may treat them as safe.

## Socket-level policies

Node type alone is insufficient for nodes such as Texture Coordinate and Geometry. The audit
uses `ShaderLinkSnapshot.from_socket` to classify only the outputs that actually contribute to
the renderer-effective Material Output.

Current Texture Coordinate policy:

| Output | Capability | Reason |
| --- | --- | --- |
| UV | `LOCAL_UV_SAFE` | reconstructed target carries export UV data |
| Camera | `CAMERA_RENDER_REQUIRED` | active camera coordinate system |
| Window | `CAMERA_RENDER_REQUIRED` | screen-space coordinate system |
| Reflection | `CAMERA_RENDER_REQUIRED` | view/reflection coordinate system |
| Object | `CAMERA_RENDER_REQUIRED` | original source/reference-object context |
| Generated | `CAMERA_RENDER_REQUIRED` | original undeformed source bounds |
| Normal | `CAMERA_RENDER_REQUIRED` | original source shading context |
| From Instancer | `GROUP_RENDER_REQUIRED` | instance context cannot be reconstructed locally |

Geometry outputs `Incoming`, `Backfacing`, `Pointiness`, and `Random Per Island` are also marked
as source/camera-bound.

## Explicit high-risk families

The audit currently records:

- Camera Data and Object Info as `CAMERA_RENDER_REQUIRED`;
- Shader to RGB as Eevee camera-render-only and unsupported for Cycles;
- Attribute and Vertex Color as camera-render-required until generic/color attributes are
  preserved on the bake target;
- Particle Info, Hair Info, Curves Info, and Point Density as `GROUP_RENDER_REQUIRED`;
- OSL Script as `UNSUPPORTED` until engine/device/source/compilation preflight exists;
- Volume and render displacement as `CAMERA_RENDER_REQUIRED`;
- incomplete recursive graph analysis as `UNSUPPORTED`;
- unclassified reachable node types as `UNSUPPORTED`.

## Deliberate non-routing status

Slice 4A is diagnostic only:

```text
analyse reachable graph
        -> capability audit report
        -> tests and logs

analyse reachable graph
        -> existing semantic dependencies
        -> existing build_texture_plan() routing
```

No production B1-B4 decision consumes the audit in this slice. This prevents a broad routing
change before Blender fixtures establish the exact engine and source-context behavior.

## Validation

Pure tests cover local Image Texture, socket-specific Texture Coordinate outputs, Camera Data,
Object Info, Shader to RGB renderer mismatch, Attribute, Particle Info, OSL, graph-analysis
issues, Volume, scene dependencies, and unknown future node types.

A manual-only Blender 4.4 fixture creates real nodes for:

- Texture Coordinate UV, Window, and From Instancer;
- Camera Data;
- Object Info Random;
- Attribute Color;
- Particle Info Age;
- Shader to RGB under Eevee and Cycles audit targets.

The fixture is wired into `.github/workflows/blender-camera-projection.yml`, whose only trigger
remains `workflow_dispatch`.

## Next slice

Slice 4B must establish one immutable renderer contract shared by:

1. the explicit export Scene;
2. renderer-specific Material Output selection;
3. capability auditing;
4. object-bake or camera-render execution.

Until that is implemented, the audit must remain diagnostic and must not automatically reroute
materials.
