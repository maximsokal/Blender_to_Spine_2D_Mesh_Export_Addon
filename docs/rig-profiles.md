# Rig profiles implementation plan

## Goal

The exporter must expose rig generation as an explicit Scene-level category independent from texture, cutting, and baking settings.

Supported profiles:

- `LEGACY_ROTATABLE_MESH` — current three-axis rotation rig. This remains the default and must stay byte-compatible with existing exports.
- `TWO_AXIS_ROTATION_SCALE` — X/Y pseudo-rotation with an independent uniform scale control, generalized from the reference Spine 4.2.43 skeleton embedded below.

## UI

Add a dedicated **Rig** foldout between **Export** and **Cut**. It contains:

- Rig profile enum.
- Control icons toggle.
- Preview animation toggle.
- A short profile-specific description.

Changing the profile invalidates cached readiness and schedules a new analysis. Reset restores `LEGACY_ROTATABLE_MESH`.

## Settings pipeline

The selected profile must travel through the complete immutable path:

`Scene RNA -> _SceneExportProfile -> ExportSettings -> A1SingleObjectExportSettings -> rig builder router -> document assembly -> Spine JSON`.

Single-object and multi-object exports use one Scene-level profile. Mixed profiles inside one document are not supported in the first implementation.

## Builder architecture

Do not branch inside the existing legacy bone and constraint builders. Add an explicit router:

- `LEGACY_ROTATABLE_MESH -> build_legacy_rig`
- `TWO_AXIS_ROTATION_SCALE -> build_two_axis_scale_rig`

The new profile reuses the stable attachment contract: one generated vertex bone per exported mesh vertex, parented to its resolved Z-group rotation bone, with one full-weight influence. The reference `TOP1..TOP4` and `BOTTOM5..BOTTOM8` bones therefore generalize to the existing per-vertex bones and are not hard-coded.

## Two-axis hierarchy

Generalized hierarchy:

```text
root
├── <prefix>_main
│   ├── <prefix>
│   │   ├── <prefix>_scale_rotate_X
│   │   │   └── <prefix>_rotate_X
│   │   │       ├── <prefix>_1_scale
│   │   │       │   └── <prefix>_1
│   │   │       ├── <prefix>_2_scale
│   │   │       │   └── <prefix>_2
│   │   │       └── ... one pair per Z group
│   │   ├── <prefix>_rotate_X_constraint
│   │   └── <prefix>_rotate_X_constraint_scale_IK
│   │       └── <prefix>_rotate_X_constraint_rotate_IK
│   │           └── <prefix>_rotate_X_constraint_IK
│   ├── <prefix>_rotation_X
│   └── <prefix>_rotation_Y
└── <prefix>_scale
```

## Constraint order

The reference evaluation order must be preserved semantically:

1. X transform constraint (`order=0`).
2. IK (`order=1`).
3. Uniform scale transform (`order=2`).
4. X depth scale transform (`order=3`).
5. Y transform constraint (`order=4`).

The scale transform targets `<prefix>_rotate_X` and every Z-group rotation bone. The second profile does not generate a Z rotation control or the three-axis scale compensator.

## Generalization rules

The embedded JSON is a behavior reference, not a literal template:

- Names are always prefixed; global names such as `scale`, `IK`, and `null` are forbidden.
- Fixed model values (`500`, `600`, `818.36`, and similar) are converted to centralized layout ratios derived from texture dimensions and Z-group offsets.
- The number of Z groups and exported vertices is unrestricted.
- The existing three-axis profile remains unchanged.

## Visual controls and preview

- Three-axis profile: current X/Y/Z/main controls and current preview.
- Two-axis profile: X/Y/scale/main controls and a profile-specific preview covering X, Y, X+Y, scale, and rotation+scale.

## Validation

Required tests:

- Scene property registration, reset, readiness invalidation, and settings propagation.
- Exact two-axis bone and constraint order.
- Arbitrary Z-group counts and vertex counts.
- No Z rotation control in the two-axis profile.
- Scale target set contains the X owner and all Z-group rotation bones.
- Spine cross-reference validation and weighted-index validation.
- Existing three-axis golden output remains unchanged.
- Blender headless export and manual Spine 4.2.43 validation for X, Y, X+Y, scale, and X+Y+scale.

## Reference Spine 4.2.43 skeleton

This is the complete user-provided reference and must remain available in the repository for implementation and regression review.

```json
{
"skeleton": {
	"hash": "6b/QcQ/2H2s",
	"spine": "4.2.43",
	"x": -251.28,
	"y": -201.47,
	"width": 502.57,
	"height": 501.47,
	"images": "./Images/",
	"audio": ""
},
"bones": [
	{ "name": "root" },
	{ "name": "BOX_RIG", "parent": "root", "color": "faff00ff", "icon": "square" },
	{ "name": "null", "parent": "BOX_RIG", "color": "faff00ff" },
	{ "name": "SCALE_ROTATE_X", "parent": "null", "scaleX": 0 },
	{ "name": "ROTATE_X", "parent": "SCALE_ROTATE_X" },
	{ "name": "TOP_SCALE", "parent": "ROTATE_X", "rotation": 90, "y": 300, "inherit": "onlyTranslation" },
	{ "name": "TOP_ROTATION", "parent": "TOP_SCALE", "rotation": -90 },
	{ "name": "TOP1", "parent": "TOP_ROTATION", "rotation": 90, "x": 249.95, "y": 250.41 },
	{ "name": "TOP2", "parent": "TOP_ROTATION", "x": 249.95, "y": -250.46 },
	{ "name": "TOP3", "parent": "TOP_ROTATION", "rotation": -90, "x": -249.99, "y": -250.46 },
	{ "name": "TOP4", "parent": "TOP_ROTATION", "x": -249.99, "y": 250.49 },
	{ "name": "BOTTOM_SCALE", "parent": "ROTATE_X", "rotation": 90, "y": -201.47, "inherit": "onlyTranslation" },
	{ "name": "BOTTOM_ROTATION", "parent": "BOTTOM_SCALE", "rotation": -90 },
	{ "name": "BOTTOM5", "parent": "BOTTOM_ROTATION", "rotation": 90, "x": 249.95, "y": 250.41 },
	{ "name": "BOTTOM6", "parent": "BOTTOM_ROTATION", "x": 249.95, "y": -250.46 },
	{ "name": "BOTTOM7", "parent": "BOTTOM_ROTATION", "rotation": -90, "x": -249.99, "y": -250.46 },
	{ "name": "BOTTOM8", "parent": "BOTTOM_ROTATION", "x": -249.99, "y": 250.49 },
	{ "name": "ROTATE_Y_CTRL", "parent": "BOX_RIG", "length": 200, "rotation": -17.43, "x": 818.36, "color": "1aff00ff" },
	{ "name": "SCALE_ROTATE_X_CONTRAINT", "parent": "null", "length": 600, "rotation": 90, "color": "abe323ff" },
	{ "name": "SCALE_IK", "parent": "null", "y": 600, "scaleX": 0 },
	{ "name": "ROTATE_IK", "parent": "SCALE_IK", "x": -600 },
	{ "name": "IK", "parent": "ROTATE_IK", "x": 600, "color": "ff3f00ff", "icon": "ik" },
	{
		"name": "ROTATE_X_CTRL",
		"parent": "BOX_RIG",
		"length": 200,
		"rotation": -134.67,
		"x": 818.36,
		"y": 100,
		"color": "ff0000ff"
	},
	{ "name": "scale", "parent": "root", "x": 1124.14, "y": 59.89, "color": "abe323ff", "icon": "square" }
],
"slots": [
	{ "name": "BOX_FRONT", "bone": "TOP_ROTATION", "attachment": "BOX" },
	{ "name": "BOX_BACK", "bone": "BOTTOM_ROTATION", "attachment": "BOX2" },
	{ "name": "BOX_SIDE1", "bone": "null", "attachment": "BOX" },
	{ "name": "BOX_SIDE2", "bone": "null", "attachment": "BOX" },
	{ "name": "BOX_SIDE3", "bone": "null", "attachment": "BOX" },
	{ "name": "BOX_SIDE4", "bone": "null", "attachment": "BOX" }
],
"ik": [
	{
		"name": "IK",
		"order": 1,
		"bones": [ "SCALE_ROTATE_X_CONTRAINT" ],
		"target": "IK",
		"compress": true,
		"stretch": true
	}
],
"transform": [
	{
		"name": "ROTATE_X_CONSTRAINT",
		"bones": [ "ROTATE_IK", "ROTATE_X" ],
		"target": "ROTATE_X_CTRL",
		"local": true,
		"relative": true,
		"x": -500,
		"y": 500,
		"scaleX": -1,
		"mixX": 0,
		"mixScaleX": 0,
		"mixShearY": 0
	},
	{
		"name": "ROTATE_Y",
		"order": 4,
		"bones": [ "TOP_ROTATION", "BOTTOM_ROTATION" ],
		"target": "ROTATE_Y_CTRL",
		"local": true,
		"relative": true,
		"x": -500,
		"y": 300,
		"mixX": 0,
		"mixScaleX": 0,
		"mixShearY": 0
	},
	{
		"name": "scale",
		"order": 2,
		"bones": [ "ROTATE_X", "TOP_ROTATION", "BOTTOM_ROTATION" ],
		"target": "scale",
		"relative": true,
		"mixRotate": 0,
		"mixX": 0,
		"mixShearY": 0
	},
	{
		"name": "SCALE_ROTATE_X_CONSTRAINT",
		"order": 3,
		"bones": [ "BOTTOM_SCALE", "TOP_SCALE" ],
		"target": "SCALE_ROTATE_X_CONTRAINT",
		"rotation": -90,
		"x": -201.47,
		"scaleX": -1,
		"mixRotate": 0,
		"mixX": 0,
		"mixShearY": 0
	}
],
"skins": [
	{
		"name": "default",
		"attachments": {
			"BOX_BACK": {
				"BOX2": {
					"type": "mesh",
					"uvs": [ 0, 0, 1, 0, 1, 1, 0, 1 ],
					"triangles": [ 3, 0, 1, 3, 1, 2 ],
					"vertices": [ 1, 16, 499.99, 0.01, 1, 1, 13, 0.09, 499.95, 1, 1, 14, -499.95, -0.04, 1, 1, 15, 0.03, 499.99, 1 ],
					"hull": 4,
					"edges": [ 0, 2, 2, 4, 4, 6, 0, 6, 6, 2 ],
					"width": 500,
					"height": 501
				}
			},
			"BOX_FRONT": {
				"BOX": {
					"type": "mesh",
					"uvs": [ 0, 0, 1, 0, 1, 1, 0, 1 ],
					"triangles": [ 3, 0, 1, 3, 1, 2 ],
					"vertices": [ 1, 10, -0.01, 0.01, 1, 1, 7, 0.09, -0.05, 1, 1, 8, 0.05, -0.04, 1, 1, 9, 0.03, -0.01, 1 ],
					"hull": 4,
					"edges": [ 0, 2, 2, 4, 4, 6, 0, 6, 6, 2 ],
					"width": 500,
					"height": 501
				}
			},
			"BOX_SIDE1": {
				"BOX": {
					"type": "mesh",
					"uvs": [ 0, 0, 1, 0, 1, 1, 0, 1 ],
					"triangles": [ 3, 1, 2, 3, 0, 1 ],
					"vertices": [ 1, 9, 0.82, -0.01, 1, 1, 8, 0.05, -0.82, 1, 1, 14, 0.05, -0.35, 1, 1, 15, 0.35, -0.01, 1 ],
					"hull": 4,
					"edges": [ 0, 2, 2, 4, 4, 6, 0, 6, 6, 2 ],
					"width": 500,
					"height": 501
				}
			},
			"BOX_SIDE2": {
				"BOX": {
					"type": "mesh",
					"uvs": [ 0, 0, 1, 0, 1, 1, 0, 1 ],
					"triangles": [ 3, 0, 1, 3, 1, 2 ],
					"vertices": [ 1, 10, -1.3, -0.49, 1, 1, 9, -0.46, -1.3, 1, 1, 15, -0.47, -0.83, 1, 1, 16, -0.83, -0.49, 1 ],
					"hull": 4,
					"edges": [ 0, 2, 2, 4, 4, 6, 0, 6, 6, 2 ],
					"width": 500,
					"height": 501
				}
			},
			"BOX_SIDE3": {
				"BOX": {
					"type": "mesh",
					"uvs": [ 0, 0, 1, 0, 1, 1, 0, 1 ],
					"triangles": [ 3, 0, 1, 3, 1, 2 ],
					"vertices": [ 1, 7, 0.87, -0.05, 1, 1, 10, -0.01, 0.79, 1, 1, 16, -0.01, 0.32, 1, 1, 13, 0.41, -0.05, 1 ],
					"hull": 4,
					"edges": [ 0, 2, 2, 4, 4, 6, 0, 6, 6, 2 ],
					"width": 500,
					"height": 501
				}
			},
			"BOX_SIDE4": {
				"BOX": {
					"type": "mesh",
					"uvs": [ 0, 0, 1, 0, 1, 1, 0, 1 ],
					"triangles": [ 3, 0, 1, 3, 1, 2 ],
					"vertices": [ 1, 8, 1.33, 0.46, 1, 1, 7, -0.41, -1.33, 1, 1, 13, -0.41, -0.86, 1, 1, 14, 0.86, 0.46, 1 ],
					"hull": 4,
					"edges": [ 0, 2, 2, 4, 4, 6, 0, 6, 6, 2 ],
					"width": 500,
					"height": 501
				}
			}
		}
	}
],
"animations": {
	"rotate both": {
		"bones": {
			"ROTATE_X_CTRL": {
				"rotate": [
					{},
					{ "time": 2, "value": 360 }
				]
			},
			"ROTATE_Y_CTRL": {
				"rotate": [
					{},
					{ "time": 2, "value": 360 }
				]
			},
			"scale": {
				"scale": [
					{ "x": 0.981, "y": 0.981 }
				]
			}
		}
	},
	"rotate both2": {
		"bones": {
			"ROTATE_X_CTRL": {
				"rotate": [
					{},
					{ "time": 2, "value": 360 }
				]
			},
			"ROTATE_Y_CTRL": {
				"rotate": [
					{},
					{ "time": 2, "value": 360 }
				]
			},
			"scale": {
				"scale": [
					{
						"x": 0.75,
						"y": 0.75,
						"curve": [ 0.167, 0.75, 0.333, 1.5, 0.167, 0.75, 0.333, 1.5 ]
					},
					{
						"time": 0.5,
						"x": 1.5,
						"y": 1.5,
						"curve": [ 0.667, 1.5, 0.833, 0.75, 0.667, 1.5, 0.833, 0.75 ]
					},
					{
						"time": 1,
						"x": 0.75,
						"y": 0.75,
						"curve": [ 1.167, 0.75, 1.333, 1.5, 1.167, 0.75, 1.333, 1.5 ]
					},
					{
						"time": 1.5,
						"x": 1.5,
						"y": 1.5,
						"curve": [ 1.667, 1.5, 1.833, 0.75, 1.667, 1.5, 1.833, 0.75 ]
					},
					{ "time": 2, "x": 0.75, "y": 0.75 }
				]
			}
		}
	},
	"rotate x": {
		"bones": {
			"ROTATE_X_CTRL": {
				"rotate": [
					{},
					{ "time": 2, "value": 360 }
				]
			},
			"scale": {
				"scale": [
					{ "x": 2, "y": 2 }
				]
			}
		}
	},
	"rotate y": {
		"bones": {
			"ROTATE_Y_CTRL": {
				"rotate": [
					{},
					{ "time": 2, "value": 360 }
				]
			},
			"scale": {
				"scale": [
					{ "x": 2, "y": 2 }
				]
			}
		}
	}
}
}
```
