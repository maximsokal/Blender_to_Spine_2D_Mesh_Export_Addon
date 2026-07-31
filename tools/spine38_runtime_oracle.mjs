#!/usr/bin/env node

/** Validate one generated Spine 3.8 JSON with a read-only 3.8 runtime entry. */

import assert from 'node:assert/strict';
import { existsSync, readFileSync, statSync } from 'node:fs';
import { resolve } from 'node:path';
import { pathToFileURL } from 'node:url';

const EXPECTED_VERSION = '3.8.99';
const RENDERABLE_TYPES = new Set(['region', 'mesh', 'linkedmesh']);
const SCALE_RESPONSE_FACTOR = 1.25;
const RESPONSE_EPSILON = 1e-6;
const ZERO_SCALE_EPSILON = 1e-8;
const MATRIX_FIELDS = Object.freeze(['worldX', 'worldY', 'a', 'b', 'c', 'd']);

function fail(message, details = undefined) {
  const error = new Error(message);
  if (details !== undefined) error.details = details;
  throw error;
}

function isRecord(value) {
  return typeof value === 'object' && value !== null && !Array.isArray(value);
}

function record(value, path) {
  if (!isRecord(value)) fail(`${path} must be a JSON object`, { value });
  return value;
}

function array(value, path) {
  if (!Array.isArray(value)) fail(`${path} must be a JSON array`, { value });
  return value;
}

function finite(value, path) {
  if (typeof value !== 'number' || !Number.isFinite(value)) {
    fail(`${path} must be finite`, { value });
  }
  return value;
}

function nonEmptyString(value, path) {
  if (typeof value !== 'string' || !value.trim()) {
    fail(`${path} must be a non-empty string`, { value });
  }
  return value;
}

function inputFile(argument, label) {
  if (typeof argument !== 'string' || !argument.trim()) fail(`Missing ${label}`);
  let path = resolve(process.cwd(), argument);
  if (!existsSync(path)) fail(`${label} does not exist: ${path}`);
  if (statSync(path).isDirectory()) path = resolve(path, 'index.js');
  if (!existsSync(path) || !statSync(path).isFile()) {
    fail(`${label} is not a file: ${path}`);
  }
  return path;
}

function mutableVector2() {
  return {
    x: 0,
    y: 0,
    set(x, y) {
      this.x = x;
      this.y = y;
      return this;
    },
  };
}

function attachmentType(attachment) {
  return typeof attachment.type === 'string' ? attachment.type : 'region';
}

function collectAtlasRegions(document) {
  const result = new Set();
  const skins = array(document.skins ?? [], 'document.skins');
  for (let skinIndex = 0; skinIndex < skins.length; skinIndex += 1) {
    const skin = record(skins[skinIndex], `document.skins[${skinIndex}]`);
    const groups = record(
      skin.attachments ?? {},
      `document.skins[${skinIndex}].attachments`,
    );
    for (const [slotName, slotValue] of Object.entries(groups)) {
      const attachments = record(
        slotValue,
        `document.skins[${skinIndex}].attachments.${slotName}`,
      );
      for (const [entryName, rawAttachment] of Object.entries(attachments)) {
        const attachment = record(
          rawAttachment,
          `document.skins[${skinIndex}].attachments.${slotName}.${entryName}`,
        );
        if ('sequence' in attachment) {
          fail('Spine 3.8 setup attachments cannot contain sequence data');
        }
        if (!RENDERABLE_TYPES.has(attachmentType(attachment))) continue;
        result.add(
          typeof attachment.path === 'string' && attachment.path
            ? attachment.path
            : entryName,
        );
      }
    }
  }
  return [...result].sort();
}

function expectedSetupAttachments(document) {
  const slots = array(document.slots ?? [], 'document.slots');
  const skins = array(document.skins ?? [], 'document.skins');
  const defaultSkin = skins.find(
    (value) => isRecord(value) && (value.name ?? 'default') === 'default',
  );
  if (!isRecord(defaultSkin)) return [];
  const groups = record(defaultSkin.attachments ?? {}, 'defaultSkin.attachments');
  const result = [];
  for (let slotIndex = 0; slotIndex < slots.length; slotIndex += 1) {
    const slot = record(slots[slotIndex], `document.slots[${slotIndex}]`);
    const slotName = nonEmptyString(slot.name, `document.slots[${slotIndex}].name`);
    if (slot.attachment === undefined || slot.attachment === null) continue;
    const attachmentName = nonEmptyString(
      slot.attachment,
      `document.slots[${slotIndex}].attachment`,
    );
    const attachment = record(
      record(groups[slotName], `defaultSkin.attachments.${slotName}`)[attachmentName],
      `defaultSkin.attachments.${slotName}.${attachmentName}`,
    );
    const type = attachmentType(attachment);
    if (RENDERABLE_TYPES.has(type)) result.push({ slotName, attachmentName, type });
  }
  return result;
}

function atlasText(regions) {
  const header = [
    'oracle.png',
    'size: 1,1',
    'format: RGBA8888',
    'filter: Linear,Linear',
    'repeat: none',
  ];
  const body = regions.flatMap((name) => [
    name,
    '  rotate: false',
    '  xy: 0, 0',
    '  size: 1, 1',
    '  orig: 1, 1',
    '  offset: 0, 0',
    '  index: -1',
  ]);
  return [...header, ...body, ''].join('\n');
}

function oracleTexture() {
  const image = Object.freeze({ width: 1, height: 1 });
  return {
    setFilters() {},
    setWraps() {},
    getImage() {
      return image;
    },
    dispose() {},
  };
}

function constraintRecords(document) {
  const result = [];
  for (const collection of ['ik', 'transform', 'path']) {
    const values = array(document[collection] ?? [], `document.${collection}`);
    for (let index = 0; index < values.length; index += 1) {
      const constraint = record(values[index], `document.${collection}[${index}]`);
      const name = nonEmptyString(
        constraint.name,
        `document.${collection}[${index}].name`,
      );
      const order = constraint.order ?? 0;
      if (!Number.isInteger(order) || order < 0) {
        fail(`document.${collection}[${index}].order must be non-negative`);
      }
      if (collection === 'transform') {
        for (const legacyField of [
          'rotateMix',
          'translateMix',
          'scaleMix',
          'shearMix',
        ]) {
          finite(constraint[legacyField], `document.transform[${index}].${legacyField}`);
        }
        for (const forbidden of [
          'mixRotate',
          'mixX',
          'mixY',
          'mixScaleX',
          'mixScaleY',
          'mixShearY',
        ]) {
          if (forbidden in constraint) {
            fail(`Spine 4.x mix field leaked into 3.8: ${forbidden}`);
          }
        }
      }
      result.push({ collection, name, order });
    }
  }
  const orders = result.map((item) => item.order);
  assert.equal(new Set(orders).size, orders.length, 'Constraint orders must be unique');
  assert.deepEqual(
    [...orders].sort((a, b) => a - b),
    Array.from({ length: orders.length }, (_, index) => index),
    'Constraint orders must form 0..N-1',
  );
  return result;
}

function scaleControlRecords(document) {
  const transforms = array(document.transform ?? [], 'document.transform');
  const result = [];
  for (let index = 0; index < transforms.length; index += 1) {
    const constraint = record(transforms[index], `document.transform[${index}]`);
    const name = nonEmptyString(
      constraint.name,
      `document.transform[${index}].name`,
    );
    if (!name.endsWith('_scale_constraint')) continue;

    const target = nonEmptyString(
      constraint.target,
      `document.transform[${index}].target`,
    );
    if (!target.endsWith('_scale')) {
      fail(`Scale control constraint '${name}' has an unexpected target`, { target });
    }

    const rotateMix = finite(
      constraint.rotateMix,
      `document.transform[${index}].rotateMix`,
    );
    const translateMix = finite(
      constraint.translateMix,
      `document.transform[${index}].translateMix`,
    );
    const scaleMix = finite(
      constraint.scaleMix,
      `document.transform[${index}].scaleMix`,
    );
    const shearMix = finite(
      constraint.shearMix,
      `document.transform[${index}].shearMix`,
    );
    if (scaleMix <= 0 || rotateMix !== 0 || translateMix !== 0 || shearMix !== 0) {
      fail(`Scale control constraint '${name}' is not scale-only`, {
        rotateMix,
        translateMix,
        scaleMix,
        shearMix,
      });
    }

    const bones = array(
      constraint.bones,
      `document.transform[${index}].bones`,
    ).map((value, boneIndex) =>
      nonEmptyString(value, `document.transform[${index}].bones[${boneIndex}]`),
    );
    if (bones.length === 0) {
      fail(`Scale control constraint '${name}' has no constrained bones`);
    }
    if (new Set(bones).size !== bones.length) {
      fail(`Scale control constraint '${name}' repeats constrained bones`, { bones });
    }

    result.push(Object.freeze({ name, target, bones: Object.freeze(bones) }));
  }

  const names = result.map((item) => item.name);
  const targets = result.map((item) => item.target);
  assert.equal(new Set(names).size, names.length, 'Scale control names must be unique');
  assert.equal(new Set(targets).size, targets.length, 'Scale control targets must be unique');
  return Object.freeze(result);
}

function runtimeConstraints(skeleton) {
  return [
    ...(skeleton.ikConstraints ?? []),
    ...(skeleton.transformConstraints ?? []),
    ...(skeleton.pathConstraints ?? []),
  ];
}

function validateUpdateCache(skeleton, expected) {
  const constraints = runtimeConstraints(skeleton);
  const expectedNames = expected.map((item) => item.name).sort();
  const runtimeNames = constraints.map((item) => item.data.name).sort();
  assert.deepEqual(runtimeNames, expectedNames, 'Runtime constraint inventory differs');
  if (!Array.isArray(skeleton._updateCache)) fail('Runtime _updateCache is missing');
  const constraintSet = new Set(constraints);
  const cached = skeleton._updateCache.filter((item) => constraintSet.has(item));
  assert.equal(cached.length, constraints.length, 'Runtime skipped constraints');
  const counts = new Map();
  for (const item of cached) {
    const name = item.data.name;
    counts.set(name, (counts.get(name) ?? 0) + 1);
  }
  for (const name of expectedNames) {
    assert.equal(counts.get(name), 1, `Constraint '${name}' cache count differs`);
  }
  return cached.length;
}

function validateMatrices(skeleton) {
  for (let index = 0; index < skeleton.bones.length; index += 1) {
    const bone = skeleton.bones[index];
    for (const [field, value] of Object.entries({
      worldX: bone.worldX,
      worldY: bone.worldY,
      a: bone.a,
      b: bone.b,
      c: bone.c,
      d: bone.d,
      ax: bone.ax,
      ay: bone.ay,
      arotation: bone.arotation,
      ascaleX: bone.ascaleX,
      ascaleY: bone.ascaleY,
    })) {
      finite(value, `runtime.bones[${index}](${bone.data.name}).${field}`);
    }
  }
}

function runtimeSetupAttachments(runtime, skeleton) {
  const result = [];
  for (let index = 0; index < skeleton.drawOrder.length; index += 1) {
    const slot = skeleton.drawOrder[index];
    const attachment = slot.getAttachment();
    if (!attachment) continue;
    let type = null;
    if (attachment instanceof runtime.RegionAttachment) type = 'region';
    else if (attachment instanceof runtime.MeshAttachment) type = 'mesh';
    else continue;
    result.push({ slotName: slot.data.name, attachmentName: attachment.name, type });
  }
  return result;
}

function validateAttachments(runtime, skeleton, expected) {
  const actual = runtimeSetupAttachments(runtime, skeleton);
  const key = (item) => `${item.slotName}\u0000${item.attachmentName}`;
  assert.deepEqual(actual.map(key).sort(), expected.map(key).sort());
  return actual;
}

function bounds(skeleton) {
  const offset = mutableVector2();
  const size = mutableVector2();
  skeleton.getBounds(offset, size);
  const result = {
    x: finite(offset.x, 'bounds.x'),
    y: finite(offset.y, 'bounds.y'),
    width: finite(size.x, 'bounds.width'),
    height: finite(size.y, 'bounds.height'),
  };
  if (result.width <= 0 || result.height <= 0) fail('Bounds are not positive', result);
  return result;
}

function runtimeSkeleton(runtime, skeletonData) {
  const skeleton = new runtime.Skeleton(skeletonData);
  if (skeletonData.defaultSkin) skeleton.setSkin(skeletonData.defaultSkin);
  skeleton.setToSetupPose();
  skeleton.updateCache();
  return skeleton;
}

function transformConstraintByName(skeleton, name) {
  const constraint = (skeleton.transformConstraints ?? []).find(
    (item) => item?.data?.name === name,
  );
  if (!constraint) fail(`Runtime transform constraint is missing: ${name}`);
  return constraint;
}

function boneByName(skeleton, name, label) {
  const bone = skeleton.findBone(name);
  if (!bone) fail(`${label} bone is missing: ${name}`);
  return bone;
}

function matrixSnapshot(bone, label) {
  const result = {};
  for (const field of MATRIX_FIELDS) {
    result[field] = finite(bone[field], `${label}.${field}`);
  }
  return Object.freeze(result);
}

function numberDiffers(left, right) {
  const scale = Math.max(1, Math.abs(left), Math.abs(right));
  return Math.abs(left - right) > RESPONSE_EPSILON * scale;
}

function matrixDiffers(left, right) {
  return MATRIX_FIELDS.some((field) => numberDiffers(left[field], right[field]));
}

function boundsDiffer(left, right) {
  return ['x', 'y', 'width', 'height'].some((field) =>
    numberDiffers(left[field], right[field]),
  );
}

function applyScaleTargets(skeleton, controls) {
  const changes = [];
  for (const control of controls) {
    const constraint = transformConstraintByName(skeleton, control.name);
    if (constraint.target?.data?.name !== control.target) {
      fail(`Runtime target differs for scale constraint '${control.name}'`, {
        expected: control.target,
        actual: constraint.target?.data?.name,
      });
    }
    const target = boneByName(skeleton, control.target, 'Scale target');
    const setupScaleX = finite(target.scaleX, `${control.target}.scaleX`);
    const setupScaleY = finite(target.scaleY, `${control.target}.scaleY`);
    if (
      Math.abs(setupScaleX) <= ZERO_SCALE_EPSILON ||
      Math.abs(setupScaleY) <= ZERO_SCALE_EPSILON
    ) {
      fail(`Scale target '${control.target}' has a zero setup scale`, {
        setupScaleX,
        setupScaleY,
      });
    }
    target.scaleX = setupScaleX * SCALE_RESPONSE_FACTOR;
    target.scaleY = setupScaleY * SCALE_RESPONSE_FACTOR;
    changes.push({
      constraint: control.name,
      target: control.target,
      setupScaleX,
      setupScaleY,
      scaledScaleX: target.scaleX,
      scaledScaleY: target.scaleY,
    });
  }
  return changes;
}

function disableScaleConstraints(skeleton, controls) {
  for (const control of controls) {
    const disabledConstraint = transformConstraintByName(skeleton, control.name);
    disabledConstraint.scaleMix = 0;
  }
}

function scaleResponse(runtime, skeletonData, controls) {
  if (controls.length === 0) {
    return {
      applicable: false,
      scaleFactor: SCALE_RESPONSE_FACTOR,
      controlCount: 0,
      respondingControlCount: 0,
      changedBoneCount: 0,
      allControlsResponded: true,
      boundsChanged: false,
      constraintAffectsBounds: false,
      matricesFinite: true,
      controls: [],
    };
  }

  const setupSkeleton = runtimeSkeleton(runtime, skeletonData);
  setupSkeleton.updateWorldTransform();
  validateMatrices(setupSkeleton);
  const setupBounds = bounds(setupSkeleton);

  const scaledSkeleton = runtimeSkeleton(runtime, skeletonData);
  const targetChanges = applyScaleTargets(scaledSkeleton, controls);
  scaledSkeleton.updateWorldTransform();
  validateMatrices(scaledSkeleton);
  const scaledBounds = bounds(scaledSkeleton);

  const disabledSkeleton = runtimeSkeleton(runtime, skeletonData);
  disableScaleConstraints(disabledSkeleton, controls);
  applyScaleTargets(disabledSkeleton, controls);
  disabledSkeleton.updateWorldTransform();
  validateMatrices(disabledSkeleton);
  const disabledConstraintBounds = bounds(disabledSkeleton);

  const changedBoneNames = new Set();
  const responseRecords = controls.map((control) => {
    const runtimeConstraint = transformConstraintByName(scaledSkeleton, control.name);
    if (runtimeConstraint.active !== true) {
      fail(`Scale control constraint '${control.name}' is inactive in updateCache()`);
    }

    const changedFromSetup = [];
    const changedByConstraint = [];
    for (const boneName of control.bones) {
      const setupMatrix = matrixSnapshot(
        boneByName(setupSkeleton, boneName, 'Setup constrained'),
        `setup.${boneName}`,
      );
      const scaledMatrix = matrixSnapshot(
        boneByName(scaledSkeleton, boneName, 'Scaled constrained'),
        `scaled.${boneName}`,
      );
      const disabledMatrix = matrixSnapshot(
        boneByName(disabledSkeleton, boneName, 'Disabled constrained'),
        `disabled.${boneName}`,
      );
      if (matrixDiffers(scaledMatrix, setupMatrix)) changedFromSetup.push(boneName);
      if (matrixDiffers(scaledMatrix, disabledMatrix)) {
        changedByConstraint.push(boneName);
        changedBoneNames.add(boneName);
      }
    }

    return {
      constraint: control.name,
      target: control.target,
      constrainedBoneCount: control.bones.length,
      changedFromSetupBoneCount: changedFromSetup.length,
      changedByConstraintBoneCount: changedByConstraint.length,
      responded:
        changedFromSetup.length > 0 &&
        changedByConstraint.length > 0,
    };
  });

  const respondingControlCount = responseRecords.filter((item) => item.responded).length;
  const allControlsResponded = respondingControlCount === controls.length;
  const boundsChanged = boundsDiffer(scaledBounds, setupBounds);
  const constraintAffectsBounds = boundsDiffer(
    scaledBounds,
    disabledConstraintBounds,
  );

  const report = {
    applicable: true,
    scaleFactor: SCALE_RESPONSE_FACTOR,
    controlCount: controls.length,
    respondingControlCount,
    changedBoneCount: changedBoneNames.size,
    allControlsResponded,
    boundsChanged,
    constraintAffectsBounds,
    matricesFinite: true,
    setupBounds,
    scaledBounds,
    disabledConstraintBounds,
    targets: targetChanges,
    controls: responseRecords,
  };

  if (!allControlsResponded) {
    fail('One or more Spine 3.8 scale controls did not affect constrained bones', report);
  }
  if (changedBoneNames.size === 0) {
    fail('Spine 3.8 scale constraints changed no constrained bones', report);
  }
  if (!boundsChanged) {
    fail('Spine 3.8 scale controls did not change render bounds', report);
  }
  if (!constraintAffectsBounds) {
    fail('Disabling Spine 3.8 scale constraints did not change scaled bounds', report);
  }
  return report;
}

async function main() {
  const jsonPath = inputFile(process.argv[2], 'Spine 3.8 JSON');
  const runtimeEntry = inputFile(process.argv[3], 'Spine 3.8 runtime entry');
  if (process.argv.length > 4) fail(`Unknown argument: ${process.argv[4]}`);

  const runtime = await import(pathToFileURL(runtimeEntry).href);
  for (const exportName of [
    'TextureAtlas',
    'AtlasAttachmentLoader',
    'SkeletonJson',
    'Skeleton',
    'RegionAttachment',
    'MeshAttachment',
  ]) {
    if (typeof runtime[exportName] !== 'function') {
      fail(`Runtime export is missing: ${exportName}`, {
        available: Object.keys(runtime).sort(),
      });
    }
  }

  const document = record(JSON.parse(readFileSync(jsonPath, 'utf8')), 'document');
  const metadata = record(document.skeleton, 'document.skeleton');
  if (metadata.spine !== EXPECTED_VERSION) {
    fail(`Expected ${EXPECTED_VERSION}, received ${String(metadata.spine)}`);
  }
  if ('constraints' in document) fail('Unified 4.3 constraints leaked into Spine 3.8');

  const records = constraintRecords(document);
  const scaleControls = scaleControlRecords(document);
  const expectedAttachments = expectedSetupAttachments(document);
  const atlas = new runtime.TextureAtlas(
    atlasText(collectAtlasRegions(document)),
    () => oracleTexture(),
  );

  try {
    const loader = new runtime.AtlasAttachmentLoader(atlas);
    const reader = new runtime.SkeletonJson(loader);
    const skeletonData = reader.readSkeletonData(document);
    const skeleton = runtimeSkeleton(runtime, skeletonData);
    const scheduled = validateUpdateCache(skeleton, records);
    skeleton.updateWorldTransform();
    validateMatrices(skeleton);
    const attachments = validateAttachments(runtime, skeleton, expectedAttachments);
    const setupBounds = bounds(skeleton);
    const scaleResponseEvidence = scaleResponse(runtime, skeletonData, scaleControls);

    console.info(
      JSON.stringify(
        {
          ok: true,
          jsonPath,
          runtimeEntry,
          version: skeletonData.version,
          counts: {
            bones: skeleton.bones.length,
            slots: skeleton.slots.length,
            skins: skeletonData.skins.length,
            ik: skeleton.ikConstraints.length,
            transform: skeleton.transformConstraints.length,
            path: skeleton.pathConstraints.length,
            atlasPages: atlas.pages.length,
            atlasRegions: atlas.regions.length,
            setupRenderableAttachments: attachments.length,
          },
          updateCache: {
            expectedConstraints: records.length,
            scheduledConstraints: scheduled,
            everyConstraintScheduledExactlyOnce: true,
          },
          matrices: { finiteBones: skeleton.bones.length, allFinite: true },
          bounds: setupBounds,
          scaleResponse: scaleResponseEvidence,
        },
        null,
        2,
      ),
    );
  } finally {
    atlas.dispose();
  }
}

main().catch((error) => {
  console.error(
    JSON.stringify(
      {
        ok: false,
        message: error instanceof Error ? error.message : String(error),
        details: error?.details,
        stack: error instanceof Error ? error.stack : undefined,
      },
      null,
      2,
    ),
  );
  process.exitCode = 1;
});
