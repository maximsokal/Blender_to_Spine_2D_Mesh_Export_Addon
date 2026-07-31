#!/usr/bin/env node

/** Validate one generated Spine 3.8 JSON with a read-only 3.8 runtime entry. */

import assert from 'node:assert/strict';
import { existsSync, readFileSync, statSync } from 'node:fs';
import { resolve } from 'node:path';
import { pathToFileURL } from 'node:url';

const EXPECTED_VERSION = '3.8.99';
const RENDERABLE_TYPES = new Set(['region', 'mesh', 'linkedmesh']);

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

function bounds(runtime, skeleton) {
  const offset = new runtime.Vector2();
  const size = new runtime.Vector2();
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
    'Vector2',
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
  const expectedAttachments = expectedSetupAttachments(document);
  const atlas = new runtime.TextureAtlas(
    atlasText(collectAtlasRegions(document)),
    () => oracleTexture(),
  );

  try {
    const loader = new runtime.AtlasAttachmentLoader(atlas);
    const reader = new runtime.SkeletonJson(loader);
    const skeletonData = reader.readSkeletonData(document);
    const skeleton = new runtime.Skeleton(skeletonData);
    if (skeletonData.defaultSkin) skeleton.setSkin(skeletonData.defaultSkin);
    skeleton.setToSetupPose();
    skeleton.updateCache();
    const scheduled = validateUpdateCache(skeleton, records);
    skeleton.updateWorldTransform();
    validateMatrices(skeleton);
    const attachments = validateAttachments(runtime, skeleton, expectedAttachments);
    const setupBounds = bounds(runtime, skeleton);

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
