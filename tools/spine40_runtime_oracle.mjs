#!/usr/bin/env node

/**
 * Validate a generated Spine 4.0 JSON with an exact read-only Spine 4.0 runtime.
 *
 * Usage:
 *   node tools/spine40_runtime_oracle.mjs <json-file> <runtime-entry> [--full]
 */

import assert from 'node:assert/strict';
import { existsSync, readFileSync, statSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';

const SCRIPT_DIR = dirname(fileURLToPath(import.meta.url));
const REPO_ROOT = resolve(SCRIPT_DIR, '..');
const EXPECTED_VERSION = '4.0.64';
const RENDERABLE_ATTACHMENT_TYPES = new Set(['region', 'mesh', 'linkedmesh']);

function fail(message, details = undefined) {
  const error = new Error(message);
  if (details !== undefined) error.details = details;
  throw error;
}

function isRecord(value) {
  return typeof value === 'object' && value !== null && !Array.isArray(value);
}

function requireRecord(value, path) {
  if (!isRecord(value)) fail(`${path} must be a JSON object`);
  return value;
}

function requireArray(value, path) {
  if (!Array.isArray(value)) fail(`${path} must be a JSON array`);
  return value;
}

function requireFinite(value, path) {
  if (typeof value !== 'number' || !Number.isFinite(value)) {
    fail(`${path} must be finite`, { value });
  }
  return value;
}

function requirePositiveInteger(value, path) {
  if (!Number.isInteger(value) || value <= 0) {
    fail(`${path} must be a positive integer`, { value });
  }
  return value;
}

function requireNonEmptyString(value, path) {
  if (typeof value !== 'string' || !value.trim()) {
    fail(`${path} must be a non-empty string`, { value });
  }
  return value;
}

function parseOptions(argumentsList) {
  if (!Array.isArray(argumentsList)) fail('argumentsList must be an array');
  let includeDetails = false;
  for (const argument of argumentsList) {
    if (argument === '--full') {
      includeDetails = true;
      continue;
    }
    fail(`Unknown oracle option: ${String(argument)}`);
  }
  return Object.freeze({ includeDetails });
}

function resolveRuntimeEntry(argument) {
  const configured = argument ?? process.env.SPINE40_RUNTIME_ENTRY;
  if (typeof configured !== 'string' || !configured.trim()) {
    fail(
      'Missing Spine 4.0 runtime entry. Pass it as the second argument or set ' +
        'SPINE40_RUNTIME_ENTRY.',
    );
  }

  let entry = resolve(process.cwd(), configured);
  if (!existsSync(entry)) fail(`Spine 4.0 runtime entry does not exist: ${entry}`);
  if (statSync(entry).isDirectory()) entry = resolve(entry, 'index.js');
  if (!existsSync(entry)) fail(`Spine 4.0 runtime index does not exist: ${entry}`);
  if (!statSync(entry).isFile()) fail(`Spine 4.0 runtime entry is not a file: ${entry}`);
  return entry;
}

function attachmentType(attachment) {
  return typeof attachment.type === 'string' ? attachment.type : 'region';
}

function collectAtlasRegions(document) {
  const regions = new Set();
  const skins = requireArray(document.skins ?? [], 'document.skins');
  for (let skinIndex = 0; skinIndex < skins.length; skinIndex += 1) {
    const skin = requireRecord(skins[skinIndex], `document.skins[${skinIndex}]`);
    const groups = requireRecord(
      skin.attachments ?? {},
      `document.skins[${skinIndex}].attachments`,
    );
    for (const [slotName, slotValue] of Object.entries(groups)) {
      const attachments = requireRecord(
        slotValue,
        `document.skins[${skinIndex}].attachments.${slotName}`,
      );
      for (const [entryName, rawAttachment] of Object.entries(attachments)) {
        const attachment = requireRecord(
          rawAttachment,
          `document.skins[${skinIndex}].attachments.${slotName}.${entryName}`,
        );
        if ('sequence' in attachment) {
          fail('Spine 4.0 acceptance does not permit setup attachment sequences', {
            skinIndex,
            slotName,
            entryName,
          });
        }
        if (!RENDERABLE_ATTACHMENT_TYPES.has(attachmentType(attachment))) continue;
        const regionName =
          typeof attachment.path === 'string' && attachment.path
            ? attachment.path
            : typeof attachment.name === 'string' && attachment.name
              ? attachment.name
              : entryName;
        regions.add(regionName);
      }
    }
  }
  return [...regions].sort();
}

function collectExpectedSetupAttachments(document) {
  const slots = requireArray(document.slots ?? [], 'document.slots');
  const skins = requireArray(document.skins ?? [], 'document.skins');
  const defaultSkinIndex = skins.findIndex(
    (skin) => isRecord(skin) && (skin.name ?? 'default') === 'default',
  );
  if (defaultSkinIndex < 0) return [];

  const defaultSkin = requireRecord(
    skins[defaultSkinIndex],
    `document.skins[${defaultSkinIndex}]`,
  );
  const groups = requireRecord(
    defaultSkin.attachments ?? {},
    `document.skins[${defaultSkinIndex}].attachments`,
  );
  const expected = [];

  for (let slotIndex = 0; slotIndex < slots.length; slotIndex += 1) {
    const slot = requireRecord(slots[slotIndex], `document.slots[${slotIndex}]`);
    const slotName = requireNonEmptyString(slot.name, `document.slots[${slotIndex}].name`);
    const attachmentName = slot.attachment;
    if (attachmentName === undefined || attachmentName === null) continue;
    requireNonEmptyString(attachmentName, `document.slots[${slotIndex}].attachment`);

    const attachments = groups[slotName];
    if (!isRecord(attachments)) {
      fail(`Setup slot '${slotName}' has no default-skin attachment table`);
    }
    const attachment = attachments[attachmentName];
    if (!isRecord(attachment)) {
      fail(`Setup attachment '${attachmentName}' is missing for slot '${slotName}'`);
    }
    const type = attachmentType(attachment);
    if (RENDERABLE_ATTACHMENT_TYPES.has(type)) {
      expected.push({ slotName, attachmentName, type });
    }
  }
  return expected;
}

function createAtlasText(regions) {
  if (!Array.isArray(regions) || !regions.every((value) => typeof value === 'string')) {
    fail('regions must be an array of strings');
  }
  const header = [
    'oracle.png',
    'size: 1,1',
    'format: RGBA8888',
    'filter: Linear,Linear',
    'repeat: none',
  ];
  const regionLines = regions.flatMap((region) => [
    region,
    '  rotate: false',
    '  xy: 0, 0',
    '  size: 1, 1',
    '  orig: 1, 1',
    '  offset: 0, 0',
    '  index: -1',
  ]);
  return [...header, ...regionLines, ''].join('\n');
}

function createOracleTexture(width, height) {
  const image = Object.freeze({
    width: requirePositiveInteger(width, 'oracle texture width'),
    height: requirePositiveInteger(height, 'oracle texture height'),
  });
  return {
    setFilters() {},
    setWraps() {},
    getImage() {
      return image;
    },
    dispose() {},
  };
}

function bindAtlasPageTextures(atlas) {
  if (!isRecord(atlas) || !Array.isArray(atlas.pages)) {
    fail('Spine 4.0 TextureAtlas must expose a pages array');
  }
  if (atlas.pages.length === 0) fail('Synthetic Spine 4.0 atlas contains no pages');

  for (let index = 0; index < atlas.pages.length; index += 1) {
    const page = requireRecord(atlas.pages[index], `atlas.pages[${index}]`);
    const width = requirePositiveInteger(page.width, `atlas.pages[${index}].width`);
    const height = requirePositiveInteger(page.height, `atlas.pages[${index}].height`);
    const texture = createOracleTexture(width, height);
    if (typeof page.setTexture === 'function') page.setTexture(texture);
    else page.texture = texture;
    if (!page.texture || typeof page.texture.getImage !== 'function') {
      fail(`atlas.pages[${index}] did not retain the assigned texture`);
    }
  }
}

function readConstraintRecords(document) {
  const records = [];
  for (const collectionName of ['ik', 'transform', 'path']) {
    const values = document[collectionName] ?? [];
    if (!Array.isArray(values)) fail(`document.${collectionName} must be an array`);
    for (let index = 0; index < values.length; index += 1) {
      const constraint = requireRecord(
        values[index],
        `document.${collectionName}[${index}]`,
      );
      const name = requireNonEmptyString(
        constraint.name,
        `document.${collectionName}[${index}].name`,
      );
      const order = constraint.order ?? 0;
      if (!Number.isInteger(order) || order < 0) {
        fail(`document.${collectionName}[${index}].order must be non-negative`, {
          value: order,
        });
      }
      records.push({ collectionName, index, name, order });
    }
  }
  return records;
}

function validateConstraintOrders(records) {
  const names = records.map((record) => record.name);
  assert.equal(new Set(names).size, names.length, 'Constraint names must be globally unique');
  const orders = records.map((record) => record.order);
  assert.equal(
    new Set(orders).size,
    orders.length,
    `Spine 4.0 requires globally unique constraint orders: ${JSON.stringify(records)}`,
  );
  assert.deepEqual(
    [...orders].sort((left, right) => left - right),
    Array.from({ length: orders.length }, (_, index) => index),
    'Spine 4.0 constraint orders must form 0..N-1',
  );
}

function summarizeConstraintOrders(records) {
  if (records.length === 0) {
    return Object.freeze({ count: 0, minimum: null, maximum: null, contiguous: true });
  }
  const orders = records.map((record) => record.order);
  return Object.freeze({
    count: records.length,
    minimum: Math.min(...orders),
    maximum: Math.max(...orders),
    contiguous: true,
  });
}

function runtimeConstraintObjects(skeleton) {
  return [
    ...(skeleton.ikConstraints ?? []),
    ...(skeleton.transformConstraints ?? []),
    ...(skeleton.pathConstraints ?? []),
  ];
}

function validateUpdateCache(skeleton, expectedRecords) {
  const constraints = runtimeConstraintObjects(skeleton);
  const expectedNames = expectedRecords.map((record) => record.name).sort();
  const runtimeNames = constraints.map((constraint) => constraint.data.name).sort();
  assert.deepEqual(runtimeNames, expectedNames, 'Runtime constraint inventory differs');
  if (!Array.isArray(skeleton._updateCache)) {
    fail('Spine 4.0 Skeleton does not expose the expected _updateCache array');
  }

  const constraintSet = new Set(constraints);
  const cached = skeleton._updateCache.filter((item) => constraintSet.has(item));
  assert.equal(
    cached.length,
    constraints.length,
    'One or more runtime constraints were skipped by Skeleton.updateCache()',
  );
  const occurrences = new Map();
  for (const constraint of cached) {
    const name = constraint.data.name;
    occurrences.set(name, (occurrences.get(name) ?? 0) + 1);
  }
  for (const name of expectedNames) {
    assert.equal(
      occurrences.get(name),
      1,
      `Constraint '${name}' must appear exactly once in the update cache`,
    );
  }
  return cached.map((constraint) => constraint.data.name);
}

function validateBoneMatrices(skeleton) {
  const snapshots = [];
  for (let index = 0; index < skeleton.bones.length; index += 1) {
    const bone = skeleton.bones[index];
    const values = {
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
    };
    for (const [field, value] of Object.entries(values)) {
      requireFinite(value, `runtime.bones[${index}](${bone.data.name}).${field}`);
    }
    snapshots.push({ name: bone.data.name, ...values });
  }
  return snapshots;
}

function collectRuntimeSetupAttachments(runtime, skeleton) {
  const result = [];
  for (let index = 0; index < skeleton.drawOrder.length; index += 1) {
    const slot = skeleton.drawOrder[index];
    const attachment = slot.getAttachment();
    if (!attachment) continue;
    let type = null;
    if (attachment instanceof runtime.RegionAttachment) type = 'region';
    else if (attachment instanceof runtime.MeshAttachment) type = 'mesh';
    else continue;
    result.push({
      drawOrderIndex: index,
      slotName: slot.data.name,
      attachmentName: attachment.name,
      type,
    });
  }
  return result;
}

function validateSetupAttachments(runtime, skeleton, expected) {
  const actual = collectRuntimeSetupAttachments(runtime, skeleton);
  const expectedKeys = expected
    .map((item) => `${item.slotName}\u0000${item.attachmentName}`)
    .sort();
  const actualKeys = actual
    .map((item) => `${item.slotName}\u0000${item.attachmentName}`)
    .sort();
  assert.deepEqual(
    actualKeys,
    expectedKeys,
    'Runtime setup renderable attachments differ from JSON setup attachments',
  );
  return actual;
}

function setupBounds(runtime, skeleton, attachments) {
  if (!Array.isArray(attachments)) fail('attachments must be an array');
  if (attachments.length === 0) return null;
  const offset = new runtime.Vector2();
  const size = new runtime.Vector2();
  skeleton.getBounds(offset, size);
  const bounds = {
    x: requireFinite(offset.x, 'runtime.bounds.x'),
    y: requireFinite(offset.y, 'runtime.bounds.y'),
    width: requireFinite(size.x, 'runtime.bounds.width'),
    height: requireFinite(size.y, 'runtime.bounds.height'),
  };
  if (bounds.width <= 0 || bounds.height <= 0) {
    fail('Runtime setup bounds must have positive width and height', bounds);
  }
  return bounds;
}

async function main() {
  const jsonArgument = process.argv[2];
  if (!jsonArgument) {
    fail(
      'Usage: node tools/spine40_runtime_oracle.mjs ' +
        '<json-file> <runtime-entry> [--full]',
    );
  }
  const options = parseOptions(process.argv.slice(4));
  const jsonPath = resolve(process.cwd(), jsonArgument);
  if (!existsSync(jsonPath) || !statSync(jsonPath).isFile()) {
    fail(`JSON file does not exist: ${jsonPath}`);
  }

  const runtimeEntry = resolveRuntimeEntry(process.argv[3]);
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
      fail(`Runtime is missing required export '${exportName}'`, {
        runtimeEntry,
        available: Object.keys(runtime).sort(),
      });
    }
  }

  const document = requireRecord(JSON.parse(readFileSync(jsonPath, 'utf8')), 'document');
  const skeletonMetadata = requireRecord(document.skeleton, 'document.skeleton');
  if (skeletonMetadata.spine !== EXPECTED_VERSION) {
    fail(
      `Expected Spine ${EXPECTED_VERSION} JSON, received ` +
        `skeleton.spine=${String(skeletonMetadata.spine)}`,
    );
  }

  const constraintRecords = readConstraintRecords(document);
  validateConstraintOrders(constraintRecords);
  const expectedAttachments = collectExpectedSetupAttachments(document);
  const atlasRegions = collectAtlasRegions(document);
  const atlas = new runtime.TextureAtlas(createAtlasText(atlasRegions));
  bindAtlasPageTextures(atlas);

  try {
    const loader = new runtime.AtlasAttachmentLoader(atlas);
    const reader = new runtime.SkeletonJson(loader);
    const skeletonData = reader.readSkeletonData(document);
    const skeleton = new runtime.Skeleton(skeletonData);
    if (skeletonData.defaultSkin) skeleton.setSkin(skeletonData.defaultSkin);
    skeleton.setToSetupPose();
    skeleton.updateCache();
    const cachedConstraints = validateUpdateCache(skeleton, constraintRecords);
    skeleton.updateWorldTransform();

    const bones = validateBoneMatrices(skeleton);
    const attachments = validateSetupAttachments(runtime, skeleton, expectedAttachments);
    const bounds = setupBounds(runtime, skeleton, attachments);
    const summary = {
      ok: true,
      outputMode: options.includeDetails ? 'full' : 'summary',
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
      constraintOrders: summarizeConstraintOrders(constraintRecords),
      updateCache: {
        expectedConstraints: constraintRecords.length,
        scheduledConstraints: cachedConstraints.length,
        everyConstraintScheduledExactlyOnce: true,
      },
      matrices: {
        finiteBones: bones.length,
        allFinite: true,
      },
      bounds,
    };

    const report = options.includeDetails
      ? {
          ...summary,
          details: {
            constraintOrders: constraintRecords,
            updateCacheConstraints: cachedConstraints,
            setupAttachments: attachments,
            bones,
          },
        }
      : summary;
    console.info(JSON.stringify(report, null, 2));
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
        repoRoot: REPO_ROOT,
      },
      null,
      2,
    ),
  );
  process.exitCode = 1;
});
