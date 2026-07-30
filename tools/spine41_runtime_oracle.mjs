#!/usr/bin/env node

/**
 * Validate an externally generated Spine 4.1 JSON with the exact vendored 4.1 runtime.
 *
 * Usage:
 *   node tools/spine41_runtime_oracle.mjs <json-file> <runtime-entry> [--full]
 *
 * The runtime path may also be supplied through SPINE41_RUNTIME_ENTRY.
 * The referenced runtime repository is read-only: this script imports runtime code and
 * creates only in-memory atlas/texture objects.
 */

import assert from 'node:assert/strict';
import { existsSync, readFileSync, statSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';

const SCRIPT_DIR = dirname(fileURLToPath(import.meta.url));
const REPO_ROOT = resolve(SCRIPT_DIR, '..');
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

function parseOutputOptions(argumentsList) {
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
  const configured = argument ?? process.env.SPINE41_RUNTIME_ENTRY;
  if (typeof configured !== 'string' || !configured.trim()) {
    fail(
      'Missing Spine 4.1 runtime entry. Pass it as the second argument or set ' +
        'SPINE41_RUNTIME_ENTRY.',
    );
  }

  let entry = resolve(process.cwd(), configured);
  if (!existsSync(entry)) fail(`Spine 4.1 runtime entry does not exist: ${entry}`);
  if (statSync(entry).isDirectory()) entry = resolve(entry, 'index.js');
  if (!existsSync(entry)) fail(`Spine 4.1 runtime index does not exist: ${entry}`);
  if (!statSync(entry).isFile()) fail(`Spine 4.1 runtime entry is not a file: ${entry}`);
  return entry;
}

function sequenceRegionNames(basePath, sequence) {
  if (!isRecord(sequence)) return [basePath];

  const count = sequence.count;
  const start = sequence.start ?? 0;
  const digits = sequence.digits ?? 0;
  if (!Number.isInteger(count) || count <= 0) {
    fail('attachment.sequence.count must be a positive integer', { basePath, sequence });
  }
  if (!Number.isInteger(start) || !Number.isInteger(digits) || digits < 0) {
    fail('attachment sequence start/digits must be valid integers', {
      basePath,
      sequence,
    });
  }

  return Array.from({ length: count }, (_, index) => {
    const frame = String(start + index).padStart(digits, '0');
    return `${basePath}${frame}`;
  });
}

function attachmentType(attachment) {
  return typeof attachment.type === 'string' ? attachment.type : 'region';
}

function collectAtlasRegions(document) {
  const result = new Set();
  const skins = requireArray(document.skins ?? [], 'document.skins');

  for (let skinIndex = 0; skinIndex < skins.length; skinIndex += 1) {
    const skin = requireRecord(skins[skinIndex], `document.skins[${skinIndex}]`);
    const attachments = requireRecord(
      skin.attachments ?? {},
      `document.skins[${skinIndex}].attachments`,
    );

    for (const [slotName, slotValue] of Object.entries(attachments)) {
      const slotAttachments = requireRecord(
        slotValue,
        `document.skins[${skinIndex}].attachments.${slotName}`,
      );
      for (const [entryName, attachmentValue] of Object.entries(slotAttachments)) {
        const attachment = requireRecord(
          attachmentValue,
          `document.skins[${skinIndex}].attachments.${slotName}.${entryName}`,
        );
        if (!RENDERABLE_ATTACHMENT_TYPES.has(attachmentType(attachment))) continue;

        const basePath =
          typeof attachment.path === 'string' && attachment.path
            ? attachment.path
            : typeof attachment.name === 'string' && attachment.name
              ? attachment.name
              : entryName;
        for (const regionName of sequenceRegionNames(basePath, attachment.sequence)) {
          result.add(regionName);
        }
      }
    }
  }

  return [...result].sort();
}

function collectExpectedSetupRenderableAttachments(document) {
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
  const attachments = requireRecord(
    defaultSkin.attachments ?? {},
    `document.skins[${defaultSkinIndex}].attachments`,
  );
  const expected = [];

  for (let slotIndex = 0; slotIndex < slots.length; slotIndex += 1) {
    const slot = requireRecord(slots[slotIndex], `document.slots[${slotIndex}]`);
    const slotName = requireNonEmptyString(slot.name, `document.slots[${slotIndex}].name`);
    const setupAttachmentName = slot.attachment;
    if (setupAttachmentName === undefined || setupAttachmentName === null) continue;
    requireNonEmptyString(
      setupAttachmentName,
      `document.slots[${slotIndex}].attachment`,
    );

    const slotAttachments = attachments[slotName];
    if (!isRecord(slotAttachments)) {
      fail(`Setup slot '${slotName}' has no attachment table in the default skin`, {
        slotName,
        setupAttachmentName,
      });
    }
    const attachment = slotAttachments[setupAttachmentName];
    if (!isRecord(attachment)) {
      fail(`Setup attachment '${setupAttachmentName}' is missing for slot '${slotName}'`, {
        slotName,
        setupAttachmentName,
      });
    }

    const type = attachmentType(attachment);
    if (!RENDERABLE_ATTACHMENT_TYPES.has(type)) continue;
    expected.push({ slotName, attachmentName: setupAttachmentName, type });
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
    fail('Spine 4.1 TextureAtlas must expose a pages array');
  }
  if (atlas.pages.length === 0) {
    fail('Synthetic Spine 4.1 atlas contains no pages');
  }

  for (let pageIndex = 0; pageIndex < atlas.pages.length; pageIndex += 1) {
    const page = requireRecord(atlas.pages[pageIndex], `atlas.pages[${pageIndex}]`);
    if (typeof page.setTexture !== 'function') {
      fail(`atlas.pages[${pageIndex}].setTexture must be a function`);
    }

    const width = requirePositiveInteger(page.width, `atlas.pages[${pageIndex}].width`);
    const height = requirePositiveInteger(page.height, `atlas.pages[${pageIndex}].height`);
    page.setTexture(createOracleTexture(width, height));

    if (!page.texture || typeof page.texture.getImage !== 'function') {
      fail(`atlas.pages[${pageIndex}] did not retain the assigned texture`);
    }
    const image = page.texture.getImage();
    if (image.width !== width || image.height !== height) {
      fail(`atlas.pages[${pageIndex}] texture image dimensions changed`, {
        expected: { width, height },
        actual: image,
      });
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
    `Spine 4.1 requires globally unique constraint orders: ${JSON.stringify(records)}`,
  );
  assert.deepEqual(
    [...orders].sort((left, right) => left - right),
    Array.from({ length: orders.length }, (_, index) => index),
    'Spine 4.1 constraint orders must form 0..N-1',
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
    ...skeleton.ikConstraints,
    ...skeleton.transformConstraints,
    ...skeleton.pathConstraints,
  ];
}

function validateUpdateCache(skeleton, expectedRecords) {
  const constraints = runtimeConstraintObjects(skeleton);
  const expectedNames = expectedRecords.map((record) => record.name).sort();
  const runtimeNames = constraints.map((constraint) => constraint.data.name).sort();
  assert.deepEqual(runtimeNames, expectedNames, 'Runtime constraint inventory differs');

  if (!Array.isArray(skeleton._updateCache)) {
    fail('Spine 4.1 Skeleton does not expose the expected _updateCache array');
  }
  const constraintSet = new Set(constraints);
  const cachedConstraints = skeleton._updateCache.filter((item) => constraintSet.has(item));
  assert.equal(
    cachedConstraints.length,
    constraints.length,
    'One or more runtime constraints were skipped by Skeleton.updateCache()',
  );

  const occurrences = new Map();
  for (const constraint of cachedConstraints) {
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

  return cachedConstraints.map((constraint) => constraint.data.name);
}

function validateBoneMatrices(skeleton) {
  const snapshots = [];
  for (let index = 0; index < skeleton.bones.length; index += 1) {
    const bone = skeleton.bones[index];
    const name = bone.data.name;
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
      requireFinite(value, `runtime.bones[${index}](${name}).${field}`);
    }
    snapshots.push({ name, ...values });
  }
  return snapshots;
}

function collectRuntimeSetupRenderableAttachments(runtime, skeleton) {
  const result = [];
  for (
    let drawOrderIndex = 0;
    drawOrderIndex < skeleton.drawOrder.length;
    drawOrderIndex += 1
  ) {
    const slot = skeleton.drawOrder[drawOrderIndex];
    const attachment = slot.getAttachment();
    if (!attachment) continue;

    let type = null;
    if (attachment instanceof runtime.RegionAttachment) type = 'region';
    else if (attachment instanceof runtime.MeshAttachment) type = 'mesh';
    else continue;

    result.push({
      drawOrderIndex,
      slotName: slot.data.name,
      attachmentName: attachment.name,
      type,
    });
  }
  return result;
}

function validateSetupAttachments(runtime, skeleton, expectedAttachments) {
  const actualAttachments = collectRuntimeSetupRenderableAttachments(runtime, skeleton);
  const expectedKeys = expectedAttachments
    .map((item) => `${item.slotName}\u0000${item.attachmentName}`)
    .sort();
  const actualKeys = actualAttachments
    .map((item) => `${item.slotName}\u0000${item.attachmentName}`)
    .sort();

  assert.deepEqual(
    actualKeys,
    expectedKeys,
    'Runtime setup renderable attachments differ from JSON setup attachments',
  );
  return actualAttachments;
}

function setupBounds(runtime, skeleton, renderableAttachments) {
  if (!Array.isArray(renderableAttachments)) {
    fail('renderableAttachments must be an array');
  }
  if (renderableAttachments.length === 0) return null;

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
      'Usage: node tools/spine41_runtime_oracle.mjs ' +
        '<json-file> <runtime-entry> [--full]',
    );
  }

  const options = parseOutputOptions(process.argv.slice(4));
  const jsonPath = resolve(process.cwd(), jsonArgument);
  if (!existsSync(jsonPath)) fail(`JSON file does not exist: ${jsonPath}`);
  if (!statSync(jsonPath).isFile()) fail(`JSON path is not a file: ${jsonPath}`);

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

  const document = requireRecord(
    JSON.parse(readFileSync(jsonPath, 'utf8')),
    'document',
  );
  const skeletonMetadata = requireRecord(document.skeleton, 'document.skeleton');
  const version = skeletonMetadata.spine;
  if (typeof version !== 'string' || !version.startsWith('4.1')) {
    fail(`Expected Spine 4.1 JSON, received skeleton.spine=${String(version)}`);
  }

  const constraintRecords = readConstraintRecords(document);
  validateConstraintOrders(constraintRecords);
  const expectedSetupAttachments = collectExpectedSetupRenderableAttachments(document);
  const atlasRegions = collectAtlasRegions(document);

  // Spine 4.1 TextureAtlas accepts atlas text only. Its page textures must be attached
  // explicitly through TextureAtlasPage.setTexture before SkeletonJson reads meshes.
  const atlas = new runtime.TextureAtlas(createAtlasText(atlasRegions));
  bindAtlasPageTextures(atlas);

  try {
    const loader = new runtime.AtlasAttachmentLoader(atlas);
    const jsonReader = new runtime.SkeletonJson(loader);
    const skeletonData = jsonReader.readSkeletonData(document);
    const skeleton = new runtime.Skeleton(skeletonData);

    if (skeletonData.defaultSkin) skeleton.setSkin(skeletonData.defaultSkin);
    skeleton.setToSetupPose();
    skeleton.updateCache();
    const cacheConstraints = validateUpdateCache(skeleton, constraintRecords);
    skeleton.updateWorldTransform();

    const boneSnapshots = validateBoneMatrices(skeleton);
    const setupAttachments = validateSetupAttachments(
      runtime,
      skeleton,
      expectedSetupAttachments,
    );
    const bounds = setupBounds(runtime, skeleton, setupAttachments);
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
        setupRenderableAttachments: setupAttachments.length,
      },
      constraintOrders: summarizeConstraintOrders(constraintRecords),
      updateCache: {
        expectedConstraints: constraintRecords.length,
        scheduledConstraints: cacheConstraints.length,
        everyConstraintScheduledExactlyOnce: true,
      },
      matrices: {
        finiteBones: boneSnapshots.length,
        allFinite: true,
      },
      bounds,
    };

    const report = options.includeDetails
      ? {
          ...summary,
          details: {
            constraintOrders: constraintRecords,
            updateCacheConstraints: cacheConstraints,
            setupAttachments,
            bones: boneSnapshots,
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
