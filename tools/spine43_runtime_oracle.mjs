#!/usr/bin/env node

/**
 * Validate a generated Spine 4.3 JSON with the official read-only 4.3 spine-core.
 *
 * Usage:
 *   node tools/spine43_runtime_oracle.mjs <json-file> <runtime-entry>
 *
 * `runtime-entry` may be a built ESM entry or spine-core/src/index.ts. Source execution
 * is configured by the Python acceptance runner through spine43_ts_source_loader.mjs.
 * This oracle reads the JSON and runtime only; atlas textures exist only in memory.
 */

import assert from 'node:assert/strict';
import { existsSync, readFileSync, statSync } from 'node:fs';
import { resolve } from 'node:path';
import { pathToFileURL } from 'node:url';

const EXPECTED_VERSION = '4.3.23';
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
  if (!isRecord(value)) fail(`${path} must be a JSON object`, { value });
  return value;
}

function requireArray(value, path) {
  if (!Array.isArray(value)) fail(`${path} must be a JSON array`, { value });
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

function resolveInputFile(argument, label) {
  if (typeof argument !== 'string' || !argument.trim()) fail(`Missing ${label}`);
  const path = resolve(process.cwd(), argument);
  if (!existsSync(path) || !statSync(path).isFile()) {
    fail(`${label} is not a file: ${path}`);
  }
  return path;
}

function attachmentType(attachment) {
  return typeof attachment.type === 'string' ? attachment.type : 'region';
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
  const defaultSkin = skins.find(
    (value) => isRecord(value) && (value.name ?? 'default') === 'default',
  );
  if (!isRecord(defaultSkin)) return [];
  const attachments = requireRecord(
    defaultSkin.attachments ?? {},
    'document.skins[default].attachments',
  );
  const result = [];

  for (let index = 0; index < slots.length; index += 1) {
    const slot = requireRecord(slots[index], `document.slots[${index}]`);
    const slotName = requireNonEmptyString(slot.name, `document.slots[${index}].name`);
    const setupName = slot.attachment;
    if (setupName === undefined || setupName === null) continue;
    requireNonEmptyString(setupName, `document.slots[${index}].attachment`);
    const slotTable = requireRecord(
      attachments[slotName],
      `document.skins[default].attachments.${slotName}`,
    );
    const attachment = requireRecord(
      slotTable[setupName],
      `document.skins[default].attachments.${slotName}.${setupName}`,
    );
    const type = attachmentType(attachment);
    if (RENDERABLE_ATTACHMENT_TYPES.has(type)) {
      result.push({ slotName, attachmentName: setupName, type });
    }
  }
  return result;
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
  const body = regions.flatMap((region) => [
    region,
    '  rotate: false',
    '  xy: 0, 0',
    '  size: 1, 1',
    '  orig: 1, 1',
    '  offset: 0, 0',
    '  index: -1',
  ]);
  return [...header, ...body, ''].join('\n');
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
  if (!isRecord(atlas) || !Array.isArray(atlas.pages) || atlas.pages.length === 0) {
    fail('Spine 4.3 TextureAtlas must expose at least one page');
  }
  for (let index = 0; index < atlas.pages.length; index += 1) {
    const page = requireRecord(atlas.pages[index], `atlas.pages[${index}]`);
    if (typeof page.setTexture !== 'function') {
      fail(`atlas.pages[${index}].setTexture must be a function`);
    }
    page.setTexture(createOracleTexture(page.width, page.height));
  }
}

function readConstraintRecords(document) {
  const constraints = requireArray(document.constraints ?? [], 'document.constraints');
  return constraints.map((value, index) => {
    const constraint = requireRecord(value, `document.constraints[${index}]`);
    return Object.freeze({
      index,
      name: requireNonEmptyString(
        constraint.name,
        `document.constraints[${index}].name`,
      ),
      type: requireNonEmptyString(
        constraint.type,
        `document.constraints[${index}].type`,
      ),
    });
  });
}

function validateConstraintInventory(skeleton, records) {
  if (!Array.isArray(skeleton.constraints)) {
    fail('Spine 4.3 Skeleton.constraints must be an array');
  }
  const expectedNames = records.map((record) => record.name);
  const runtimeNames = skeleton.constraints.map((constraint, index) =>
    requireNonEmptyString(
      constraint?.data?.name,
      `runtime.constraints[${index}].data.name`,
    ),
  );
  assert.deepEqual(
    runtimeNames,
    expectedNames,
    'Runtime unified constraint inventory/order differs from JSON',
  );

  if (!Array.isArray(skeleton._updateCache)) {
    fail('Spine 4.3 Skeleton._updateCache must be an array');
  }
  const constraintSet = new Set(skeleton.constraints);
  const cachedConstraints = skeleton._updateCache.filter((item) => constraintSet.has(item));
  assert.equal(
    cachedConstraints.length,
    skeleton.constraints.length,
    'One or more Spine 4.3 constraints were skipped by updateCache()',
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
  return runtimeNames;
}

function validateBoneMatrices(skeleton) {
  const snapshots = [];
  for (let index = 0; index < skeleton.bones.length; index += 1) {
    const bone = skeleton.bones[index];
    const name = requireNonEmptyString(
      bone?.data?.name,
      `runtime.bones[${index}].data.name`,
    );
    const pose = requireRecord(
      bone.appliedPose,
      `runtime.bones[${index}](${name}).appliedPose`,
    );
    const values = {
      x: pose.x,
      y: pose.y,
      rotation: pose.rotation,
      scaleX: pose.scaleX,
      scaleY: pose.scaleY,
      shearX: pose.shearX,
      shearY: pose.shearY,
      worldX: pose.worldX,
      worldY: pose.worldY,
      a: pose.a,
      b: pose.b,
      c: pose.c,
      d: pose.d,
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
  for (let index = 0; index < skeleton.slots.length; index += 1) {
    const slot = skeleton.slots[index];
    const attachment = slot?.appliedPose?.attachment ?? null;
    if (!attachment) continue;
    let type = null;
    if (attachment instanceof runtime.RegionAttachment) type = 'region';
    else if (attachment instanceof runtime.MeshAttachment) type = 'mesh';
    else continue;
    result.push({
      slotIndex: index,
      slotName: slot.data.name,
      attachmentName: attachment.name,
      type,
    });
  }
  return result;
}

function validateSetupAttachments(runtime, skeleton, expected) {
  const actual = collectRuntimeSetupRenderableAttachments(runtime, skeleton);
  const expectedKeys = expected
    .map((item) => `${item.slotName}\u0000${item.attachmentName}`)
    .sort();
  const actualKeys = actual
    .map((item) => `${item.slotName}\u0000${item.attachmentName}`)
    .sort();
  assert.deepEqual(
    actualKeys,
    expectedKeys,
    'Runtime setup attachments differ from JSON setup attachments',
  );
  return actual;
}

function setupBounds(runtime, skeleton, attachments) {
  if (!Array.isArray(attachments) || attachments.length === 0) {
    fail('Runtime must expose at least one setup renderable attachment');
  }
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
  const jsonPath = resolveInputFile(process.argv[2], 'Spine 4.3 JSON path');
  const runtimeEntry = resolveInputFile(process.argv[3], 'Spine 4.3 runtime entry');
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
      fail(`Runtime is missing required export '${exportName}'`, {
        runtimeEntry,
        available: Object.keys(runtime).sort(),
      });
    }
  }
  if (!isRecord(runtime.Physics) || typeof runtime.Physics.none !== 'number') {
    fail("Runtime is missing required enum 'Physics.none'");
  }

  const document = requireRecord(
    JSON.parse(readFileSync(jsonPath, 'utf8')),
    'document',
  );
  const metadata = requireRecord(document.skeleton, 'document.skeleton');
  const version = requireNonEmptyString(metadata.spine, 'document.skeleton.spine');
  if (version !== EXPECTED_VERSION) {
    fail(`Expected Spine ${EXPECTED_VERSION} JSON, received ${version}`);
  }

  const records = readConstraintRecords(document);
  const expectedAttachments = collectExpectedSetupRenderableAttachments(document);
  const atlas = new runtime.TextureAtlas(createAtlasText(collectAtlasRegions(document)));
  bindAtlasPageTextures(atlas);

  try {
    const loader = new runtime.AtlasAttachmentLoader(atlas);
    const reader = new runtime.SkeletonJson(loader);
    const skeletonData = reader.readSkeletonData(document);
    const skeleton = new runtime.Skeleton(skeletonData);
    if (skeletonData.defaultSkin) skeleton.setSkin(skeletonData.defaultSkin);
    skeleton.setupPose();
    skeleton.updateCache();
    skeleton.updateWorldTransform(runtime.Physics.none);

    const constraintNames = validateConstraintInventory(skeleton, records);
    const boneSnapshots = validateBoneMatrices(skeleton);
    const setupAttachments = validateSetupAttachments(
      runtime,
      skeleton,
      expectedAttachments,
    );
    const bounds = setupBounds(runtime, skeleton, setupAttachments);
    const typeCounts = Object.fromEntries(
      [...new Set(records.map((record) => record.type))].map((type) => [
        type,
        records.filter((record) => record.type === type).length,
      ]),
    );

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
            constraints: skeleton.constraints.length,
            ik: typeCounts.ik ?? 0,
            transform: typeCounts.transform ?? 0,
            atlasPages: atlas.pages.length,
            atlasRegions: atlas.regions.length,
            setupRenderableAttachments: setupAttachments.length,
          },
          updateCache: {
            expectedConstraints: records.length,
            scheduledConstraints: constraintNames.length,
            everyConstraintScheduledExactlyOnce: true,
          },
          matrices: {
            finiteBones: boneSnapshots.length,
            allFinite: true,
          },
          bounds,
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
