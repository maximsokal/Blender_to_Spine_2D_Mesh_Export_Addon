#!/usr/bin/env node

/**
 * Validate an externally generated Spine 4.1 JSON with the exact vendored 4.1 runtime.
 *
 * Usage:
 *   node tools/spine41_runtime_oracle.mjs <json-file> <runtime-entry>
 *
 * runtime-entry must point to the self-contained ESM entry, for example:
 *   ../Spine2D_curve_optimization/vendor/spine-webgl-41/index.js
 *
 * The runtime path may also be supplied through SPINE41_RUNTIME_ENTRY.
 */

import assert from 'node:assert/strict';
import { existsSync, readFileSync, statSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';

const SCRIPT_DIR = dirname(fileURLToPath(import.meta.url));
const REPO_ROOT = resolve(SCRIPT_DIR, '..');

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

function resolveRuntimeEntry(argument) {
  const configured = argument ?? process.env.SPINE41_RUNTIME_ENTRY;
  if (!configured || !configured.trim()) {
    fail(
      'Missing Spine 4.1 runtime entry. Pass it as the second argument or set ' +
      'SPINE41_RUNTIME_ENTRY.',
    );
  }

  let entry = resolve(process.cwd(), configured);
  if (!existsSync(entry)) fail(`Spine 4.1 runtime entry does not exist: ${entry}`);
  if (statSync(entry).isDirectory()) entry = resolve(entry, 'index.js');
  if (!existsSync(entry)) fail(`Spine 4.1 runtime index does not exist: ${entry}`);
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
        const type = typeof attachment.type === 'string' ? attachment.type : 'region';
        if (!['region', 'mesh', 'linkedmesh'].includes(type)) continue;

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

function createAtlasText(regions) {
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
      if (typeof constraint.name !== 'string' || !constraint.name) {
        fail(`document.${collectionName}[${index}].name must be non-empty`);
      }
      const order = constraint.order ?? 0;
      if (!Number.isInteger(order) || order < 0) {
        fail(`document.${collectionName}[${index}].order must be non-negative`, {
          value: order,
        });
      }
      records.push({ collectionName, index, name: constraint.name, order });
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

function setupBounds(runtime, skeleton) {
  const hasRenderableAttachment = skeleton.drawOrder.some((slot) => {
    const attachment = slot.getAttachment();
    return attachment && ['RegionAttachment', 'MeshAttachment'].includes(
      attachment.constructor?.name,
    );
  });
  if (!hasRenderableAttachment) return null;

  const offset = new runtime.Vector2();
  const size = new runtime.Vector2();
  skeleton.getBounds(offset, size);
  return {
    x: requireFinite(offset.x, 'runtime.bounds.x'),
    y: requireFinite(offset.y, 'runtime.bounds.y'),
    width: requireFinite(size.x, 'runtime.bounds.width'),
    height: requireFinite(size.y, 'runtime.bounds.height'),
  };
}

async function main() {
  const jsonArgument = process.argv[2];
  if (!jsonArgument) {
    fail('Usage: node tools/spine41_runtime_oracle.mjs <json-file> <runtime-entry>');
  }

  const jsonPath = resolve(process.cwd(), jsonArgument);
  if (!existsSync(jsonPath)) fail(`JSON file does not exist: ${jsonPath}`);
  const runtimeEntry = resolveRuntimeEntry(process.argv[3]);
  const runtime = await import(pathToFileURL(runtimeEntry).href);

  for (const exportName of [
    'TextureAtlas',
    'AtlasAttachmentLoader',
    'SkeletonJson',
    'Skeleton',
    'Vector2',
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

  const atlas = new runtime.TextureAtlas(
    createAtlasText(collectAtlasRegions(document)),
    () => ({
      setFilters() {},
      setWraps() {},
      getImage: () => ({ width: 1, height: 1 }),
      dispose() {},
    }),
  );

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

    const report = {
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
      },
      constraintOrders: constraintRecords,
      updateCacheConstraints: cacheConstraints,
      bounds: setupBounds(runtime, skeleton),
      bones: validateBoneMatrices(skeleton),
    };
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
