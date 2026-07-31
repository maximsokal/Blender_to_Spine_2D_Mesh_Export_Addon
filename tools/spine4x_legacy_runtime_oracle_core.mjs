/** Shared exact-runtime baseline for legacy-collection Spine 4.0/4.2 JSON. */

import assert from 'node:assert/strict';
import { existsSync, readFileSync, statSync } from 'node:fs';
import { resolve } from 'node:path';
import { pathToFileURL } from 'node:url';

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

function requireNonEmptyString(value, path) {
  if (typeof value !== 'string' || !value.trim()) {
    fail(`${path} must be a non-empty string`, { value });
  }
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

function resolveInputFile(value, label) {
  if (typeof value !== 'string' || !value.trim()) fail(`${label} is required`);
  let candidate = resolve(process.cwd(), value);
  if (!existsSync(candidate)) fail(`${label} does not exist: ${candidate}`);
  if (statSync(candidate).isDirectory()) candidate = resolve(candidate, 'index.js');
  if (!existsSync(candidate) || !statSync(candidate).isFile()) {
    fail(`${label} is not a file: ${candidate}`);
  }
  return candidate;
}

function attachmentType(attachment) {
  return typeof attachment.type === 'string' ? attachment.type : 'region';
}

function sequenceRegionNames(basePath, sequence) {
  if (!isRecord(sequence)) return [basePath];
  const count = requirePositiveInteger(sequence.count, 'attachment.sequence.count');
  const start = sequence.start ?? 0;
  const digits = sequence.digits ?? 0;
  if (!Number.isInteger(start) || !Number.isInteger(digits) || digits < 0) {
    fail('attachment sequence start/digits must be valid integers', { sequence });
  }
  return Array.from({ length: count }, (_, index) => {
    const frame = String(start + index).padStart(digits, '0');
    return `${basePath}${frame}`;
  });
}

function collectAtlasRegions(document) {
  const regions = new Set();
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
      for (const [entryName, rawAttachment] of Object.entries(slotAttachments)) {
        const attachment = requireRecord(
          rawAttachment,
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
          regions.add(regionName);
        }
      }
    }
  }
  return [...regions].sort();
}

function collectExpectedSetupAttachments(document) {
  const slots = requireArray(document.slots ?? [], 'document.slots');
  const skins = requireArray(document.skins ?? [], 'document.skins');
  const defaultSkin = skins.find(
    (candidate) => isRecord(candidate) && (candidate.name ?? 'default') === 'default',
  );
  if (!defaultSkin) return [];
  const attachments = requireRecord(defaultSkin.attachments ?? {}, 'defaultSkin.attachments');
  const result = [];
  for (let index = 0; index < slots.length; index += 1) {
    const slot = requireRecord(slots[index], `document.slots[${index}]`);
    const slotName = requireNonEmptyString(slot.name, `document.slots[${index}].name`);
    const setupName = slot.attachment;
    if (setupName === undefined || setupName === null) continue;
    requireNonEmptyString(setupName, `document.slots[${index}].attachment`);
    const slotAttachments = attachments[slotName];
    if (!isRecord(slotAttachments) || !isRecord(slotAttachments[setupName])) {
      fail(`Setup attachment '${setupName}' is missing for slot '${slotName}'`);
    }
    const type = attachmentType(slotAttachments[setupName]);
    if (RENDERABLE_ATTACHMENT_TYPES.has(type)) {
      result.push({ slotName, attachmentName: setupName, type });
    }
  }
  return result;
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
    fail('TextureAtlas must expose at least one page');
  }
  for (let index = 0; index < atlas.pages.length; index += 1) {
    const page = requireRecord(atlas.pages[index], `atlas.pages[${index}]`);
    const width = requirePositiveInteger(page.width, `atlas.pages[${index}].width`);
    const height = requirePositiveInteger(page.height, `atlas.pages[${index}].height`);
    const texture = createOracleTexture(width, height);
    if (typeof page.setTexture === 'function') page.setTexture(texture);
    else page.texture = texture;
    if (!page.texture || typeof page.texture.getImage !== 'function') {
      fail(`atlas.pages[${index}] did not retain its texture`);
    }
  }
}

function readConstraintRecords(document) {
  const records = [];
  for (const collectionName of ['ik', 'transform', 'path']) {
    const values = document[collectionName] ?? [];
    if (!Array.isArray(values)) fail(`document.${collectionName} must be an array`);
    for (let index = 0; index < values.length; index += 1) {
      const constraint = requireRecord(values[index], `document.${collectionName}[${index}]`);
      const name = requireNonEmptyString(
        constraint.name,
        `document.${collectionName}[${index}].name`,
      );
      const order = constraint.order ?? 0;
      if (!Number.isInteger(order) || order < 0) {
        fail(`document.${collectionName}[${index}].order must be non-negative`);
      }
      records.push({ collectionName, name, order });
    }
  }
  const names = records.map((record) => record.name);
  assert.equal(new Set(names).size, names.length, 'Constraint names must be globally unique');
  const orders = records.map((record) => record.order);
  assert.equal(new Set(orders).size, orders.length, 'Constraint orders must be unique');
  assert.deepEqual(
    [...orders].sort((left, right) => left - right),
    Array.from({ length: orders.length }, (_, index) => index),
    'Constraint orders must form 0..N-1',
  );
  return records;
}

function runtimeConstraintObjects(skeleton) {
  return [
    ...(Array.isArray(skeleton.ikConstraints) ? skeleton.ikConstraints : []),
    ...(Array.isArray(skeleton.transformConstraints) ? skeleton.transformConstraints : []),
    ...(Array.isArray(skeleton.pathConstraints) ? skeleton.pathConstraints : []),
  ];
}

function validateUpdateCache(skeleton, expectedRecords) {
  const constraints = runtimeConstraintObjects(skeleton);
  const expectedNames = expectedRecords.map((record) => record.name).sort();
  const runtimeNames = constraints.map((constraint) => constraint.data.name).sort();
  assert.deepEqual(runtimeNames, expectedNames, 'Runtime constraint inventory differs');
  if (!Array.isArray(skeleton._updateCache)) {
    fail('Runtime Skeleton does not expose _updateCache');
  }
  const constraintSet = new Set(constraints);
  const cached = skeleton._updateCache.filter((item) => constraintSet.has(item));
  assert.equal(cached.length, constraints.length, 'Runtime skipped one or more constraints');
  const occurrences = new Map();
  for (const constraint of cached) {
    const name = constraint.data.name;
    occurrences.set(name, (occurrences.get(name) ?? 0) + 1);
  }
  for (const name of expectedNames) {
    assert.equal(occurrences.get(name), 1, `Constraint '${name}' must be cached once`);
  }
  return cached.map((constraint) => constraint.data.name);
}

function setSetupPose(skeleton) {
  if (typeof skeleton.setToSetupPose === 'function') skeleton.setToSetupPose();
  else if (typeof skeleton.setupPose === 'function') skeleton.setupPose();
  else fail('Runtime Skeleton has no setup-pose method');
}

function updateWorldTransform(runtime, skeleton) {
  if (isRecord(runtime.Physics) && typeof runtime.Physics.none === 'number') {
    skeleton.updateWorldTransform(runtime.Physics.none);
  } else {
    skeleton.updateWorldTransform();
  }
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
    };
    for (const [field, value] of Object.entries(values)) {
      requireFinite(value, `runtime.bones[${index}].${field}`);
    }
    snapshots.push({ name: bone.data.name, ...values });
  }
  return snapshots;
}

function runtimeSetupAttachments(runtime, skeleton) {
  const result = [];
  const slots = Array.isArray(skeleton.drawOrder) ? skeleton.drawOrder : skeleton.slots;
  for (let index = 0; index < slots.length; index += 1) {
    const slot = slots[index];
    const attachment =
      typeof slot.getAttachment === 'function'
        ? slot.getAttachment()
        : slot?.appliedPose?.attachment ?? null;
    if (!attachment) continue;
    let type = null;
    if (attachment instanceof runtime.RegionAttachment) type = 'region';
    else if (attachment instanceof runtime.MeshAttachment) type = 'mesh';
    else continue;
    result.push({ slotName: slot.data.name, attachmentName: attachment.name, type });
  }
  return result;
}

function validateSetupAttachments(runtime, skeleton, expected) {
  const actual = runtimeSetupAttachments(runtime, skeleton);
  const expectedKeys = expected
    .map((item) => `${item.slotName}\u0000${item.attachmentName}`)
    .sort();
  const actualKeys = actual
    .map((item) => `${item.slotName}\u0000${item.attachmentName}`)
    .sort();
  assert.deepEqual(actualKeys, expectedKeys, 'Runtime setup attachments differ');
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
    fail('Runtime bounds must be positive', bounds);
  }
  return bounds;
}

export async function runLegacySpine4xOracle(options) {
  const configuration = requireRecord(options, 'options');
  const expectedVersion = requireNonEmptyString(
    configuration.expectedVersion,
    'options.expectedVersion',
  );
  const expectedFamily = requireNonEmptyString(
    configuration.expectedFamily,
    'options.expectedFamily',
  );
  const jsonPath = resolveInputFile(configuration.jsonArgument, 'JSON path');
  const runtimeEntry = resolveInputFile(configuration.runtimeArgument, 'runtime entry');
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
        available: Object.keys(runtime).sort(),
      });
    }
  }

  const document = requireRecord(JSON.parse(readFileSync(jsonPath, 'utf8')), 'document');
  const metadata = requireRecord(document.skeleton, 'document.skeleton');
  const version = requireNonEmptyString(metadata.spine, 'document.skeleton.spine');
  if (version !== expectedVersion || !version.startsWith(expectedFamily)) {
    fail(`Expected Spine ${expectedVersion} JSON, received ${version}`);
  }
  if (Object.prototype.hasOwnProperty.call(document, 'constraints')) {
    fail('Unified constraints are not valid for this legacy-collection oracle');
  }

  const records = readConstraintRecords(document);
  const expectedAttachments = collectExpectedSetupAttachments(document);
  const atlas = new runtime.TextureAtlas(createAtlasText(collectAtlasRegions(document)));
  bindAtlasPageTextures(atlas);

  try {
    const loader = new runtime.AtlasAttachmentLoader(atlas);
    const reader = new runtime.SkeletonJson(loader);
    const skeletonData = reader.readSkeletonData(document);
    const skeleton = new runtime.Skeleton(skeletonData);
    if (skeletonData.defaultSkin) skeleton.setSkin(skeletonData.defaultSkin);
    setSetupPose(skeleton);
    skeleton.updateCache();
    const cacheConstraints = validateUpdateCache(skeleton, records);
    updateWorldTransform(runtime, skeleton);
    const bones = validateBoneMatrices(skeleton);
    const attachments = validateSetupAttachments(
      runtime,
      skeleton,
      expectedAttachments,
    );
    const bounds = setupBounds(runtime, skeleton, attachments);

    return {
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
        scheduledConstraints: cacheConstraints.length,
        everyConstraintScheduledExactlyOnce: true,
      },
      matrices: {
        finiteBones: bones.length,
        allFinite: true,
      },
      bounds,
    };
  } finally {
    if (typeof atlas.dispose === 'function') atlas.dispose();
  }
}

export function writeOracleFailure(error) {
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
}
