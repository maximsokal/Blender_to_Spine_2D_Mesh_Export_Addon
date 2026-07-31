#!/usr/bin/env node
/** Trace the first Spine 4.2 update-cache item that creates a non-finite bone matrix. */

import { existsSync, readFileSync, statSync } from 'node:fs';
import { resolve } from 'node:path';
import { pathToFileURL } from 'node:url';

const RENDERABLE_ATTACHMENT_TYPES = new Set(['region', 'mesh', 'linkedmesh']);
const MATRIX_FIELDS = ['worldX', 'worldY', 'a', 'b', 'c', 'd'];

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
  }
}

function setSetupPose(skeleton) {
  if (typeof skeleton.setToSetupPose === 'function') skeleton.setToSetupPose();
  else if (typeof skeleton.setupPose === 'function') skeleton.setupPose();
  else fail('Runtime Skeleton has no setup-pose method');
}

function diagnosticNumber(value) {
  if (typeof value !== 'number') return value;
  if (Number.isNaN(value)) return 'NaN';
  if (value === Number.POSITIVE_INFINITY) return 'Infinity';
  if (value === Number.NEGATIVE_INFINITY) return '-Infinity';
  return value;
}

function boneDiagnostic(bone, index) {
  const data = bone?.data;
  return {
    index,
    boneName: data?.name ?? null,
    parentName: bone?.parent?.data?.name ?? null,
    setup: {
      x: diagnosticNumber(data?.x),
      y: diagnosticNumber(data?.y),
      rotation: diagnosticNumber(data?.rotation),
      scaleX: diagnosticNumber(data?.scaleX),
      scaleY: diagnosticNumber(data?.scaleY),
      shearX: diagnosticNumber(data?.shearX),
      shearY: diagnosticNumber(data?.shearY),
      transformMode: data?.transformMode ?? null,
    },
    local: {
      x: diagnosticNumber(bone?.x),
      y: diagnosticNumber(bone?.y),
      rotation: diagnosticNumber(bone?.rotation),
      scaleX: diagnosticNumber(bone?.scaleX),
      scaleY: diagnosticNumber(bone?.scaleY),
      shearX: diagnosticNumber(bone?.shearX),
      shearY: diagnosticNumber(bone?.shearY),
    },
    applied: {
      ax: diagnosticNumber(bone?.ax),
      ay: diagnosticNumber(bone?.ay),
      arotation: diagnosticNumber(bone?.arotation),
      ascaleX: diagnosticNumber(bone?.ascaleX),
      ascaleY: diagnosticNumber(bone?.ascaleY),
      ashearX: diagnosticNumber(bone?.ashearX),
      ashearY: diagnosticNumber(bone?.ashearY),
    },
    world: {
      worldX: diagnosticNumber(bone?.worldX),
      worldY: diagnosticNumber(bone?.worldY),
      a: diagnosticNumber(bone?.a),
      b: diagnosticNumber(bone?.b),
      c: diagnosticNumber(bone?.c),
      d: diagnosticNumber(bone?.d),
    },
  };
}

function constraintCollections(skeleton) {
  return [
    ['ik', skeleton.ikConstraints],
    ['transform', skeleton.transformConstraints],
    ['path', skeleton.pathConstraints],
    ['physics', skeleton.physicsConstraints],
  ];
}

function constraintDiagnostic(collection, constraint, index) {
  const data = constraint?.data;
  return {
    kind: 'constraint',
    collection,
    index,
    name: data?.name ?? null,
    order: data?.order ?? null,
    local: data?.local ?? null,
    relative: data?.relative ?? null,
    bones: Array.isArray(constraint?.bones)
      ? constraint.bones.map((bone) => bone?.data?.name ?? null)
      : constraint?.bone?.data?.name
        ? [constraint.bone.data.name]
        : [],
    target:
      constraint?.target?.data?.name ??
      constraint?.target?.bone?.data?.name ??
      null,
  };
}

function runtimeConstraintDiagnostics(skeleton) {
  const result = [];
  for (const [collection, constraints] of constraintCollections(skeleton)) {
    if (!Array.isArray(constraints)) continue;
    for (let index = 0; index < constraints.length; index += 1) {
      result.push(constraintDiagnostic(collection, constraints[index], index));
    }
  }
  result.sort((left, right) => {
    const leftOrder = Number.isInteger(left.order) ? left.order : Number.MAX_SAFE_INTEGER;
    const rightOrder = Number.isInteger(right.order) ? right.order : Number.MAX_SAFE_INTEGER;
    if (leftOrder !== rightOrder) return leftOrder - rightOrder;
    if (left.collection !== right.collection) {
      return left.collection.localeCompare(right.collection);
    }
    return left.index - right.index;
  });
  return result;
}

function updateItemDiagnostic(skeleton, item, cacheIndex) {
  const boneIndex = skeleton.bones.indexOf(item);
  if (boneIndex >= 0) {
    return {
      kind: 'bone',
      cacheIndex,
      bone: boneDiagnostic(item, boneIndex),
    };
  }
  for (const [collection, constraints] of constraintCollections(skeleton)) {
    if (!Array.isArray(constraints)) continue;
    const index = constraints.indexOf(item);
    if (index >= 0) {
      return {
        cacheIndex,
        ...constraintDiagnostic(collection, item, index),
      };
    }
  }
  return {
    kind: item?.constructor?.name ?? typeof item,
    cacheIndex,
    name: item?.data?.name ?? null,
  };
}

function firstNonFiniteBone(skeleton) {
  for (let index = 0; index < skeleton.bones.length; index += 1) {
    const bone = skeleton.bones[index];
    for (const field of MATRIX_FIELDS) {
      const value = bone?.[field];
      if (typeof value !== 'number' || !Number.isFinite(value)) {
        return {
          ...boneDiagnostic(bone, index),
          field,
          value: diagnosticNumber(value),
        };
      }
    }
  }
  return null;
}

function initializeAppliedPose(skeleton) {
  for (const bone of skeleton.bones) {
    bone.ax = bone.x;
    bone.ay = bone.y;
    bone.arotation = bone.rotation;
    bone.ascaleX = bone.scaleX;
    bone.ascaleY = bone.scaleY;
    bone.ashearX = bone.shearX;
    bone.ashearY = bone.shearY;
  }
}

function traceUpdateCache(runtime, skeleton) {
  if (!Array.isArray(skeleton._updateCache)) {
    fail('Runtime Skeleton does not expose _updateCache');
  }
  if (!isRecord(runtime.Physics) || typeof runtime.Physics.none !== 'number') {
    fail('Spine 4.2 runtime does not expose Physics.none');
  }

  initializeAppliedPose(skeleton);
  const initialFailure = firstNonFiniteBone(skeleton);
  if (initialFailure !== null) {
    return {
      found: true,
      stage: 'BEFORE_UPDATE_CACHE',
      bone: initialFailure,
      item: null,
      cachePrefix: [],
    };
  }

  const cachePrefix = [];
  for (let cacheIndex = 0; cacheIndex < skeleton._updateCache.length; cacheIndex += 1) {
    const item = skeleton._updateCache[cacheIndex];
    const itemDiagnostic = updateItemDiagnostic(skeleton, item, cacheIndex);
    if (typeof item?.update !== 'function') {
      fail(`updateCache[${cacheIndex}] has no update() method`, { item: itemDiagnostic });
    }

    item.update(runtime.Physics.none);
    cachePrefix.push(itemDiagnostic);
    const firstFailure = firstNonFiniteBone(skeleton);
    if (firstFailure !== null) {
      const firstNearby = Math.max(0, firstFailure.index - 2);
      const lastNearby = Math.min(skeleton.bones.length, firstFailure.index + 3);
      return {
        found: true,
        stage: 'UPDATE_CACHE_ITEM',
        cacheIndex,
        item: itemDiagnostic,
        bone: firstFailure,
        nearbyBones: skeleton.bones
          .slice(firstNearby, lastNearby)
          .map((bone, offset) => boneDiagnostic(bone, firstNearby + offset)),
        cachePrefix,
      };
    }
  }

  return {
    found: false,
    stage: 'COMPLETE',
    cachePrefix,
  };
}

async function main(argv) {
  if (!Array.isArray(argv) || argv.length !== 2) {
    fail('Usage: node tools/spine42_runtime_step_trace.mjs <json> <runtime-entry>');
  }
  const jsonPath = resolveInputFile(argv[0], 'JSON path');
  const runtimeEntry = resolveInputFile(argv[1], 'runtime entry');
  const runtime = await import(pathToFileURL(runtimeEntry).href);

  for (const exportName of [
    'TextureAtlas',
    'AtlasAttachmentLoader',
    'SkeletonJson',
    'Skeleton',
  ]) {
    if (typeof runtime[exportName] !== 'function') {
      fail(`Runtime is missing required export '${exportName}'`, {
        available: Object.keys(runtime).sort(),
      });
    }
  }

  const document = requireRecord(JSON.parse(readFileSync(jsonPath, 'utf8')), 'document');
  const skeletonMetadata = requireRecord(document.skeleton, 'document.skeleton');
  const jsonVersion = requireNonEmptyString(
    skeletonMetadata.spine,
    'document.skeleton.spine',
  );
  if (jsonVersion !== '4.2.43') {
    fail(`Expected Spine 4.2.43 JSON, received ${jsonVersion}`);
  }

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

    const trace = traceUpdateCache(runtime, skeleton);
    console.log(
      JSON.stringify(
        {
          ok: true,
          jsonPath,
          runtimeEntry,
          version: skeletonData.version,
          trace,
          constraints: runtimeConstraintDiagnostics(skeleton),
        },
        null,
        2,
      ),
    );
  } finally {
    if (typeof atlas.dispose === 'function') atlas.dispose();
  }
}

main(process.argv.slice(2)).catch((error) => {
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
