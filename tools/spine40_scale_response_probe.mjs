#!/usr/bin/env node

/**
 * Validate generated two-axis scale controls with the exact read-only Spine 4.0 runtime.
 *
 * Usage:
 *   node tools/spine40_scale_response_probe.mjs <json-file> <runtime-entry>
 *
 * The probe changes only in-memory runtime bones. It never writes to the JSON file or to
 * the external runtime repository.
 */

import { existsSync, readFileSync, statSync } from 'node:fs';
import { resolve } from 'node:path';
import { pathToFileURL } from 'node:url';

const SCALE_SUFFIX = '_scale';
const SCALE_FACTORS = Object.freeze([0.5, 1.5, 2.0]);
const RELATIVE_TOLERANCE = 0.005;
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

function createAtlasText(regions) {
  const header = [
    'scale-probe.png',
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
    fail('Spine 4.0 TextureAtlas must expose at least one page');
  }
  for (let index = 0; index < atlas.pages.length; index += 1) {
    const page = requireRecord(atlas.pages[index], `atlas.pages[${index}]`);
    if (typeof page.setTexture !== 'function') {
      fail(`atlas.pages[${index}].setTexture must be a function`);
    }
    page.setTexture(createOracleTexture(page.width, page.height));
  }
}

function collectScaleControlSpecs(document) {
  const transforms = requireArray(document.transform ?? [], 'document.transform');
  const slots = requireArray(document.slots ?? [], 'document.slots');
  const boneNames = new Set(
    requireArray(document.bones ?? [], 'document.bones').map((value, index) =>
      requireNonEmptyString(
        requireRecord(value, `document.bones[${index}]`).name,
        `document.bones[${index}].name`,
      ),
    ),
  );
  const slotNames = slots.map((value, index) =>
    requireNonEmptyString(
      requireRecord(value, `document.slots[${index}]`).name,
      `document.slots[${index}].name`,
    ),
  );

  const result = [];
  for (let index = 0; index < transforms.length; index += 1) {
    const constraint = requireRecord(
      transforms[index],
      `document.transform[${index}]`,
    );
    const name = requireNonEmptyString(
      constraint.name,
      `document.transform[${index}].name`,
    );
    const target = requireNonEmptyString(
      constraint.target,
      `document.transform[${index}].target`,
    );
    if (name !== target || !name.endsWith(SCALE_SUFFIX)) continue;
    if (constraint.relative !== true) {
      fail(`Scale constraint '${name}' must remain relative-world`, { constraint });
    }
    if (constraint.local === true) {
      fail(`Scale constraint '${name}' must not use local evaluation`, { constraint });
    }
    const prefix = name.slice(0, -SCALE_SUFFIX.length);
    if (!prefix) fail(`Scale constraint '${name}' has an empty prefix`);
    const driver = `${prefix}_scale_rotate_X`;
    const unsafeDriver = `${prefix}_rotate_X`;
    const constrainedBones = requireArray(
      constraint.bones,
      `document.transform[${index}].bones`,
    );
    if (!constrainedBones.includes(driver) || constrainedBones.includes(unsafeDriver)) {
      fail(`Scale constraint '${name}' uses the wrong Spine 4.0 driver`, {
        expectedDriver: driver,
        unsafeDriver,
        actual: constrainedBones,
      });
    }
    if (!boneNames.has(name) || !boneNames.has(`${prefix}_main`)) {
      fail(`Scale constraint '${name}' is missing its control or main bone`);
    }
    const ownedSlots = slotNames.filter((slotName) => slotName.startsWith(`${prefix}_`));
    if (ownedSlots.length === 0) {
      fail(`Scale control '${name}' owns no setup slots`, { prefix, slotNames });
    }
    result.push(
      Object.freeze({
        prefix,
        controlBoneName: name,
        mainBoneName: `${prefix}_main`,
        slotNames: Object.freeze(ownedSlots),
      }),
    );
  }

  if (result.length === 0) {
    fail('No generated two-axis scale controls were found');
  }
  const controls = result.map((item) => item.controlBoneName);
  if (new Set(controls).size !== controls.length) {
    fail('Generated scale control names must be unique', { controls });
  }
  return result;
}

function validateBoneMatrices(skeleton, label) {
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
      requireFinite(value, `${label}.bones[${index}](${bone.data.name}).${field}`);
    }
  }
}

function isolateSlots(skeleton, allowedSlotNames) {
  const allowed = new Set(allowedSlotNames);
  let visible = 0;
  for (const slot of skeleton.slots) {
    if (allowed.has(slot.data.name)) {
      if (slot.getAttachment()) visible += 1;
      continue;
    }
    if (typeof slot.setAttachment !== 'function') {
      fail(`Runtime slot '${slot.data.name}' has no setAttachment method`);
    }
    slot.setAttachment(null);
  }
  if (visible === 0) {
    fail('Scale probe selected no setup renderable attachments', {
      allowedSlotNames: [...allowed],
    });
  }
}

function selectedBounds(runtime, skeleton, label) {
  const offset = new runtime.Vector2();
  const size = new runtime.Vector2();
  skeleton.getBounds(offset, size);
  const bounds = {
    x: requireFinite(offset.x, `${label}.x`),
    y: requireFinite(offset.y, `${label}.y`),
    width: requireFinite(size.x, `${label}.width`),
    height: requireFinite(size.y, `${label}.height`),
  };
  if (bounds.width <= 0 || bounds.height <= 0) {
    fail(`${label} must have positive width and height`, bounds);
  }
  return bounds;
}

function approximatelyEqual(actual, expected, scale, label) {
  const tolerance = Math.max(1e-4, Math.abs(scale) * RELATIVE_TOLERANCE);
  if (Math.abs(actual - expected) > tolerance) {
    fail(`${label} differs from uniform-scale expectation`, {
      actual,
      expected,
      tolerance,
    });
  }
}

function setupIsolatedPose(runtime, skeleton, spec) {
  skeleton.setToSetupPose();
  skeleton.updateWorldTransform();
  validateBoneMatrices(skeleton, `${spec.prefix}.setup`);
  isolateSlots(skeleton, spec.slotNames);
  const main = skeleton.findBone(spec.mainBoneName);
  if (!main) fail(`Runtime is missing main bone '${spec.mainBoneName}'`);
  return {
    pivotX: requireFinite(main.worldX, `${spec.mainBoneName}.worldX`),
    pivotY: requireFinite(main.worldY, `${spec.mainBoneName}.worldY`),
    bounds: selectedBounds(runtime, skeleton, `${spec.prefix}.setupBounds`),
  };
}

function scaledIsolatedPose(runtime, skeleton, spec, factor) {
  skeleton.setToSetupPose();
  const control = skeleton.findBone(spec.controlBoneName);
  if (!control) fail(`Runtime is missing scale control '${spec.controlBoneName}'`);
  control.scaleX *= factor;
  control.scaleY *= factor;
  skeleton.updateWorldTransform();
  validateBoneMatrices(skeleton, `${spec.prefix}.scale${factor}`);
  isolateSlots(skeleton, spec.slotNames);
  return selectedBounds(runtime, skeleton, `${spec.prefix}.scale${factor}.bounds`);
}

function probeScaleControls(runtime, skeleton, specs) {
  const probes = [];
  for (const spec of specs) {
    const setup = setupIsolatedPose(runtime, skeleton, spec);
    const samples = [];
    for (const factor of SCALE_FACTORS) {
      const bounds = scaledIsolatedPose(runtime, skeleton, spec, factor);
      const expected = {
        x: setup.pivotX + (setup.bounds.x - setup.pivotX) * factor,
        y: setup.pivotY + (setup.bounds.y - setup.pivotY) * factor,
        width: setup.bounds.width * factor,
        height: setup.bounds.height * factor,
      };
      approximatelyEqual(
        bounds.x,
        expected.x,
        setup.bounds.width,
        `${spec.prefix}.x@${factor}`,
      );
      approximatelyEqual(
        bounds.y,
        expected.y,
        setup.bounds.height,
        `${spec.prefix}.y@${factor}`,
      );
      approximatelyEqual(
        bounds.width,
        expected.width,
        setup.bounds.width,
        `${spec.prefix}.width@${factor}`,
      );
      approximatelyEqual(
        bounds.height,
        expected.height,
        setup.bounds.height,
        `${spec.prefix}.height@${factor}`,
      );
      samples.push(Object.freeze({ factor, bounds, expected }));
    }
    probes.push(
      Object.freeze({
        prefix: spec.prefix,
        controlBoneName: spec.controlBoneName,
        mainBoneName: spec.mainBoneName,
        slotCount: spec.slotNames.length,
        setup: Object.freeze(setup),
        samples: Object.freeze(samples),
      }),
    );
  }
  skeleton.setToSetupPose();
  skeleton.updateWorldTransform();
  return Object.freeze({
    controls: probes.length,
    factors: SCALE_FACTORS,
    allFinite: true,
    allUniformAroundMain: true,
    probes: Object.freeze(probes),
  });
}

async function main() {
  const jsonPath = resolveInputFile(process.argv[2], 'Spine JSON path');
  const runtimeEntry = resolveInputFile(process.argv[3], 'Spine 4.0 runtime entry');
  if (process.argv.length > 4) fail(`Unknown argument: ${process.argv[4]}`);

  const runtime = await import(pathToFileURL(runtimeEntry).href);
  for (const exportName of [
    'TextureAtlas',
    'AtlasAttachmentLoader',
    'SkeletonJson',
    'Skeleton',
    'Vector2',
  ]) {
    if (typeof runtime[exportName] !== 'function') {
      fail(`Runtime is missing required export '${exportName}'`);
    }
  }

  const document = requireRecord(
    JSON.parse(readFileSync(jsonPath, 'utf8')),
    'document',
  );
  const metadata = requireRecord(document.skeleton, 'document.skeleton');
  const version = requireNonEmptyString(metadata.spine, 'document.skeleton.spine');
  if (!version.startsWith('4.0')) {
    fail(`Expected Spine 4.0 JSON, received ${version}`);
  }
  const specs = collectScaleControlSpecs(document);
  const atlas = new runtime.TextureAtlas(createAtlasText(collectAtlasRegions(document)));
  bindAtlasPageTextures(atlas);

  try {
    const loader = new runtime.AtlasAttachmentLoader(atlas);
    const reader = new runtime.SkeletonJson(loader);
    const skeletonData = reader.readSkeletonData(document);
    const skeleton = new runtime.Skeleton(skeletonData);
    if (skeletonData.defaultSkin) skeleton.setSkin(skeletonData.defaultSkin);
    skeleton.updateCache();
    const scaleBehavior = probeScaleControls(runtime, skeleton, specs);
    console.info(
      JSON.stringify(
        {
          ok: true,
          jsonPath,
          runtimeEntry,
          version: skeletonData.version,
          scaleBehavior,
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
