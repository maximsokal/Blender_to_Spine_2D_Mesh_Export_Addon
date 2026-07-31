#!/usr/bin/env node

/**
 * Resolve the official Spine 4.3 TypeScript source tree without building it in place.
 *
 * The 4.3 sources use ESM specifiers ending in `.js`, while a clean checkout contains
 * sibling `.ts` files. Node 24 can transform TypeScript when started with
 * `--experimental-transform-types`; this loader redirects only missing relative `.js`
 * imports inside the explicitly allowed runtime source root to existing `.ts` files.
 *
 * The loader performs no writes and refuses to redirect outside
 * SPINE43_RUNTIME_SOURCE_ROOT.
 */

import { existsSync, statSync } from 'node:fs';
import { dirname, isAbsolute, relative, resolve as resolvePath } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';

function fail(message) {
  throw new Error(message);
}

function resolveAllowedRoot() {
  const configured = process.env.SPINE43_RUNTIME_SOURCE_ROOT;
  if (typeof configured !== 'string' || !configured.trim()) {
    fail('SPINE43_RUNTIME_SOURCE_ROOT must identify spine-core/src');
  }
  const root = resolvePath(configured);
  if (!existsSync(root) || !statSync(root).isDirectory()) {
    fail(`SPINE43_RUNTIME_SOURCE_ROOT is not a directory: ${root}`);
  }
  return root;
}

const ALLOWED_ROOT = resolveAllowedRoot();

function isInsideAllowedRoot(path) {
  const rel = relative(ALLOWED_ROOT, path);
  return (
    rel === '' ||
    (!isAbsolute(rel) && rel !== '..' && !rel.startsWith(`..${process.platform === 'win32' ? '\\' : '/'}`))
  );
}

function redirectedTypeScriptPath(specifier, parentURL) {
  if (
    typeof specifier !== 'string' ||
    (!specifier.startsWith('./') && !specifier.startsWith('../')) ||
    !specifier.endsWith('.js') ||
    typeof parentURL !== 'string' ||
    !parentURL.startsWith('file:')
  ) {
    return null;
  }

  const parentPath = fileURLToPath(parentURL);
  if (!isInsideAllowedRoot(parentPath)) return null;

  const candidate = resolvePath(
    dirname(parentPath),
    `${specifier.slice(0, -3)}.ts`,
  );
  if (!isInsideAllowedRoot(candidate)) return null;
  if (!existsSync(candidate) || !statSync(candidate).isFile()) return null;
  return candidate;
}

export async function resolve(specifier, context, nextResolve) {
  try {
    return await nextResolve(specifier, context);
  } catch (error) {
    const candidate = redirectedTypeScriptPath(specifier, context.parentURL);
    if (candidate === null) throw error;
    return {
      url: pathToFileURL(candidate).href,
      shortCircuit: true,
    };
  }
}
