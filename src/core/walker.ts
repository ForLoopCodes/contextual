// Gitignore-aware recursive directory walker with depth control
// Returns filtered file paths respecting project ignore patterns (nested-gitignore-aware)

import { readdir, readFile, realpath, stat } from "fs/promises";
import { join, relative, resolve, sep } from "path";
import ignore, { type Ignore } from "ignore";

export interface WalkOptions {
  targetPath?: string;
  depthLimit?: number;
  rootDir: string;
}

export interface FileEntry {
  path: string;
  relativePath: string;
  isDirectory: boolean;
  depth: number;
}

interface IgnoreScope {
  scopeRoot: string;
  ig: Ignore;
}

type IgnoreChain = IgnoreScope[];

const ALWAYS_IGNORE = new Set([
  "node_modules",
  ".git",
  ".svn",
  ".hg",
  "__pycache__",
  ".DS_Store",
  "dist",
  "build",
  ".next",
  ".nuxt",
  "target",
  ".mcp_data",
  ".mcp-shadow-history",
  "coverage",
  ".cache",
  ".turbo",
  ".parcel-cache",
]);

async function loadLocalScope(dir: string): Promise<IgnoreScope | null> {
  try {
    const content = await readFile(join(dir, ".gitignore"), "utf-8");
    return { scopeRoot: dir, ig: ignore().add(content) };
  } catch {
    return null;
  }
}

function isIgnoredInChain(absPath: string, isDir: boolean, chain: IgnoreChain): boolean {
  // Walk scopes from outermost to innermost. Each scope's patterns are evaluated
  // against paths relative to that scope's directory. Later scopes can re-include
  // paths that earlier scopes excluded (gitignore negation crosses scope boundaries).
  let state: "ignored" | "included" = "included";
  for (const scope of chain) {
    let rel = relative(scope.scopeRoot, absPath).replace(/\\/g, "/");
    if (!rel || rel.startsWith("..")) continue;
    // Mark directories with a trailing slash so anchored directory patterns
    // like `/build/` match the directory itself (and short-circuit descent).
    if (isDir) rel += "/";
    const result = scope.ig.test(rel);
    if (result.unignored) state = "included";
    else if (result.ignored) state = "ignored";
  }
  return state === "ignored";
}

async function walkRecursive(
  dir: string,
  rootDir: string,
  chain: IgnoreChain,
  depth: number,
  maxDepth: number,
  results: FileEntry[],
): Promise<void> {
  if (maxDepth > 0 && depth > maxDepth) return;

  const entries = await readdir(dir, { withFileTypes: true }).catch(() => []);
  for (const entry of entries) {
    if (ALWAYS_IGNORE.has(entry.name) || entry.name.startsWith(".")) continue;

    const fullPath = join(dir, entry.name);
    const relPath = relative(rootDir, fullPath).replace(/\\/g, "/");
    const isDir = entry.isDirectory();
    if (isIgnoredInChain(fullPath, isDir, chain)) continue;

    results.push({ path: fullPath, relativePath: relPath, isDirectory: isDir, depth });

    if (isDir) {
      const localScope = await loadLocalScope(fullPath);
      const childChain = localScope ? [...chain, localScope] : chain;
      await walkRecursive(fullPath, rootDir, childChain, depth + 1, maxDepth, results);
    }
  }
}

export async function walkDirectory(options: WalkOptions): Promise<FileEntry[]> {
  const rootDir = resolve(options.rootDir);
  const startDir = options.targetPath ? resolve(rootDir, options.targetPath) : rootDir;
  const results: FileEntry[] = [];

  try {
    await stat(startDir);
  } catch {
    return results;
  }

  // Build the initial chain from rootDir down to startDir so ancestor scopes apply
  // at the start of the walk.
  const chain: IgnoreChain = [];
  const rootScope = await loadLocalScope(rootDir);
  if (rootScope) chain.push(rootScope);

  if (startDir !== rootDir) {
    const segments = relative(rootDir, startDir).split(/[\\/]/).filter(Boolean);
    let cursor = rootDir;
    for (const segment of segments) {
      cursor = join(cursor, segment);
      const scope = await loadLocalScope(cursor);
      if (scope) chain.push(scope);
    }
  }

  await walkRecursive(startDir, rootDir, chain, 0, options.depthLimit ?? 0, results);
  return results;
}

export function groupByDirectory(entries: FileEntry[]): Map<string, FileEntry[]> {
  const groups = new Map<string, FileEntry[]>();
  for (const entry of entries) {
    const dir = entry.relativePath.includes("/")
      ? entry.relativePath.substring(0, entry.relativePath.lastIndexOf("/"))
      : ".";
    const existing = groups.get(dir) ?? [];
    existing.push(entry);
    groups.set(dir, existing);
  }
  return groups;
}

let GLOBAL_EXTRA_ROOTS: string[] = [];

export function setExtraRoots(paths: string[]): void {
  GLOBAL_EXTRA_ROOTS = [...paths];
}

export function getExtraRoots(): string[] {
  return [...GLOBAL_EXTRA_ROOTS];
}

export interface WalkRootsOptions {
  rootDir: string;
  extraRoots?: string[];
  depthLimit?: number;
  targetPath?: string;
}

export async function walkRoots(options: WalkRootsOptions): Promise<FileEntry[]> {
  const rootDir = resolve(options.rootDir);
  let rootReal = rootDir;
  try {
    rootReal = await realpath(rootDir);
  } catch {
    // rootDir doesn't exist; fall through with unresolved value
  }

  const extraRoots = options.extraRoots ?? GLOBAL_EXTRA_ROOTS;
  const seen = new Set<string>();
  const results: FileEntry[] = [];

  const primary = await walkDirectory({
    rootDir,
    depthLimit: options.depthLimit,
    targetPath: options.targetPath,
  });
  for (const entry of primary) {
    if (seen.has(entry.path)) continue;
    seen.add(entry.path);
    results.push(entry);
  }

  // targetPath constrains the primary walk only — extraRoots are always walked in full.
  for (const extra of extraRoots) {
    const extraAbs = resolve(rootDir, extra);
    let extraReal = extraAbs;
    try {
      extraReal = await realpath(extraAbs);
    } catch {
      // doesn't exist; will fall through to the prefix check on unresolved path
    }
    if (extraReal !== rootReal && !extraReal.startsWith(rootReal + sep)) {
      throw new Error(`walkRoots: extraRoot "${extra}" resolves outside workspace root`);
    }
    const depthOffset = relative(rootReal, extraReal).split(/[\\/]/).filter(Boolean).length;
    const extraEntries = await walkDirectory({
      rootDir: extraReal,
      depthLimit: options.depthLimit,
    });
    for (const entry of extraEntries) {
      if (seen.has(entry.path)) continue;
      seen.add(entry.path);
      const workspaceRel = relative(rootReal, entry.path).replace(/\\/g, "/");
      results.push({
        path: entry.path,
        relativePath: workspaceRel,
        isDirectory: entry.isDirectory,
        depth: entry.depth + depthOffset,
      });
    }
  }

  return results;
}
