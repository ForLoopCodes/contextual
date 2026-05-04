// Gitignore-aware recursive directory walker with depth control
// Returns filtered file paths respecting project ignore patterns

import { readdir, readFile, realpath, stat } from "fs/promises";
import { isAbsolute, join, relative, resolve } from "path";
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

function isWithinRoot(rootDir: string, targetPath: string): boolean {
  const relPath = relative(rootDir, targetPath);
  return relPath === "" || (!relPath.startsWith("..") && !isAbsolute(relPath));
}

async function loadIgnoreRules(rootDir: string): Promise<Ignore> {
  const ig = ignore();
  try {
    const content = await readFile(join(rootDir, ".gitignore"), "utf-8");
    ig.add(content);
  } catch {
  }
  return ig;
}

async function walkRecursive(
  dir: string,
  rootDir: string,
  ig: Ignore,
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
    if (ig.ignores(relPath)) continue;

    const isDir = entry.isDirectory();
    results.push({ path: fullPath, relativePath: relPath, isDirectory: isDir, depth });

    if (isDir) await walkRecursive(fullPath, rootDir, ig, depth + 1, maxDepth, results);
  }
}

export async function walkDirectory(options: WalkOptions): Promise<FileEntry[]> {
  const rootDir = resolve(options.rootDir);
  const startDir = options.targetPath ? resolve(rootDir, options.targetPath) : rootDir;
  const results: FileEntry[] = [];

  let rootRealPath: string;
  let startRealPath: string;
  try {
    [rootRealPath, startRealPath] = await Promise.all([realpath(rootDir), realpath(startDir)]);
    await stat(startRealPath);
  } catch {
    return results;
  }

  if (!isWithinRoot(rootRealPath, startRealPath)) {
    throw new Error(`Path traversal denied: "${options.targetPath}" resolves outside root directory`);
  }

  const ig = await loadIgnoreRules(rootRealPath);
  await walkRecursive(startRealPath, rootRealPath, ig, 0, options.depthLimit ?? 0, results);
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
