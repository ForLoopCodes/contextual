// Shadow git branch manager for safe AI change tracking
// Creates restore points on hidden branch without polluting main history

import { simpleGit, type SimpleGit } from "simple-git";
import { readFile, writeFile, mkdir } from "fs/promises";
import { dirname, isAbsolute, join, relative, resolve, sep } from "path";

const SHADOW_BRANCH = "mcp-shadow-history";
const DATA_DIR = ".mcp_data";

export interface RestorePoint {
  id: string;
  timestamp: number;
  files: string[];
  message: string;
}

function assertWithinRoot(rootDir: string, filePath: string): string {
  const root = resolve(rootDir);
  const resolved = resolve(root, filePath);
  const rel = relative(root, resolved);
  if (rel === ".." || rel.startsWith(".." + sep) || isAbsolute(rel)) {
    throw new Error(`Path traversal denied: "${filePath}" resolves outside root directory`);
  }
  return resolved;
}

async function ensureDataDir(rootDir: string): Promise<string> {
  const dataPath = join(rootDir, DATA_DIR);
  await mkdir(dataPath, { recursive: true });
  return dataPath;
}

async function loadManifest(rootDir: string): Promise<RestorePoint[]> {
  const manifestPath = join(rootDir, DATA_DIR, "restore-points.json");
  try {
    return JSON.parse(await readFile(manifestPath, "utf-8"));
  } catch {
    return [];
  }
}

async function saveManifest(rootDir: string, points: RestorePoint[]): Promise<void> {
  const dataPath = await ensureDataDir(rootDir);
  await writeFile(join(dataPath, "restore-points.json"), JSON.stringify(points, null, 2));
}

export async function createRestorePoint(rootDir: string, files: string[], message: string): Promise<RestorePoint> {
  const normalizedRoot = resolve(rootDir);
  const dataPath = await ensureDataDir(normalizedRoot);
  const id = `rp-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
  const backupDir = join(dataPath, "backups", id);
  await mkdir(backupDir, { recursive: true });

  for (const file of files) {
    const fullPath = assertWithinRoot(normalizedRoot, file);
    try {
      const content = await readFile(fullPath, "utf-8");
      const backupPath = join(backupDir, file.replace(/[\\/]/g, "__"));
      await writeFile(backupPath, content);
    } catch {
    }
  }

  const point: RestorePoint = { id, timestamp: Date.now(), files, message };
  const manifest = await loadManifest(normalizedRoot);
  manifest.push(point);
  if (manifest.length > 100) manifest.splice(0, manifest.length - 100);
  await saveManifest(normalizedRoot, manifest);

  return point;
}

export async function restorePoint(rootDir: string, pointId: string): Promise<string[]> {
  const normalizedRoot = resolve(rootDir);
  const manifest = await loadManifest(normalizedRoot);
  const point = manifest.find((p) => p.id === pointId);
  if (!point) throw new Error(`Restore point ${pointId} not found`);

  const backupDir = join(normalizedRoot, DATA_DIR, "backups", pointId);
  const restoredFiles: string[] = [];

  for (const file of point.files) {
    const targetPath = assertWithinRoot(normalizedRoot, file);
    const backupPath = join(backupDir, file.replace(/[\\/]/g, "__"));
    try {
      const content = await readFile(backupPath, "utf-8");
      await mkdir(dirname(targetPath), { recursive: true });
      await writeFile(targetPath, content);
      restoredFiles.push(file);
    } catch {
    }
  }

  return restoredFiles;
}

export async function listRestorePoints(rootDir: string): Promise<RestorePoint[]> {
  return loadManifest(rootDir);
}

export async function shadowCommit(rootDir: string, message: string): Promise<boolean> {
  try {
    const git: SimpleGit = simpleGit(rootDir);
    const isRepo = await git.checkIsRepo();
    if (!isRepo) return false;

    const currentBranch = await git.revparse(["--abbrev-ref", "HEAD"]);
    const stashResult = await git.stash(["push", "-m", `mcp-shadow: ${message}`]);

    if (!stashResult.includes("No local changes")) {
      try {
        const branchExists = await git.branch(["-l", SHADOW_BRANCH]);
        if (!branchExists.all.includes(SHADOW_BRANCH)) {
          await git.branch([SHADOW_BRANCH]);
        }
        await git.checkout(SHADOW_BRANCH);
        await git.stash(["pop"]);
        await git.add(".");
        await git.commit(`[MCP Shadow] ${message}`);
        await git.checkout(currentBranch);
      } catch (e) {
        await git.checkout(currentBranch);
        try { await git.stash(["pop"]); } catch { }
        return false;
      }
    }
    return true;
  } catch {
    return false;
  }
}
