// CLI/env argument parsing for the extraRoots config.
// Pure module — no side effects, safe to import from tests.

import { realpathSync, statSync } from "fs";
import { delimiter, isAbsolute, resolve, sep } from "path";

export interface ParseExtraRootsInput {
  argv: string[];
  env: NodeJS.ProcessEnv | Record<string, string | undefined>;
  rootDir: string;
}

export interface ParseExtraRootsResult {
  accepted: string[];
  warnings: string[];
}

function extractIncludeFlags(argv: string[]): string[] {
  const out: string[] = [];
  for (let i = 0; i < argv.length; i++) {
    if (argv[i] === "--include" && i + 1 < argv.length) {
      out.push(argv[i + 1]);
      i++;
    } else if (argv[i].startsWith("--include=")) {
      out.push(argv[i].slice("--include=".length));
    }
  }
  return out;
}

export function parseExtraRoots(input: ParseExtraRootsInput): ParseExtraRootsResult {
  const accepted: string[] = [];
  const warnings: string[] = [];
  const rootAbs = resolve(input.rootDir);
  let rootReal = rootAbs;
  try {
    rootReal = realpathSync(rootAbs);
  } catch {
    // rootDir doesn't exist; fall through
  }

  const fromCli = extractIncludeFlags(input.argv);
  const raw = fromCli.length > 0
    ? fromCli
    : (input.env.CONTEXTPLUS_EXTRA_ROOTS ?? "")
        .split(delimiter)
        .filter((s) => s.length > 0);

  for (const entry of raw) {
    const abs = isAbsolute(entry) ? entry : resolve(rootAbs, entry);
    let real = abs;
    try {
      real = realpathSync(abs);
    } catch {
      // doesn't exist - statSync below will catch and warn
    }

    if (real === rootReal) {
      warnings.push(`contextplus: extraRoot '${entry}' equals the workspace root — skipping`);
      continue;
    }
    if (!real.startsWith(rootReal + sep)) {
      warnings.push(`contextplus: extraRoot '${entry}' is outside the workspace root — skipping`);
      continue;
    }
    let stats;
    try {
      stats = statSync(real);
    } catch {
      warnings.push(`contextplus: extraRoot '${entry}' does not exist — skipping`);
      continue;
    }
    if (!stats.isDirectory()) {
      warnings.push(`contextplus: extraRoot '${entry}' is not a directory — skipping`);
      continue;
    }
    accepted.push(real);
  }

  return { accepted, warnings };
}
