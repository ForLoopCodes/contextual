import { describe, it, before, after } from "node:test";
import assert from "node:assert/strict";
import { writeFile, mkdir, rm } from "fs/promises";
import { join, delimiter } from "path";
import { parseExtraRoots } from "../../build/core/extra-roots.js";

const FIX = join(process.cwd(), "test", "_cli_fixtures");

describe("parseExtraRoots", () => {
  before(async () => {
    await rm(FIX, { recursive: true, force: true });
    await mkdir(join(FIX, "a"), { recursive: true });
    await mkdir(join(FIX, "b"), { recursive: true });
    await writeFile(join(FIX, "not-a-dir.txt"), "");
  });

  after(async () => {
    await rm(FIX, { recursive: true, force: true });
  });

  it("parses repeated --include flags", () => {
    const result = parseExtraRoots({
      argv: ["--include", "a", "--include", "b"],
      env: {},
      rootDir: FIX,
    });
    assert.deepEqual(result.accepted.sort(), [join(FIX, "a"), join(FIX, "b")].sort());
    assert.equal(result.warnings.length, 0);
  });

  it("falls back to env var when no --include flag is present", () => {
    const result = parseExtraRoots({
      argv: [],
      env: { CONTEXTPLUS_EXTRA_ROOTS: ["a", "b"].join(delimiter) },
      rootDir: FIX,
    });
    assert.deepEqual(result.accepted.sort(), [join(FIX, "a"), join(FIX, "b")].sort());
  });

  it("CLI wins entirely when both --include and env are set", () => {
    const result = parseExtraRoots({
      argv: ["--include", "a"],
      env: { CONTEXTPLUS_EXTRA_ROOTS: "b" },
      rootDir: FIX,
    });
    assert.deepEqual(result.accepted, [join(FIX, "a")]);
  });

  it("warns and drops non-existent paths", () => {
    const result = parseExtraRoots({
      argv: ["--include", "missing"],
      env: {},
      rootDir: FIX,
    });
    assert.equal(result.accepted.length, 0);
    assert.equal(result.warnings.length, 1);
    assert.match(result.warnings[0], /missing/);
  });

  it("warns and drops files (not directories)", () => {
    const result = parseExtraRoots({
      argv: ["--include", "not-a-dir.txt"],
      env: {},
      rootDir: FIX,
    });
    assert.equal(result.accepted.length, 0);
    assert.equal(result.warnings.length, 1);
    assert.match(result.warnings[0], /not a directory/i);
  });

  it("warns and drops paths outside the workspace root", () => {
    const result = parseExtraRoots({
      argv: ["--include", "/tmp"],
      env: {},
      rootDir: FIX,
    });
    assert.equal(result.accepted.length, 0);
    assert.equal(result.warnings.length, 1);
    assert.match(result.warnings[0], /outside/i);
  });

  it("rejects the workspace root itself", () => {
    const result = parseExtraRoots({
      argv: ["--include", "."],
      env: {},
      rootDir: FIX,
    });
    assert.equal(result.accepted.length, 0);
    assert.equal(result.warnings.length, 1);
  });

  it("skips empty entries in env list", () => {
    const result = parseExtraRoots({
      argv: [],
      env: { CONTEXTPLUS_EXTRA_ROOTS: `a${delimiter}${delimiter}b` },
      rootDir: FIX,
    });
    assert.deepEqual(result.accepted.sort(), [join(FIX, "a"), join(FIX, "b")].sort());
  });

  it("rejects symlinks that point outside the workspace root", async () => {
    const { symlink } = await import("fs/promises");
    const OUTSIDE = join(FIX, "..", "_outside_target");
    await rm(OUTSIDE, { recursive: true, force: true });
    await mkdir(OUTSIDE, { recursive: true });
    try {
      await symlink(OUTSIDE, join(FIX, "bad-link"));
    } catch {
      // If symlink creation fails (e.g., on a filesystem that doesn't support symlinks),
      // skip the test gracefully.
      return;
    }

    const result = parseExtraRoots({
      argv: ["--include", "bad-link"],
      env: {},
      rootDir: FIX,
    });

    assert.equal(result.accepted.length, 0, "symlink to outside should be rejected");
    assert.equal(result.warnings.length, 1);
    assert.match(result.warnings[0], /outside/i);

    await rm(join(FIX, "bad-link"));
    await rm(OUTSIDE, { recursive: true, force: true });
  });
});
