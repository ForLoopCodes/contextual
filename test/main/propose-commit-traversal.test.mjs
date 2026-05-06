// PoC test: CWE-22 Path Traversal in proposeCommit
// Demonstrates that an agent-supplied filePath with "../" can escape rootDir
// and write arbitrary files outside the project root.

import { describe, it, after, before } from "node:test";
import assert from "node:assert/strict";
import { proposeCommit } from "../../build/tools/propose-commit.js";
import { readFile, mkdir, rm, stat } from "fs/promises";
import { join, resolve } from "path";

const FIXTURE_ROOT = join(process.cwd(), "test", "_traversal_fixtures");
const PROJECT_DIR = join(FIXTURE_ROOT, "project");
const ESCAPE_TARGET = join(FIXTURE_ROOT, "escaped.txt");

describe("CWE-22: Path traversal in proposeCommit", async () => {
  before(async () => {
    await rm(FIXTURE_ROOT, { recursive: true, force: true });
    await mkdir(PROJECT_DIR, { recursive: true });
  });

  it("should reject file_path that escapes rootDir via ../", async () => {
    const maliciousPath = "../escaped.txt";
    const maliciousContent = "PWNED - arbitrary file write outside project root";

    let threw = false;
    try {
      await proposeCommit({
        rootDir: PROJECT_DIR,
        filePath: maliciousPath,
        newContent: maliciousContent,
      });
    } catch (e) {
      threw = true;
      assert.ok(
        e.message.toLowerCase().includes("traversal") ||
        e.message.toLowerCase().includes("outside") ||
        e.message.toLowerCase().includes("path"),
        `Expected path traversal error, got: ${e.message}`
      );
    }

    // Verify the file was NOT written outside rootDir
    let fileExists = false;
    try {
      await stat(ESCAPE_TARGET);
      fileExists = true;
    } catch {
      fileExists = false;
    }

    // Either it should have thrown, or the file should not exist
    assert.ok(threw || !fileExists,
      "proposeCommit should either throw on path traversal or not write the file outside rootDir");

    // Stronger: it MUST throw
    assert.ok(threw,
      "proposeCommit MUST throw an error when file_path escapes rootDir");
  });

  it("should reject absolute paths outside rootDir", async () => {
    const absoluteEscapePath = join(FIXTURE_ROOT, "escaped_abs.txt");

    let threw = false;
    try {
      await proposeCommit({
        rootDir: PROJECT_DIR,
        filePath: absoluteEscapePath,
        newContent: "PWNED via absolute path",
      });
    } catch (e) {
      threw = true;
    }

    assert.ok(threw,
      "proposeCommit MUST throw when an absolute path outside rootDir is provided");
  });

  it("should reject encoded traversal like ../../", async () => {
    const maliciousPath = "subdir/../../escaped_deep.txt";

    let threw = false;
    try {
      await proposeCommit({
        rootDir: PROJECT_DIR,
        filePath: maliciousPath,
        newContent: "PWNED via nested traversal",
      });
    } catch (e) {
      threw = true;
    }

    assert.ok(threw,
      "proposeCommit MUST throw for nested directory traversal");
  });

  it("should still allow valid paths inside rootDir", async () => {
    const content = "// Valid file\n// FEATURE: Test\n\nconst x = 1;\n";
    const result = await proposeCommit({
      rootDir: PROJECT_DIR,
      filePath: "valid/test.ts",
      newContent: content,
    });
    assert.ok(result.includes("✅") || result.includes("saved"),
      "Valid paths inside rootDir should succeed");
    const written = await readFile(join(PROJECT_DIR, "valid", "test.ts"), "utf-8");
    assert.equal(written, content);
  });

  after(async () => {
    await rm(FIXTURE_ROOT, { recursive: true, force: true });
  });
});
