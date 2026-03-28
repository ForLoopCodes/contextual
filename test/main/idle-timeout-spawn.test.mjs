import { describe, it } from "node:test";
import assert from "node:assert/strict";
import { spawn } from "node:child_process";
import { resolve, dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { mkdtempSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import {
  createIdleMonitor,
} from "../../build/core/process-lifecycle.js";

const PROJECT_ROOT = resolve(dirname(fileURLToPath(import.meta.url)), "../..");

function wait(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function createTestScript(withFix) {
  const buildPath = join(PROJECT_ROOT, "build/core/process-lifecycle.js").replace(/\\/g, "/");
  return `
    import { createIdleMonitor } from "file://${buildPath}";

    const idleMonitor = createIdleMonitor({
      timeoutMs: 200,
      onIdle: () => {
        process.stderr.write("IDLE_SHUTDOWN\\n");
        process.exit(0);
      },
      ${withFix ? 'isTransportAlive: () => process.stdin.readable && !process.stdin.destroyed,' : ''}
    });

    process.stderr.write("STARTED\\n");
    const keepAlive = setInterval(() => {}, 1000);
    setTimeout(() => {
      idleMonitor.stop();
      clearInterval(keepAlive);
      process.stderr.write("SURVIVED\\n");
      process.exit(0);
    }, 1500);
  `;
}

function runHarness(withFix) {
  return new Promise((resolve) => {
    const tmpDir = mkdtempSync(join(tmpdir(), "cp-test-"));
    const scriptPath = join(tmpDir, "harness.mjs");
    writeFileSync(scriptPath, createTestScript(withFix));

    const child = spawn("node", [scriptPath], {
      stdio: ["pipe", "pipe", "pipe"],
    });

    let stderr = "";
    child.stderr.on("data", (d) => { stderr += d.toString(); });

    child.on("exit", (code) => {
      resolve({ code, stderr });
    });
  });
}

describe("idle-timeout transport-aware fix", () => {
  it("does NOT fire onIdle when isTransportAlive returns true", async () => {
    let idleFired = 0;
    const monitor = createIdleMonitor({
      timeoutMs: 30,
      onIdle: () => { idleFired += 1; },
      isTransportAlive: () => true,
    });
    await wait(80);
    assert.equal(idleFired, 0, "onIdle should not fire when transport is alive");
    monitor.stop();
  });

  it("fires onIdle when isTransportAlive returns false", async () => {
    let idleFired = 0;
    const monitor = createIdleMonitor({
      timeoutMs: 30,
      onIdle: () => { idleFired += 1; },
      isTransportAlive: () => false,
    });
    await wait(80);
    assert.equal(idleFired, 1, "onIdle should fire when transport is dead");
    monitor.stop();
  });

  it("fires onIdle normally when no isTransportAlive provided (backward compat)", async () => {
    let idleFired = 0;
    const monitor = createIdleMonitor({
      timeoutMs: 30,
      onIdle: () => { idleFired += 1; },
    });
    await wait(80);
    assert.equal(idleFired, 1, "onIdle should fire with no transport check");
    monitor.stop();
  });

  it("reschedules then fires when transport dies after initial alive check", async () => {
    let transportAlive = true;
    let idleFired = 0;
    const monitor = createIdleMonitor({
      timeoutMs: 30,
      onIdle: () => { idleFired += 1; },
      isTransportAlive: () => transportAlive,
    });
    await wait(50);
    assert.equal(idleFired, 0, "should not fire while transport alive");
    transportAlive = false;
    await wait(50);
    assert.equal(idleFired, 1, "should fire after transport dies");
    monitor.stop();
  });

  it("touch resets the idle timer even with transport check", async () => {
    let idleFired = 0;
    const monitor = createIdleMonitor({
      timeoutMs: 40,
      onIdle: () => { idleFired += 1; },
      isTransportAlive: () => false,
    });
    await wait(20);
    monitor.touch();
    await wait(20);
    assert.equal(idleFired, 0, "touch should reset timer");
    await wait(30);
    assert.equal(idleFired, 1, "should fire after full timeout post-touch");
    monitor.stop();
  });

  it("spawn: without isTransportAlive, server exits on idle with stdin open", async () => {
    const result = await runHarness(false);
    assert.equal(result.code, 0);
    assert.ok(result.stderr.includes("IDLE_SHUTDOWN"),
      "server idle-shutdown with stdin open (no transport check)");
    assert.ok(!result.stderr.includes("SURVIVED"),
      "server died before survival window");
  });

  it("spawn: with isTransportAlive, server survives idle when stdin is open", async () => {
    const result = await runHarness(true);
    assert.equal(result.code, 0);
    assert.ok(!result.stderr.includes("IDLE_SHUTDOWN"),
      "server should NOT idle-shutdown when transport alive");
    assert.ok(result.stderr.includes("SURVIVED"),
      "server should survive past idle timeout");
  });
});
