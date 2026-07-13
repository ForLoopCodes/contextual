import { describe, it, before, after } from "node:test";
import { mkdir, writeFile, rm } from "fs/promises";
import { join, resolve } from "path";

const { walkDirectory, groupByDirectory } =
  await import("../../build/core/walker.js");

const FIXTURE = resolve("test/_demo_walker_fixtures");

before(async () => {
  await mkdir(join(FIXTURE, "src", "components"), { recursive: true });
  await mkdir(join(FIXTURE, "docs"), { recursive: true });
  await writeFile(join(FIXTURE, "README.md"), "# Demo");
  await writeFile(join(FIXTURE, "src", "index.ts"), "export {};");
  await writeFile(join(FIXTURE, "src", "app.ts"), "export {};");
  await writeFile(
    join(FIXTURE, "src", "components", "Button.tsx"),
    "export {};",
  );
  await writeFile(
    join(FIXTURE, "src", "components", "Modal.tsx"),
    "export {};",
  );
  await writeFile(join(FIXTURE, "docs", "guide.md"), "# Guide");
  await writeFile(join(FIXTURE, ".gitignore"), "docs/\n");
});

after(async () => {
  await rm(FIXTURE, { recursive: true, force: true });
});

describe("DEMO: walkDirectory", () => {
  it("INPUT: full walk with depth_limit=3", async () => {
    const input = { rootDir: FIXTURE, depthLimit: 3 };
    console.log("\n--- INPUT ---");
    console.log(JSON.stringify(input, null, 2));

    const entries = await walkDirectory(input);
    const output = entries.map((e) => ({
      path: e.relativePath,
      isDir: e.isDirectory,
      depth: e.depth,
    }));

    console.log("\n--- OUTPUT ---");
    console.log(JSON.stringify(output, null, 2));
    console.log("--- END ---\n");
  });

  it("INPUT: scoped to 'src' with depth_limit=1", async () => {
    const input = { rootDir: FIXTURE, targetPath: "src", depthLimit: 1 };
    console.log("\n--- INPUT ---");
    console.log(JSON.stringify(input, null, 2));

    const entries = await walkDirectory(input);
    const output = entries.map((e) => ({
      path: e.relativePath,
      isDir: e.isDirectory,
      depth: e.depth,
    }));

    console.log("\n--- OUTPUT ---");
    console.log(JSON.stringify(output, null, 2));
    console.log("--- END ---\n");
  });
});

describe("DEMO: groupByDirectory", () => {
  it("INPUT: group walker entries by directory", async () => {
    const entries = await walkDirectory({ rootDir: FIXTURE, depthLimit: 3 });
    const files = entries.filter((e) => !e.isDirectory);

    console.log("\n--- INPUT (file paths) ---");
    console.log(files.map((f) => f.relativePath));

    const grouped = groupByDirectory(files);
    const output = Object.fromEntries(
      [...grouped.entries()].map(([k, v]) => [k, v.map((f) => f.relativePath)]),
    );

    console.log("\n--- OUTPUT ---");
    console.log(JSON.stringify(output, null, 2));
    console.log("--- END ---\n");
  });
});

describe("DEMO: tessera-style monorepo (issue #38)", () => {
  const ROOT = join(process.cwd(), "test", "_demo_tessera");

  before(async () => {
    await rm(ROOT, { recursive: true, force: true });
    await mkdir(join(ROOT, "docs"), { recursive: true });
    await mkdir(join(ROOT, "repos", "lacuna", "src"), { recursive: true });
    await mkdir(join(ROOT, "repos", "graphrag-core", "src"), { recursive: true });
    await mkdir(join(ROOT, "repos", "lacuna", "build"), { recursive: true });
    await writeFile(join(ROOT, ".gitignore"), "repos/\n");
    await writeFile(join(ROOT, "docs", "readme.md"), "docs");
    await writeFile(join(ROOT, "repos", "lacuna", ".gitignore"), "build/\n");
    await writeFile(join(ROOT, "repos", "lacuna", "src", "main.py"), "m");
    await writeFile(join(ROOT, "repos", "lacuna", "build", "junk.py"), "j");
    await writeFile(join(ROOT, "repos", "graphrag-core", "src", "core.py"), "c");
  });

  after(async () => {
    await rm(ROOT, { recursive: true, force: true });
  });

  it("workspace alone indexes only docs/ (the broken case before this PR)", async () => {
    const { walkDirectory } = await import("../../build/core/walker.js");
    const entries = await walkDirectory({ rootDir: ROOT });
    console.log("[demo] without extraRoots: " + entries.map((e) => e.relativePath).sort().join(", "));
  });

  it("with extraRoots, both sub-repos are indexed and each respects its own .gitignore", async () => {
    const { walkRoots } = await import("../../build/core/walker.js");
    const entries = await walkRoots({
      rootDir: ROOT,
      extraRoots: ["repos/lacuna", "repos/graphrag-core"],
    });
    console.log("[demo] with extraRoots:    " + entries.map((e) => e.relativePath).sort().join(", "));
  });
});
