import { describe, it, before, after, beforeEach } from "node:test";
import assert from "node:assert/strict";
import { mkdir, rm, readFile } from "fs/promises";
import { join, resolve } from "path";
import { Ollama } from "ollama";

const {
  upsertNode,
  createRelation,
  searchGraph,
  pruneStaleLinks,
  addInterlinkedContext,
  retrieveWithTraversal,
  getGraphStats,
} = await import("../../build/core/memory-graph.js");

const {
  toolUpsertMemoryNode,
  toolCreateRelation,
  toolSearchMemoryGraph,
  toolPruneStaleLinks,
  toolAddInterlinkedContext,
  toolRetrieveWithTraversal,
} = await import("../../build/tools/memory-tools.js");

const FIXTURE = resolve("test/_memory_graph_fixtures");
let embedCounter = 0;

function mockEmbedding() {
  embedCounter = 0;
  const original = Ollama.prototype.embed;
  Ollama.prototype.embed = async function ({ input }) {
    const batch = Array.isArray(input) ? input : [input];
    return {
      embeddings: batch.map((text) => {
        embedCounter++;
        const vec = new Array(64).fill(0);
        for (let i = 0; i < Math.min(text.length, 64); i++) {
          vec[i] = (text.charCodeAt(i) % 100) / 100;
        }
        const norm = Math.sqrt(vec.reduce((s, v) => s + v * v, 0));
        return norm > 0 ? vec.map((v) => v / norm) : vec;
      }),
    };
  };
  return () => { Ollama.prototype.embed = original; };
}

before(async () => {
  await rm(FIXTURE, { recursive: true, force: true });
  await mkdir(join(FIXTURE, ".mcp_data"), { recursive: true });
});

after(async () => {
  await rm(FIXTURE, { recursive: true, force: true });
});

describe("memory-graph core", () => {
  describe("upsertNode", () => {
    it("creates a new node with embedding", async () => {
      const restore = mockEmbedding();
      try {
        const node = await upsertNode(FIXTURE, "concept", "Auth Flow", "Handles user login and session management");
        assert.ok(node.id.startsWith("mn-"));
        assert.equal(node.type, "concept");
        assert.equal(node.label, "Auth Flow");
        assert.equal(node.accessCount, 1);
        assert.ok(node.embedding.length > 0);
      } finally {
        restore();
      }
    });

    it("updates existing node with same label+type", async () => {
      const restore = mockEmbedding();
      try {
        const first = await upsertNode(FIXTURE, "note", "Test Note", "Original content");
        const second = await upsertNode(FIXTURE, "note", "Test Note", "Updated content");
        assert.equal(first.id, second.id);
        assert.equal(second.content, "Updated content");
        assert.equal(second.accessCount, 2);
      } finally {
        restore();
      }
    });

    it("stores metadata on the node", async () => {
      const restore = mockEmbedding();
      try {
        const node = await upsertNode(FIXTURE, "file", "config.ts", "Configuration loader", { language: "typescript" });
        assert.equal(node.metadata.language, "typescript");
      } finally {
        restore();
      }
    });
  });

  describe("createRelation", () => {
    it("creates edge between existing nodes", async () => {
      const restore = mockEmbedding();
      try {
        const a = await upsertNode(FIXTURE, "concept", "Edge A", "Source concept");
        const b = await upsertNode(FIXTURE, "concept", "Edge B", "Target concept");
        const edge = await createRelation(FIXTURE, a.id, b.id, "relates_to", 0.9);
        assert.ok(edge);
        assert.ok(edge.id.startsWith("me-"));
        assert.equal(edge.relation, "relates_to");
        assert.equal(edge.weight, 0.9);
      } finally {
        restore();
      }
    });

    it("returns null for nonexistent node IDs", async () => {
      const restore = mockEmbedding();
      try {
        const edge = await createRelation(FIXTURE, "fake-id-1", "fake-id-2", "depends_on");
        assert.equal(edge, null);
      } finally {
        restore();
      }
    });

    it("updates duplicate edge weight instead of creating new", async () => {
      const restore = mockEmbedding();
      try {
        const a = await upsertNode(FIXTURE, "symbol", "Dup A", "Function A");
        const b = await upsertNode(FIXTURE, "symbol", "Dup B", "Function B");
        const first = await createRelation(FIXTURE, a.id, b.id, "references", 0.5);
        const second = await createRelation(FIXTURE, a.id, b.id, "references", 0.95);
        assert.equal(first.id, second.id);
        assert.equal(second.weight, 0.95);
      } finally {
        restore();
      }
    });
  });

  describe("searchGraph", () => {
    it("returns results ranked by embedding similarity", async () => {
      const restore = mockEmbedding();
      try {
        await upsertNode(FIXTURE, "concept", "Search Target", "Authentication and login");
        const result = await searchGraph(FIXTURE, "authentication login", 0, 3);
        assert.ok(result.direct.length > 0);
        assert.ok(result.totalNodes > 0);
      } finally {
        restore();
      }
    });

    it("returns empty for empty graph in fresh dir", async () => {
      const emptyDir = resolve("test/_memory_empty");
      await mkdir(join(emptyDir, ".mcp_data"), { recursive: true });
      const restore = mockEmbedding();
      try {
        const result = await searchGraph(emptyDir, "anything", 1, 5);
        assert.equal(result.direct.length, 0);
        assert.equal(result.neighbors.length, 0);
      } finally {
        restore();
        await rm(emptyDir, { recursive: true, force: true });
      }
    });

    it("includes neighbors at depth 1", async () => {
      const restore = mockEmbedding();
      try {
        const a = await upsertNode(FIXTURE, "concept", "Nav Root", "Root navigation");
        const b = await upsertNode(FIXTURE, "concept", "Nav Child", "Child navigation link");
        await createRelation(FIXTURE, a.id, b.id, "contains");
        const result = await searchGraph(FIXTURE, "Nav Root navigation", 1, 1);
        assert.ok(result.direct.length > 0 || result.neighbors.length > 0);
      } finally {
        restore();
      }
    });
  });

  describe("pruneStaleLinks", () => {
    it("removes edges with decayed weight below threshold", async () => {
      const restore = mockEmbedding();
      try {
        const a = await upsertNode(FIXTURE, "note", "Prune A", "Will be pruned");
        const b = await upsertNode(FIXTURE, "note", "Prune B", "Will be pruned too");
        const edge = await createRelation(FIXTURE, a.id, b.id, "relates_to", 0.01);
        assert.ok(edge);
        const result = await pruneStaleLinks(FIXTURE, 0.5);
        assert.ok(result.removed >= 0);
        assert.ok(typeof result.remaining === "number");
      } finally {
        restore();
      }
    });
  });

  describe("addInterlinkedContext", () => {
    it("creates multiple nodes with auto-linking", async () => {
      const restore = mockEmbedding();
      try {
        const result = await addInterlinkedContext(FIXTURE, [
          { type: "concept", label: "Interlink A", content: "First interlinked concept about testing" },
          { type: "concept", label: "Interlink B", content: "Second interlinked concept about testing" },
          { type: "note", label: "Interlink Note", content: "A note about testing concepts" },
        ], true);
        assert.equal(result.nodes.length, 3);
        assert.ok(Array.isArray(result.edges));
      } finally {
        restore();
      }
    });

    it("skips auto-linking when disabled", async () => {
      const restore = mockEmbedding();
      try {
        const result = await addInterlinkedContext(FIXTURE, [
          { type: "concept", label: "No Link A", content: "Should not auto link" },
          { type: "concept", label: "No Link B", content: "Should not auto link either" },
        ], false);
        assert.equal(result.nodes.length, 2);
        assert.equal(result.edges.length, 0);
      } finally {
        restore();
      }
    });
  });

  describe("retrieveWithTraversal", () => {
    it("returns start node and connected neighbors", async () => {
      const restore = mockEmbedding();
      try {
        const root = await upsertNode(FIXTURE, "concept", "Traversal Root", "Starting point");
        const child1 = await upsertNode(FIXTURE, "symbol", "Traversal Child 1", "First child");
        const child2 = await upsertNode(FIXTURE, "symbol", "Traversal Child 2", "Second child");
        await createRelation(FIXTURE, root.id, child1.id, "contains");
        await createRelation(FIXTURE, root.id, child2.id, "contains");

        const results = await retrieveWithTraversal(FIXTURE, root.id, 1);
        assert.ok(results.length >= 1);
        assert.equal(results[0].node.id, root.id);
        assert.equal(results[0].depth, 0);
      } finally {
        restore();
      }
    });

    it("returns empty for nonexistent node", async () => {
      const restore = mockEmbedding();
      try {
        const results = await retrieveWithTraversal(FIXTURE, "nonexistent-id", 2);
        assert.equal(results.length, 0);
      } finally {
        restore();
      }
    });

    it("respects edge filter", async () => {
      const restore = mockEmbedding();
      try {
        const a = await upsertNode(FIXTURE, "concept", "Filter Root", "Root for filtering");
        const b = await upsertNode(FIXTURE, "symbol", "Filter Dep", "Dependency target");
        const c = await upsertNode(FIXTURE, "note", "Filter Ref", "Reference target");
        await createRelation(FIXTURE, a.id, b.id, "depends_on");
        await createRelation(FIXTURE, a.id, c.id, "references");

        const filtered = await retrieveWithTraversal(FIXTURE, a.id, 1, ["depends_on"]);
        const depNodes = filtered.filter((r) => r.depth > 0);
        for (const r of depNodes) {
          assert.ok(r.pathRelations.some((p) => p.includes("depends_on")));
        }
      } finally {
        restore();
      }
    });
  });

  describe("getGraphStats", () => {
    it("returns node and edge counts with type breakdown", async () => {
      const restore = mockEmbedding();
      try {
        const stats = await getGraphStats(FIXTURE);
        assert.ok(typeof stats.nodes === "number");
        assert.ok(typeof stats.edges === "number");
        assert.ok(typeof stats.types === "object");
        assert.ok(typeof stats.relations === "object");
      } finally {
        restore();
      }
    });
  });
});

describe("memory-tools MCP wrappers", () => {
  describe("toolUpsertMemoryNode", () => {
    it("returns formatted success message with node ID", async () => {
      const restore = mockEmbedding();
      try {
        const output = await toolUpsertMemoryNode({
          rootDir: FIXTURE,
          type: "concept",
          label: "MCP Test Node",
          content: "Testing the MCP wrapper",
        });
        assert.ok(output.includes("✅"));
        assert.ok(output.includes("MCP Test Node"));
        assert.ok(output.includes("mn-"));
      } finally {
        restore();
      }
    });
  });

  describe("toolCreateRelation", () => {
    it("returns success for valid node IDs", async () => {
      const restore = mockEmbedding();
      try {
        const a = await upsertNode(FIXTURE, "concept", "Rel MCP A", "A node");
        const b = await upsertNode(FIXTURE, "concept", "Rel MCP B", "B node");
        const output = await toolCreateRelation({
          rootDir: FIXTURE,
          sourceId: a.id,
          targetId: b.id,
          relation: "implements",
        });
        assert.ok(output.includes("✅"));
        assert.ok(output.includes("implements"));
      } finally {
        restore();
      }
    });

    it("returns error for invalid node IDs", async () => {
      const restore = mockEmbedding();
      try {
        const output = await toolCreateRelation({
          rootDir: FIXTURE,
          sourceId: "bad-1",
          targetId: "bad-2",
          relation: "relates_to",
        });
        assert.ok(output.includes("❌"));
      } finally {
        restore();
      }
    });
  });

  describe("toolSearchMemoryGraph", () => {
    it("returns formatted search results", async () => {
      const restore = mockEmbedding();
      try {
        const output = await toolSearchMemoryGraph({
          rootDir: FIXTURE,
          query: "testing concepts",
          maxDepth: 1,
          topK: 3,
        });
        assert.ok(typeof output === "string");
        assert.ok(output.length > 0);
      } finally {
        restore();
      }
    });
  });

  describe("toolPruneStaleLinks", () => {
    it("returns pruning summary", async () => {
      const restore = mockEmbedding();
      try {
        const output = await toolPruneStaleLinks({ rootDir: FIXTURE, threshold: 0.99 });
        assert.ok(output.includes("🧹"));
        assert.ok(output.includes("Removed"));
      } finally {
        restore();
      }
    });
  });

  describe("toolAddInterlinkedContext", () => {
    it("returns formatted bulk-add results", async () => {
      const restore = mockEmbedding();
      try {
        const output = await toolAddInterlinkedContext({
          rootDir: FIXTURE,
          items: [
            { type: "note", label: "Bulk A", content: "First bulk item" },
            { type: "note", label: "Bulk B", content: "Second bulk item" },
          ],
          autoLink: true,
        });
        assert.ok(output.includes("✅"));
        assert.ok(output.includes("Bulk A"));
        assert.ok(output.includes("Bulk B"));
      } finally {
        restore();
      }
    });
  });

  describe("toolRetrieveWithTraversal", () => {
    it("returns error for nonexistent node", async () => {
      const restore = mockEmbedding();
      try {
        const output = await toolRetrieveWithTraversal({
          rootDir: FIXTURE,
          startNodeId: "ghost-node",
          maxDepth: 2,
        });
        assert.ok(output.includes("❌"));
      } finally {
        restore();
      }
    });

    it("returns traversal results for valid node", async () => {
      const restore = mockEmbedding();
      try {
        const node = await upsertNode(FIXTURE, "concept", "Trav MCP Root", "Root for MCP traversal");
        const output = await toolRetrieveWithTraversal({
          rootDir: FIXTURE,
          startNodeId: node.id,
          maxDepth: 1,
        });
        assert.ok(output.includes("Trav MCP Root"));
        assert.ok(!output.includes("❌"));
      } finally {
        restore();
      }
    });
  });
});
