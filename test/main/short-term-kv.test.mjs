import { describe, it, beforeEach } from "node:test";
import assert from "node:assert/strict";
import { mkdir, rm } from "fs/promises";
import { join, resolve } from "path";
import { Ollama } from "ollama";

const {
  ShortTermKVStore,
  getStore,
  dropStore,
  resetStore,
  _sessionStoresMap,
} = await import("../../build/core/short-term-kv.js");

const {
  mergeRankHits,
  searchGraph,
  upsertNode,
  createRelation,
} = await import("../../build/core/memory-graph.js");

const {
  resolveContext,
  promoteToLongTerm,
  getMemoryStatus,
} = await import("../../build/core/solution-engine.js");

const {
  toolInitSilo,
  toolPeek,
  toolSet,
  toolFlush,
  toolResolveContext,
  toolMemoryStatus,
} = await import("../../build/tools/memory-tools.js");

beforeEach(() => {
  _sessionStoresMap.clear();
});

describe("ShortTermKVStore — peek/set/flush", () => {
  it("peek miss on empty store", () => {
    const store = new ShortTermKVStore();
    const [hit, value, index] = store.peek("missing");
    assert.equal(hit, false);
    assert.equal(value, null);
    assert.equal(index, null);
  });

  it("peek hit after set", () => {
    const store = new ShortTermKVStore();
    store.set("url", "https://example.com");
    const [hit, value, index] = store.peek("url");
    assert.equal(hit, true);
    assert.equal(value, "https://example.com");
    assert.equal(index, 0);
  });

  it("set update keeps same index", () => {
    const store = new ShortTermKVStore();
    store.set("k", "v1");
    const idx = store.set("k", "v2");
    assert.equal(idx, 0);
    assert.equal(store.peek("k")[1], "v2");
  });

  it("flush clears all slots", () => {
    const store = new ShortTermKVStore();
    store.set("a", "1");
    store.set("b", "2");
    assert.equal(store.flush(), 2);
    assert.equal(store.size, 0);
    assert.equal(store.peek("a")[0], false);
  });
});

describe("ShortTermKVStore — LRU eviction", () => {
  it("evicts least-recently-used when over silo_size", () => {
    const store = new ShortTermKVStore(2);
    store.set("a", "1");
    store.set("b", "2");
    store.peek("a"); // touch a so b is LRU
    store.set("c", "3");
    assert.equal(store.size, 2);
    assert.equal(store.has("b"), false);
    assert.equal(store.has("a"), true);
    assert.equal(store.has("c"), true);
  });
});

describe("ShortTermKVStore — pending / resolve", () => {
  it("peekContext returns pending when marked", () => {
    const store = new ShortTermKVStore();
    store.markPending("job", "job");
    const result = store.peekContext("job");
    assert.equal(result.hit, true);
    assert.equal(result.status, "pending");
    assert.equal(result.promise_id, "job");
  });

  it("resolve with payload stores value and clears pending", () => {
    const store = new ShortTermKVStore();
    store.markPending("job", "job");
    const resolved = store.resolve("job", "done");
    assert.equal(resolved.status, "resolved");
    assert.equal(resolved.payload, "done");
    const peek = store.peekContext("job");
    assert.equal(peek.hit, true);
    assert.equal(peek.value, "done");
  });
});

describe("session helpers", () => {
  it("resetStore clears on init", () => {
    const first = resetStore("s1", 8);
    first.set("x", "1");
    const second = resetStore("s1", 16);
    assert.equal(second.size, 0);
    assert.equal(second.siloSize, 16);
    assert.equal(getStore("s1").peek("x")[0], false);
  });

  it("dropStore removes session", () => {
    resetStore("s2").set("a", "b");
    assert.equal(dropStore("s2"), 1);
    assert.equal(dropStore("s2"), 0);
  });
});

describe("mergeRankHits unified ranking", () => {
  it("sorts by relevanceScore desc and dedupes by id", () => {
    const mk = (id, label, score, depth = 0) => ({
      node: { id, label, type: "concept", content: label, embedding: [], createdAt: 0, lastAccessed: 0, accessCount: 1, metadata: {} },
      depth,
      pathRelations: [],
      relevanceScore: score,
    });
    const ranked = mergeRankHits([
      mk("a", "A", 50, 0),
      mk("b", "B", 90, 1),
      mk("a", "A", 40, 0),
      mk("c", "C", 70, 1),
    ], 2);
    assert.equal(ranked.length, 2);
    assert.equal(ranked[0].node.id, "b");
    assert.equal(ranked[1].node.id, "c");
  });

  it("high-scoring neighbor outranks lower direct in top_k", () => {
    const mk = (id, label, score, depth) => ({
      node: { id, label, type: "note", content: label, embedding: [], createdAt: 0, lastAccessed: 0, accessCount: 1, metadata: {} },
      depth,
      pathRelations: [],
      relevanceScore: score,
    });
    // Unified ranking: neighbor with higher score wins top_k slot
    const ranked = mergeRankHits([
      mk("direct-low", "DirectLow", 60, 0),
      mk("neighbor-high", "NeighborHigh", 95, 1),
    ], 1);
    assert.equal(ranked[0].node.id, "neighbor-high");
    assert.equal(ranked[0].depth, 1);
  });
});

describe("tool wrappers — KV", () => {
  it("init/peek/set/flush round-trip", async () => {
    const init = JSON.parse(await toolInitSilo({ sessionId: "t1", siloSize: 32 }));
    assert.equal(init.status, "initialized");
    assert.equal(init.silo_size, 32);

    assert.equal(JSON.parse(await toolPeek({ sessionId: "t1", key: "k" })).hit, false);
    const stored = JSON.parse(await toolSet({ sessionId: "t1", key: "k", value: "v" }));
    assert.equal(stored.status, "stored");
    const hit = JSON.parse(await toolPeek({ sessionId: "t1", key: "k" }));
    assert.equal(hit.hit, true);
    assert.equal(hit.value, "v");

    const flushed = JSON.parse(await toolFlush({ sessionId: "t1" }));
    assert.equal(flushed.status, "flushed");
    assert.equal(flushed.cleared_count, 1);
  });
});

const FIXTURE = resolve("test/_short_term_kv_fixtures");

function mockEmbedding() {
  const original = Ollama.prototype.embed;
  Ollama.prototype.embed = async function ({ input }) {
    const batch = Array.isArray(input) ? input : [input];
    return {
      embeddings: batch.map((text) => {
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

describe("solution engine", () => {
  beforeEach(async () => {
    await rm(FIXTURE, { recursive: true, force: true });
    await mkdir(join(FIXTURE, ".mcp_data"), { recursive: true });
    _sessionStoresMap.clear();
  });

  it("resolveContext hits short-term first", async () => {
    const store = resetStore("sol1");
    store.set("auth", "cached-auth-flow");
    const result = await resolveContext(FIXTURE, "auth", store);
    assert.equal(result.source, "short_term");
    assert.equal(result.value, "cached-auth-flow");
    assert.equal(result.score, 1);
  });

  it("resolveContext falls back to long-term graph", async () => {
    const restore = mockEmbedding();
    try {
      const store = resetStore("sol2");
      await upsertNode(FIXTURE, "concept", "Auth Flow", "Handles login sessions");
      const result = await resolveContext(FIXTURE, "Auth Flow login", store);
      assert.equal(result.source, "long_term");
      assert.ok(result.value);
      assert.ok(result.score > 0);
    } finally {
      restore();
    }
  });

  it("promoteToLongTerm creates graph node", async () => {
    const restore = mockEmbedding();
    try {
      const promoted = await promoteToLongTerm(FIXTURE, "Promoted Key", "Promoted content", "note");
      assert.equal(promoted.promoted, true);
      assert.ok(promoted.nodeId?.startsWith("mn-"));
    } finally {
      restore();
    }
  });

  it("memory_status reports both layers", async () => {
    const restore = mockEmbedding();
    try {
      const store = resetStore("sol3", 64);
      store.set("a", "1");
      await upsertNode(FIXTURE, "note", "Status Note", "status content");
      const status = await getMemoryStatus(FIXTURE, store);
      assert.equal(status.shortTerm.slots, 1);
      assert.equal(status.shortTerm.siloSize, 64);
      assert.ok(status.longTerm.nodes >= 1);
      assert.equal(status.promotionThreshold, 3);
    } finally {
      restore();
    }
  });

  it("toolResolveContext JSON round-trip", async () => {
    await toolInitSilo({ sessionId: "sol4" });
    await toolSet({ sessionId: "sol4", key: "x", value: "y" });
    const raw = await toolResolveContext({ rootDir: FIXTURE, sessionId: "sol4", key: "x" });
    const parsed = JSON.parse(raw);
    assert.equal(parsed.source, "short_term");
    const status = JSON.parse(await toolMemoryStatus({ rootDir: FIXTURE, sessionId: "sol4" }));
    assert.equal(status.shortTerm.slots, 1);
  });
});

describe("searchGraph + mergeRankHits integration", () => {
  it("returns neighbors that can outrank weak directs in merged top_k helper", async () => {
    const dir = resolve("test/_rank_fix_fixtures");
    await rm(dir, { recursive: true, force: true });
    await mkdir(join(dir, ".mcp_data"), { recursive: true });
    const restore = mockEmbedding();
    try {
      const root = await upsertNode(dir, "concept", "RankRootExactQueryZZ", "RankRootExactQueryZZ unique seed");
      const neighbor = await upsertNode(dir, "concept", "RankNeighbor", "unrelated other topic");
      await createRelation(dir, root.id, neighbor.id, "relates_to", 1.0);
      const result = await searchGraph(dir, "RankRootExactQueryZZ", 1, 5);
      assert.ok(result.direct.length + result.neighbors.length >= 1);
      const merged = mergeRankHits([...result.direct, ...result.neighbors], 5);
      assert.ok(merged.length >= 1);
      // Scores must be non-increasing
      for (let i = 1; i < merged.length; i++) {
        assert.ok(merged[i - 1].relevanceScore >= merged[i].relevanceScore);
      }
    } finally {
      restore();
      await rm(dir, { recursive: true, force: true });
    }
  });
});
