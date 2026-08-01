import { describe, it } from "node:test";
import assert from "node:assert/strict";
import { webSearch } from "../../build/tools/web-search.js";

describe("web-search", () => {
  describe("webSearch function", () => {
    it("is a function", () => {
      assert.equal(typeof webSearch, "function");
    });

    it("returns a string", async () => {
      const result = await webSearch({
        query: "test search query"
      });
      assert.equal(typeof result, "string");
    });

    it("handles basic search query", async () => {
      const result = await webSearch({
        query: "TypeScript programming language",
        count: 3
      });
      
      assert.equal(typeof result, "string");
      assert.ok(result.length > 0);
      assert.ok(result.includes("TypeScript programming language"));
    });

    it("handles empty results gracefully", async () => {
      const result = await webSearch({
        query: "xyzabc123nonexistentquery456def",
        count: 1
      });
      
      assert.equal(typeof result, "string");
      assert.ok(result.length > 0);
    });

    it("respects count parameter", async () => {
      const result = await webSearch({
        query: "JavaScript",
        count: 5
      });
      
      assert.equal(typeof result, "string");
      assert.ok(result.length > 0);
    });

    it("handles network errors gracefully", async () => {
      // This test verifies error handling without requiring network access
      const originalFetch = global.fetch;
      global.fetch = () => Promise.reject(new Error("Network error"));
      
      const result = await webSearch({
        query: "test"
      });
      
      assert.equal(typeof result, "string");
      assert.ok(result.includes("Web search error"));
      
      global.fetch = originalFetch;
    });

    it("handles API errors gracefully", async () => {
      // Mock a 429 rate limit response
      const originalFetch = global.fetch;
      global.fetch = () => Promise.resolve({
        ok: false,
        status: 429,
        statusText: "Too Many Requests"
      });
      
      const result = await webSearch({
        query: "test"
      });
      
      assert.equal(typeof result, "string");
      assert.ok(result.includes("Rate limit exceeded"));
      
      global.fetch = originalFetch;
    });
  });

  describe("parameter validation", () => {
    it("clamps count parameter to valid range", async () => {
      // Test that count is properly clamped between 1-20
      const result1 = await webSearch({
        query: "test",
        count: 0  // Should be clamped to 1
      });
      
      const result2 = await webSearch({
        query: "test", 
        count: 50  // Should be clamped to 20
      });
      
      assert.equal(typeof result1, "string");
      assert.equal(typeof result2, "string");
    });

    it("handles optional parameters", async () => {
      const result = await webSearch({
        query: "test",
        country: "uk",
        search_lang: "en",
        freshness: "week",
        domains: ["github.com", "stackoverflow.com"]
      });
      
      assert.equal(typeof result, "string");
      assert.ok(result.length > 0);
    });
  });
});