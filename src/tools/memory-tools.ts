// MCP tool wrappers for memory graph operations and interlinked RAG
// FEATURE: Memory Tools — upsert, relate, search, prune, interlink, traverse

import type { NodeType, RelationType, TraversalResult } from "../core/memory-graph.js";
import { upsertNode, createRelation, searchGraph, pruneStaleLinks, addInterlinkedContext, retrieveWithTraversal, getGraphStats, mergeRankHits } from "../core/memory-graph.js";
import { getStore, resetStore, dropStore, type PeekContextResult } from "../core/short-term-kv.js";
import { resolveContext, promoteToLongTerm, getMemoryStatus } from "../core/solution-engine.js";

export interface UpsertMemoryNodeOptions {
  rootDir: string;
  type: NodeType;
  label: string;
  content: string;
  metadata?: Record<string, string>;
}

export interface CreateRelationOptions {
  rootDir: string;
  sourceId: string;
  targetId: string;
  relation: RelationType;
  weight?: number;
  metadata?: Record<string, string>;
}

export interface SearchMemoryGraphOptions {
  rootDir: string;
  query: string;
  maxDepth?: number;
  topK?: number;
  edgeFilter?: RelationType[];
}

export interface PruneStaleLinksOptions {
  rootDir: string;
  threshold?: number;
}

export interface AddInterlinkedContextOptions {
  rootDir: string;
  items: Array<{ type: NodeType; label: string; content: string; metadata?: Record<string, string> }>;
  autoLink?: boolean;
}

export interface RetrieveWithTraversalOptions {
  rootDir: string;
  startNodeId: string;
  maxDepth?: number;
  edgeFilter?: RelationType[];
}

function formatTraversalResult(result: TraversalResult): string {
  return [
    `  [${result.node.type}] ${result.node.label} (depth: ${result.depth}, score: ${result.relevanceScore})`,
    `    Content: ${result.node.content.slice(0, 120)}${result.node.content.length > 120 ? "..." : ""}`,
    result.pathRelations.length > 1 ? `    Path: ${result.pathRelations.join(" ")}` : "",
    `    ID: ${result.node.id} | Accessed: ${result.node.accessCount}x`,
  ].filter(Boolean).join("\n");
}

export async function toolUpsertMemoryNode(options: UpsertMemoryNodeOptions): Promise<string> {
  const node = await upsertNode(options.rootDir, options.type, options.label, options.content, options.metadata);
  const stats = await getGraphStats(options.rootDir);
  return [
    `✅ Memory node upserted: ${node.label}`,
    `  ID: ${node.id}`,
    `  Type: ${node.type}`,
    `  Access count: ${node.accessCount}`,
    `\nGraph: ${stats.nodes} nodes, ${stats.edges} edges`,
  ].join("\n");
}

export async function toolCreateRelation(options: CreateRelationOptions): Promise<string> {
  const edge = await createRelation(options.rootDir, options.sourceId, options.targetId, options.relation, options.weight, options.metadata);
  if (!edge) return `❌ Failed: one or both node IDs not found (source: ${options.sourceId}, target: ${options.targetId})`;

  const stats = await getGraphStats(options.rootDir);
  return [
    `✅ Relation created: ${options.sourceId} --[${edge.relation}]--> ${options.targetId}`,
    `  Edge ID: ${edge.id}`,
    `  Weight: ${edge.weight}`,
    `\nGraph: ${stats.nodes} nodes, ${stats.edges} edges`,
  ].join("\n");
}

export async function toolSearchMemoryGraph(options: SearchMemoryGraphOptions): Promise<string> {
  const result = await searchGraph(options.rootDir, options.query, options.maxDepth, options.topK, options.edgeFilter);
  if (result.direct.length === 0 && result.neighbors.length === 0) {
    return `No memory nodes found for: "${options.query}"\nGraph has ${result.totalNodes} nodes, ${result.totalEdges} edges.`;
  }

  const topK = options.topK ?? 5;
  const ranked = mergeRankHits([...result.direct, ...result.neighbors], topK);
  const sections: string[] = [
    `Memory Graph Search: "${options.query}"`,
    `Graph: ${result.totalNodes} nodes, ${result.totalEdges} edges`,
    `Ranked (merged direct+neighbor by relevance, top ${topK}):\n`,
  ];
  for (const hit of ranked) sections.push(formatTraversalResult(hit));

  if (result.neighbors.length > 0) {
    sections.push("\nLinked Neighbors (full):");
    for (const neighbor of result.neighbors) sections.push(formatTraversalResult(neighbor));
  }

  return sections.join("\n");
}

export async function toolPruneStaleLinks(options: PruneStaleLinksOptions): Promise<string> {
  const result = await pruneStaleLinks(options.rootDir, options.threshold);
  return [
    `🧹 Pruning complete`,
    `  Removed: ${result.removed} stale links/orphan nodes`,
    `  Remaining edges: ${result.remaining}`,
  ].join("\n");
}

export async function toolAddInterlinkedContext(options: AddInterlinkedContextOptions): Promise<string> {
  const result = await addInterlinkedContext(options.rootDir, options.items, options.autoLink);
  const sections = [
    `✅ Added ${result.nodes.length} interlinked nodes`,
    result.edges.length > 0 ? `  Auto-linked: ${result.edges.length} similarity edges (threshold ≥ 0.72)` : "  No auto-links above threshold",
    "\nNodes:",
  ];

  for (const node of result.nodes) {
    sections.push(`  [${node.type}] ${node.label} → ${node.id}`);
  }

  if (result.edges.length > 0) {
    sections.push("\nEdges:");
    for (const edge of result.edges) {
      sections.push(`  ${edge.source} --[${edge.relation} w:${Math.round(edge.weight * 100) / 100}]--> ${edge.target}`);
    }
  }

  const stats = await getGraphStats(options.rootDir);
  sections.push(`\nGraph total: ${stats.nodes} nodes, ${stats.edges} edges`);
  return sections.join("\n");
}

export async function toolRetrieveWithTraversal(options: RetrieveWithTraversalOptions): Promise<string> {
  const results = await retrieveWithTraversal(options.rootDir, options.startNodeId, options.maxDepth, options.edgeFilter);
  if (results.length === 0) return `❌ Node not found: ${options.startNodeId}`;

  const sections = [`Traversal from: ${results[0].node.label} (depth limit: ${options.maxDepth ?? 2})\n`];
  for (const result of results) sections.push(formatTraversalResult(result));

  return sections.join("\n");
}

// --- Short-term KV + solution engine wrappers (ported from ppm/mcp) ---

export async function toolInitSilo(options: { sessionId: string; siloSize?: number }): Promise<string> {
  const store = resetStore(options.sessionId, options.siloSize ?? 256);
  return JSON.stringify({
    status: "initialized",
    session_id: options.sessionId,
    silo_size: store.siloSize,
    cleared: true,
  });
}

export async function toolPeek(options: { sessionId: string; key: string }): Promise<string> {
  const store = getStore(options.sessionId);
  const result: PeekContextResult = store.peekContext(options.key);
  return JSON.stringify(result);
}

export async function toolSet(options: { sessionId: string; key: string; value: string }): Promise<string> {
  const store = getStore(options.sessionId);
  const index = store.set(options.key, options.value);
  return JSON.stringify({ status: "stored", index });
}

export async function toolResolve(options: {
  sessionId: string;
  promiseId: string;
  payload?: string;
}): Promise<string> {
  const store = getStore(options.sessionId);
  const result = store.resolve(options.promiseId, options.payload);
  return JSON.stringify(result);
}

export async function toolFlush(options: { sessionId: string }): Promise<string> {
  const cleared = dropStore(options.sessionId);
  return JSON.stringify({ status: "flushed", cleared_count: cleared });
}

export async function toolResolveContext(options: {
  rootDir: string;
  sessionId: string;
  key: string;
}): Promise<string> {
  const store = getStore(options.sessionId);
  const result = await resolveContext(options.rootDir, options.key, store);
  return JSON.stringify(result);
}

export async function toolPromoteToLongTerm(options: {
  rootDir: string;
  sessionId: string;
  key: string;
  value: string;
  nodeType?: NodeType;
}): Promise<string> {
  const store = getStore(options.sessionId);
  if (!store.has(options.key)) {
    store.set(options.key, options.value);
  }
  const result = await promoteToLongTerm(
    options.rootDir,
    options.key,
    options.value,
    options.nodeType ?? "concept",
  );
  return JSON.stringify(result);
}

export async function toolMemoryStatus(options: {
  rootDir: string;
  sessionId: string;
}): Promise<string> {
  const store = getStore(options.sessionId);
  const result = await getMemoryStatus(options.rootDir, store);
  return JSON.stringify(result, null, 2);
}

export { mergeRankHits };
