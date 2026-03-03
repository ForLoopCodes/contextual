// MCP tool wrappers for memory graph operations and interlinked RAG
// FEATURE: Memory Tools — upsert, relate, search, prune, interlink, traverse

import type { NodeType, RelationType, TraversalResult } from "../core/memory-graph.js";
import { upsertNode, createRelation, searchGraph, pruneStaleLinks, addInterlinkedContext, retrieveWithTraversal, getGraphStats } from "../core/memory-graph.js";

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
  if (result.direct.length === 0) return `No memory nodes found for: "${options.query}"\nGraph has ${result.totalNodes} nodes, ${result.totalEdges} edges.`;

  const sections: string[] = [`Memory Graph Search: "${options.query}"`, `Graph: ${result.totalNodes} nodes, ${result.totalEdges} edges\n`];

  sections.push("Direct Matches:");
  for (const hit of result.direct) sections.push(formatTraversalResult(hit));

  if (result.neighbors.length > 0) {
    sections.push("\nLinked Neighbors:");
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
