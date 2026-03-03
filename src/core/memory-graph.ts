// In-memory property graph with JSON persistence for linking memory nodes
// FEATURE: Memory Graph — traversal, decay scoring, auto-similarity edges

import { readFile, writeFile } from "fs/promises";
import { join } from "path";
import { fetchEmbedding, ensureMcpDataDir } from "./embeddings.js";

export type NodeType = "concept" | "file" | "symbol" | "note";
export type RelationType = "relates_to" | "depends_on" | "implements" | "references" | "similar_to" | "contains";

export interface MemoryNode {
  id: string;
  type: NodeType;
  label: string;
  content: string;
  embedding: number[];
  createdAt: number;
  lastAccessed: number;
  accessCount: number;
  metadata: Record<string, string>;
}

export interface MemoryEdge {
  id: string;
  source: string;
  target: string;
  relation: RelationType;
  weight: number;
  createdAt: number;
  metadata: Record<string, string>;
}

interface GraphStore {
  nodes: Record<string, MemoryNode>;
  edges: Record<string, MemoryEdge>;
}

export interface TraversalResult {
  node: MemoryNode;
  depth: number;
  pathRelations: string[];
  relevanceScore: number;
}

export interface GraphSearchResult {
  direct: TraversalResult[];
  neighbors: TraversalResult[];
  totalNodes: number;
  totalEdges: number;
}

const GRAPH_FILE = "memory-graph.json";
const CACHE_DIR = ".mcp_data";
const DECAY_LAMBDA = 0.05;
const SIMILARITY_THRESHOLD = 0.72;
const STALE_THRESHOLD = 0.15;

let graphCache = new Map<string, GraphStore>();
let savePending = new Map<string, boolean>();
let saveTimeout = new Map<string, ReturnType<typeof setTimeout>>();

function generateId(prefix: string): string {
  return `${prefix}-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
}

function cosine(a: number[], b: number[]): number {
  const len = Math.min(a.length, b.length);
  if (len === 0) return 0;
  let dot = 0, normA = 0, normB = 0;
  for (let i = 0; i < len; i++) {
    dot += a[i] * b[i];
    normA += a[i] * a[i];
    normB += b[i] * b[i];
  }
  const denom = Math.sqrt(normA) * Math.sqrt(normB);
  return denom === 0 ? 0 : dot / denom;
}

function decayWeight(edge: MemoryEdge): number {
  const daysSinceCreation = (Date.now() - edge.createdAt) / 86_400_000;
  return edge.weight * Math.exp(-DECAY_LAMBDA * daysSinceCreation);
}

async function loadGraph(rootDir: string): Promise<GraphStore> {
  if (graphCache.has(rootDir)) return graphCache.get(rootDir)!;
  try {
    const raw = JSON.parse(await readFile(join(rootDir, CACHE_DIR, GRAPH_FILE), "utf-8"));
    const store: GraphStore = {
      nodes: raw?.nodes && typeof raw.nodes === "object" ? raw.nodes : {},
      edges: raw?.edges && typeof raw.edges === "object" ? raw.edges : {},
    };
    graphCache.set(rootDir, store);
  } catch {
    graphCache.set(rootDir, { nodes: {}, edges: {} });
  }
  return graphCache.get(rootDir)!;
}

async function persistGraph(rootDir: string): Promise<void> {
  const store = graphCache.get(rootDir);
  if (!store) return;
  await ensureMcpDataDir(rootDir);
  await writeFile(join(rootDir, CACHE_DIR, GRAPH_FILE), JSON.stringify(store, null, 2));
}

function scheduleSave(rootDir: string): void {
  const existing = saveTimeout.get(rootDir);
  if (existing) clearTimeout(existing);
  savePending.set(rootDir, true);
  saveTimeout.set(rootDir, setTimeout(() => {
    if (savePending.get(rootDir)) {
      persistGraph(rootDir).catch(() => {}).finally(() => savePending.set(rootDir, false));
    }
  }, 500));
}

function getEdgesForNode(graph: GraphStore, nodeId: string): MemoryEdge[] {
  return Object.values(graph.edges).filter(e => e.source === nodeId || e.target === nodeId);
}

function getNeighborId(edge: MemoryEdge, fromId: string): string {
  return edge.source === fromId ? edge.target : edge.source;
}

export async function upsertNode(rootDir: string, type: NodeType, label: string, content: string, metadata?: Record<string, string>): Promise<MemoryNode> {
  const graph = await loadGraph(rootDir);
  const existing = Object.values(graph.nodes).find(n => n.label === label && n.type === type);

  if (existing) {
    existing.content = content;
    existing.lastAccessed = Date.now();
    existing.accessCount++;
    if (metadata) Object.assign(existing.metadata, metadata);
    existing.embedding = (await fetchEmbedding(`${label} ${content}`))[0];
    scheduleSave(rootDir);
    return existing;
  }

  const node: MemoryNode = {
    id: generateId("mn"),
    type,
    label,
    content,
    embedding: (await fetchEmbedding(`${label} ${content}`))[0],
    createdAt: Date.now(),
    lastAccessed: Date.now(),
    accessCount: 1,
    metadata: metadata ?? {},
  };
  graph.nodes[node.id] = node;
  scheduleSave(rootDir);
  return node;
}

export async function createRelation(rootDir: string, sourceId: string, targetId: string, relation: RelationType, weight?: number, metadata?: Record<string, string>): Promise<MemoryEdge | null> {
  const graph = await loadGraph(rootDir);
  if (!graph.nodes[sourceId] || !graph.nodes[targetId]) return null;

  const duplicate = Object.values(graph.edges).find(e =>
    e.source === sourceId && e.target === targetId && e.relation === relation
  );
  if (duplicate) {
    duplicate.weight = weight ?? duplicate.weight;
    if (metadata) Object.assign(duplicate.metadata, metadata);
    scheduleSave(rootDir);
    return duplicate;
  }

  const edge: MemoryEdge = {
    id: generateId("me"),
    source: sourceId,
    target: targetId,
    relation,
    weight: weight ?? 1.0,
    createdAt: Date.now(),
    metadata: metadata ?? {},
  };
  graph.edges[edge.id] = edge;
  scheduleSave(rootDir);
  return edge;
}

export async function searchGraph(rootDir: string, query: string, maxDepth: number = 1, topK: number = 5, edgeFilter?: RelationType[]): Promise<GraphSearchResult> {
  const graph = await loadGraph(rootDir);
  const nodes = Object.values(graph.nodes);
  if (nodes.length === 0) return { direct: [], neighbors: [], totalNodes: 0, totalEdges: 0 };

  const [queryVec] = await fetchEmbedding(query);
  const scored = nodes.map(n => ({ node: n, score: cosine(queryVec, n.embedding) }))
    .sort((a, b) => b.score - a.score);

  const directHits = scored.slice(0, topK).map(({ node, score }) => {
    node.lastAccessed = Date.now();
    return {
      node,
      depth: 0,
      pathRelations: [] as string[],
      relevanceScore: Math.round(score * 1000) / 10,
    };
  });

  const neighborResults: TraversalResult[] = [];
  const visited = new Set(directHits.map(h => h.node.id));

  for (const hit of directHits) {
    traverseNeighbors(graph, hit.node.id, queryVec, 1, maxDepth, [hit.node.label], visited, neighborResults, edgeFilter);
  }

  neighborResults.sort((a, b) => b.relevanceScore - a.relevanceScore);

  scheduleSave(rootDir);
  return {
    direct: directHits,
    neighbors: neighborResults.slice(0, topK * 2),
    totalNodes: nodes.length,
    totalEdges: Object.keys(graph.edges).length,
  };
}

function traverseNeighbors(
  graph: GraphStore, nodeId: string, queryVec: number[], depth: number, maxDepth: number,
  pathLabels: string[], visited: Set<string>, results: TraversalResult[], edgeFilter?: RelationType[],
): void {
  if (depth > maxDepth) return;

  for (const edge of getEdgesForNode(graph, nodeId)) {
    if (edgeFilter && !edgeFilter.includes(edge.relation)) continue;
    const neighborId = getNeighborId(edge, nodeId);
    if (visited.has(neighborId)) continue;

    const neighbor = graph.nodes[neighborId];
    if (!neighbor) continue;

    visited.add(neighborId);
    const similarity = cosine(queryVec, neighbor.embedding);
    const edgeDecay = decayWeight(edge);
    const relevance = similarity * 0.6 + (edgeDecay / Math.max(edge.weight, 0.01)) * 0.4;

    results.push({
      node: neighbor,
      depth,
      pathRelations: [...pathLabels, `--[${edge.relation}]-->`, neighbor.label],
      relevanceScore: Math.round(relevance * 1000) / 10,
    });

    neighbor.lastAccessed = Date.now();
    traverseNeighbors(graph, neighborId, queryVec, depth + 1, maxDepth, [...pathLabels, `--[${edge.relation}]-->`, neighbor.label], visited, results, edgeFilter);
  }
}

export async function pruneStaleLinks(rootDir: string, threshold?: number): Promise<{ removed: number; remaining: number }> {
  const graph = await loadGraph(rootDir);
  const cutoff = threshold ?? STALE_THRESHOLD;
  const toRemove: string[] = [];

  for (const [edgeId, edge] of Object.entries(graph.edges)) {
    if (decayWeight(edge) < cutoff) toRemove.push(edgeId);
  }

  for (const id of toRemove) delete graph.edges[id];

  const orphanNodeIds = Object.keys(graph.nodes).filter(nodeId =>
    getEdgesForNode(graph, nodeId).length === 0
      && graph.nodes[nodeId].accessCount <= 1
      && (Date.now() - graph.nodes[nodeId].lastAccessed) > 7 * 86_400_000
  );
  for (const id of orphanNodeIds) delete graph.nodes[id];

  scheduleSave(rootDir);
  return { removed: toRemove.length + orphanNodeIds.length, remaining: Object.keys(graph.edges).length };
}

export async function addInterlinkedContext(rootDir: string, items: Array<{ type: NodeType; label: string; content: string; metadata?: Record<string, string> }>, autoLink: boolean = true): Promise<{ nodes: MemoryNode[]; edges: MemoryEdge[] }> {
  const createdNodes: MemoryNode[] = [];
  for (const item of items) {
    createdNodes.push(await upsertNode(rootDir, item.type, item.label, item.content, item.metadata));
  }

  const createdEdges: MemoryEdge[] = [];

  if (autoLink && createdNodes.length > 1) {
    for (let i = 0; i < createdNodes.length; i++) {
      for (let j = i + 1; j < createdNodes.length; j++) {
        const similarity = cosine(createdNodes[i].embedding, createdNodes[j].embedding);
        if (similarity >= SIMILARITY_THRESHOLD) {
          const edge = await createRelation(rootDir, createdNodes[i].id, createdNodes[j].id, "similar_to", similarity);
          if (edge) createdEdges.push(edge);
        }
      }
    }
  }

  const graph = await loadGraph(rootDir);
  const existingNodes = Object.values(graph.nodes)
    .filter(n => !createdNodes.find(cn => cn.id === n.id))
    .slice(0, 200);
  if (autoLink) {
    for (const newNode of createdNodes) {
      for (const existing of existingNodes) {
        const similarity = cosine(newNode.embedding, existing.embedding);
        if (similarity >= SIMILARITY_THRESHOLD) {
          const edge = await createRelation(rootDir, newNode.id, existing.id, "similar_to", similarity);
          if (edge) createdEdges.push(edge);
        }
      }
    }
  }

  return { nodes: createdNodes, edges: createdEdges };
}

export async function retrieveWithTraversal(rootDir: string, startNodeId: string, maxDepth: number = 2, edgeFilter?: RelationType[]): Promise<TraversalResult[]> {
  const graph = await loadGraph(rootDir);
  const startNode = graph.nodes[startNodeId];
  if (!startNode) return [];

  startNode.lastAccessed = Date.now();
  startNode.accessCount++;

  const results: TraversalResult[] = [{
    node: startNode,
    depth: 0,
    pathRelations: [startNode.label],
    relevanceScore: 100,
  }];

  const visited = new Set([startNodeId]);
  collectTraversal(graph, startNodeId, 1, maxDepth, [startNode.label], visited, results, edgeFilter);

  scheduleSave(rootDir);
  return results;
}

function collectTraversal(
  graph: GraphStore, nodeId: string, depth: number, maxDepth: number,
  pathLabels: string[], visited: Set<string>, results: TraversalResult[], edgeFilter?: RelationType[],
): void {
  if (depth > maxDepth) return;

  for (const edge of getEdgesForNode(graph, nodeId)) {
    if (edgeFilter && !edgeFilter.includes(edge.relation)) continue;
    const neighborId = getNeighborId(edge, nodeId);
    if (visited.has(neighborId)) continue;

    const neighbor = graph.nodes[neighborId];
    if (!neighbor) continue;

    visited.add(neighborId);
    neighbor.lastAccessed = Date.now();

    const decayed = decayWeight(edge);
    const depthPenalty = 1 / (1 + depth * 0.3);
    const score = decayed * depthPenalty * 100;

    results.push({
      node: neighbor,
      depth,
      pathRelations: [...pathLabels, `--[${edge.relation}]-->`, neighbor.label],
      relevanceScore: Math.round(score * 10) / 10,
    });

    collectTraversal(graph, neighborId, depth + 1, maxDepth, [...pathLabels, `--[${edge.relation}]-->`, neighbor.label], visited, results, edgeFilter);
  }
}

export async function getGraphStats(rootDir: string): Promise<{ nodes: number; edges: number; types: Record<string, number>; relations: Record<string, number> }> {
  const graph = await loadGraph(rootDir);
  const types: Record<string, number> = {};
  const relations: Record<string, number> = {};

  for (const node of Object.values(graph.nodes)) types[node.type] = (types[node.type] ?? 0) + 1;
  for (const edge of Object.values(graph.edges)) relations[edge.relation] = (relations[edge.relation] ?? 0) + 1;

  return { nodes: Object.keys(graph.nodes).length, edges: Object.keys(graph.edges).length, types, relations };
}
