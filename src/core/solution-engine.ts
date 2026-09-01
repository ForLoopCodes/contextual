// Unified short-term KV + long-term graph resolution
// FEATURE: Solution Engine — resolve_context, promote_to_long_term, memory_status

import { ShortTermKVStore } from "./short-term-kv.js";
import {
  upsertNode,
  searchGraph,
  getGraphStats,
  mergeRankHits,
  type NodeType,
} from "./memory-graph.js";

const PROMOTION_THRESHOLD = 3;

/**
 * Resolve context by checking short-term KV first, then long-term graph.
 * Graph hits use merge-ranked direct+neighbor results (pmll ranking fix).
 */
export async function resolveContext(
  rootDir: string,
  key: string,
  store: ShortTermKVStore,
): Promise<{ source: "short_term" | "long_term" | "miss"; value: string | null; score: number }> {
  const [hit, value] = store.peek(key);
  if (hit && value !== null) {
    return { source: "short_term", value, score: 1.0 };
  }

  const graphResult = await searchGraph(rootDir, key, 1, 5);
  const ranked = mergeRankHits([...graphResult.direct, ...graphResult.neighbors], 1);
  if (ranked.length > 0) {
    const top = ranked[0];
    return {
      source: "long_term",
      value: top.node.content,
      score: top.relevanceScore / 100,
    };
  }

  return { source: "miss", value: null, score: 0 };
}

/** Promote a short-term KV entry into the persistent memory graph. */
export async function promoteToLongTerm(
  rootDir: string,
  key: string,
  value: string,
  nodeType: NodeType = "concept",
  metadata?: Record<string, string>,
): Promise<{ promoted: boolean; nodeId: string | null }> {
  const node = await upsertNode(rootDir, nodeType, key, value, {
    ...(metadata ?? {}),
    promoted_from: "short_term_kv",
  });
  return { promoted: true, nodeId: node.id };
}

/** Unified status view of both memory layers. */
export async function getMemoryStatus(
  rootDir: string,
  store: ShortTermKVStore,
): Promise<{
  shortTerm: { slots: number; siloSize: number; pending: number };
  longTerm: { nodes: number; edges: number; types: Record<string, number> };
  promotionThreshold: number;
}> {
  const stats = await getGraphStats(rootDir);
  return {
    shortTerm: {
      slots: store.size,
      siloSize: store.siloSize,
      pending: store.pendingCount,
    },
    longTerm: {
      nodes: stats.nodes,
      edges: stats.edges,
      types: stats.types,
    },
    promotionThreshold: PROMOTION_THRESHOLD,
  };
}
