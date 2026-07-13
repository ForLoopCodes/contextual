// Identifier-level semantic retrieval with call-site ranking and line metadata
// FEATURE: Symbol intelligence via semantic search over definitions and usages

import { readFile } from "fs/promises";
import { walkRoots } from "../core/walker.js";
import { analyzeFile, flattenSymbols, isSupportedFile } from "../core/parser.js";
import {
  fetchEmbedding,
  getEmbeddingBatchSize,
  loadEmbeddingCache,
  saveEmbeddingCache,
  type EmbeddingCache,
} from "../core/embeddings.js";
import { resolve } from "path";

export interface SemanticIdentifierSearchOptions {
  rootDir: string;
  query: string;
  topK?: number;
  topCallsPerIdentifier?: number;
  includeKinds?: string[];
  semanticWeight?: number;
  keywordWeight?: number;
}

interface IdentifierDoc {
  id: string;
  path: string;
  header: string;
  name: string;
  kind: string;
  line: number;
  endLine: number;
  signature: string;
  parentName?: string;
  text: string;
}

interface RankedIdentifier {
  doc: IdentifierDoc;
  semanticScore: number;
  keywordScore: number;
  score: number;
}

interface CallSite {
  file: string;
  line: number;
  context: string;
  semanticScore: number;
  keywordScore: number;
  score: number;
}

interface IdentifierIndex {
  docs: IdentifierDoc[];
  vectors: number[][];
  fileLines: Map<string, string[]>;
}

const IDENTIFIER_CACHE_FILE = "identifier-embeddings-cache.json";
const CALLSITE_CACHE_PREFIX = "callsite:";
const INDEX_TTL_MS = 60_000;

let cachedRootDir: string | null = null;
let cachedAt = 0;
let cachedIndex: IdentifierIndex | null = null;

function hashContent(text: string): string {
  let h = 0;
  for (let i = 0; i < text.length; i++) h = ((h << 5) - h + text.charCodeAt(i)) | 0;
  return h.toString(36);
}

function splitTerms(text: string): string[] {
  return text
    .replace(/([a-z])([A-Z])/g, "$1 $2")
    .replace(/([A-Z])([A-Z][a-z])/g, "$1 $2")
    .toLowerCase()
    .split(/[^a-z0-9_]+/)
    .filter((token) => token.length > 1);
}

function cosine(a: number[], b: number[]): number {
  let dot = 0;
  let normA = 0;
  let normB = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
    normA += a[i] * a[i];
    normB += b[i] * b[i];
  }
  const denom = Math.sqrt(normA) * Math.sqrt(normB);
  return denom === 0 ? 0 : dot / denom;
}

function clamp01(value: number): number {
  if (value <= 0) return 0;
  if (value >= 1) return 1;
  return value;
}

function normalizeWeight(value: number | undefined, fallback: number): number {
  if (value === undefined || !Number.isFinite(value) || value < 0) return fallback;
  return value;
}

function formatLineRange(line: number, endLine: number): string {
  return endLine > line ? `L${line}-L${endLine}` : `L${line}`;
}

function escapeRegex(input: string): string {
  return input.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

function getKeywordCoverage(queryTerms: Set<string>, input: string): number {
  if (queryTerms.size === 0) return 0;
  const docTerms = new Set(splitTerms(input));
  let matched = 0;
  for (const term of queryTerms) {
    if (docTerms.has(term)) matched++;
  }
  return matched / queryTerms.size;
}

function isDefinitionLine(line: string, symbolName: string): boolean {
  const escaped = escapeRegex(symbolName);
  const patterns = [
    new RegExp(`(?:function|class|enum|interface|struct|type|trait|fn|def|func)\\s+${escaped}`),
    new RegExp(`(?:const|let|var|pub|export)\\s+(?:async\\s+)?(?:function\\s+)?${escaped}`),
  ];
  return patterns.some((pattern) => pattern.test(line));
}

function normalizeKinds(kinds?: string[]): Set<string> | null {
  if (!kinds || kinds.length === 0) return null;
  const normalized = kinds.map((k) => k.trim().toLowerCase()).filter(Boolean);
  return normalized.length > 0 ? new Set(normalized) : null;
}

function normalizeRelativePath(path: string): string {
  return path.replace(/\\/g, "/");
}

function removeFileScopedCacheEntries(cache: EmbeddingCache, relativePath: string): void {
  const definitionPrefix = `id:${relativePath}:`;
  const callsitePrefix = `${CALLSITE_CACHE_PREFIX}${relativePath}:`;
  for (const key of Object.keys(cache)) {
    if (key.startsWith(definitionPrefix) || key.startsWith(callsitePrefix)) {
      delete cache[key];
    }
  }
}

async function buildIdentifierDocsForFile(rootDir: string, relativePath: string): Promise<IdentifierDoc[]> {
  const normalized = normalizeRelativePath(relativePath);
  const fullPath = resolve(rootDir, normalized);
  if (!isSupportedFile(fullPath)) return [];

  try {
    const analysis = await analyzeFile(fullPath);
    const flat = flattenSymbols(analysis.symbols);
    return flat.map((symbol) => ({
      id: `${normalized}:${symbol.name}:${symbol.line}`,
      path: normalized,
      header: analysis.header,
      name: symbol.name,
      kind: symbol.kind,
      line: symbol.line,
      endLine: symbol.endLine,
      signature: symbol.signature,
      parentName: symbol.parentName,
      text: `${symbol.name} ${symbol.kind} ${symbol.signature} ${normalized} ${analysis.header} ${symbol.parentName ?? ""}`,
    }));
  } catch {
    return [];
  }
}

async function buildIdentifierIndex(rootDir: string): Promise<IdentifierIndex> {
  if (cachedIndex && cachedRootDir === rootDir && Date.now() - cachedAt < INDEX_TTL_MS) {
    return cachedIndex;
  }

  const entries = await walkRoots({ rootDir, depthLimit: 0 });
  const files = entries.filter((entry) => !entry.isDirectory && isSupportedFile(entry.path));
  const docs: IdentifierDoc[] = [];
  const fileLines = new Map<string, string[]>();

  for (const file of files) {
    try {
      const content = await readFile(file.path, "utf-8");
      fileLines.set(file.relativePath, content.split("\n"));
      const analysis = await analyzeFile(file.path);
      const flat = flattenSymbols(analysis.symbols);
      for (const symbol of flat) {
        docs.push({
          id: `${file.relativePath}:${symbol.name}:${symbol.line}`,
          path: file.relativePath,
          header: analysis.header,
          name: symbol.name,
          kind: symbol.kind,
          line: symbol.line,
          endLine: symbol.endLine,
          signature: symbol.signature,
          parentName: symbol.parentName,
          text: `${symbol.name} ${symbol.kind} ${symbol.signature} ${file.relativePath} ${analysis.header} ${symbol.parentName ?? ""}`,
        });
      }
    } catch {
    }
  }

  if (docs.length === 0) {
    const empty: IdentifierIndex = { docs: [], vectors: [], fileLines };
    cachedIndex = empty;
    cachedRootDir = rootDir;
    cachedAt = Date.now();
    return empty;
  }

  const cache = await loadEmbeddingCache(rootDir, IDENTIFIER_CACHE_FILE);
  const vectors: number[][] = new Array(docs.length);
  const uncached: { idx: number; key: string; hash: string; text: string }[] = [];

  for (let i = 0; i < docs.length; i++) {
    const text = docs[i].text;
    const hash = hashContent(text);
    const key = `id:${docs[i].id}`;
    if (cache[key]?.hash === hash) {
      vectors[i] = cache[key].vector;
    } else {
      uncached.push({ idx: i, key, hash, text });
    }
  }

  if (uncached.length > 0) {
    const batchSize = getEmbeddingBatchSize();
    for (let i = 0; i < uncached.length; i += batchSize) {
      const batch = uncached.slice(i, i + batchSize);
      const embeddings = await fetchEmbedding(batch.map((entry) => entry.text));
      for (let j = 0; j < batch.length; j++) {
        vectors[batch[j].idx] = embeddings[j];
        cache[batch[j].key] = { hash: batch[j].hash, vector: embeddings[j] };
      }
    }
    await saveEmbeddingCache(rootDir, cache, IDENTIFIER_CACHE_FILE);
  }

  const index: IdentifierIndex = { docs, vectors, fileLines };
  cachedIndex = index;
  cachedRootDir = rootDir;
  cachedAt = Date.now();
  return index;
}

async function rankCallSites(
  rootDir: string,
  cache: EmbeddingCache,
  queryTerms: Set<string>,
  queryVec: number[],
  symbol: IdentifierDoc,
  fileLines: Map<string, string[]>,
  limit: number,
): Promise<{ sites: CallSite[]; total: number }> {
  const callPattern = symbol.kind === "function" || symbol.kind === "method"
    ? new RegExp(`\\b${escapeRegex(symbol.name)}\\s*\\(`)
    : new RegExp(`\\b${escapeRegex(symbol.name)}\\b`);

  const candidates: { file: string; line: number; context: string; keywordScore: number }[] = [];

  for (const [file, lines] of fileLines) {
    for (let i = 0; i < lines.length; i++) {
      const raw = lines[i];
      if (!callPattern.test(raw)) {
        callPattern.lastIndex = 0;
        continue;
      }
      callPattern.lastIndex = 0;

      if (file === symbol.path && i + 1 === symbol.line) continue;
      if (isDefinitionLine(raw, symbol.name)) continue;

      const context = raw.trim().slice(0, 220);
      const keywordScore = getKeywordCoverage(queryTerms, `${file} ${context}`);
      candidates.push({
        file,
        line: i + 1,
        context,
        keywordScore,
      });
    }
  }

  if (candidates.length === 0) return { sites: [], total: 0 };

  const embedBudget = Math.max(30, limit * 4);
  const sampled = candidates
    .slice()
    .sort((a, b) => b.keywordScore - a.keywordScore)
    .slice(0, Math.min(embedBudget, candidates.length));

  const uncached: { key: string; hash: string; text: string }[] = [];
  const keyedCandidates: { candidate: (typeof sampled)[number]; key: string; hash: string }[] = [];

  for (const candidate of sampled) {
    const key = `${CALLSITE_CACHE_PREFIX}${candidate.file}:${candidate.line}`;
    const text = `${candidate.file} ${candidate.context}`;
    const hash = hashContent(text);
    keyedCandidates.push({ candidate, key, hash });
    if (cache[key]?.hash !== hash) {
      uncached.push({ key, hash, text });
    }
  }

  if (uncached.length > 0) {
    const batchSize = getEmbeddingBatchSize();
    for (let i = 0; i < uncached.length; i += batchSize) {
      const batch = uncached.slice(i, i + batchSize);
      const embeddings = await fetchEmbedding(batch.map((item) => item.text));
      for (let j = 0; j < batch.length; j++) {
        cache[batch[j].key] = { hash: batch[j].hash, vector: embeddings[j] };
      }
    }
  }

  const ranked: CallSite[] = keyedCandidates.map(({ candidate, key }) => {
    const vector = cache[key]?.vector;
    const semanticScore = vector ? Math.max(cosine(queryVec, vector), 0) : 0;
    const score = clamp01(semanticScore * 0.82 + candidate.keywordScore * 0.18);
    return {
      file: candidate.file,
      line: candidate.line,
      context: candidate.context,
      semanticScore,
      keywordScore: candidate.keywordScore,
      score,
    };
  });

  return {
    sites: ranked.sort((a, b) => b.score - a.score).slice(0, Math.max(1, limit)),
    total: candidates.length,
  };
}

export async function semanticIdentifierSearch(options: SemanticIdentifierSearchOptions): Promise<string> {
  const topK = Math.max(1, Math.floor(options.topK ?? 5));
  const topCalls = Math.max(1, Math.floor(options.topCallsPerIdentifier ?? 10));
  const semanticWeight = normalizeWeight(options.semanticWeight, 0.78);
  const keywordWeight = normalizeWeight(options.keywordWeight, 0.22);
  const includeKinds = normalizeKinds(options.includeKinds);

  const index = await buildIdentifierIndex(options.rootDir);
  if (index.docs.length === 0) {
    return "No supported identifiers found for semantic identifier search.";
  }

  const [queryVec] = await fetchEmbedding(options.query);
  const queryTerms = new Set(splitTerms(options.query));

  const scored: RankedIdentifier[] = [];
  for (let i = 0; i < index.docs.length; i++) {
    const doc = index.docs[i];
    if (includeKinds && !includeKinds.has(doc.kind.toLowerCase())) continue;

    const semanticScore = Math.max(cosine(queryVec, index.vectors[i]), 0);
    const keywordScore = getKeywordCoverage(queryTerms, `${doc.name} ${doc.signature} ${doc.path} ${doc.header}`);
    const totalWeight = semanticWeight + keywordWeight;
    const score = totalWeight > 0
      ? clamp01((semanticWeight * semanticScore + keywordWeight * keywordScore) / totalWeight)
      : semanticScore;

    scored.push({ doc, semanticScore, keywordScore, score });
  }

  if (scored.length === 0) {
    return "No identifiers matched the requested kind filters.";
  }

  const top = scored.sort((a, b) => b.score - a.score).slice(0, topK);
  const cache = await loadEmbeddingCache(options.rootDir, IDENTIFIER_CACHE_FILE);

  const lines: string[] = [
    `Top ${top.length} identifier matches for: "${options.query}"`,
    "",
  ];

  for (let i = 0; i < top.length; i++) {
    const item = top[i];
    const range = formatLineRange(item.doc.line, item.doc.endLine);
    lines.push(`${i + 1}. ${item.doc.kind} ${item.doc.name} - ${item.doc.path} (${range})`);
    lines.push(`   Score: ${Math.round(item.score * 1000) / 10}% | Semantic: ${Math.round(item.semanticScore * 1000) / 10}% | Keyword: ${Math.round(item.keywordScore * 1000) / 10}%`);
    lines.push(`   Signature: ${item.doc.signature}`);
    if (item.doc.parentName) lines.push(`   Parent: ${item.doc.parentName}`);

    const calls = await rankCallSites(
      options.rootDir,
      cache,
      queryTerms,
      queryVec,
      item.doc,
      index.fileLines,
      topCalls,
    );

    if (calls.sites.length === 0) {
      lines.push("   Calls: none found");
      lines.push("");
      continue;
    }

    lines.push(`   Calls (${calls.sites.length}/${calls.total}):`);
    for (let j = 0; j < calls.sites.length; j++) {
      const site = calls.sites[j];
      lines.push(`     ${j + 1}. ${site.file}:L${site.line} (${Math.round(site.score * 1000) / 10}%) ${site.context}`);
    }
    lines.push("");
  }

  await saveEmbeddingCache(options.rootDir, cache, IDENTIFIER_CACHE_FILE);
  return lines.join("\n");
}

export function invalidateIdentifierSearchCache(): void {
  cachedRootDir = null;
  cachedAt = 0;
  cachedIndex = null;
}

export async function refreshIdentifierEmbeddings(options: { rootDir: string; relativePaths: string[] }): Promise<number> {
  const uniquePaths = Array.from(new Set(options.relativePaths.map(normalizeRelativePath).filter(Boolean)));
  if (uniquePaths.length === 0) return 0;

  const cache = await loadEmbeddingCache(options.rootDir, IDENTIFIER_CACHE_FILE);
  const pending: { key: string; hash: string; text: string }[] = [];

  for (const relativePath of uniquePaths) {
    removeFileScopedCacheEntries(cache, relativePath);
    const docs = await buildIdentifierDocsForFile(options.rootDir, relativePath);
    for (const doc of docs) {
      const key = `id:${doc.id}`;
      const hash = hashContent(doc.text);
      pending.push({ key, hash, text: doc.text });
    }
  }

  if (pending.length > 0) {
    const batchSize = getEmbeddingBatchSize();
    for (let i = 0; i < pending.length; i += batchSize) {
      const batch = pending.slice(i, i + batchSize);
      const vectors = await fetchEmbedding(batch.map((entry) => entry.text));
      for (let j = 0; j < batch.length; j++) {
        cache[batch[j].key] = { hash: batch[j].hash, vector: vectors[j] };
      }
    }
  }

  await saveEmbeddingCache(options.rootDir, cache, IDENTIFIER_CACHE_FILE);
  invalidateIdentifierSearchCache();
  return pending.length;
}
