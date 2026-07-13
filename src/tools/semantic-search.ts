// Ollama-powered semantic search over file headers and symbol names
// Uses vector embeddings with cosine similarity for concept matching

import { walkRoots } from "../core/walker.js";
import { analyzeFile, flattenSymbols, isSupportedFile } from "../core/parser.js";
import {
  fetchEmbedding,
  getEmbeddingBatchSize,
  loadEmbeddingCache,
  saveEmbeddingCache,
  SearchIndex,
  type SearchDocument,
  type SearchQueryOptions,
} from "../core/embeddings.js";
import { readFile, stat } from "fs/promises";
import { extname, resolve } from "path";

export interface SemanticSearchOptions {
  rootDir: string;
  query: string;
  topK?: number;
  semanticWeight?: number;
  keywordWeight?: number;
  minSemanticScore?: number;
  minKeywordScore?: number;
  minCombinedScore?: number;
  requireKeywordMatch?: boolean;
  requireSemanticMatch?: boolean;
}

let cachedIndex: SearchIndex | null = null;
let cachedRootDir: string | null = null;
let lastIndexTime = 0;

const INDEX_TTL_MS = 60000;
const SEARCH_CACHE_FILE = "embeddings-cache.json";
const TEXT_INDEX_EXTENSIONS = new Set([
  ".md",
  ".txt",
  ".json",
  ".jsonc",
  ".geojson",
  ".csv",
  ".tsv",
  ".ndjson",
  ".yaml",
  ".yml",
  ".toml",
  ".lock",
  ".env",
]);
const MAX_TEXT_DOC_CHARS = 4000;
const DEFAULT_MAX_EMBED_FILE_SIZE = 50 * 1024;

function isTextIndexCandidate(filePath: string): boolean {
  return TEXT_INDEX_EXTENSIONS.has(extname(filePath).toLowerCase());
}

function toIntegerOr(value: string | undefined, fallback: number): number {
  if (!value) return fallback;
  const parsed = Number.parseInt(value, 10);
  return Number.isFinite(parsed) ? parsed : fallback;
}

function getMaxEmbedFileSize(): number {
  return Math.max(1024, toIntegerOr(process.env.CONTEXTPLUS_MAX_EMBED_FILE_SIZE, DEFAULT_MAX_EMBED_FILE_SIZE));
}

function extractPlainTextHeader(content: string): string {
  const lines = content.split("\n");
  const headerLines: string[] = [];
  for (const line of lines) {
    const trimmed = line.trim();
    if (!trimmed) continue;
    headerLines.push(trimmed.slice(0, 120));
    if (headerLines.length >= 2) break;
  }
  return headerLines.join(" | ");
}

function hashContent(text: string): string {
  let h = 0;
  for (let i = 0; i < text.length; i++) h = ((h << 5) - h + text.charCodeAt(i)) | 0;
  return h.toString(36);
}

function normalizeRelativePath(path: string): string {
  return path.replace(/\\/g, "/");
}

async function buildSearchDocumentForFile(rootDir: string, relativePath: string): Promise<SearchDocument | null> {
  const normalized = normalizeRelativePath(relativePath);
  const fullPath = resolve(rootDir, normalized);

  if (isTextIndexCandidate(fullPath)) {
    try {
      if ((await stat(fullPath)).size > getMaxEmbedFileSize()) return null;
      const raw = await readFile(fullPath, "utf-8");
      const content = raw.slice(0, MAX_TEXT_DOC_CHARS);
      return {
        path: normalized,
        header: extractPlainTextHeader(content),
        symbols: [],
        content,
      };
    } catch {
      return null;
    }
  }

  if (!isSupportedFile(fullPath)) return null;

  try {
    const analysis = await analyzeFile(fullPath);
    const flatSymbols = flattenSymbols(analysis.symbols);
    return {
      path: normalized,
      header: analysis.header,
      symbols: flatSymbols.map((s) => s.name),
      symbolEntries: flatSymbols.map((s) => ({
        name: s.name,
        kind: s.kind,
        line: s.line,
        endLine: s.endLine,
        signature: s.signature,
      })),
      content: flatSymbols.map((s) => s.signature).join(" "),
    };
  } catch {
    return null;
  }
}

async function buildIndex(rootDir: string): Promise<SearchIndex> {
  if (cachedIndex && cachedRootDir === rootDir && Date.now() - lastIndexTime < INDEX_TTL_MS) {
    return cachedIndex;
  }

  const entries = await walkRoots({ rootDir, depthLimit: 0 });
  const files = entries.filter((e) => !e.isDirectory);

  const docs: SearchDocument[] = [];
  for (const file of files) {
    const doc = await buildSearchDocumentForFile(rootDir, file.relativePath);
    if (doc) docs.push(doc);
  }

  const index = new SearchIndex();
  await index.index(docs, rootDir);
  cachedIndex = index;
  cachedRootDir = rootDir;
  lastIndexTime = Date.now();

  return index;
}

export async function semanticCodeSearch(options: SemanticSearchOptions): Promise<string> {
  const index = await buildIndex(options.rootDir);
  const searchOptions: SearchQueryOptions = {
    topK: options.topK,
    semanticWeight: options.semanticWeight,
    keywordWeight: options.keywordWeight,
    minSemanticScore: options.minSemanticScore,
    minKeywordScore: options.minKeywordScore,
    minCombinedScore: options.minCombinedScore,
    requireKeywordMatch: options.requireKeywordMatch,
    requireSemanticMatch: options.requireSemanticMatch,
  };
  const results = await index.search(options.query, searchOptions);

  if (results.length === 0) return "No matching files found for the given query.";

  const lines: string[] = [`Top ${results.length} hybrid matches for: "${options.query}"\n`];

  for (let i = 0; i < results.length; i++) {
    const r = results[i];
    lines.push(`${i + 1}. ${r.path} (${r.score}% total)`);
    lines.push(`   Semantic: ${r.semanticScore}% | Keyword: ${r.keywordScore}%`);
    if (r.header) lines.push(`   Header: ${r.header}`);
    if (r.matchedSymbols.length > 0) lines.push(`   Matched symbols: ${r.matchedSymbols.join(", ")}`);
    if (r.matchedSymbolLocations.length > 0) lines.push(`   Definition lines: ${r.matchedSymbolLocations.join(", ")}`);
    lines.push("");
  }

  return lines.join("\n");
}

export function invalidateSearchCache(): void {
  cachedIndex = null;
  cachedRootDir = null;
  lastIndexTime = 0;
}

export async function refreshFileSearchEmbeddings(options: { rootDir: string; relativePaths: string[] }): Promise<number> {
  const uniquePaths = Array.from(new Set(options.relativePaths.map(normalizeRelativePath).filter(Boolean)));
  if (uniquePaths.length === 0) return 0;

  const cache = await loadEmbeddingCache(options.rootDir, SEARCH_CACHE_FILE);
  const pending: { path: string; hash: string; text: string }[] = [];

  for (const relativePath of uniquePaths) {
    const doc = await buildSearchDocumentForFile(options.rootDir, relativePath);
    if (!doc) {
      delete cache[relativePath];
      continue;
    }

    const text = `${doc.header} ${doc.symbols.join(" ")} ${doc.content}`;
    const hash = hashContent(text);
    if (cache[relativePath]?.hash === hash) continue;
    pending.push({ path: relativePath, hash, text });
  }

  if (pending.length > 0) {
    const batchSize = getEmbeddingBatchSize();
    for (let i = 0; i < pending.length; i += batchSize) {
      const batch = pending.slice(i, i + batchSize);
      const vectors = await fetchEmbedding(batch.map((entry) => entry.text));
      for (let j = 0; j < batch.length; j++) {
        cache[batch[j].path] = { hash: batch[j].hash, vector: vectors[j] };
      }
    }
  }

  await saveEmbeddingCache(options.rootDir, cache, SEARCH_CACHE_FILE);
  invalidateSearchCache();
  return pending.length;
}
