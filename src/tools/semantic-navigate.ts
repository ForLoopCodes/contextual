// Semantic project navigator using spectral clustering and provider-agnostic labeling
// Browse codebase by meaning: embeds files, clusters vectors, generates labels

import { walkRoots } from "../core/walker.js";
import { analyzeFile, flattenSymbols, isSupportedFile } from "../core/parser.js";
import { fetchEmbedding } from "../core/embeddings.js";
import { readFile } from "fs/promises";
import { spectralCluster, findPathPattern } from "../core/clustering.js";
import { extname } from "path";

export interface SemanticNavigateOptions {
  rootDir: string;
  maxDepth?: number;
  maxClusters?: number;
}

interface FileInfo {
  relativePath: string;
  header: string;
  content: string;
  symbolPreview: string[];
}

interface ClusterNode {
  label: string;
  pathPattern: string | null;
  files: FileInfo[];
  children: ClusterNode[];
}

const EMBED_PROVIDER = (process.env.CONTEXTPLUS_EMBED_PROVIDER ?? "ollama").toLowerCase();
const EMBED_MODEL = process.env.OLLAMA_EMBED_MODEL ?? "nomic-embed-text";
const CHAT_MODEL = process.env.OLLAMA_CHAT_MODEL ?? "llama3.2";
const OPENAI_CHAT_MODEL = process.env.CONTEXTPLUS_OPENAI_CHAT_MODEL ?? process.env.OPENAI_CHAT_MODEL ?? "gpt-4o-mini";
const OPENAI_API_KEY = process.env.CONTEXTPLUS_OPENAI_API_KEY ?? process.env.OPENAI_API_KEY ?? "";
const OPENAI_BASE_URL = process.env.CONTEXTPLUS_OPENAI_BASE_URL ?? process.env.OPENAI_BASE_URL ?? "https://api.openai.com/v1";
const MAX_FILES_PER_LEAF = 20;
const NON_CODE_NAVIGATE_EXTENSIONS = new Set([
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

type OllamaChatClient = { chat: (params: Record<string, unknown>) => Promise<{ message: { content: string } }> };
let ollamaClient: OllamaChatClient | null = null;

async function getOllamaClient(): Promise<OllamaChatClient> {
  if (!ollamaClient) {
    const { Ollama } = await import("ollama");
    ollamaClient = new Ollama({ host: process.env.OLLAMA_HOST }) as unknown as OllamaChatClient;
  }
  return ollamaClient;
}

async function fetchEmbeddings(inputs: string[]): Promise<number[][]> {
  return fetchEmbedding(inputs);
}

function isNavigableSourceCandidate(filePath: string): boolean {
  return isSupportedFile(filePath) && !NON_CODE_NAVIGATE_EXTENSIONS.has(extname(filePath).toLowerCase());
}

async function chatCompletion(prompt: string): Promise<string> {
  if (EMBED_PROVIDER === "openai") {
    const url = `${OPENAI_BASE_URL.replace(/\/+$/, "")}/chat/completions`;
    const response = await fetch(url, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "Authorization": `Bearer ${OPENAI_API_KEY}`,
      },
      body: JSON.stringify({
        model: OPENAI_CHAT_MODEL,
        messages: [{ role: "user", content: prompt }],
        stream: false,
      }),
    });

    if (!response.ok) {
      const body = await response.text().catch(() => "");
      throw new Error(`OpenAI chat API error ${response.status}: ${body}`);
    }

    const data = await response.json() as { choices: { message: { content: string } }[] };
    return data.choices[0]?.message?.content ?? "";
  }

  const client = await getOllamaClient();
  const response = await client.chat({
    model: CHAT_MODEL,
    messages: [{ role: "user", content: prompt }],
    stream: false,
  });
  return response.message.content;
}

async function embedFilesWithFallback(files: FileInfo[]): Promise<{ files: FileInfo[]; vectors: number[][]; skipped: number }> {
  if (files.length === 0) return { files: [], vectors: [], skipped: 0 };
  const texts = files.map((file) => `${file.header} ${file.relativePath} ${file.content}`);

  try {
    return { files, vectors: await fetchEmbeddings(texts), skipped: 0 };
  } catch (error) {
    const keptFiles: FileInfo[] = [];
    const vectors: number[][] = [];

    for (let i = 0; i < files.length; i++) {
      try {
        const [vector] = await fetchEmbeddings([texts[i]]);
        keptFiles.push(files[i]);
        vectors.push(vector);
      } catch {
      }
    }

    if (keptFiles.length === 0) throw error;
    return { files: keptFiles, vectors, skipped: files.length - keptFiles.length };
  }
}

function extractHeader(content: string): string {
  const lines = content.split("\n");
  const headerLines: string[] = [];
  for (const line of lines.slice(0, 5)) {
    const trimmed = line.trim();
    if (trimmed.startsWith("//") || trimmed.startsWith("#") || trimmed.startsWith("--")) {
      headerLines.push(trimmed.replace(/^\/\/\s*|^#\s*|^--\s*/, ""));
    } else if (trimmed.length > 0) {
      break;
    }
  }
  return headerLines.join(" ").substring(0, 200);
}

function formatLineRange(line: number, endLine: number): string {
  return endLine > line ? `L${line}-L${endLine}` : `L${line}`;
}

async function labelSiblingClusters(clusters: { files: FileInfo[]; pathPattern: string | null }[]): Promise<string[]> {
  if (clusters.length === 0) return [];
  if (clusters.length === 1) {
    const pp = clusters[0].pathPattern;
    if (pp) return [pp];
    return [clusters[0].files.map((f) => f.relativePath.split("/").pop()).join(", ").substring(0, 40)];
  }

  const clusterDescriptions = clusters.map((c, i) => {
    const fileList = c.files.map((f) => `${f.relativePath}: ${f.header || "no description"}`).join("\n  ");
    const pattern = c.pathPattern ? ` (pattern: ${c.pathPattern})` : "";
    return `Cluster ${i + 1}${pattern}:\n  ${fileList}`;
  });

  const prompt = `You are labeling clusters of code files. For each cluster below, produce EXACTLY one JSON array of objects, each with:
- "overarchingTheme": a sentence about the cluster's theme
- "distinguishingFeature": what makes this cluster unique vs siblings
- "label": EXACTLY 2 words describing the cluster

${clusterDescriptions.join("\n\n")}

Respond with ONLY a JSON array of ${clusters.length} objects. No other text.`;

  try {
    const response = await chatCompletion(prompt);
    const jsonMatch = response.match(/\[[\s\S]*\]/);
    if (!jsonMatch) return clusters.map((_, i) => `Cluster ${i + 1}`);
    const labels = JSON.parse(jsonMatch[0]) as { label: string }[];
    return labels.map((l, i) => {
      const pp = clusters[i].pathPattern;
      const base = l.label || `Cluster ${i + 1}`;
      return pp ? `${base} (${pp})` : base;
    });
  } catch {
    return clusters.map((c, i) => c.pathPattern ?? `Cluster ${i + 1}`);
  }
}

async function buildHierarchy(files: FileInfo[], vectors: number[][], maxClusters: number, depth: number, maxDepth: number): Promise<ClusterNode> {
  if (files.length <= MAX_FILES_PER_LEAF || depth >= maxDepth) {
    return {
      label: "",
      pathPattern: findPathPattern(files.map((f) => f.relativePath)),
      files,
      children: [],
    };
  }

  const clusterResults = spectralCluster(vectors, maxClusters);

  if (clusterResults.length <= 1) {
    return {
      label: "",
      pathPattern: findPathPattern(files.map((f) => f.relativePath)),
      files,
      children: [],
    };
  }

  const childMetas = clusterResults.map((cluster) => ({
    files: cluster.indices.map((i) => files[i]),
    vectors: cluster.indices.map((i) => vectors[i]),
    pathPattern: findPathPattern(cluster.indices.map((i) => files[i].relativePath)),
  }));

  const labels = await labelSiblingClusters(childMetas.map((c) => ({ files: c.files, pathPattern: c.pathPattern })));

  const children: ClusterNode[] = [];
  for (let i = 0; i < childMetas.length; i++) {
    const child = await buildHierarchy(childMetas[i].files, childMetas[i].vectors, maxClusters, depth + 1, maxDepth);
    child.label = labels[i];
    children.push(child);
  }

  return {
    label: "",
    pathPattern: findPathPattern(files.map((f) => f.relativePath)),
    files: [],
    children,
  };
}

function renderClusterTree(node: ClusterNode, indent: number = 0): string {
  const pad = "  ".repeat(indent);
  let result = "";

  if (node.label) {
    result += `${pad}[${node.label}]\n`;
  }

  if (node.children.length > 0) {
    for (const child of node.children) {
      result += renderClusterTree(child, indent + 1);
    }
  } else {
    for (const file of node.files) {
      const label = file.header ? ` - ${file.header}` : "";
      const symbols = file.symbolPreview.length > 0 ? ` | symbols: ${file.symbolPreview.join(", ")}` : "";
      result += `${pad}  ${file.relativePath}${label}${symbols}\n`;
    }
  }

  return result;
}

export async function semanticNavigate(options: SemanticNavigateOptions): Promise<string> {
  const maxClusters = options.maxClusters ?? 20;
  const maxDepth = options.maxDepth ?? 3;

  const entries = await walkRoots({ rootDir: options.rootDir, depthLimit: 0 });
  const fileEntries = entries.filter((e) => !e.isDirectory && isNavigableSourceCandidate(e.path));

  if (fileEntries.length === 0) return "No supported source files found in the project.";

  const files: FileInfo[] = [];
  for (const entry of fileEntries) {
    try {
      const content = await readFile(entry.path, "utf-8");
      let header = extractHeader(content);
      let symbolPreview: string[] = [];
      try {
        const analysis = await analyzeFile(entry.path);
        if (analysis.header) header = analysis.header;
        symbolPreview = flattenSymbols(analysis.symbols)
          .slice(0, 3)
          .map((s) => `${s.name}@${formatLineRange(s.line, s.endLine)}`);
      } catch {
      }
      files.push({
        relativePath: entry.relativePath,
        header,
        content: content.substring(0, 500),
        symbolPreview,
      });
    } catch {
    }
  }

  if (files.length === 0) return "Could not read any source files.";

  let embeddableFiles: FileInfo[] = files;
  let vectors: number[][] = [];
  let skippedForEmbedding = 0;
  try {
    const embedded = await embedFilesWithFallback(files);
    embeddableFiles = embedded.files;
    vectors = embedded.vectors;
    skippedForEmbedding = embedded.skipped;
  } catch (err) {
    const providerHint = EMBED_PROVIDER === "openai"
      ? `Check CONTEXTPLUS_OPENAI_API_KEY and CONTEXTPLUS_OPENAI_BASE_URL.`
      : `Make sure Ollama is running (check OLLAMA_HOST) and that the embedding model configured in OLLAMA_EMBED_MODEL is available.`;
    return `Embedding provider (${EMBED_PROVIDER}) not available: ${err instanceof Error ? err.message : String(err)}\n${providerHint}`;
  }

  if (embeddableFiles.length === 0) return "No embeddable source files found in the project.";

  if (embeddableFiles.length <= MAX_FILES_PER_LEAF) {
    let fileLabels: string[];
    try {
      const prompt = `For each file below, produce a 3-7 word description. Return ONLY a JSON array of strings.\n\n${embeddableFiles.map((f) => `${f.relativePath}: ${f.header}`).join("\n")}`;
      const response = await chatCompletion(prompt);
      const match = response.match(/\[[\s\S]*\]/);
      fileLabels = match ? JSON.parse(match[0]) : embeddableFiles.map((f) => f.header);
    } catch {
      fileLabels = embeddableFiles.map((f) => f.header);
    }

    const summary = skippedForEmbedding > 0
      ? `Semantic Navigator: ${embeddableFiles.length} files (${skippedForEmbedding} skipped due embedding limits)\n`
      : `Semantic Navigator: ${embeddableFiles.length} files\n`;
    const lines = [summary];
    for (let i = 0; i < embeddableFiles.length; i++) {
      const symbols = embeddableFiles[i].symbolPreview.length > 0 ? ` | symbols: ${embeddableFiles[i].symbolPreview.join(", ")}` : "";
      lines.push(`  ${embeddableFiles[i].relativePath} - ${fileLabels[i] || embeddableFiles[i].header}${symbols}`);
    }
    return lines.join("\n");
  }

  const tree = await buildHierarchy(embeddableFiles, vectors, maxClusters, 0, maxDepth);
  tree.label = "Project";

  const summary = skippedForEmbedding > 0
    ? `Semantic Navigator: ${embeddableFiles.length} files organized by meaning (${skippedForEmbedding} skipped due embedding limits)`
    : `Semantic Navigator: ${embeddableFiles.length} files organized by meaning`;

  return `${summary}\n\n${renderClusterTree(tree)}`;
}
