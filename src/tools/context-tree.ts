// Structural tree generator with file headers, symbols, and depth control
// Dynamic token-aware pruning: Level 0 (files only) to Level 2 (deep context)

import { walkRoots, type FileEntry } from "../core/walker.js";
import { analyzeFile, formatSymbol, isSupportedFile } from "../core/parser.js";

export interface ContextTreeOptions {
  rootDir: string;
  targetPath?: string;
  depthLimit?: number;
  includeSymbols?: boolean;
  maxTokens?: number;
}

interface TreeNode {
  name: string;
  relativePath: string;
  isDirectory: boolean;
  header?: string;
  symbols?: string;
  children: TreeNode[];
}

const CHARS_PER_TOKEN = 4;

function estimateTokens(text: string): number {
  return Math.ceil(text.length / CHARS_PER_TOKEN);
}

async function buildTree(entries: FileEntry[], _rootDir: string, includeSymbols: boolean): Promise<TreeNode> {
  const root: TreeNode = { name: ".", relativePath: ".", isDirectory: true, children: [] };
  const dirMap = new Map<string, TreeNode>();
  dirMap.set(".", root);

  const sortedEntries = entries.sort((a, b) => a.depth - b.depth || a.relativePath.localeCompare(b.relativePath));

  for (const entry of sortedEntries) {
    const parts = entry.relativePath.split("/");
    const parentPath = parts.length > 1 ? parts.slice(0, -1).join("/") : ".";
    let parent = dirMap.get(parentPath);
    if (!parent) {
      parent = root;
    }

    const node: TreeNode = {
      name: parts[parts.length - 1],
      relativePath: entry.relativePath,
      isDirectory: entry.isDirectory,
      children: [],
    };

    if (!entry.isDirectory && isSupportedFile(entry.path)) {
      try {
        const analysis = await analyzeFile(entry.path);
        node.header = analysis.header || undefined;
        if (includeSymbols && analysis.symbols.length > 0) {
          node.symbols = analysis.symbols.map((s) => formatSymbol(s, 0)).join("\n");
        }
      } catch { }
    }

    parent.children.push(node);
    if (entry.isDirectory) {
      dirMap.set(entry.relativePath, node);
    }
  }

  return root;
}

function renderTree(node: TreeNode, indent: number = 0): string {
  let result = "";
  const pad = "  ".repeat(indent);

  if (indent === 0) {
    result = `${node.name}/\n`;
  } else if (node.isDirectory) {
    result = `${pad}${node.name}/\n`;
  } else {
    result = `${pad}${node.name}`;
    if (node.header) result += ` | ${node.header}`;
    result += "\n";
    if (node.symbols) {
      for (const line of node.symbols.split("\n")) {
        result += `${pad}  ${line}\n`;
      }
    }
  }

  for (const child of node.children) {
    result += renderTree(child, indent + 1);
  }
  return result;
}

function pruneSymbols(node: TreeNode): void {
  node.symbols = undefined;
  for (const child of node.children) pruneSymbols(child);
}

function pruneHeaders(node: TreeNode): void {
  node.header = undefined;
  node.symbols = undefined;
  for (const child of node.children) pruneHeaders(child);
}

export async function getContextTree(options: ContextTreeOptions): Promise<string> {
  const entries = await walkRoots({
    rootDir: options.rootDir,
    targetPath: options.targetPath,
    depthLimit: options.depthLimit,
  });

  const includeSymbols = options.includeSymbols !== false;
  const tree = await buildTree(entries, options.rootDir, includeSymbols);
  const maxTokens = options.maxTokens ?? 20000;

  let rendered = renderTree(tree);
  if (estimateTokens(rendered) <= maxTokens) return rendered;

  pruneSymbols(tree);
  rendered = renderTree(tree);
  if (estimateTokens(rendered) <= maxTokens) return `[Level 1: Headers only, symbols pruned to fit ${maxTokens} tokens]\n\n${rendered}`;

  pruneHeaders(tree);
  rendered = renderTree(tree);
  return `[Level 0: File names only, project too large for ${maxTokens} tokens]\n\n${rendered}`;
}
