#!/usr/bin/env node
// Context+ MCP - Semantic codebase navigator for AI agents
// Structural AST tree, blast radius, semantic search, commit gatekeeper

import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import { mkdir, readFile, writeFile } from "fs/promises";
import { dirname, resolve } from "path";
import { fileURLToPath } from "url";
import { z } from "zod";
import { startEmbeddingTracker } from "./core/embedding-tracker.js";
import { isBrokenPipeError, runCleanup } from "./core/process-lifecycle.js";
import { getContextTree } from "./tools/context-tree.js";
import { getFileSkeleton } from "./tools/file-skeleton.js";
import { ensureMcpDataDir } from "./core/embeddings.js";
import { semanticCodeSearch, invalidateSearchCache } from "./tools/semantic-search.js";
import { semanticIdentifierSearch, invalidateIdentifierSearchCache } from "./tools/semantic-identifiers.js";
import { getBlastRadius } from "./tools/blast-radius.js";
import { runStaticAnalysis } from "./tools/static-analysis.js";
import { proposeCommit } from "./tools/propose-commit.js";
import { listRestorePoints, restorePoint } from "./git/shadow.js";
import { semanticNavigate } from "./tools/semantic-navigate.js";
import { getFeatureHub } from "./tools/feature-hub.js";

type AgentTarget = "claude" | "cursor" | "vscode" | "windsurf" | "opencode";

const AGENT_CONFIG_PATH: Record<AgentTarget, string> = {
  claude: ".mcp.json",
  cursor: ".cursor/mcp.json",
  vscode: ".vscode/mcp.json",
  windsurf: ".windsurf/mcp.json",
  opencode: "opencode.json",
};

const SUB_COMMANDS = ["init", "skeleton", "tree"];
const passthroughArgs = process.argv.slice(2);
const ROOT_DIR = passthroughArgs[0] && !SUB_COMMANDS.includes(passthroughArgs[0])
  ? resolve(passthroughArgs[0])
  : process.cwd();
const INSTRUCTIONS_PATH = resolve(dirname(fileURLToPath(import.meta.url)), "../INSTRUCTIONS.md");
const INSTRUCTIONS_RESOURCE_URI = "contextplus://instructions";

function parseAgentTarget(input?: string): AgentTarget {
  const normalized = (input ?? "claude").toLowerCase();
  if (normalized === "claude" || normalized === "claude-code") return "claude";
  if (normalized === "cursor") return "cursor";
  if (normalized === "vscode" || normalized === "vs-code" || normalized === "vs") return "vscode";
  if (normalized === "windsurf") return "windsurf";
  if (normalized === "opencode" || normalized === "open-code") return "opencode";
  throw new Error(`Unsupported coding agent \"${input}\". Use one of: claude, cursor, vscode, windsurf, opencode.`);
}

function parseRunner(args: string[]): "npx" | "bunx" {
  const explicit = args.find((arg) => arg.startsWith("--runner="));
  if (explicit) {
    const value = explicit.split("=")[1];
    if (value === "npx" || value === "bunx") return value;
    throw new Error(`Unsupported runner \"${value}\". Use --runner=npx or --runner=bunx.`);
  }
  const runnerFlagIndex = args.findIndex((arg) => arg === "--runner");
  if (runnerFlagIndex >= 0) {
    const value = args[runnerFlagIndex + 1];
    if (value === "npx" || value === "bunx") return value;
    throw new Error(`Unsupported runner \"${value}\". Use --runner=npx or --runner=bunx.`);
  }
  const userAgent = (process.env.npm_config_user_agent ?? "").toLowerCase();
  const execPath = (process.env.npm_execpath ?? "").toLowerCase();
  if (userAgent.includes("bun/") || execPath.includes("bun")) return "bunx";
  return "npx";
}

function buildMcpConfig(runner: "npx" | "bunx") {
  const commandArgs = runner === "npx" ? ["-y", "contextplus"] : ["contextplus"];
  return JSON.stringify(
    {
      mcpServers: {
        contextplus: {
          command: runner,
          args: commandArgs,
          env: {
            OLLAMA_EMBED_MODEL: "nomic-embed-text",
            OLLAMA_CHAT_MODEL: "gemma2:27b",
            OLLAMA_API_KEY: "YOUR_OLLAMA_API_KEY",
            CONTEXTPLUS_EMBED_BATCH_SIZE: "8",
            CONTEXTPLUS_EMBED_TRACKER: "true",
          },
        },
      },
    },
    null,
    2,
  );
}

function buildOpenCodeConfig(runner: "npx" | "bunx") {
  const command = runner === "npx" ? ["npx", "-y", "contextplus"] : ["bunx", "contextplus"];
  return JSON.stringify(
    {
      $schema: "https://opencode.ai/config.json",
      mcp: {
        contextplus: {
          type: "local",
          command,
          enabled: true,
          environment: {
            OLLAMA_EMBED_MODEL: "nomic-embed-text",
            OLLAMA_CHAT_MODEL: "gemma2:27b",
            OLLAMA_API_KEY: "YOUR_OLLAMA_API_KEY",
            CONTEXTPLUS_EMBED_BATCH_SIZE: "8",
            CONTEXTPLUS_EMBED_TRACKER: "true",
          },
        },
      },
    },
    null,
    2,
  );
}

async function runInitCommand(args: string[]) {
  const nonFlags = args.filter((arg) => !arg.startsWith("--"));
  const target = parseAgentTarget(nonFlags[0]);
  const runner = parseRunner(args);
  const outputPath = resolve(process.cwd(), AGENT_CONFIG_PATH[target]);
  const content = target === "opencode" ? buildOpenCodeConfig(runner) : buildMcpConfig(runner);
  await mkdir(dirname(outputPath), { recursive: true });
  await writeFile(outputPath, `${content}\n`, "utf8");
  console.error(`Context+ initialized for ${target} using ${runner}.`);
  console.error(`Wrote MCP config: ${outputPath}`);
}

const server = new McpServer({
  name: "contextplus",
  version: "1.0.0",
}, {
  capabilities: { logging: {} },
});

server.resource(
  "contextplus_instructions",
  INSTRUCTIONS_RESOURCE_URI,
  async (uri) => ({
    contents: [{
      uri: uri.href,
      mimeType: "text/markdown",
      text: await readFile(INSTRUCTIONS_PATH, "utf8"),
    }],
  }),
);

server.tool(
  "get_context_tree",
  "Get the structural tree of the project with file headers, function names, classes, enums, and line ranges. " +
  "Automatically reads 2-line headers for file purpose. Dynamic token-aware pruning: " +
  "Level 2 (deep symbols) -> Level 1 (headers only) -> Level 0 (file names only) based on project size.",
  {
    target_path: z.string().optional().describe("Specific directory or file to analyze (relative to project root). Defaults to root."),
    depth_limit: z.number().optional().describe("How many folder levels deep to scan. Use 1-2 for large projects."),
    include_symbols: z.boolean().optional().describe("Include function/class/enum names in the tree. Defaults to true."),
    max_tokens: z.number().optional().describe("Maximum tokens for output. Auto-prunes if exceeded. Default: 20000."),
  },
  async ({ target_path, depth_limit, include_symbols, max_tokens }) => ({
    content: [{
      type: "text" as const,
      text: await getContextTree({
        rootDir: ROOT_DIR,
        targetPath: target_path,
        depthLimit: depth_limit,
        includeSymbols: include_symbols,
        maxTokens: max_tokens,
      }),
    }],
  }),
);

server.tool(
  "semantic_identifier_search",
  "Search semantic intent at identifier level (functions, methods, classes, variables) with definition lines and ranked call sites. " +
  "Uses embeddings over symbol signatures and source context, then returns line-numbered definition/call chains.",
  {
    query: z.string().describe("Natural language intent to match identifiers and usages."),
    top_k: z.number().optional().describe("How many identifiers to return. Default: 5."),
    top_calls_per_identifier: z.number().optional().describe("How many ranked call sites per identifier. Default: 10."),
    include_kinds: z.array(z.string()).optional().describe("Optional kinds filter, e.g. [\"function\", \"method\", \"variable\"]."),
    semantic_weight: z.number().optional().describe("Weight for semantic similarity score. Default: 0.78."),
    keyword_weight: z.number().optional().describe("Weight for keyword overlap score. Default: 0.22."),
  },
  async ({ query, top_k, top_calls_per_identifier, include_kinds, semantic_weight, keyword_weight }) => ({
    content: [{
      type: "text" as const,
      text: await semanticIdentifierSearch({
        rootDir: ROOT_DIR,
        query,
        topK: top_k,
        topCallsPerIdentifier: top_calls_per_identifier,
        includeKinds: include_kinds,
        semanticWeight: semantic_weight,
        keywordWeight: keyword_weight,
      }),
    }],
  }),
);

server.tool(
  "get_file_skeleton",
  "Get detailed function signatures, class methods, and type definitions of a specific file WITHOUT reading the full body. " +
  "Shows the API surface: function names, parameters, return types, and line ranges. Perfect for understanding how to use code without loading it all.",
  {
    file_path: z.string().describe("Path to the file to inspect (relative to project root)."),
  },
  async ({ file_path }) => ({
    content: [{
      type: "text" as const,
      text: await getFileSkeleton({ rootDir: ROOT_DIR, filePath: file_path }),
    }],
  }),
);

server.tool(
  "semantic_code_search",
  "Search the codebase by MEANING, not just exact variable names. Uses Ollama embeddings over file headers and symbol names. " +
  "Example: searching 'user authentication' finds files about login, sessions, JWT even if those exact words aren't used, with matched definition lines.",
  {
    query: z.string().describe("Natural language description of what you're looking for. Example: 'how are transactions signed'"),
    top_k: z.number().optional().describe("Number of matches to return. Default: 5."),
    semantic_weight: z.number().optional().describe("Weight for embedding similarity in hybrid ranking. Default: 0.72."),
    keyword_weight: z.number().optional().describe("Weight for keyword overlap in hybrid ranking. Default: 0.28."),
    min_semantic_score: z.number().optional().describe("Minimum semantic score filter. Accepts 0-1 or 0-100."),
    min_keyword_score: z.number().optional().describe("Minimum keyword score filter. Accepts 0-1 or 0-100."),
    min_combined_score: z.number().optional().describe("Minimum final score filter. Accepts 0-1 or 0-100."),
    require_keyword_match: z.boolean().optional().describe("When true, only return files with keyword overlap."),
    require_semantic_match: z.boolean().optional().describe("When true, only return files with positive semantic similarity."),
  },
  async ({
    query,
    top_k,
    semantic_weight,
    keyword_weight,
    min_semantic_score,
    min_keyword_score,
    min_combined_score,
    require_keyword_match,
    require_semantic_match,
  }) => ({
    content: [{
      type: "text" as const,
      text: await semanticCodeSearch({
        rootDir: ROOT_DIR,
        query,
        topK: top_k,
        semanticWeight: semantic_weight,
        keywordWeight: keyword_weight,
        minSemanticScore: min_semantic_score,
        minKeywordScore: min_keyword_score,
        minCombinedScore: min_combined_score,
        requireKeywordMatch: require_keyword_match,
        requireSemanticMatch: require_semantic_match,
      }),
    }],
  }),
);

server.tool(
  "get_blast_radius",
  "Before deleting or modifying code, check the BLAST RADIUS. Traces every file and line where a specific symbol " +
  "(function, class, variable) is imported or used. Prevents orphaned code. Also warns if usage count is low (candidate for inlining).",
  {
    symbol_name: z.string().describe("The function, class, or variable name to trace across the codebase."),
    file_context: z.string().optional().describe("The file where the symbol is defined. Excludes the definition line from results."),
  },
  async ({ symbol_name, file_context }) => ({
    content: [{
      type: "text" as const,
      text: await getBlastRadius({ rootDir: ROOT_DIR, symbolName: symbol_name, fileContext: file_context }),
    }],
  }),
);

server.tool(
  "run_static_analysis",
  "Run the project's native linter/compiler to find unused variables, dead code, type errors, and syntax issues. " +
  "Delegates detection to deterministic tools instead of LLM guessing. Supports TypeScript, Python, Rust, Go.",
  {
    target_path: z.string().optional().describe("Specific file or folder to lint (relative to root). Omit for full project."),
  },
  async ({ target_path }) => ({
    content: [{
      type: "text" as const,
      text: await runStaticAnalysis({ rootDir: ROOT_DIR, targetPath: target_path }),
    }],
  }),
);

server.tool(
  "propose_commit",
  "The ONLY way to write code. Validates the code against strict rules before saving: " +
  "2-line header comments, no inline comments, max nesting depth, max file length. " +
  "Creates a shadow restore point before writing. REJECTS code that violates formatting rules.",
  {
    file_path: z.string().describe("Where to save the file (relative to project root)."),
    new_content: z.string().describe("The complete file content to save."),
  },
  async ({ file_path, new_content }) => {
    invalidateSearchCache();
    invalidateIdentifierSearchCache();
    return {
      content: [{
        type: "text" as const,
        text: await proposeCommit({ rootDir: ROOT_DIR, filePath: file_path, newContent: new_content }),
      }],
    };
  },
);

server.tool(
  "list_restore_points",
  "List all shadow restore points created by propose_commit. Each point captures the file state before the AI made changes. " +
  "Use this to find a restore point ID for undoing a bad change.",
  {},
  async () => {
    const points = await listRestorePoints(ROOT_DIR);
    if (points.length === 0) return { content: [{ type: "text" as const, text: "No restore points found." }] };

    const lines = points.map((p) =>
      `${p.id} | ${new Date(p.timestamp).toISOString()} | ${p.files.join(", ")} | ${p.message}`,
    );
    return { content: [{ type: "text" as const, text: `Restore Points (${points.length}):\n\n${lines.join("\n")}` }] };
  },
);

server.tool(
  "undo_change",
  "Restore files to their state before a specific AI change. Uses the shadow restore point system. " +
  "Does NOT affect git history. Call list_restore_points first to find the point ID.",
  {
    point_id: z.string().describe("The restore point ID (format: rp-timestamp-hash). Get from list_restore_points."),
  },
  async ({ point_id }) => {
    const restored = await restorePoint(ROOT_DIR, point_id);
    invalidateSearchCache();
    invalidateIdentifierSearchCache();
    return {
      content: [{
        type: "text" as const,
        text: restored.length > 0
          ? `Restored ${restored.length} file(s):\n${restored.join("\n")}`
          : "No files were restored. The backup may be empty.",
      }],
    };
  },
);

server.tool(
  "semantic_navigate",
  "Browse the codebase by MEANING, not directory structure. Uses spectral clustering on Ollama embeddings to group " +
  "semantically related files into labeled clusters. Inspired by Gabriella Gonzalez's semantic navigator. " +
  "Requires Ollama running with an embedding model and a chat model for labeling.",
  {
    max_depth: z.number().optional().describe("Maximum nesting depth of clusters. Default: 3."),
    max_clusters: z.number().optional().describe("Maximum sub-clusters per level. Default: 20."),
  },
  async ({ max_depth, max_clusters }) => ({
    content: [{
      type: "text" as const,
      text: await semanticNavigate({ rootDir: ROOT_DIR, maxDepth: max_depth, maxClusters: max_clusters }),
    }],
  }),
);

server.tool(
  "get_feature_hub",
  "Obsidian-style feature hub navigator. Hub files are .md files containing [[path/to/file]] wikilinks that act as a Map of Content. " +
  "Modes: (1) No args = list all hubs, (2) hub_path or feature_name = show hub with bundled skeletons of all linked files, " +
  "(3) show_orphans = find files not linked to any hub. Prevents orphaned code and enables graph-based codebase navigation.",
  {
    hub_path: z.string().optional().describe("Path to a specific hub .md file (relative to root)."),
    feature_name: z.string().optional().describe("Feature name to search for. Finds matching hub file automatically."),
    show_orphans: z.boolean().optional().describe("If true, lists all source files not linked to any feature hub."),
  },
  async ({ hub_path, feature_name, show_orphans }) => ({
    content: [{
      type: "text" as const,
      text: await getFeatureHub({
        rootDir: ROOT_DIR,
        hubPath: hub_path,
        featureName: feature_name,
        showOrphans: show_orphans,
      }),
    }],
  }),
);

async function main() {
  const args = process.argv.slice(2);
  if (args[0] === "init") {
    await runInitCommand(args.slice(1));
    return;
  }
  if (args[0] === "skeleton" || args[0] === "tree") {
    const targetRoot = args[1] ? resolve(args[1]) : process.cwd();
    const tree = await getContextTree({
      rootDir: targetRoot,
      includeSymbols: true,
      maxTokens: 50000,
    });
    process.stdout.write(tree + "\n");
    return;
  }
  await ensureMcpDataDir(ROOT_DIR);
  const trackerEnabled = (process.env.CONTEXTPLUS_EMBED_TRACKER ?? "true").toLowerCase() !== "false";
  const stopTracker = trackerEnabled
    ? startEmbeddingTracker({
      rootDir: ROOT_DIR,
      debounceMs: Number.parseInt(process.env.CONTEXTPLUS_EMBED_TRACKER_DEBOUNCE_MS ?? "700", 10),
      maxFilesPerTick: Number.parseInt(process.env.CONTEXTPLUS_EMBED_TRACKER_MAX_FILES ?? "8", 10),
    })
    : () => { };
  const transport = new StdioServerTransport();
  await server.connect(transport);

  let shuttingDown = false;
  const closeServer = async () => {
    const closable = server as unknown as { close?: () => Promise<void> | void };
    if (typeof closable.close === "function") {
      await closable.close();
    }
  };
  const closeTransport = async () => {
    const closable = transport as unknown as { close?: () => Promise<void> | void };
    if (typeof closable.close === "function") {
      await closable.close();
    }
  };
  const shutdown = async (reason: string, exitCode: number = 0) => {
    if (shuttingDown) return;
    shuttingDown = true;
    console.error(`Context+ MCP shutdown requested: ${reason}`);
    await runCleanup({ stopTracker, closeServer, closeTransport });
    process.exit(exitCode);
  };
  const requestShutdown = (reason: string, exitCode: number = 0) => {
    void shutdown(reason, exitCode);
  };

  process.once("SIGINT", () => requestShutdown("SIGINT", 0));
  process.once("SIGTERM", () => requestShutdown("SIGTERM", 0));
  process.once("exit", () => stopTracker());
  process.stdin.once("end", () => requestShutdown("stdin-end", 0));
  process.stdin.once("close", () => requestShutdown("stdin-close", 0));
  process.stdin.once("error", (error) => {
    if (isBrokenPipeError(error)) requestShutdown("stdin-error", 0);
  });
  process.stdout.once("error", (error) => {
    if (isBrokenPipeError(error)) requestShutdown("stdout-error", 0);
  });
  process.stderr.once("error", (error) => {
    if (isBrokenPipeError(error)) requestShutdown("stderr-error", 0);
  });

  console.error(`Context+ MCP server running on stdio | root: ${ROOT_DIR}`);
}

main().catch((error) => {
  console.error("Fatal error:", error);
  process.exit(1);
});
