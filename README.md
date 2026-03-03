# Context+

Semantic Intelligence for Large-Scale Engineering.

Context+ is an MCP server designed for developers who demand 99% accuracy. By combining RAG, Tree-sitter AST, Spectral Clustering, and Obsidian-style linking, Context+ turns a massive codebase into a searchable, hierarchical feature graph.

https://github.com/user-attachments/assets/a97a451f-c9b4-468d-b036-15b65fc13e79

## Tools

### Discovery

| Tool                         | Description                                                                                                                                                      |
| ---------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `get_context_tree`           | Structural AST tree of a project with file headers and symbol ranges (line numbers for functions/classes/methods). Dynamic pruning shrinks output automatically. |
| `get_file_skeleton`          | Function signatures, class methods, and type definitions with line ranges, without reading full bodies. Shows the API surface.                                   |
| `semantic_code_search`       | Search by meaning, not exact text. Uses embeddings over file headers/symbols and returns matched symbol definition lines.                                        |
| `semantic_identifier_search` | Identifier-level semantic retrieval for functions/classes/variables with ranked call sites and line numbers.                                                     |
| `semantic_navigate`          | Browse codebase by meaning using spectral clustering. Groups semantically related files into labeled clusters.                                                   |

### Analysis

| Tool                  | Description                                                                                                                   |
| --------------------- | ----------------------------------------------------------------------------------------------------------------------------- |
| `get_blast_radius`    | Trace every file and line where a symbol is imported or used. Prevents orphaned references.                                   |
| `run_static_analysis` | Run native linters and compilers to find unused variables, dead code, and type errors. Supports TypeScript, Python, Rust, Go. |

### Code Ops

| Tool              | Description                                                                                                              |
| ----------------- | ------------------------------------------------------------------------------------------------------------------------ |
| `propose_commit`  | The only way to write code. Validates against strict rules before saving. Creates a shadow restore point before writing. |
| `get_feature_hub` | Obsidian-style feature hub navigator. Hubs are `.md` files with `[[wikilinks]]` that map features to code files.         |

### Version Control

| Tool                  | Description                                                                                                |
| --------------------- | ---------------------------------------------------------------------------------------------------------- |
| `list_restore_points` | List all shadow restore points created by `propose_commit`. Each captures file state before AI changes.    |
| `undo_change`         | Restore files to their state before a specific AI change. Uses shadow restore points. Does not affect git. |

### Memory & RAG

| Tool                       | Description                                                                                                       |
| -------------------------- | ----------------------------------------------------------------------------------------------------------------- |
| `upsert_memory_node`       | Create or update a memory node (concept, file, symbol, note) with auto-generated embeddings.                      |
| `create_relation`          | Create typed edges between nodes (relates_to, depends_on, implements, references, similar_to, contains).          |
| `search_memory_graph`      | Semantic search with graph traversal — finds direct matches then walks 1st/2nd-degree neighbors.                  |
| `prune_stale_links`        | Remove decayed edges (e^(-λt) below threshold) and orphan nodes with low access counts.                           |
| `add_interlinked_context`  | Bulk-add nodes with auto-similarity linking (cosine ≥ 0.72 creates edges automatically).                         |
| `retrieve_with_traversal`  | Start from a node and walk outward — returns all reachable neighbors scored by decay and depth.                   |

## Setup

### Quick Start (npx / bunx)

No installation needed. Add Context+ to your IDE MCP config.

For Claude Code, Cursor, and Windsurf, use `mcpServers`:

```json
{
  "mcpServers": {
    "contextplus": {
      "command": "bunx",
      "args": ["contextplus"],
      "env": {
        "OLLAMA_EMBED_MODEL": "nomic-embed-text",
        "OLLAMA_CHAT_MODEL": "gemma2:27b",
        "OLLAMA_API_KEY": "YOUR_OLLAMA_API_KEY"
      }
    }
  }
}
```

For VS Code (`.vscode/mcp.json`), use `servers` and `inputs`:

```json
{
  "servers": {
    "contextplus": {
      "type": "stdio",
      "command": "bunx",
      "args": ["contextplus"],
      "env": {
        "OLLAMA_EMBED_MODEL": "nomic-embed-text",
        "OLLAMA_CHAT_MODEL": "gemma2:27b",
        "OLLAMA_API_KEY": "YOUR_OLLAMA_API_KEY"
      }
    }
  },
  "inputs": []
}
```

If you prefer `npx`, use:

- `"command": "npx"`
- `"args": ["-y", "contextplus"]`

Or generate the MCP config file directly in your current directory:

```bash
npx -y contextplus init claude
bunx contextplus init cursor
npx -y contextplus init opencode
```

Supported coding agent names: `claude`, `cursor`, `vscode`, `windsurf`, `opencode`.

Config file locations:

| IDE         | Config File          |
| ----------- | -------------------- |
| Claude Code | `.mcp.json`          |
| Cursor      | `.cursor/mcp.json`   |
| VS Code     | `.vscode/mcp.json`   |
| Windsurf    | `.windsurf/mcp.json` |
| OpenCode    | `opencode.json`      |

### CLI Subcommands

- `init [target]` - Generate MCP configuration (targets: `claude`, `cursor`, `vscode`, `windsurf`, `opencode`).
- `skeleton [path]` or `tree [path]` - **(New)** View the structural tree of a project with file headers and symbol definitions directly in your terminal.
- `[path]` - Start the MCP server (stdio) for the specified path (defaults to current directory).

### From Source

```bash
npm install
npm run build
```

## Architecture

Three layers built with TypeScript over stdio using the Model Context Protocol SDK:

**Core** (`src/core/`) - Multi-language AST parsing (tree-sitter, 43 extensions), gitignore-aware traversal, Ollama vector embeddings with disk cache, wikilink hub graph, in-memory property graph with decay scoring.

**Tools** (`src/tools/`) - 17 MCP tools exposing structural, semantic, operational, and memory graph capabilities.

**Git** (`src/git/`) - Shadow restore point system for undo without touching git history.

**Runtime Cache** (`.mcp_data/`) - created on server startup; stores reusable file, identifier, and call-site embeddings to avoid repeated GPU/CPU embedding work. A realtime tracker refreshes changed files/functions incrementally.

## Config

| Variable                                | Default            | Description                                                   |
| --------------------------------------- | ------------------ | ------------------------------------------------------------- |
| `OLLAMA_EMBED_MODEL`                    | `nomic-embed-text` | Embedding model                                               |
| `OLLAMA_API_KEY`                        | -                  | Ollama Cloud API key                                          |
| `OLLAMA_CHAT_MODEL`                     | `llama3.2`         | Chat model for cluster labeling                               |
| `CONTEXTPLUS_EMBED_BATCH_SIZE`          | `8`                | Embedding batch size per GPU call, clamped to 5-10            |
| `CONTEXTPLUS_EMBED_TRACKER`             | `true`             | Enable realtime embedding refresh on file changes             |
| `CONTEXTPLUS_EMBED_TRACKER_MAX_FILES`   | `8`                | Max changed files processed per tracker tick, clamped to 5-10 |
| `CONTEXTPLUS_EMBED_TRACKER_DEBOUNCE_MS` | `700`              | Debounce window before tracker refresh                        |

## Test

```bash
npm test
npm run test:demo
npm run test:all
```
