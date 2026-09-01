# Context+ MCP - Agent Workflow

## Purpose

Context+ gives you structural awareness of the entire codebase without reading every file. These tools replace your default search and read operations — use them as your primary interface to the codebase.

## Short-Term KV Memory (native Context+)

Before every expensive MCP tool invocation, agents MUST `peek` the short-term KV silo. This cuts redundant expensive calls. Native core files: `src/core/short-term-kv.ts`, `src/core/solution-engine.ts`, `src/core/memory-graph.ts` (`mergeRankHits` for dual-layer `resolve_context`), wrappers in `src/tools/memory-tools.ts`, registered in `src/index.ts`.

**Token burn (existing PPM numbers only):** [three-way-speed-comparison.md](https://github.com/drQedwards/PPM/blob/main/mcp/benchmarks/three-way-speed-comparison.md) — TypeScript avg test execution: baseline **~302ms**, Context+-only **~63ms**, PMLL/peek-only **~26ms**, combined **~36ms**; peek cache hits **~0ms**. Prefer `peek` / `resolve_context` so graph work is paid once.

**Retrieval (not agent accuracy):** labeled P@k / R@k / MRR harness on pmll/PPM ([benchmark_retrieval.md](https://github.com/drQedwards/pmll/blob/main/mcp/benchmarks/benchmark_retrieval.md) + [run_retrieval_stub.py](https://github.com/drQedwards/pmll/blob/main/mcp/benchmarks/run_retrieval_stub.py)) — toy stub; **not** agent task success; do not cite 99%/99.99%.

**Optional complementary:** [pmll-memory-mcp](https://www.npmjs.com/package/pmll-memory-mcp) (`npx pmll-memory-mcp`) for full Q-promise / Python SQLite P0 durability — see [drQedwards/PPM](https://github.com/drQedwards/PPM) / [drQedwards/pmll](https://github.com/drQedwards/pmll).

| Tool      | Input                                              | Description                                                |
|-----------|----------------------------------------------------|------------------------------------------------------------|
| `init`    | `session_id: str`, `silo_size: int = 256`          | Initialize / reset the session KV silo                     |
| `peek`    | `session_id: str`, `key: str`                      | Non-destructive cache lookup + pending check (before every expensive tool) |
| `set`     | `session_id: str`, `key: str`, `value: str`        | Store a key-value pair after a cache miss                  |
| `resolve` | `session_id: str`, `promise_id: str`               | Check or resolve a pending continuation                    |
| `flush`   | `session_id: str`                                  | Clear all silo slots at task completion                    |

### The `peek()` Pattern

Call `peek` before every expensive tool invocation:

1. **`init`** once at task start to set up the session silo
2. **`peek`** before each expensive call — if hit, use the cached value; if pending, wait with `resolve`
3. **`set`** after a cache miss to populate the silo for future agents/subtasks
4. **`resolve`** to check or fulfill pending continuations
5. **`flush`** at task end to clear all session slots

This pattern ensures that Context+ tool results, Playwright page contents, and other expensive outputs are cached and reused across subtasks rather than re-fetched.

## Tool Priority (Mandatory)

You MUST use Context+ tools instead of native equivalents. Only fall back to native tools when a Context+ tool cannot fulfill the specific need.

| Instead of…              | MUST use…                    | Why                                          |
|--------------------------|------------------------------|----------------------------------------------|
| `grep`, `rg`, `ripgrep`  | `semantic_code_search`       | Finds by meaning, not just string match      |
| `find`, `ls`, `glob`     | `get_context_tree`           | Returns structure with symbols + line ranges |
| `cat`, `head`, read file | `get_file_skeleton` first    | Signatures without wasting context on bodies |
| manual symbol tracing    | `get_blast_radius`           | Traces all usages across the entire codebase |
| keyword search           | `semantic_identifier_search` | Ranked definitions + call chains             |
| directory browsing       | `semantic_navigate`          | Browse by meaning, not file paths            |

## Workflow

1. Start every task with `get_context_tree` or `get_file_skeleton` for structural overview
2. Use `semantic_code_search` or `semantic_identifier_search` to find code by meaning
3. Run `get_blast_radius` BEFORE modifying or deleting any symbol
4. Prefer structural tools over full-file reads — only read full files when signatures are insufficient
5. Run `run_static_analysis` after writing code
6. Use `search_memory_graph` at task start for prior context, `upsert_memory_node` after completing work

## Execution Rules

- Think less, execute sooner: make the smallest safe change that can be validated quickly
- Batch independent reads/searches in parallel — do not serialize them
- If a command fails, diagnose once, pivot strategy, continue — cap retries to 1-2
- Keep outputs concise: short status updates, no verbose reasoning

## Tool Reference

### PMLL Short-Term KV Memory

| Tool      | When to Use                                                                  |
|-----------|------------------------------------------------------------------------------|
| `init`    | Once at task start. Reset the session short-term KV silo.                    |
| `peek`    | Before every expensive MCP tool call. Non-destructive cache + pending check. |
| `set`     | After a cache miss. Store the result so future agents/subtasks skip the call. |
| `resolve` | When a key is pending. Check or fulfill the continuation.                    |
| `flush`   | At task end. Clear all silo slots for the session.                           |

### Context+ Structural Tools

| Tool                        | When to Use                                                  |
|-----------------------------|--------------------------------------------------------------|
| `get_context_tree`          | Start of every task. Map files + symbols with line ranges.   |
| `get_file_skeleton`         | Before full reads. Get signatures + line ranges first.       |
| `semantic_code_search`      | Find relevant files by concept.                              |
| `semantic_identifier_search`| Find functions/classes/variables and their call chains.      |
| `semantic_navigate`         | Browse codebase by meaning, not directory structure.         |
| `get_blast_radius`          | Before deleting or modifying any symbol.                     |
| `get_feature_hub`           | Browse feature graph hubs. Find orphaned files.              |
| `run_static_analysis`       | After writing code. Catch errors deterministically.          |
| `propose_commit`            | Validate and save file changes.                              |
| `list_restore_points`       | See undo history.                                            |
| `undo_change`               | Revert a change without touching git.                        |

### Long-Term Memory Graph

| Tool                        | When to Use                                                  |
|-----------------------------|--------------------------------------------------------------|
| `upsert_memory_node`        | Create/update memory nodes (concept, file, symbol, note).    |
| `create_relation`           | Create typed edges between memory nodes.                     |
| `search_memory_graph`       | Semantic search + graph traversal across neighbors.          |
| `prune_stale_links`         | Remove decayed edges and orphan nodes.                       |
| `add_interlinked_context`   | Bulk-add nodes with auto-similarity linking.                 |
| `retrieve_with_traversal`   | Walk outward from a node, return scored neighbors.           |

### Solution Engine

| Tool                   | When to Use                                                           |
|------------------------|-----------------------------------------------------------------------|
| `resolve_context`      | Unified context lookup — checks short-term KV first, falls back to long-term semantic graph. |
| `promote_to_long_term` | Promote a frequently-accessed short-term KV entry to the long-term memory graph. |
| `memory_status`        | Get a unified view of both short-term (KV cache) and long-term (semantic graph) memory layers. |

## Anti-Patterns

1. Reading entire files without checking the skeleton first
2. Deleting functions without checking blast radius
3. Running independent commands sequentially when they can be parallelized
4. Repeating failed commands without changing approach
5. Calling expensive MCP tools without calling `peek` first to check the cache
6. Forgetting to call `init` at task start or `flush` at task end, causing silent cache misses or stale data across sessions
7. Storing frequently-accessed payloads only in short-term KV instead of promoting them to long-term memory with `promote_to_long_term`
8. Calling `search_memory_graph` or `retrieve_with_traversal` directly instead of using `resolve_context`, which checks both memory layers in one call
9. Ignoring `pending` status from `peek` and re-issuing the same expensive call instead of waiting with `resolve`
