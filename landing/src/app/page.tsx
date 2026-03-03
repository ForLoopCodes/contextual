import Background from "../components/Background";
import Header from "../components/Header";
import IdeSetup from "../components/IdeSetup";
import InstructionsSection from "../components/InstructionsSection";
import IsometricDiagram from "../components/IsometricDiagram";
import ToolDiagram from "../components/ToolDiagram";

export const dynamic = "force-dynamic";

const toolRefRows = [
  {
    name: "get_context_tree",
    desc: "Get the structural AST tree of a project with file headers plus line-numbered function/class/method symbols. Dynamic token-aware pruning shrinks output automatically.",
    input:
      "{\n  target_path?: string,\n  depth_limit?: number,\n  include_symbols?: boolean,\n  max_tokens?: number\n}",
    output:
      '"src/\n  index.ts - Entry point\n    function: getStars() (L170-L181)\n    function: Home() (L183-L760)\n  utils/\n    parser.ts - AST parsing\n      function: parseFile() (L22-L84)\n      function: walkTree() (L86-L132)"',
  },
  {
    name: "get_file_skeleton",
    desc: "Get function signatures, class methods, and type definitions of a file with line ranges, without reading the full body.",
    input: "{ file_path: string }",
    output:
      '"[function] L12-L58 export function parseFile(\n  filePath: string,\n  options?: ParseOptions\n): Promise<AST>;\n\n[class] L60-L130 export class Walker;\n  [method] L72-L94 walk(node: Node): void;\n  [method] L96-L118 getSymbols(): Symbol[];"',
  },
  {
    name: "semantic_code_search",
    desc: "Search the codebase by meaning, not exact text. Uses embeddings over file headers/symbols and returns matched definition lines.",
    input: "{ query: string, top_k?: number }",
    output:
      '"1. src/auth/jwt.ts (94.0% total)\n   Semantic: 91.5% | Keyword: 96.2%\n   Definition lines: verifyToken@L20-L58, signToken@L60-L102\n2. src/auth/session.ts (87.4% total)\n   Definition lines: createSession@L12-L42"',
  },
  {
    name: "semantic_identifier_search",
    desc: "Find closest functions/classes/variables by meaning, then return ranked definition and call-chain locations with line numbers. Uses realtime-refreshed identifier embeddings.",
    input:
      "{\n  query: string,\n  top_k?: number,\n  top_calls_per_identifier?: number,\n  include_kinds?: string[]\n}",
    output:
      '"1. function verifyToken - src/auth/jwt.ts (L20-L58)\n   Score: 92.4%\n   Calls (3/3):\n     1. src/middleware/guard.ts:L33 (88.1%) verifyToken(token)\n     2. src/routes/api.ts:L12 (84.7%) const user = verifyToken(raw)\n2. variable tokenExpiry - src/auth/config.ts (L8)"',
  },
  {
    name: "get_blast_radius",
    desc: "Before modifying code, trace every file and line where a symbol is imported or used. Prevents orphaned references.",
    input: "{\n  symbol_name: string,\n  file_context?: string\n}",
    output:
      '"parseFile - 7 usages\n  src/index.ts:14  import { parseFile }\n  src/tools/tree.ts:8  const ast = parseFile(p)\n  src/tools/skeleton.ts:22  parseFile(path)\n  test/parser.test.ts:5  import { parseFile }"',
  },
  {
    name: "run_static_analysis",
    desc: "Run the native linter or compiler to find unused variables, dead code, and type errors. Supports TypeScript, Python, Rust, Go.",
    input: "{ target_path?: string }",
    output:
      '"src/utils.ts:14:5\n  error TS2345: Argument of type string\n  is not assignable to parameter\n\nsrc/old.ts:1:1\n  warning: file has no exports"',
  },
  {
    name: "propose_commit",
    desc: "The only way to write code. Validates against strict rules before saving. Creates a shadow restore point before writing.",
    input: "{\n  file_path: string,\n  new_content: string\n}",
    output:
      '"✓ Header comment present\n✓ No inline comments\n✓ Max nesting depth: 3\n✓ File length: 142 lines\n\nSaved src/tools/search.ts\nRestore point: rp-1719384000-a3f2"',
  },
  {
    name: "list_restore_points",
    desc: "List all shadow restore points created by propose_commit. Each captures file state before AI changes.",
    input: "{ }",
    output:
      '"rp-1719384000-a3f2 | 2025-06-26\n  src/tools/search.ts | refactor search\n\nrp-1719383000-b7c1 | 2025-06-26\n  src/index.ts | add new tool"',
  },
  {
    name: "undo_change",
    desc: "Restore files to their state before a specific AI change. Uses shadow restore points. Does not affect git.",
    input: "{ point_id: string }",
    output: '"Restored 1 file(s):\n  src/tools/search.ts"',
  },
  {
    name: "semantic_navigate",
    desc: "Browse codebase by meaning using spectral clustering. Groups semantically related files into labeled clusters.",
    input: "{\n  max_depth?: number,\n  max_clusters?: number\n}",
    output:
      '"Authentication (4 files)\n  src/auth/jwt.ts\n  src/auth/session.ts\n  src/middleware/guard.ts\n  src/models/user.ts\n\nParsing (3 files)\n  src/core/parser.ts\n  src/core/tree-sitter.ts\n  src/core/walker.ts"',
  },
  {
    name: "get_feature_hub",
    desc: "Obsidian-style feature hub navigator. Hubs are .md files with [[wikilinks]] that map features to code files.",
    input:
      "{\n  hub_path?: string,\n  feature_name?: string,\n  show_orphans?: boolean\n}",
    output:
      '"## auth.md\n[[src/auth/jwt.ts]]\n  → verifyToken(token: string)\n  → signToken(payload: object)\n\n[[src/auth/session.ts]]\n  → createSession(userId: string)\n  → destroySession(id: string)"',
  },
];

async function getStars(): Promise<number> {
  try {
    const res = await fetch(
      "https://api.github.com/repos/ForLoopCodes/contextplus",
      { cache: "no-store" },
    );
    if (!res.ok) {
      return 0;
    }
    const data = await res.json();
    return data.stargazers_count ?? 0;
  } catch {
    return 0;
  }
}

export default async function Home() {
  const stars = await getStars();

  return (
    <div className="relative w-full min-h-screen">
      <Background />
      <Header stars={stars} />

      <div
        className="hero-diagram-row"
        style={{
          display: "flex",
          alignItems: "center",
          padding: "0 100px",
          gap: 60,
          zIndex: 1,
          position: "relative",
        }}
      >
        <section
          className="hero-section flex flex-col relative"
          style={{ gap: 24, flex: "1 1 auto", minWidth: 0, maxWidth: 630 }}
        >
          <h1
            className="hero-title font-light"
            style={{
              fontSize: 56,
              lineHeight: "72px",
              letterSpacing: "-0.02em",
              color: "var(--text-muted)",
              fontFamily: "var(--font-geist-sans)",
            }}
          >
            Semantic Intelligence for
            <br />
            <span style={{ color: "var(--text-primary)" }}>
              Large-Scale Engineering.
            </span>
          </h1>
          <p
            className="hero-text font-light"
            style={{
              fontSize: 18,
              lineHeight: "28px",
              letterSpacing: "-0.02em",
              background:
                "linear-gradient(180deg, var(--text-primary) 0%, var(--gradient-end) 100%)",
              WebkitBackgroundClip: "text",
              WebkitTextFillColor: "transparent",
              backgroundClip: "text",
            }}
          >
            Context+ is an MCP server designed for developers who demand 99%
            accuracy. By combining RAG, Treesitter AST & Spectral
            Clustering, Context+ turns a massive codebase into a searchable,
            hierarchical graph.
          </p>
        </section>
      </div>

      <IsometricDiagram />

      <div
        className="diagram-outer"
        style={{
          position: "relative",
          zIndex: 1,
          width: "fit-content",
          margin: "0 auto",
          minHeight: "100vh",
          display: "flex",
          flexDirection: "column",
          justifyContent: "center",
        }}
      >
        <ToolDiagram />
      </div>

      <IdeSetup />

      <InstructionsSection />

      <section
        className="tools-ref"
        style={{
          position: "relative",
          zIndex: 1,
          padding: "40px 100px 80px",
          width: "100%",
          maxWidth: 1200,
          marginLeft: "auto",
          marginRight: "auto",
        }}
      >
        <p
          style={{
            fontSize: 18,
            fontWeight: 300,
            lineHeight: "28px",
            fontFamily: "var(--font-geist-pixel-square)",
            letterSpacing: "-0.02em",
            background:
              "linear-gradient(180deg, var(--text-primary) 0%, var(--gradient-end) 100%)",
            WebkitBackgroundClip: "text",
            WebkitTextFillColor: "transparent",
            backgroundClip: "text" as const,
            maxWidth: 630,
            marginLeft: "auto",
            marginRight: "auto",
            textAlign: "center" as const,
            marginBottom: 40,
          }}
        >
          Context+ guarantees minimal context bloat. It gives your agent deep
          semantic understanding of your codebase, from AST parsing and symbol
          navigation to blast radius analysis and commit validation. Nothing
          misses the context, with RAG.
        </p>
        <table
          style={{
            width: "100%",
            borderCollapse: "collapse",
            borderSpacing: 0,
          }}
        >
          <tbody>
            {toolRefRows.map(({ name, desc, input, output }) => (
              <tr key={name} style={{ verticalAlign: "top" }}>
                <td
                  style={{
                    padding: "24px 32px 24px 0",
                    borderBottom: "1px solid var(--table-border)",
                    whiteSpace: "nowrap",
                  }}
                >
                  <code
                    style={{
                      fontFamily: "var(--font-geist-pixel-square)",
                      fontSize: 14,
                      fontWeight: 500,
                      color: "var(--text-primary)",
                      letterSpacing: "-0.02em",
                    }}
                  >
                    {name}
                  </code>
                </td>
                <td
                  style={{
                    padding: "24px 0",
                    borderBottom: "1px solid var(--table-border)",
                  }}
                >
                  <p
                    style={{
                      fontSize: 14,
                      fontWeight: 300,
                      lineHeight: "22px",
                      color: "var(--text-body)",
                      marginBottom: 16,
                      fontFamily: "var(--font-geist-sans)",
                    }}
                  >
                    {desc}
                  </p>
                  <div
                    className="code-pair"
                    style={{ display: "flex", gap: 12, width: "100%" }}
                  >
                    <pre
                      style={{
                        flex: 1,
                        fontFamily: "var(--font-geist-mono)",
                        fontSize: 12,
                        fontWeight: 300,
                        lineHeight: "18px",
                        color: "var(--text-code)",
                        background: "var(--code-bg)",
                        backdropFilter: "blur(8px)",
                        WebkitBackdropFilter: "blur(8px)",
                        borderRadius: 8,
                        padding: "12px 16px",
                        overflow: "auto",
                        whiteSpace: "pre-wrap",
                        wordBreak: "break-word",
                        margin: 0,
                      }}
                    >
                      <span
                        style={{
                          fontSize: 10,
                          fontWeight: 500,
                          color: "var(--text-faint)",
                          display: "block",
                          marginBottom: 6,
                          fontFamily: "var(--font-geist-mono)",
                        }}
                      >
                        INPUT
                      </span>
                      {input}
                    </pre>
                    <pre
                      style={{
                        flex: 2,
                        fontFamily: "var(--font-geist-mono)",
                        fontSize: 12,
                        fontWeight: 300,
                        lineHeight: "18px",
                        color: "var(--text-code)",
                        background: "var(--code-bg)",
                        backdropFilter: "blur(8px)",
                        WebkitBackdropFilter: "blur(8px)",
                        borderRadius: 8,
                        padding: "12px 16px",
                        overflow: "auto",
                        whiteSpace: "pre-wrap",
                        wordBreak: "break-word",
                        margin: 0,
                      }}
                    >
                      <span
                        style={{
                          fontSize: 10,
                          fontWeight: 500,
                          color: "var(--text-faint)",
                          display: "block",
                          marginBottom: 6,
                          fontFamily: "var(--font-geist-mono)",
                        }}
                      >
                        OUTPUT
                      </span>
                      {output}
                    </pre>
                  </div>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </section>

      <section
        className="quote-section"
        style={{
          position: "relative",
          zIndex: 1,
          minHeight: "100vh",
          display: "flex",
          flexDirection: "column",
          alignItems: "center",
          justifyContent: "center",
          padding: "80px 100px",
          textShadow: "0 4px 12px var(--quote-shadow)",
        }}
      >
        <p
          style={{
            fontFamily: "var(--font-geist-sans)",
            fontSize: 48,
            fontWeight: 300,
            lineHeight: "68px",
            letterSpacing: "-0.05em",
            textAlign: "center",
            maxWidth: 900,
            background:
              "linear-gradient(180deg, var(--text-primary) 0%, var(--gradient-end) 100%)",
            WebkitBackgroundClip: "text",
            WebkitTextFillColor: "transparent",
            backgroundClip: "text" as const,
          }}
        >
          &ldquo;Context engineering is the delicate art and science of filling
          the context window with just the right information for the next
          step.&rdquo;
        </p>
        <a
          href="https://x.com/karpathy/status/1937902205765607626"
          target="_blank"
          rel="noopener noreferrer"
          style={{
            marginTop: 24,
            fontSize: 16,
            fontWeight: 300,
            color: "var(--text-secondary)",
            textDecoration: "none",
            fontFamily: "var(--font-geist-pixel-square)",
            letterSpacing: "-0.02em",
          }}
        >
          - Andrej Karpathy
        </a>
      </section>

      <footer
        className="site-footer"
        style={{
          position: "relative",
          zIndex: 1,
          padding: "80px 100px",
          display: "flex",
          justifyContent: "space-between",
          alignItems: "center",
          borderTop: "1.5px solid var(--footer-border)",
          background: "var(--nav-bg)",
          backdropFilter: "blur(12px)",
          WebkitBackdropFilter: "blur(12px)",
        }}
      >
        <span
          className="font-light"
          style={{
            fontSize: 22,
            lineHeight: "28px",
            color: "var(--text-primary)",
          }}
        >
          Context+
        </span>
        <div style={{ display: "flex", gap: 16, alignItems: "center" }}>
          <a
            href="https://www.npmjs.com/package/contextplus"
            target="_blank"
            rel="noopener noreferrer"
            className="flex items-center"
          >
            <svg
              width="20"
              height="20"
              viewBox="0 0 256 256"
              fill="var(--icon-color)"
            >
              <path d="M0 256V0h256v256H0zm41-41h57.5V71.2H141V215h34V41H41v174z" />
            </svg>
          </a>
          <a
            href="https://github.com/ForLoopCodes/contextplus"
            target="_blank"
            rel="noopener noreferrer"
            className="flex items-center"
            style={{ gap: 8 }}
          >
            <svg
              width="18"
              height="18"
              viewBox="0 0 24 24"
              fill="none"
              stroke="var(--icon-color)"
              strokeWidth="2"
              strokeLinecap="round"
              strokeLinejoin="round"
            >
              <path d="M9 19c-5 1.5-5-2.5-7-3m14 6v-3.87a3.37 3.37 0 0 0-.94-2.61c3.14-.35 6.44-1.54 6.44-7A5.44 5.44 0 0 0 20 4.77 5.07 5.07 0 0 0 19.91 1S18.73.65 16 2.48a13.38 13.38 0 0 0-7 0C6.27.65 5.09 1 5.09 1A5.07 5.07 0 0 0 5 4.77a5.44 5.44 0 0 0-1.5 3.78c0 5.42 3.3 6.61 6.44 7A3.37 3.37 0 0 0 9 18.13V22" />
            </svg>
            <span
              className="font-light"
              style={{
                fontSize: 18,
                lineHeight: "24px",
                color: "var(--text-primary)",
              }}
            >
              {stars}
            </span>
          </a>
        </div>
      </footer>
    </div>
  );
}
