"use client";

import { useEffect, useState } from "react";

const toolGroupsLight = [
  {
    name: "Discovery",
    color: "#000000",
    layout: "grid" as const,
    tools: [
      { color: "#000000", label: "Context Tree" },
      { color: "#111111", label: "File Skeleton" },
      { color: "#222222", label: "Semantic Search" },
      { color: "#333333", label: "Semantic Identifiers" },
    ],
  },
  {
    name: "Analysis",
    color: "#444444",
    layout: "column" as const,
    tools: [
      { color: "#444444", label: "Blast Radius" },
      { color: "#555555", label: "Static Analysis" },
    ],
  },
  {
    name: "Code Ops",
    color: "#666666",
    layout: "column" as const,
    tools: [
      { color: "#666666", label: "Propose Commit" },
      { color: "#777777", label: "Feature Hub" },
    ],
  },
  {
    name: "Version Control",
    color: "#888888",
    layout: "column" as const,
    tools: [
      { color: "#888888", label: "Restore Points" },
      { color: "#999999", label: "Undo Change" },
    ],
  },
];

const toolGroupsDark = [
  {
    name: "Discovery",
    color: "#ffffff",
    layout: "grid" as const,
    tools: [
      { color: "#ffffff", label: "Context Tree" },
      { color: "#eeeeee", label: "File Skeleton" },
      { color: "#dddddd", label: "Semantic Search" },
      { color: "#cccccc", label: "Semantic Identifiers" },
    ],
  },
  {
    name: "Analysis",
    color: "#bbbbbb",
    layout: "column" as const,
    tools: [
      { color: "#bbbbbb", label: "Blast Radius" },
      { color: "#aaaaaa", label: "Static Analysis" },
    ],
  },
  {
    name: "Code Ops",
    color: "#999999",
    layout: "column" as const,
    tools: [
      { color: "#999999", label: "Propose Commit" },
      { color: "#888888", label: "Feature Hub" },
    ],
  },
  {
    name: "Version Control",
    color: "#777777",
    layout: "column" as const,
    tools: [
      { color: "#777777", label: "Restore Points" },
      { color: "#666666", label: "Undo Change" },
    ],
  },
];

export default function ToolDiagram() {
  const [isDark, setIsDark] = useState(false);

  useEffect(() => {
    const checkTheme = () => {
      const theme = document.documentElement.getAttribute("data-theme");
      setIsDark(theme === "dark");
    };
    checkTheme();
    const observer = new MutationObserver(checkTheme);
    observer.observe(document.documentElement, {
      attributes: true,
      attributeFilter: ["data-theme"],
    });
    return () => observer.disconnect();
  }, []);

  const toolGroups = isDark ? toolGroupsDark : toolGroupsLight;

  return (
    <>
      <div
        style={{
          display: "flex",
          alignItems: "center",
          gap: 12,
          marginBottom: 16,
        }}
      >
        <div
          style={{
            flex: 1,
            height: 8,
            minWidth: 40,
            overflow: "hidden",
            position: "relative",
          }}
        >
          <svg
            style={{
              position: "absolute",
              inset: 0,
              width: "100%",
              height: "100%",
            }}
            xmlns="http://www.w3.org/2000/svg"
          >
            <defs>
              <pattern
                id="sep-left"
                width="6"
                height="6"
                patternUnits="userSpaceOnUse"
                patternTransform="rotate(45)"
              >
                <line
                  x1="0"
                  y1="0"
                  x2="0"
                  y2="6"
                  stroke={isDark ? "#666666" : "#333333"}
                  strokeWidth="1.5"
                />
              </pattern>
            </defs>
            <rect width="100%" height="100%" fill="url(#sep-left)" />
          </svg>
        </div>
        <span
          style={{
            fontSize: 13,
            fontWeight: 500,
            color: "var(--text-primary)",
            fontFamily: "var(--font-geist-pixel-square)",
            letterSpacing: "-0.02em",
            whiteSpace: "nowrap",
          }}
        >
          Context+ MCP
        </span>
        <div
          style={{
            flex: 1,
            height: 8,
            minWidth: 40,
            overflow: "hidden",
            position: "relative",
          }}
        >
          <svg
            style={{
              position: "absolute",
              inset: 0,
              width: "100%",
              height: "100%",
            }}
            xmlns="http://www.w3.org/2000/svg"
          >
            <defs>
              <pattern
                id="sep-right"
                width="6"
                height="6"
                patternUnits="userSpaceOnUse"
                patternTransform="rotate(45)"
              >
                <line
                  x1="0"
                  y1="0"
                  x2="0"
                  y2="6"
                  stroke={isDark ? "#888888" : "#888888"}
                  strokeWidth="1.5"
                />
              </pattern>
            </defs>
            <rect width="100%" height="100%" fill="url(#sep-right)" />
          </svg>
        </div>
      </div>
      <div
        className="diagram-groups"
        style={{ display: "flex", gap: 16, alignItems: "flex-start" }}
      >
        {toolGroups.map(({ name, color, layout, tools }) => (
          <div
            key={name}
            style={{ display: "flex", flexDirection: "column", gap: 8 }}
          >
            <span
              style={{
                fontSize: 13,
                fontWeight: 500,
                color,
                fontFamily: "var(--font-geist-pixel-square)",
                letterSpacing: "-0.02em",
              }}
            >
              {name}
            </span>
            <div
              className={
                layout === "grid" ? "discovery-grid" : "group-inner-col"
              }
              style={{
                display: layout === "grid" ? "grid" : "flex",
                ...(layout === "grid"
                  ? { gridTemplateColumns: "1fr 1fr" }
                  : { flexDirection: "column" as const }),
                gap: 10,
                border: "1.5px solid var(--panel-border)",
                backdropFilter: "blur(8px)",
                WebkitBackdropFilter: "blur(8px)",
                background: "var(--panel-bg)",
                borderRadius: 20,
                padding: 20,
              }}
            >
              {tools.map(({ color: toolColor, label }) => (
                <div
                  key={label}
                  style={{
                    display: "flex",
                    flexDirection: "column",
                    gap: 6,
                  }}
                >
                  <span
                    style={{
                      fontSize: 13,
                      fontWeight: 500,
                      color: toolColor,
                      fontFamily: "var(--font-geist-pixel-square)",
                      letterSpacing: "-0.02em",
                    }}
                  >
                    {label}
                  </span>
                  <div
                    className="tool-square"
                    style={{
                      boxSizing: "border-box",
                      width: 126,
                      height: 126,
                      border: `1.5px solid ${toolColor}`,
                      borderRadius: 14,
                      overflow: "hidden",
                      position: "relative",
                    }}
                  >
                    <svg
                      style={{
                        position: "absolute",
                        inset: 0,
                        width: "100%",
                        height: "100%",
                      }}
                      xmlns="http://www.w3.org/2000/svg"
                    >
                      <defs>
                        <pattern
                          id={`diag-${label.replace(/\s/g, "")}-${isDark ? "dark" : "light"}`}
                          width="6"
                          height="6"
                          patternUnits="userSpaceOnUse"
                          patternTransform="rotate(45)"
                        >
                          <line
                            x1="0"
                            y1="0"
                            x2="0"
                            y2="6"
                            stroke={toolColor}
                            strokeWidth="1.5"
                          />
                        </pattern>
                      </defs>
                      <rect
                        width="100%"
                        height="100%"
                        fill={`url(#diag-${label.replace(/\s/g, "")}-${isDark ? "dark" : "light"})`}
                      />
                    </svg>
                  </div>
                </div>
              ))}
            </div>
          </div>
        ))}
      </div>
      <div style={{ marginTop: 16, position: "relative" }}>
        <div
          style={{
            height: 100,
            borderRadius: 20,
            overflow: "hidden",
            position: "relative",
            background: "var(--panel-bg)",
          }}
        >
          <svg
            style={{ position: "absolute", inset: 0, width: "100%", height: "100%" }}
            xmlns="http://www.w3.org/2000/svg"
          >
            <defs>
              <pattern
                id={`diag-rag-bg-${isDark ? "dark" : "light"}`}
                width="6"
                height="6"
                patternUnits="userSpaceOnUse"
                patternTransform="rotate(45)"
              >
                <line
                  x1="0"
                  y1="0"
                  x2="0"
                  y2="6"
                  stroke={isDark ? "#000000" : "#000000"}
                  strokeWidth="1"
                />
              </pattern>
            </defs>
            <rect width="100%" height="100%" fill={`url(#diag-rag-bg-${isDark ? "dark" : "light"})`} />
          </svg>
          <span
            style={{
              position: "absolute",
              top: "50%",
              left: "50%",
              transform: "translate(-50%, -50%)",
              fontSize: 13,
              fontWeight: 500,
              color: isDark ? "#ffffff" : "#000000",
              fontFamily: "var(--font-geist-pixel-square)",
              letterSpacing: "-0.02em",
              backdropFilter: "blur(8px)",
              WebkitBackdropFilter: "blur(8px)",
              background: "var(--panel-bg)",
              borderRadius: 9999,
              padding: "2px 8px",
              border: "none",
            }}
          >
            RAG Functions
          </span>
        </div>
      </div>
    </>
  );
}
