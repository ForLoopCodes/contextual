"use client";

import { useRef, useState, useEffect, useCallback } from "react";

function useTheme() {
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

  return isDark;
}

const functions = [
  {
    id: "context-tree",
    label: "Context Tree",
    desc: "Hierarchical feature graph of the codebase",
    color: "#000000",
  },
  {
    id: "file-skeleton",
    label: "File Skeleton",
    desc: "Structural outline of any source file",
    color: "#1a1a1a",
  },
  {
    id: "semantic-search",
    label: "Semantic Search",
    desc: "File-level semantic search with definition lines",
    color: "#333333",
  },
  {
    id: "identifier-search",
    label: "Identifier Search",
    desc: "Function and variable search with ranked call chains",
    color: "#3a3a3a",
  },
  {
    id: "semantic-navigate",
    label: "Semantic Navigate",
    desc: "Jump to related code by meaning",
    color: "#444444",
  },
  {
    id: "blast-radius",
    label: "Blast Radius",
    desc: "Impact analysis for code changes",
    color: "#555555",
  },
  {
    id: "static-analysis",
    label: "Static Analysis",
    desc: "AST-powered code inspection",
    color: "#666666",
  },
  {
    id: "feature-hub",
    label: "Feature Hub",
    desc: "Cluster features by semantic similarity",
    color: "#777777",
  },
  {
    id: "propose-commit",
    label: "Propose Commit",
    desc: "AI-generated commit messages",
    color: "#888888",
  },
  {
    id: "git-shadow",
    label: "Git Shadow",
    desc: "Lightweight git worktree operations",
    color: "#999999",
  },
  {
    id: "memory-search",
    label: "Memory Search",
    desc: "RAG-powered search with graph traversal",
    color: "#2a4a6a",
  },
  {
    id: "memory-upsert",
    label: "Memory Upsert",
    desc: "Persist learned context as graph nodes",
    color: "#3a5a7a",
  },
  {
    id: "memory-traverse",
    label: "Graph Traverse",
    desc: "Walk linked nodes by relation and decay",
    color: "#4a6a8a",
  },
];

function getCardConfig(width: number) {
  if (width >= 1800) return { cardSize: 360, stackDx: -24, stackDy: 24 };
  if (width >= 1400) return { cardSize: 300, stackDx: -21, stackDy: 21 };
  if (width >= 1025) return { cardSize: 250, stackDx: -18, stackDy: 18 };
  if (width >= 850) return { cardSize: 220, stackDx: -16, stackDy: 16 };
  if (width >= 500) return { cardSize: 240, stackDx: -16, stackDy: 16 };
  return { cardSize: 220, stackDx: -14, stackDy: 14 };
}

export default function IsometricDiagram() {
  const containerRef = useRef<HTMLDivElement>(null);
  const [hoveredIdx, setHoveredIdx] = useState<number | null>(null);
  const [order, setOrder] = useState(() =>
    functions.map((_: unknown, i: number) => i),
  );
  const [animPhase, setAnimPhase] = useState<"fade-out" | "fade-in" | null>(
    null,
  );
  const [animating, setAnimating] = useState(false);
  const [animCardId, setAnimCardId] = useState<number | null>(null);
  const [windowWidth, setWindowWidth] = useState(1200);
  const isDark = useTheme();

  useEffect(() => {
    const update = () => setWindowWidth(window.innerWidth);
    update();
    window.addEventListener("resize", update);
    return () => window.removeEventListener("resize", update);
  }, []);

  const {
    cardSize: CARD_SIZE,
    stackDx: STACK_DX,
    stackDy: STACK_DY,
  } = getCardConfig(windowWidth);
  const isMobile = windowWidth < 500;

  const cycleTopCard = useCallback(() => {
    if (animating) return;
    const topGroupIdx = order[0];
    setAnimating(true);
    setHoveredIdx(null);
    setAnimCardId(topGroupIdx);

    setAnimPhase("fade-out");

    setTimeout(() => {
      setOrder((prev) => {
        const [top, ...rest] = prev;
        return [...rest, top];
      });
      setAnimPhase("fade-in");

      setTimeout(() => {
        setAnimPhase(null);
        setAnimating(false);
        setAnimCardId(null);
      }, 400);
    }, 400);
  }, [animating, order]);

  const handleClick = (_visualIdx: number) => {
    cycleTopCard();
  };

  const isPaused = hoveredIdx !== null || animating;
  useEffect(() => {
    if (isPaused) return;
    const timer = setInterval(() => {
      cycleTopCard();
    }, 3000);
    return () => clearInterval(timer);
  }, [isPaused, cycleTopCard]);

  const maxDx = Math.abs(STACK_DX) * (functions.length - 1);
  const maxDy = Math.abs(STACK_DY) * (functions.length - 1);
  const stackW = CARD_SIZE + maxDx;
  const stackH = CARD_SIZE + maxDy;

  const isLargeDesktop = windowWidth >= 1400 && windowWidth < 1800;
  const isSmallDesktop = windowWidth >= 850 && windowWidth < 1400;
  const isVerySmallDesktop = windowWidth >= 850 && windowWidth < 1250;
  const isLargeMobile = windowWidth >= 500 && windowWidth < 850;

  return (
    <div
      className="iso-diagram"
      ref={containerRef}
      style={{
        position: "relative",
        zIndex: 1,
        marginTop: isMobile
          ? -50
          : isLargeMobile
            ? -100
            : isVerySmallDesktop
              ? -120
              : isSmallDesktop
                ? -120
                : isLargeDesktop
                  ? -400
                  : -600,
        display: "flex",
        alignItems: isMobile ? "center" : "flex-end",
        aspectRatio: isMobile ? "1 / 1" : undefined,
        justifyContent: isMobile ? "center" : "flex-end",
        marginBottom: `${isMobile ? 20 : isVerySmallDesktop ? 60 : isSmallDesktop ? 120 : 300}px !important`,
        padding: isMobile ? "0 20px 30px" : "0 100px 40px 100px",
        overflow: isMobile ? "hidden" : undefined,
      }}
    >
      <div
        style={{
          perspective: 2200,
          flexShrink: 0,
        }}
      >
        <div
          style={{
            position: "relative",
            width: stackH,
            height: stackH,
            transformStyle: "preserve-3d",
            transform: "rotateX(45deg) rotateZ(-45deg)",
          }}
        >
          {order.map((groupIdx, visualIdx) => {
            const g = functions[groupIdx];
            const isHovered = hoveredIdx === visualIdx;
            const isAnimCard = groupIdx === animCardId;

            const xPos = maxDx + visualIdx * STACK_DX;
            const yPos = visualIdx * STACK_DY;

            const t = visualIdx / (functions.length - 1);
            const grayLight = Math.round(t * 210);
            const grayDark = Math.round(255 - t * 180);
            const gray = isDark ? grayDark : grayLight;
            const borderColor = `rgb(${gray},${gray},${gray})`;

            const cardBg = isDark
              ? "rgba(20,20,20,0.85)"
              : "rgba(239,239,239,0.8)";
            const labelBg = isDark
              ? "rgba(20,20,20,0.8)"
              : "rgba(239,239,239,0.7)";
            const cardShadow = isHovered
              ? isDark
                ? "0 16px 40px rgba(0,0,0,0.4)"
                : "0 16px 40px rgba(0,0,0,0.18)"
              : isDark
                ? "0 2px 8px rgba(0,0,0,0.2)"
                : "0 2px 8px rgba(0,0,0,0.04)";

            const isFadingOut = animPhase === "fade-out" && isAnimCard;
            const isFadingIn = animPhase === "fade-in" && isAnimCard;

            return (
              <div
                key={g.id}
                onMouseEnter={() =>
                  visualIdx === 0 && !animating ? setHoveredIdx(0) : undefined
                }
                onMouseLeave={() => setHoveredIdx(null)}
                onClick={() =>
                  visualIdx === 0 && !animating ? handleClick(0) : undefined
                }
                style={{
                  position: "absolute",
                  left: xPos,
                  top: yPos,
                  width: CARD_SIZE,
                  height: CARD_SIZE,
                  borderRadius: 18,
                  border: `1.5px solid ${borderColor}`,
                  overflow: "visible",
                  cursor: visualIdx === 0 && !animating ? "pointer" : "default",
                  pointerEvents:
                    visualIdx === 0 && !animating ? "auto" : "none",
                  transition:
                    "top 0.4s cubic-bezier(0.4,0,0.2,1), left 0.4s cubic-bezier(0.4,0,0.2,1), box-shadow 0.25s ease, transform 0.3s ease, opacity 0.4s ease, filter 0.4s ease, border-color 0.4s ease, width 0.3s ease, height 0.3s ease",
                  transform: isFadingOut
                    ? "translateZ(60px) translateY(-40px)"
                    : isHovered
                      ? "translateZ(30px)"
                      : "translateZ(0)",
                  opacity: isFadingOut ? 0 : isFadingIn ? 0.5 : 1,
                  filter: isFadingOut ? "blur(8px)" : "blur(0px)",
                  boxShadow: cardShadow,
                  background: cardBg,
                  backdropFilter: "blur(6px)",
                  WebkitBackdropFilter: "blur(6px)",
                  zIndex: isHovered ? 10 : functions.length - visualIdx,
                }}
              >
                <svg
                  style={{
                    position: "absolute",
                    inset: 0,
                    width: "100%",
                    height: "100%",
                    borderRadius: 18,
                    overflow: "hidden",
                  }}
                  xmlns="http://www.w3.org/2000/svg"
                >
                  <defs>
                    <pattern
                      id={`iso-diag-${g.id}`}
                      width="6"
                      height="6"
                      patternUnits="userSpaceOnUse"
                      patternTransform="rotate(0)"
                    >
                      <line
                        x1="0"
                        y1="0"
                        x2="0"
                        y2="6"
                        stroke={borderColor}
                        strokeWidth="1.5"
                        strokeOpacity={isHovered ? "0.5" : "0.25"}
                      />
                    </pattern>
                  </defs>
                  <rect
                    width="100%"
                    height="100%"
                    fill={`url(#iso-diag-${g.id})`}
                  />
                </svg>
                <span
                  style={{
                    position: "absolute",
                    left: 10,
                    bottom: 0,
                    fontSize: isMobile ? 9 : 11,
                    fontWeight: 300,
                    color: borderColor,
                    fontFamily: "var(--font-geist-mono)",
                    letterSpacing: "-0.01em",
                    whiteSpace: "nowrap",
                    pointerEvents: "none",
                    background: labelBg,
                    backdropFilter: "blur(4px)",
                    WebkitBackdropFilter: "blur(4px)",
                    padding: "2px 6px",
                    borderRadius: 8,
                    transition: "color 0.4s ease, opacity 0.4s ease",
                  }}
                >
                  {g.label}
                </span>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
}
