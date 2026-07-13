// Dependency graph analyzer to trace symbol usage across the codebase
// Finds every file and line where a function, class, or variable is referenced

import { walkRoots } from "../core/walker.js";
import { isSupportedFile } from "../core/parser.js";
import { readFile } from "fs/promises";

export interface BlastRadiusOptions {
  rootDir: string;
  symbolName: string;
  fileContext?: string;
}

interface SymbolUsage {
  file: string;
  line: number;
  context: string;
}

export async function getBlastRadius(options: BlastRadiusOptions): Promise<string> {
  const entries = await walkRoots({ rootDir: options.rootDir, depthLimit: 0 });
  const files = entries.filter((e) => !e.isDirectory && isSupportedFile(e.path));
  const usages: SymbolUsage[] = [];
  const symbolPattern = new RegExp(`\\b${escapeRegex(options.symbolName)}\\b`, "g");

  for (const file of files) {
    try {
      const content = await readFile(file.path, "utf-8");
      const lines = content.split("\n");

      for (let i = 0; i < lines.length; i++) {
        if (symbolPattern.test(lines[i])) {
          const isDefinition = options.fileContext && file.relativePath === options.fileContext && isDefinitionLine(lines[i], options.symbolName);
          if (!isDefinition) {
            usages.push({
              file: file.relativePath,
              line: i + 1,
              context: lines[i].trim().substring(0, 120),
            });
          }
          symbolPattern.lastIndex = 0;
        }
      }
    } catch {
    }
  }

  if (usages.length === 0) return `Symbol "${options.symbolName}" is not used anywhere in the codebase.`;

  const byFile = new Map<string, SymbolUsage[]>();
  for (const u of usages) {
    const existing = byFile.get(u.file) ?? [];
    existing.push(u);
    byFile.set(u.file, existing);
  }

  const lines: string[] = [
    `Blast radius for "${options.symbolName}": ${usages.length} usages in ${byFile.size} files\n`,
  ];

  for (const [file, fileUsages] of byFile) {
    lines.push(`  ${file}:`);
    for (const u of fileUsages) {
      lines.push(`    L${u.line}: ${u.context}`);
    }
  }

  if (usages.length <= 1) {
    lines.push(`\n⚠ LOW USAGE: This symbol is used only ${usages.length} time(s). Consider inlining if it's under 20 lines.`);
  }

  return lines.join("\n");
}

function escapeRegex(str: string): string {
  return str.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

function isDefinitionLine(line: string, symbolName: string): boolean {
  const definitionPatterns = [
    new RegExp(`(?:function|class|enum|interface|struct|type|trait|fn|def|func)\\s+${escapeRegex(symbolName)}`),
    new RegExp(`(?:const|let|var|pub|export)\\s+(?:async\\s+)?(?:function\\s+)?${escapeRegex(symbolName)}`),
  ];
  return definitionPatterns.some((p) => p.test(line));
}
