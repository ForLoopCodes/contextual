// Code commit gatekeeper enforcing 2-line headers, no inline comments
// Validates abstraction rules and creates shadow restore points before writing

import { writeFile, mkdir } from "fs/promises";
import { resolve, dirname, extname } from "path";
import { createRestorePoint } from "../git/shadow.js";
import { isSupportedFile } from "../core/parser.js";

export interface ProposeCommitOptions {
  rootDir: string;
  filePath: string;
  newContent: string;
}

interface ValidationError {
  rule: string;
  message: string;
  line?: number;
}

function validateHeader(lines: string[], ext: string): ValidationError[] {
  const errors: ValidationError[] = [];
  const commentPrefixes: Record<string, string> = {
    ".ts": "//", ".tsx": "//", ".js": "//", ".jsx": "//",
    ".rs": "//", ".go": "//", ".c": "//", ".cpp": "//",
    ".java": "//", ".cs": "//", ".swift": "//", ".kt": "//",
    ".py": "#", ".rb": "#", ".lua": "--", ".zig": "//",
  };

  const prefix = commentPrefixes[ext];
  if (!prefix) return errors;

  const headerLines = lines.slice(0, 5).filter((l) => l.startsWith(prefix));
  if (headerLines.length < 2) {
    errors.push({
      rule: "header",
      message: `Missing 2-line file header. First 2 lines must be ${prefix} comments explaining the file.`,
    });
  }

  if (headerLines.length >= 2 && !headerLines[1].toUpperCase().includes("FEATURE:")) {
    errors.push({
      rule: "feature-tag",
      message: `Line 2 should include a FEATURE: tag (e.g., "${prefix} FEATURE: Feature Name"). Links files to feature hubs.`,
    });
  }

  return errors;
}

function validateNoInlineComments(lines: string[], ext: string): ValidationError[] {
  const errors: ValidationError[] = [];
  const isScriptLang = [".py", ".rb"].includes(ext);
  const commentPrefix = isScriptLang ? "#" : "//";

  for (let i = 2; i < lines.length; i++) {
    const trimmed = lines[i].trim();
    if (trimmed.startsWith(commentPrefix) && !trimmed.startsWith("#!") && !trimmed.startsWith("#include")) {
      errors.push({
        rule: "no-comments",
        message: `Unauthorized comment found on line ${i + 1}: ${trimmed.substring(0, 80)}`,
        line: i + 1,
      });
    }
  }

  return errors;
}

function validateAbstraction(lines: string[]): ValidationError[] {
  const errors: ValidationError[] = [];
  let nestingDepth = 0;
  let maxNesting = 0;

  for (let i = 0; i < lines.length; i++) {
    const line = lines[i];
    nestingDepth += (line.match(/{/g) || []).length;
    nestingDepth -= (line.match(/}/g) || []).length;
    maxNesting = Math.max(maxNesting, nestingDepth);
  }

  if (maxNesting > 6) {
    errors.push({
      rule: "nesting",
      message: `Nesting depth of ${maxNesting} detected. Maximum recommended is 3-4 levels. Flatten the structure.`,
    });
  }

  if (lines.length > 1000) {
    errors.push({
      rule: "file-length",
      message: `File is ${lines.length} lines. Maximum recommended is 500-1000. Consider splitting.`,
    });
  }

  return errors;
}

export async function proposeCommit(options: ProposeCommitOptions): Promise<string> {
  const rootDir = resolve(options.rootDir);
  const fullPath = resolve(rootDir, options.filePath);

  if (fullPath !== rootDir && !fullPath.startsWith(rootDir + "/")) {
    throw new Error(`Path traversal detected: resolved path escapes project root`);
  }

  const ext = extname(fullPath);
  const lines = options.newContent.split("\n");
  const allErrors: ValidationError[] = [];

  if (isSupportedFile(fullPath)) {
    allErrors.push(...validateHeader(lines, ext));
    allErrors.push(...validateNoInlineComments(lines, ext));
  }
  allErrors.push(...validateAbstraction(lines));

  const commentErrors = allErrors.filter((e) => e.rule === "no-comments");
  if (commentErrors.length > 5) {
    return [
      `REJECTED: ${allErrors.length} violations found.\n`,
      ...allErrors.slice(0, 10).map((e) => `  ❌ [${e.rule}] ${e.message}`),
      allErrors.length > 10 ? `  ... and ${allErrors.length - 10} more violations` : "",
      "\nFix all violations and resubmit.",
    ].join("\n");
  }

  const warnings = allErrors.filter((e) => e.rule !== "no-comments" || commentErrors.length <= 5);

  await createRestorePoint(options.rootDir, [options.filePath], `Pre-commit: ${options.filePath}`);
  await mkdir(dirname(fullPath), { recursive: true });
  await writeFile(fullPath, options.newContent, "utf-8");

  const result = [`✅ File saved: ${options.filePath}`];
  if (warnings.length > 0) {
    result.push(`\n⚠ ${warnings.length} warning(s):`);
    for (const w of warnings) result.push(`  ⚠ [${w.rule}] ${w.message}`);
  }
  result.push(`\nRestore point created. Use undo tools if needed.`);

  return result.join("\n");
}
