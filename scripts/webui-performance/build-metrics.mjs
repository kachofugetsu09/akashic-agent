import { gzipSync } from "node:zlib";
import { readFileSync, readdirSync, statSync } from "node:fs";
import { relative, resolve, sep } from "node:path";

const SCRIPT_PATTERN = /<script\b[^>]*\bsrc=["']([^"']+\.js)["'][^>]*>/giu;
const STYLESHEET_PATTERN = /<link\b(?=[^>]*\brel=["']stylesheet["'])(?=[^>]*\bhref=["']([^"']+\.css)["'])[^>]*>/giu;

export function collectBuildMetrics(outputDirectory, entrypoint) {
  const files = listFiles(outputDirectory);
  const entryHtml = readFileSync(resolve(outputDirectory, entrypoint), "utf8");
  const entryJavaScript = referencedAssets(entryHtml, SCRIPT_PATTERN);
  const entryStylesheets = referencedAssets(entryHtml, STYLESHEET_PATTERN);
  const records = files.map((file) => assetRecord(outputDirectory, file));

  return {
    entrypoint,
    initialJavaScript: summarizeSelected(records, entryJavaScript),
    initialStylesheets: summarizeSelected(records, entryStylesheets),
    artifacts: summarizeAssets(records),
  };
}

export function compareBuildMetrics(current, baseline) {
  const checks = [];
  for (const target of Object.keys(baseline.build.budgets).sort()) {
    const metrics = current[target];
    const budgets = baseline.build.budgets[target];
    if (!metrics) throw new Error(`缺少构建目标: ${target}`);
    checks.push(checkBudget(target, "initialJavaScript.gzipBytes", metrics.initialJavaScript.gzipBytes, budgets.initialJavaScriptGzipBytes));
    checks.push(checkBudget(target, "initialStylesheets.gzipBytes", metrics.initialStylesheets.gzipBytes, budgets.initialStylesheetsGzipBytes));
    checks.push(checkBudget(target, "artifacts.javascript.gzipBytes", metrics.artifacts.javascript.gzipBytes, budgets.javascriptGzipBytes));
    checks.push(checkBudget(target, "artifacts.rawBytes", metrics.artifacts.rawBytes, budgets.artifactRawBytes));
    checks.push(checkBudget(target, "artifacts.fileCount", metrics.artifacts.fileCount, budgets.fileCount));
  }
  return checks;
}

export function createBuildBaseline({ sourceCommit, toolchain, targets }) {
  const budgets = Object.fromEntries(Object.entries(targets).map(([name, metrics]) => [name, {
    initialJavaScriptGzipBytes: byteBudget(metrics.initialJavaScript.gzipBytes, 0.05),
    initialStylesheetsGzipBytes: byteBudget(metrics.initialStylesheets.gzipBytes, 0.05),
    javascriptGzipBytes: byteBudget(metrics.artifacts.javascript.gzipBytes, 0.05),
    artifactRawBytes: byteBudget(metrics.artifacts.rawBytes, 0.05),
    fileCount: metrics.artifacts.fileCount + Math.max(3, Math.ceil(metrics.artifacts.fileCount * 0.02)),
  }]));
  return {
    schemaVersion: 1,
    sourceCommit,
    toolchain,
    build: {
      reference: targets,
      budgets,
    },
    browser: {
      status: "unmeasured",
      sampling: {
        runs: 5,
        aggregate: "median",
        tail: "p75",
      },
      reference: null,
      budgets: null,
    },
  };
}

function listFiles(root) {
  const files = [];
  const visit = (directory) => {
    for (const entry of readdirSync(directory, { withFileTypes: true })) {
      const path = resolve(directory, entry.name);
      if (entry.isDirectory()) visit(path);
      else if (entry.isFile()) files.push(path);
    }
  };
  visit(root);
  return files.sort();
}

function referencedAssets(html, pattern) {
  return new Set([...html.matchAll(pattern)].map((match) => {
    const pathname = new URL(match[1], "https://perf.invalid").pathname;
    return pathname.split("/").filter(Boolean).at(-1);
  }));
}

function assetRecord(root, file) {
  const bytes = readFileSync(file);
  return {
    file: relative(root, file).split(sep).join("/"),
    rawBytes: statSync(file).size,
    gzipBytes: gzipSync(bytes, { level: 9 }).byteLength,
  };
}

function summarizeSelected(records, names) {
  const selected = records.filter((record) => names.has(record.file.split("/").at(-1)));
  if (selected.length !== names.size) throw new Error("入口 HTML 引用了不存在的构建资源");
  return summarizeRecords(selected);
}

function summarizeAssets(records) {
  const javascript = records.filter((record) => record.file.endsWith(".js"));
  const stylesheets = records.filter((record) => record.file.endsWith(".css"));
  return {
    ...summarizeRecords(records),
    javascript: summarizeRecords(javascript),
    stylesheets: summarizeRecords(stylesheets),
    largestJavaScript: javascript
      .toSorted((left, right) => right.gzipBytes - left.gzipBytes)
      .slice(0, 8),
  };
}

function summarizeRecords(records) {
  return {
    fileCount: records.length,
    rawBytes: records.reduce((sum, record) => sum + record.rawBytes, 0),
    gzipBytes: records.reduce((sum, record) => sum + record.gzipBytes, 0),
  };
}

function byteBudget(value, growth) {
  return Math.ceil((value * (1 + growth)) / 1024) * 1024;
}

function checkBudget(target, metric, actual, maximum) {
  return {
    target,
    metric,
    actual,
    maximum,
    passed: actual <= maximum,
    delta: actual - maximum,
  };
}
