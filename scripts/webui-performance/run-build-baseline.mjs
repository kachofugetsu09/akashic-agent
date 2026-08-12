import { mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { dirname, resolve } from "node:path";
import { spawnSync } from "node:child_process";
import { fileURLToPath } from "node:url";

import {
  collectBuildMetrics,
  compareBuildMetrics,
  createBuildBaseline,
} from "./build-metrics.mjs";

const here = dirname(fileURLToPath(import.meta.url));
const repoRoot = resolve(here, "..", "..");
const baselinePath = resolve(here, "baseline.json");
const updateBaseline = process.argv.includes("--update-baseline");
const buildRoot = mkdtempSync(resolve(tmpdir(), "akashic-webui-build-"));

try {
  const targets = {
    desktop: buildTarget("frontend/chat/vite.config.ts", "index.html", resolve(buildRoot, "desktop")),
    mobile: buildTarget("frontend/chat/vite.mobile.config.ts", "mobile.html", resolve(buildRoot, "mobile")),
  };
  const toolchain = readToolchain();
  if (updateBaseline) {
    const previous = readBaselineIfPresent();
    const next = createBuildBaseline({ sourceCommit: gitCommit(), toolchain, targets });
    if (previous?.browser?.status === "measured") next.browser = previous.browser;
    writeFileSync(baselinePath, `${JSON.stringify(next, null, 2)}\n`);
    console.log(`已更新构建性能基线: ${baselinePath}`);
    printSummary(targets);
  } else {
    const baseline = JSON.parse(readFileSync(baselinePath, "utf8"));
    const checks = compareBuildMetrics(targets, baseline);
    printSummary(targets);
    for (const check of checks) {
      console.log(`${check.passed ? "PASS" : "FAIL"} ${check.target}.${check.metric}: ${check.actual} <= ${check.maximum}`);
    }
    if (checks.some((check) => !check.passed)) process.exitCode = 1;
  }
} finally {
  rmSync(buildRoot, { recursive: true, force: true });
}

function buildTarget(config, entrypoint, outputDirectory) {
  const vite = resolve(repoRoot, "node_modules/vite/bin/vite.js");
  const result = spawnSync(process.execPath, [vite, "build", "--config", config, "--outDir", outputDirectory, "--emptyOutDir"], {
    cwd: repoRoot,
    encoding: "utf8",
    stdio: ["ignore", "pipe", "pipe"],
  });
  if (result.status !== 0) {
    process.stderr.write(result.stdout);
    process.stderr.write(result.stderr);
    throw new Error(`${config} 构建失败`);
  }
  return collectBuildMetrics(outputDirectory, entrypoint);
}

function readToolchain() {
  const packageJson = JSON.parse(readFileSync(resolve(repoRoot, "package.json"), "utf8"));
  return {
    node: process.version,
    react: packageJson.dependencies.react,
    vite: packageJson.devDependencies.vite,
  };
}

function gitCommit() {
  const result = spawnSync("git", ["rev-parse", "HEAD"], { cwd: repoRoot, encoding: "utf8" });
  if (result.status !== 0) throw new Error("无法读取 Git commit");
  return result.stdout.trim();
}

function readBaselineIfPresent() {
  try {
    return JSON.parse(readFileSync(baselinePath, "utf8"));
  } catch (error) {
    if (error?.code === "ENOENT") return null;
    throw error;
  }
}

function printSummary(targets) {
  for (const [name, target] of Object.entries(targets)) {
    console.log(`${name}: initial-js=${target.initialJavaScript.gzipBytes}B gzip, initial-css=${target.initialStylesheets.gzipBytes}B gzip, artifacts=${target.artifacts.rawBytes}B/${target.artifacts.fileCount} files`);
  }
}
