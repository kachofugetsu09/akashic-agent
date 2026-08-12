import assert from "node:assert/strict";
import { mkdirSync, mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { resolve } from "node:path";
import test from "node:test";

import { collectBuildMetrics, compareBuildMetrics, createBuildBaseline } from "./build-metrics.mjs";

test("collectBuildMetrics separates initial assets from lazy chunks", () => {
  const directory = mkdtempSync(resolve(tmpdir(), "akashic-build-metrics-test-"));
  try {
    mkdirSync(resolve(directory, "nested"));
    writeFileSync(resolve(directory, "index.html"), '<link rel="stylesheet" href="/assets/main.css"><script type="module" src="/assets/main.js"></script>');
    writeFileSync(resolve(directory, "main.js"), "export const main = true;\n");
    writeFileSync(resolve(directory, "main.css"), ".main { color: black; }\n");
    writeFileSync(resolve(directory, "nested/lazy.js"), `export const lazy = "${"deterministic-lazy-chunk-".repeat(20)}";\n`);

    const metrics = collectBuildMetrics(directory, "index.html");

    assert.equal(metrics.initialJavaScript.fileCount, 1);
    assert.equal(metrics.initialStylesheets.fileCount, 1);
    assert.equal(metrics.artifacts.javascript.fileCount, 2);
    assert.equal(metrics.artifacts.fileCount, 4);
    assert.equal(metrics.artifacts.largestJavaScript[0].file, "nested/lazy.js");
  } finally {
    rmSync(directory, { recursive: true, force: true });
  }
});

test("createBuildBaseline gives an explicit five-percent byte budget", () => {
  const target = {
    initialJavaScript: { gzipBytes: 10_000 },
    initialStylesheets: { gzipBytes: 2_000 },
    artifacts: {
      rawBytes: 100_000,
      fileCount: 10,
      javascript: { gzipBytes: 30_000 },
    },
  };
  const baseline = createBuildBaseline({ sourceCommit: "abc", toolchain: {}, targets: { desktop: target } });
  const budget = baseline.build.budgets.desktop;

  assert.equal(budget.initialJavaScriptGzipBytes, 11_264);
  assert.equal(budget.fileCount, 13);
  assert.equal(baseline.browser.status, "unmeasured");
});

test("compareBuildMetrics reports every exceeded budget", () => {
  const baseline = {
    build: {
      budgets: {
        desktop: {
          initialJavaScriptGzipBytes: 10,
          initialStylesheetsGzipBytes: 10,
          javascriptGzipBytes: 10,
          artifactRawBytes: 10,
          fileCount: 1,
        },
      },
    },
  };
  const current = {
    desktop: {
      initialJavaScript: { gzipBytes: 11 },
      initialStylesheets: { gzipBytes: 9 },
      artifacts: { javascript: { gzipBytes: 12 }, rawBytes: 10, fileCount: 2 },
    },
  };

  const failures = compareBuildMetrics(current, baseline).filter((check) => !check.passed);

  assert.deepEqual(failures.map((check) => check.metric), [
    "initialJavaScript.gzipBytes",
    "artifacts.javascript.gzipBytes",
    "artifacts.fileCount",
  ]);
});
