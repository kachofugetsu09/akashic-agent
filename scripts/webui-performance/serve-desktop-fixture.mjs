import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { dirname, resolve } from "node:path";
import { spawnSync } from "node:child_process";
import { fileURLToPath } from "node:url";

import { startDesktopFixtureServer } from "./desktop-fixture-server.mjs";
import { loadReplayTurn } from "./replay-turn.mjs";

const here = dirname(fileURLToPath(import.meta.url));
const repoRoot = resolve(here, "..", "..");
const buildRoot = mkdtempSync(resolve(tmpdir(), "akashic-webui-t3-"));
const output = resolve(buildRoot, "desktop");
const port = Number(process.env.AKASHIC_WEBUI_FIXTURE_PORT || "4173");
const historyCount = Number(process.env.AKASHIC_WEBUI_HISTORY_COUNT || "100");
const replayTurn = process.env.AKASHIC_WEBUI_REPLAY_TURN
  ? loadReplayTurn(resolve(process.env.AKASHIC_WEBUI_REPLAY_TURN))
  : null;
let fixture;

try {
  const vite = resolve(repoRoot, "node_modules/vite/bin/vite.js");
  const build = spawnSync(process.execPath, [
    vite,
    "build",
    "--config",
    "frontend/chat/vite.config.ts",
    "--outDir",
    output,
    "--emptyOutDir",
  ], { cwd: repoRoot, encoding: "utf8" });
  if (build.status !== 0) throw new Error(`desktop fixture build failed\n${build.stdout}\n${build.stderr}`);
  fixture = await startDesktopFixtureServer(output, { port, historyCount, replayTurn });
  console.log(JSON.stringify({ event: "webui.fixture_ready", origin: fixture.origin }));
  await waitForSignal();
} finally {
  await fixture?.close();
  rmSync(buildRoot, { recursive: true, force: true });
}

function waitForSignal() {
  return new Promise((resolveSignal) => {
    process.once("SIGINT", resolveSignal);
    process.once("SIGTERM", resolveSignal);
  });
}
