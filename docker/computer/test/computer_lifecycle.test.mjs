import test from "node:test";
import assert from "node:assert/strict";
import { execFile } from "node:child_process";
import { chmod, mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { promisify } from "node:util";
import { ComputerLifecycle } from "../lifecycle.mjs";

const exec = promisify(execFile);

async function waitForFile(path, timeoutMs = 3000) {
  const deadline = Date.now() + timeoutMs;
  while (Date.now() < deadline) {
    try {
      return await readFile(path, "utf8");
    } catch {
      await new Promise((r) => setTimeout(r, 20));
    }
  }
  throw new Error(`Timed out waiting for ${path}`);
}

test("gateway stream token uses hex without leading hyphen", async () => {
  const gatewaySource = await readFile(new URL("../gateway.mjs", import.meta.url), "utf8");
  assert.match(gatewaySource, /COMPUTER_STREAM_TOKEN\s*\|\|=\s*randomBytes\(24\)\.toString\("hex"\)/);
  assert.doesNotMatch(gatewaySource, /COMPUTER_STREAM_TOKEN\s*\|\|=\s*randomBytes\(24\)\.toString\("base64url"\)/);
});

test("start.sh binds master-token with equals and cleans Singleton locks in --runtime", async () => {
  const startSource = await readFile(new URL("../start.sh", import.meta.url), "utf8");
  assert.match(startSource, /--master-token="\$COMPUTER_STREAM_TOKEN"/);
  const runtimeSection = startSource.split('if [ "${1:-}" != "--runtime" ]; then')[1];
  assert.match(runtimeSection, /\/data\/profile\/SingletonCookie/);
  assert.match(runtimeSection, /\/data\/profile\/SingletonLock/);
  assert.match(runtimeSection, /\/data\/profile\/SingletonSocket/);
});

test("argparse accepts leading-hyphen token when passed via --master-token=value", async () => {
  const { stdout } = await exec("python3", [
    "-c",
    "import argparse, sys; p = argparse.ArgumentParser(); p.add_argument('--master-token'); print(p.parse_args(sys.argv[1:]).master_token)",
    "--master-token=-leading-hyphen-token",
  ]);
  assert.equal(stdout.trim(), "-leading-hyphen-token");
});

test("ComputerLifecycle startup rollback sends SIGTERM and returns to sleeping without locking failed", async () => {
  const dir = await mkdtemp(join(tmpdir(), "computer-lifecycle-term-"));
  const runtimeScript = join(dir, "runtime.sh");
  const ready = join(dir, "ready");
  const marker = join(dir, "cleaned");
  await writeFile(
    runtimeScript,
    `#!/bin/sh
on_term() {
  echo "term" > "${marker}"
  exit 0
}
trap on_term TERM INT
echo "ok" > "${ready}"
while true; do sleep 0.1; done
`,
  );
  await chmod(runtimeScript, 0o755);

  const lifecycle = new ComputerLifecycle({
    start: async () => {
      await waitForFile(ready);
      throw new Error("Computer stream returned 503");
    },
    stop: async () => {},
    abandon: async () => {},
    idleMs: 60000,
    runtimeCommand: runtimeScript,
    runtimeArgs: [],
    stopWaitMs: 1000,
    killWaitMs: 1000,
  });

  try {
    await assert.rejects(lifecycle.wake(), /Computer stream returned 503/);
    assert.equal(lifecycle.state, "sleeping");
    assert.equal((await readFile(marker, "utf8")).trim(), "term");
  } finally {
    clearInterval(lifecycle.timer);
    await rm(dir, { recursive: true, force: true });
  }
});

test("ComputerLifecycle SIGKILL path drains child process group before failing", async () => {
  const dir = await mkdtemp(join(tmpdir(), "computer-lifecycle-kill-"));
  const runtimeScript = join(dir, "stubborn.sh");
  const ready = join(dir, "ready");
  await writeFile(
    runtimeScript,
    `#!/bin/sh
trap '' TERM
echo "ok" > "${ready}"
sleep 30 &
wait
`,
  );
  await chmod(runtimeScript, 0o755);

  const lifecycle = new ComputerLifecycle({
    start: async () => {
      await waitForFile(ready);
      throw new Error("health timeout");
    },
    stop: async () => {},
    abandon: async () => {},
    idleMs: 60000,
    runtimeCommand: runtimeScript,
    runtimeArgs: [],
    stopWaitMs: 250,
    killWaitMs: 2000,
  });

  try {
    await assert.rejects(
      lifecycle.wake(),
      /Computer startup failed: health timeout; cleanup failed: Computer runtime required SIGKILL/,
    );
    assert.equal(lifecycle.state, "failed");
    assert.equal(lifecycle.child, null);
  } finally {
    clearInterval(lifecycle.timer);
    await rm(dir, { recursive: true, force: true });
  }
});
