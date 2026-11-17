import test from "node:test";
import assert from "node:assert/strict";
import { chmod, mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { ComputerLifecycle } from "../lifecycle.mjs";

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
