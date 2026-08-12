import assert from "node:assert/strict";
import { mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { resolve } from "node:path";
import test from "node:test";

import WebSocket from "ws";

import { startDesktopFixtureServer } from "./desktop-fixture-server.mjs";

test("desktop fixture serves both profiles and a real WebSocket stream", async () => {
  const root = mkdtempSync(resolve(tmpdir(), "akashic-desktop-fixture-test-"));
  writeFileSync(resolve(root, "index.html"), "<!doctype html><title>fixture</title>");
  const fixture = await startDesktopFixtureServer(root);
  try {
    const sessions = await fetch(`${fixture.origin}/api/chat/sessions`).then((response) => response.json());
    assert.deepEqual(sessions.items.map((item) => item.first_message_content), [
      "性能基线会话",
      "纯文本性能会话",
    ]);
    const plain = await fetch(`${fixture.origin}/api/chat/sessions/perf-session-plain/messages`).then((response) => response.json());
    assert.equal(plain.items.length, 100);
    assert.equal(plain.items.some((item) => item.tool_chain.length > 0), false);
    const runtimeDocuments = await fetch(`${fixture.origin}/api/chat/runtime/documents`).then((response) => response.json());
    assert.deepEqual(runtimeDocuments.items.map((item) => item.id), ["projectneed", "workflow"]);
    const runtimeMcp = await fetch(`${fixture.origin}/api/chat/runtime/mcp?owner_id=core&name=filesystem`).then((response) => response.json());
    assert.equal(runtimeMcp.markdown, "## filesystem\n\nMCP 详情夹具。");
    const upload = await fetch(`${fixture.origin}/api/chat/uploads?filename=fixture.txt`, { method: "POST", body: "fixture" }).then((response) => response.json());
    assert.equal(upload.upload_path, "uploads/fixture.txt");

    const socket = new WebSocket(`ws://127.0.0.1:${fixture.port}/ws`);
    await new Promise((resolveOpen, reject) => {
      socket.once("open", resolveOpen);
      socket.once("error", reject);
    });
    const frames = [];
    socket.on("message", (data) => frames.push(JSON.parse(String(data))));
    const response = await fetch(`${fixture.origin}/__fixture/stream?count=3&delta=x&interval_ms=0`, { method: "POST" });
    assert.equal(response.status, 200);
    await new Promise((resolveFrames) => {
      const poll = () => frames.length === 5 ? resolveFrames() : setTimeout(poll, 5);
      poll();
    });
    assert.deepEqual(frames.map(({ type }) => type), [
      "turn.started",
      "answer.delta",
      "answer.delta",
      "answer.delta",
      "message.final",
    ]);
    assert.equal(frames.at(-1).content, "xxx");
    socket.close();
  } finally {
    await fixture.close();
    rmSync(root, { recursive: true, force: true });
  }
});
