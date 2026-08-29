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
    assert.equal(plain.items.length, 50);
    assert.deepEqual([plain.items[0].seq, plain.items.at(-1).seq], [50, 99]);
    assert.equal(plain.before_seq, 50);
    assert.equal(plain.has_more, true);
    const older = await fetch(`${fixture.origin}/api/chat/sessions/perf-session-plain/messages?page_size=50&before_seq=50`).then((response) => response.json());
    assert.deepEqual([older.items[0].seq, older.items.at(-1).seq], [0, 49]);
    assert.equal(older.has_more, false);
    assert.equal(plain.items.some((item) => item.tool_chain.length > 0), false);
    const runtimeDocuments = await fetch(`${fixture.origin}/api/chat/runtime/documents`).then((response) => response.json());
    assert.deepEqual(runtimeDocuments.items.map((item) => item.id), ["projectneed", "workflow"]);
    const runtimeMcp = await fetch(`${fixture.origin}/api/chat/runtime/mcp?owner_id=core&name=filesystem`).then((response) => response.json());
    assert.equal(runtimeMcp.markdown, "## filesystem\n\nMCP 详情夹具。");
    const upload = await fetch(`${fixture.origin}/api/chat/uploads?filename=fixture.txt`, { method: "POST", body: "fixture" }).then((response) => response.json());
    assert.equal(upload.upload_path, "uploads/fixture.txt");
    const pairing = await fetch(`${fixture.origin}/api/chat/mobile-pairing`, { method: "POST" }).then((response) => response.json());
    assert.equal(pairing.protocol_version, 1);
    const claim = await fetch(`${fixture.origin}/api/chat/mobile-pairing/${pairing.pairing_id}`).then((response) => response.json());
    assert.equal(claim.confirmation_code, "358864");
    const device = await fetch(`${fixture.origin}/api/chat/mobile-pairing/${pairing.pairing_id}/approve`, { method: "POST" }).then((response) => response.json());
    assert.deepEqual(device, { device_id: "pixel-7", display_name: "Pixel 7" });
    const settings = await fetch(`${fixture.origin}/api/settings/model/catalog`).then((response) => response.json());
    assert.equal(settings.models.length, 48);
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

test("desktop fixture replays one rich turn over a long stored history", async () => {
  const root = mkdtempSync(resolve(tmpdir(), "akashic-desktop-replay-test-"));
  writeFileSync(resolve(root, "index.html"), "<!doctype html><title>fixture</title>");
  const replayTurn = {
    content: "final answer",
    stages: [{
      text: "stage text",
      reasoning: "thinking",
      calls: [{
        callId: "call-1", name: "probe", status: "success",
        arguments: { scope: "fixture" }, finalArguments: { scope: "fixture" }, result: "done",
      }],
    }],
  };
  const fixture = await startDesktopFixtureServer(root, { historyCount: 1_975, replayTurn });
  try {
    const history = await fetch(`${fixture.origin}/api/chat/sessions/perf-session/messages`).then((response) => response.json());
    assert.equal(history.total, 1_975);
    assert.deepEqual([history.items[0].seq, history.items.at(-1).seq], [1_925, 1_974]);
    const socket = new WebSocket(`ws://127.0.0.1:${fixture.port}/ws`);
    await new Promise((resolveOpen, reject) => {
      socket.once("open", resolveOpen);
      socket.once("error", reject);
    });
    const frames = [];
    socket.on("message", (data) => frames.push(JSON.parse(String(data))));
    const response = await fetch(`${fixture.origin}/__fixture/stream?mode=replay&characters_per_second=10000&chunk_characters=1`, { method: "POST" });
    assert.equal(response.status, 200);
    const summary = await response.json();
    assert.deepEqual({ stages: summary.stageCount, calls: summary.callCount }, { stages: 1, calls: 1 });
    assert.equal(summary.chunkCharacters, 1);
    assert.equal(summary.deltaCount, "thinkingstage textfinal answer".length);
    assert.equal(
      frames.filter((frame) => frame.delta !== undefined).every((frame) => frame.delta.length === 1),
      true,
    );
    assert.equal(frames[0].type, "turn.started");
    assert.equal(frames.at(-1).type, "message.final");
    assert.equal(frames.filter(({ type }) => type === "react.thinking.delta").length, "thinking".length);
    assert.equal(
      frames.filter(({ type }) => type === "answer.delta").length,
      "stage textfinal answer".length,
    );
    assert.equal(frames.filter(({ type }) => type === "react.tool.started").length, 1);
    assert.equal(frames.filter(({ type }) => type === "react.tool.completed").length, 1);
    assert.equal(frames.at(-1).content, "final answer");
    socket.close();
  } finally {
    await fixture.close();
    rmSync(root, { recursive: true, force: true });
  }
});
