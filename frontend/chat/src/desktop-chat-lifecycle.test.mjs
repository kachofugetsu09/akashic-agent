import assert from "node:assert/strict";
import { createRequire } from "node:module";
import test from "node:test";
import { build } from "esbuild";
import { JSDOM } from "jsdom";
import React, { act } from "react";
import { createRoot } from "react-dom/client";

// 编译真实 hook 及其本地依赖；React 与 DOM renderer 使用同一份模块。
const compiled = await build({
  entryPoints: [new URL("./use-desktop-chat-controller.ts", import.meta.url).pathname],
  bundle: true,
  write: false,
  platform: "node",
  format: "cjs",
  packages: "external",
  loader: { ".css": "empty", ".svg": "dataurl" },
});
const controllerModule = { exports: {} };
new Function("require", "module", "exports", compiled.outputFiles[0].text)(
  createRequire(import.meta.url), controllerModule, controllerModule.exports,
);
const { useDesktopChatController } = controllerModule.exports;

/** 挂载真实 React controller，只控制网络完成顺序和时钟。 */
async function mountChat(t, strict = false) {
  const dom = new JSDOM("<div id='root'></div>", { url: "http://localhost/" });
  const sockets = [];
  const requests = [];
  class Socket extends EventTarget {
    static CONNECTING = 0;
    static OPEN = 1;
    static CLOSED = 3;
    readyState = Socket.CONNECTING;
    sent = [];
    constructor(url) {
      super();
      this.url = url;
      sockets.push(this);
    }
    open() {
      this.readyState = Socket.OPEN;
      const event = new Event("open");
      this.onopen?.(event);
      this.dispatchEvent(event);
    }
    close(code = 1000, reason = "") {
      if (this.readyState === Socket.CLOSED) return;
      this.readyState = Socket.CLOSED;
      const event = Object.assign(new Event("close"), { code, reason });
      this.onclose?.(event);
      this.dispatchEvent(event);
    }
    send(text) { this.sent.push(text); }
  }
  t.mock.timers.enable({ apis: ["setTimeout", "setInterval", "Date"] });
  dom.window.setTimeout = globalThis.setTimeout;
  dom.window.clearTimeout = globalThis.clearTimeout;
  dom.window.setInterval = globalThis.setInterval;
  dom.window.clearInterval = globalThis.clearInterval;
  const globals = {
    window: dom.window,
    document: dom.window.document,
    WebSocket: Socket,
    IS_REACT_ACT_ENVIRONMENT: true,
    fetch(url, options = {}) {
      assert.equal(url, "/api/shell/state");
      return new Promise((resolve, reject) => {
        const request = {
          signal: options.signal,
          finish: (body) => resolve(new Response(JSON.stringify(body))),
        };
        options.signal?.addEventListener("abort", () => {
          reject(new DOMException("请求已取消", "AbortError"));
        }, { once: true });
        requests.push(request);
      });
    },
  };
  const previous = new Map(Object.keys(globals).map((key) => [key, Object.getOwnPropertyDescriptor(globalThis, key)]));
  for (const [key, value] of Object.entries(globals)) {
    Object.defineProperty(globalThis, key, { value, configurable: true, writable: true });
  }
  const root = createRoot(dom.window.document.getElementById("root"));
  let mounted = true;
  let controller;
  function Chat() {
    const chat = useDesktopChatController();
    controller = chat;
    return React.createElement("output", null, chat.shellState?.status ?? "loading");
  }
  const unmount = async () => {
    if (!mounted) return;
    mounted = false;
    await act(async () => root.unmount());
  };
  t.after(async () => {
    await unmount();
    dom.window.close();
    for (const [key, descriptor] of previous) {
      if (descriptor) Object.defineProperty(globalThis, key, descriptor);
      else delete globalThis[key];
    }
  });
  const content = React.createElement(Chat);
  await act(async () => root.render(strict ? React.createElement(React.StrictMode, null, content) : content));
  return {
    sockets, requests, unmount,
    controller: () => controller,
    state: () => dom.window.document.querySelector("output")?.textContent,
    tick: (ms) => act(async () => t.mock.timers.tick(ms)),
  };
}

test("重连后的聊天连接随卸载关闭，卸载后不再重连", async (t) => {
  const chat = await mountChat(t);
  assert.equal(chat.sockets.length, 1);
  await act(async () => {
    chat.sockets[0].open();
    chat.sockets[0].close(1006);
  });
  await chat.tick(30_000);
  assert.equal(chat.sockets.length, 2);
  await act(async () => chat.sockets[1].open());
  await chat.unmount();
  assert.equal(chat.sockets[1].readyState, 3, "必须关闭重连得到的当前 socket");
  await chat.tick(60_000);
  assert.equal(chat.sockets.length, 2, "卸载不得留下后台重连任务");
});

test("状态轮询等待上次完成，卸载取消正在进行的请求", async (t) => {
  const chat = await mountChat(t);
  assert.equal(chat.requests.length, 1);
  await chat.tick(2_400);
  assert.equal(chat.requests.length, 1, "慢请求期间不得继续发起轮询");
  await act(async () => chat.requests[0].finish({ status: "needs_setup", configured: false, chatReady: false }));
  assert.equal(chat.state(), "needs_setup");
  await chat.tick(1_200);
  assert.equal(chat.requests.length, 2);
  await act(async () => chat.requests[1].finish({ status: "starting", configured: true, chatReady: false }));
  assert.equal(chat.state(), "starting");
  await chat.tick(1_200);
  assert.equal(chat.requests.length, 3);
  await chat.unmount();
  assert.equal(chat.requests[2].signal?.aborted, true);
  await chat.tick(60_000);
  assert.equal(chat.requests.length, 3);
});

test("断线重试保留十二次上限，成功连接后重新计数", async (t) => {
  const chat = await mountChat(t);
  const fail = async () => {
    await act(async () => chat.sockets.at(-1).close(1006));
    await chat.tick(30_000);
  };
  for (let attempt = 0; attempt < 5; attempt += 1) await fail();
  assert.equal(chat.sockets.length, 6);
  await act(async () => chat.sockets.at(-1).open());
  for (let attempt = 0; attempt < 12; attempt += 1) {
    await fail();
    assert.equal(chat.sockets.length, 7 + attempt);
  }
  await fail();
  assert.equal(chat.sockets.length, 18);
  assert.equal(chat.controller().error, "聊天连接已断开，请刷新页面重试");
  await chat.tick(60_000);
  assert.equal(chat.sockets.length, 18);
});

test("StrictMode 重挂载与重试等待中的卸载不留下连接或监听器", async (t) => {
  const chat = await mountChat(t, true);
  assert.equal(chat.sockets.length, 2);
  assert.equal(chat.sockets[0].readyState, 3);
  assert.equal(chat.requests[0].signal.aborted, true);
  await act(async () => chat.sockets[1].close(1006));
  await chat.unmount();
  await chat.tick(60_000);
  assert.equal(chat.sockets.length, 2);
  for (const socket of chat.sockets) {
    assert.equal(socket.readyState, 3);
    assert.equal(socket.onmessage, null);
    assert.equal(socket.onopen, null);
    assert.equal(socket.onclose, null);
  }
});
