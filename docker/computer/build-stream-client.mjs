import { readFile, writeFile } from "node:fs/promises";

// Blob iframe 没有普通路由前缀；只向固定版本补充显式 WebSocket URL。
const path = process.argv[2];
const source = await readFile(path, "utf8");
const original = 'let websocketEndpointURL = new URL(`${ws_protocol}${window.location.host}${pathname}`);';
if (source.split(original).length !== 2) {
  throw new Error("Selkies 2.0.0 WebSocket entry changed; check the integration patch");
}
await writeFile(path, source.replace(original,
  'let websocketEndpointURL = new URL(window.__SELKIES_WEBSOCKET_URL__ || `${ws_protocol}${window.location.host}${pathname}`);'));
// 显式地址已经包含 Dashboard 的 generation 身份和完整路径。
const patched = await readFile(path, "utf8");
const append = "websocketEndpointURL.pathname += 'api/websockets';";
if (patched.split(append).length !== 2) {
  throw new Error("Selkies 2.0.0 WebSocket path changed; check the integration patch");
}
await writeFile(path, patched.replace(append,
  "if (!window.__SELKIES_WEBSOCKET_URL__) websocketEndpointURL.pathname += 'api/websockets';"));

// 重连不按每个 Blob URL 创建新的 localStorage 命名空间。
const utilPath = path.replace(/selkies-ws-core\.js$/, "lib/util.js");
const util = await readFile(utilPath, "utf8");
const storage = 'const urlForKey = window.location.origin + window.location.pathname;';
if (util.split(storage).length !== 2) {
  throw new Error("Selkies 2.0.0 storage entry changed; check the integration patch");
}
await writeFile(utilPath, util.replace(storage,
  'const urlForKey = window.__SELKIES_STORAGE_URL__ || window.location.origin + window.location.pathname;'));

// Ctrl+V 必须等待分块文字发送完毕，不能只等待 postMessage 入队。
const clipboard = 'sendExplicitClipboard(message.text);';
const current = await readFile(path, "utf8");
if (current.split(clipboard).length !== 2) {
  throw new Error("Selkies 2.0.0 clipboard entry changed; check the integration patch");
}
await writeFile(path, current.replace(clipboard, `
      let clipboardError = "";
      sendExplicitClipboard(message.text, undefined, (reason, code) => {
        if (code !== "clipboardSkipUnchanged") clipboardError = reason;
      }).then(() => window.parent.postMessage({
        type: "computerClipboardSent", id: message.requestId, error: clipboardError
      }, window.location.origin)).catch(error => window.parent.postMessage({
        type: "computerClipboardSent", id: message.requestId, error: String(error)
      }, window.location.origin));`));
