import { readFile, writeFile } from "node:fs/promises";

// 固定版本的上游源码只按精确原文替换；原文变了就停止构建，提示重新检查集成补丁。
async function patch(path, original, replacement, label) {
  const source = await readFile(path, "utf8");
  if (source.split(original).length !== 2) {
    throw new Error(`Selkies 2.0.0 ${label} changed; check the integration patch`);
  }
  await writeFile(path, source.replace(original, replacement));
}

const path = process.argv[2];
const webrtcPath = path.replace(/selkies-ws-core\.js$/, "selkies-wr-core.js");

// Blob iframe 没有普通路由前缀；只向固定版本补充显式 WebSocket URL。
await patch(path, 'let websocketEndpointURL = new URL(`${ws_protocol}${window.location.host}${pathname}`);',
  'let websocketEndpointURL = new URL(window.__SELKIES_WEBSOCKET_URL__ || `${ws_protocol}${window.location.host}${pathname}`);',
  "WebSocket entry");
// 显式地址已经包含 Dashboard 的 generation 身份和完整路径。
await patch(path, "websocketEndpointURL.pathname += 'api/websockets';",
  "if (!window.__SELKIES_WEBSOCKET_URL__) websocketEndpointURL.pathname += 'api/websockets';",
  "WebSocket path");

// 重连不按每个 Blob URL 创建新的 localStorage 命名空间。
await patch(path.replace(/selkies-ws-core\.js$/, "lib/util.js"),
  "const urlForKey = window.location.origin + window.location.pathname;",
  "const urlForKey = window.__SELKIES_STORAGE_URL__ || window.location.origin + window.location.pathname;",
  "storage entry");

// Ctrl+V 必须等待分块文字发送完毕，不能只等待 postMessage 入队。
const clipboardSent = (send) => `
      let clipboardError = "";
      ${send}(message.text, undefined, (reason, code) => {
        if (code !== "clipboardSkipUnchanged") clipboardError = reason;
      }).then(() => window.parent.postMessage({
        type: "computerClipboardSent", id: message.requestId, error: clipboardError
      }, window.location.origin)).catch(error => window.parent.postMessage({
        type: "computerClipboardSent", id: message.requestId, error: String(error)
      }, window.location.origin));`;
await patch(path, "sendExplicitClipboard(message.text);", clipboardSent("sendExplicitClipboard"), "clipboard entry");
await patch(webrtcPath, "localClipboardSender.sendExplicit(message.text);",
  clipboardSent("localClipboardSender.sendExplicit"), "WebRTC clipboard entry");

// WebRTC：信令经 Dashboard 转发，ICE 配置由面板注入，不再请求流服务自己的 /api/turn。
await patch(webrtcPath,
  'var url = new URL(protocol + window.location.host + pathname + "api/" + appName + "/signaling/");',
  'var url = new URL(window.__SELKIES_SIGNALING_URL__ || protocol + window.location.host + pathname + "api/" + appName + "/signaling/");',
  "WebRTC signaling entry");
await patch(webrtcPath, 'fetch(getRoutePrefix() + "/api/turn", { headers: sessionAuthHeaders() })',
  '(window.__SELKIES_RTC_CONFIG__ ? Promise.resolve(new Response(JSON.stringify(window.__SELKIES_RTC_CONFIG__)))'
    + ' : fetch(getRoutePrefix() + "/api/turn", { headers: sessionAuthHeaders() }))',
  "WebRTC TURN entry");
// 面板读取连接状态、统计和数据通道，用于就绪判断与链路降档。
await patch(webrtcPath, "webrtc = new WebRTCClient(signaling, videoElement, 1);",
  "webrtc = new WebRTCClient(signaling, videoElement, 1); window.selkiesWebRTC = webrtc;",
  "WebRTC client entry");
// 键鼠授权（Core 按 Agent 是否在用切换令牌）到达后面板才放行输入，免得刚放行时的第一下操作被丢弃。
await patch(webrtcPath, "collabInputGranted = granted;", "collabInputGranted = granted; window.selkiesInputGranted = granted;",
  "WebRTC input access entry");
// 断线重连归面板所有（它还要决定是否回退 WebSocket），客户端不自行刷新 iframe。
const reload = "window.__SELKIES_RECONNECT__ ? window.__SELKIES_RECONNECT__() : location.reload();";
await patch(webrtcPath, "console.warn('[webrtc] no video after resuming; reloading to reconnect.');\n\t\tlocation.reload();",
  `console.warn('[webrtc] no video after resuming; reloading to reconnect.');\n\t\t${reload}`, "WebRTC resume reload");
await patch(webrtcPath, "console.warn(`[webrtc] connection ${st}; reloading to reconnect.`);\n\t\t\t\t\t\t\tlocation.reload();",
  `console.warn(\`[webrtc] connection \${st}; reloading to reconnect.\`);\n\t\t\t\t\t\t\t${reload}`, "WebRTC failure reload");
await patch(webrtcPath, "\t\t\t\tlocation.reload();\n\t\t\t};\n\t\t\twebrtc = new WebRTCClient",
  `\t\t\t\t${reload}\n\t\t\t};\n\t\t\twebrtc = new WebRTCClient`, "WebRTC signaling reload");
