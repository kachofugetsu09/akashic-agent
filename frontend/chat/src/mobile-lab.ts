import "./mobile-lab.css";

import { MOBILE_NATIVE_METHODS, type MobileNativeMethod } from "./mobile-bridge";
import {
  appendUserTurn,
  createLabSnapshot,
  createStreamPatch,
  createTerminalPatch,
  LAB_STREAM_TEXT,
  type LabScenarioId,
} from "./mobile-lab-fixtures";

interface BridgeEnvelope {
  v: number;
  generation_id: string;
  nonce: string;
  method: MobileNativeMethod;
  args: unknown[];
}

interface DevicePreset {
  label: string;
  width: number;
  height: number;
}

const DEVICES: Record<string, DevicePreset> = {
  pixel: { label: "Pixel · 412 × 915", width: 412, height: 915 },
  compact: { label: "小屏 · 360 × 800", width: 360, height: 800 },
  tablet: { label: "平板 · 768 × 1024", width: 768, height: 1024 },
};

const root = document.getElementById("mobile-lab");
if (!root) throw new Error("Mobile Lab root 不存在");

root.innerHTML = `
  <main class="lab-shell">
    <header class="lab-header">
      <div>
        <p class="lab-kicker">AKASHIC / MOBILE WEBUI</p>
        <h1>Browser Lab</h1>
      </div>
      <div class="lab-header__status">
        <span class="lab-status-dot" aria-hidden="true"></span>
        <span id="lab-status">正在启动真实 Mobile WebUI</span>
      </div>
      <button class="lab-quiet-button" id="focus-button" type="button">只看手机</button>
    </header>

    <aside class="lab-panel lab-panel--controls" aria-label="验收场景">
      <section class="lab-control-group">
        <p class="lab-label">场景</p>
        <div class="lab-segmented" role="group" aria-label="选择场景">
          <button class="is-active" data-scenario="conversation" type="button">日常对话</button>
          <button data-scenario="stream" type="button">流式生成</button>
          <button data-scenario="long" type="button">长会话</button>
          <button data-scenario="reconnecting" type="button">重连</button>
        </div>
      </section>

      <section class="lab-control-group">
        <label class="lab-label" for="lab-device">设备画布</label>
        <select id="lab-device">
          <option value="pixel">Pixel · 412 × 915</option>
          <option value="compact">小屏 · 360 × 800</option>
          <option value="tablet">平板 · 768 × 1024</option>
        </select>
      </section>

      <section class="lab-control-group">
        <p class="lab-label">主题</p>
        <div class="lab-theme-row" role="group" aria-label="选择主题">
          <button class="is-active" data-theme="light" type="button"><span class="theme-swatch theme-swatch--blue"></span>浅蓝</button>
          <button data-theme="warm-paper" type="button"><span class="theme-swatch theme-swatch--warm"></span>暖色</button>
          <button data-theme="dark" type="button"><span class="theme-swatch theme-swatch--dark"></span>深色</button>
        </div>
      </section>

      <section class="lab-control-group lab-control-group--action">
        <button class="lab-primary-button" id="stream-button" type="button">播放一次流式回答</button>
        <button class="lab-secondary-button" id="reset-button" type="button">恢复当前场景</button>
      </section>

      <p class="lab-note">里面是生产聊天界面。外面的控制台只扮演 Android 能力层。</p>
    </aside>

    <section class="lab-stage" aria-label="手机预览">
      <div class="device" id="lab-device-frame">
        <div class="device__rail" aria-hidden="true"><span></span><i></i></div>
        <iframe
          id="lab-frame"
          title="Akashic Mobile WebUI"
          src="./mobile-lab-frame.html?generation_id=browser-lab&amp;nonce=browser-lab"
        ></iframe>
      </div>
    </section>

    <aside class="lab-panel lab-panel--activity" aria-label="Bridge 活动">
      <div class="lab-activity-heading">
        <div>
          <p class="lab-label">Bridge activity</p>
          <strong>浏览器能力记录</strong>
        </div>
        <span>${MOBILE_NATIVE_METHODS.length} methods</span>
      </div>
      <ol id="lab-activity" aria-live="polite"></ol>
      <div class="lab-boundary">
        <span>浏览器负责</span>
        <p>布局、样式、消息、流式渲染和已有 Bridge 交互。</p>
        <span>Android 负责</span>
        <p>相机、通知、系统文件、密钥、Room 和 WebView 生命周期。</p>
      </div>
    </aside>
  </main>
`;

const frame = requireElement<HTMLIFrameElement>("lab-frame");
const deviceFrame = requireElement<HTMLElement>("lab-device-frame");
const status = requireElement<HTMLElement>("lab-status");
const activity = requireElement<HTMLOListElement>("lab-activity");
const streamButton = requireElement<HTMLButtonElement>("stream-button");
const focusButton = requireElement<HTMLButtonElement>("focus-button");

let scenario: LabScenarioId = "conversation";
let snapshot = createLabSnapshot(scenario);
let streamTimer: number | null = null;
let activitySequence = 0;

bindControls();
applyFocusMode(new URL(window.location.href).searchParams.get("focus") === "1");

window.addEventListener("message", (event: MessageEvent<unknown>) => {
  if (event.origin !== window.location.origin || event.source !== frame.contentWindow) return;
  const data = event.data;
  if (!isRecord(data) || data.type !== "akashic.mobile-lab.bridge" || typeof data.payload !== "string") return;
  const envelope = parseBridgeEnvelope(data.payload);
  recordBridgeCall(envelope);
  handleBridgeCall(envelope);
});

function bindControls(): void {
  document.querySelectorAll<HTMLButtonElement>("[data-scenario]").forEach((button) => {
    button.addEventListener("click", () => {
      scenario = requireScenario(button.dataset.scenario);
      setActiveButton("[data-scenario]", button);
      resetScenario();
      if (scenario === "stream") window.setTimeout(() => startStream(LAB_STREAM_TEXT), 180);
    });
  });
  document.querySelectorAll<HTMLButtonElement>("[data-theme]").forEach((button) => {
    button.addEventListener("click", () => {
      const theme = button.dataset.theme;
      if (!theme) throw new Error("主题按钮缺少 theme");
      setActiveButton("[data-theme]", button);
      postNativeMessage("mobile.theme", theme);
      setStatus(`已切换 ${button.textContent?.trim() ?? theme}主题`);
    });
  });
  requireElement<HTMLSelectElement>("lab-device").addEventListener("change", (event) => {
    const select = event.currentTarget;
    if (!(select instanceof HTMLSelectElement)) return;
    const preset = DEVICES[select.value];
    if (!preset) throw new Error(`未知设备预设: ${select.value}`);
    deviceFrame.style.setProperty("--device-width", `${preset.width}px`);
    deviceFrame.style.setProperty("--device-height", `${preset.height}px`);
    setStatus(`画布已切换为 ${preset.label}`);
  });
  streamButton.addEventListener("click", () => startStream(LAB_STREAM_TEXT));
  requireElement<HTMLButtonElement>("reset-button").addEventListener("click", resetScenario);
  focusButton.addEventListener("click", () => applyFocusMode(!document.body.classList.contains("is-focus-mode")));
}

function handleBridgeCall(envelope: BridgeEnvelope): void {
  switch (envelope.method) {
    case "requestSnapshot":
      deliverSnapshot();
      return;
    case "reportHealthy":
      document.body.dataset.labReady = "true";
      setStatus("真实 Mobile WebUI 已就绪");
      return;
    case "sendMessage": {
      const [requestId, , text] = envelope.args;
      if (typeof requestId !== "string" || typeof text !== "string") {
        throw new Error("sendMessage 参数无效");
      }
      childWindow()?.AkashicMobile?.receiveSendResult(requestId, true);
      snapshot = appendUserTurn(snapshot, text.trim() || "（附件消息）");
      deliverSnapshot();
      window.setTimeout(() => startStream("这条回复由 Browser Bridge 接住发送动作后生成。视觉和交互走的仍然是生产 Mobile WebUI。"), 160);
      return;
    }
    case "stopTurn":
      stopStream(true);
      return;
    case "setTheme": {
      const [theme] = envelope.args;
      if (typeof theme === "string") postNativeMessage("mobile.theme", theme);
      return;
    }
    case "copyText": {
      const [text] = envelope.args;
      if (typeof text === "string" && navigator.clipboard) {
        void navigator.clipboard.writeText(text).catch((error: unknown) => {
          setStatus(`浏览器剪贴板不可用：${errorMessage(error)}`, true);
        });
      }
      return;
    }
    case "saveComposerDraft":
    case "saveReadingPosition":
    case "markSessionReadThrough":
    case "navigationTargetHandled":
    case "setWebHistoryActive":
    case "performActionHaptic":
      return;
    default:
      setStatus(`${envelope.method} 需要 Android 原生环境`, true);
  }
}

function resetScenario(): void {
  stopTimer();
  snapshot = createLabSnapshot(scenario);
  streamButton.disabled = false;
  streamButton.textContent = "播放一次流式回答";
  deliverSnapshot();
  setStatus(`已载入${scenarioLabel(scenario)}`);
}

function startStream(text: string): void {
  stopTimer();
  if (!snapshot.composer.isStreaming) {
    const last = snapshot.messages.at(-1);
    if (!last || last.role !== "assistant" || last.content !== "") {
      snapshot = appendUserTurn(snapshot, "请演示一次流式回答");
      deliverSnapshot();
    }
  }
  const characters = Array.from(text);
  let index = 0;
  let content = "";
  streamButton.disabled = true;
  streamButton.textContent = "正在播放…";
  setStatus("正在注入真实 stream patch");
  streamTimer = window.setInterval(() => {
    const delta = characters[index];
    if (delta === undefined) {
      stopTimer();
      const terminal = createTerminalPatch(snapshot, content);
      childWindow()?.AkashicMobile?.receiveStreamPatch(terminal);
      applyTerminalToLocalSnapshot(terminal);
      streamButton.disabled = false;
      streamButton.textContent = "再播放一次";
      setStatus("流式回答已到达 terminal");
      return;
    }
    content += delta;
    childWindow()?.AkashicMobile?.receiveStreamPatch(createStreamPatch(snapshot, index + 1, delta));
    index += 1;
  }, 38);
}

function stopStream(interrupted: boolean): void {
  if (streamTimer === null) return;
  stopTimer();
  const currentMessage = snapshot.messages.at(-1);
  if (!currentMessage) return;
  const renderedText = childWindow()?.document.querySelector(`[data-message-id="${currentMessage.id}"]`)?.textContent ?? "";
  const terminal = createTerminalPatch(snapshot, renderedText.trim(), interrupted);
  childWindow()?.AkashicMobile?.receiveStreamPatch(terminal);
  applyTerminalToLocalSnapshot(terminal);
  streamButton.disabled = false;
  streamButton.textContent = "再播放一次";
  setStatus("已按 Bridge stopTurn 中止流式回答");
}

function applyTerminalToLocalSnapshot(terminal: ReturnType<typeof createTerminalPatch>): void {
  const state = terminal.state;
  snapshot = {
    ...snapshot,
    connection: state.connection,
    sessions: state.sessions,
    messages: [...snapshot.messages.slice(0, -1), terminal.message],
    composer: state.composer,
  };
}

function deliverSnapshot(): void {
  const mobile = childWindow()?.AkashicMobile;
  if (!mobile) return;
  mobile.receiveSnapshot(structuredClone(snapshot));
}

function postNativeMessage(type: string, payload: unknown): void {
  frame.contentWindow?.postMessage(JSON.stringify({ type, payload }), window.location.origin);
}

function recordBridgeCall(envelope: BridgeEnvelope): void {
  activitySequence += 1;
  const item = document.createElement("li");
  const sequence = document.createElement("span");
  const method = document.createElement("strong");
  const detail = document.createElement("small");
  sequence.textContent = String(activitySequence).padStart(2, "0");
  method.textContent = envelope.method;
  detail.textContent = summarizeArgs(envelope.args);
  item.append(sequence, method, detail);
  activity.prepend(item);
  while (activity.childElementCount > 12) activity.lastElementChild?.remove();
}

function parseBridgeEnvelope(value: string): BridgeEnvelope {
  const raw: unknown = JSON.parse(value);
  if (!isRecord(raw) || raw.v !== 1 || typeof raw.method !== "string" || !Array.isArray(raw.args)) {
    throw new Error("Browser Bridge 收到无效 envelope");
  }
  if (!MOBILE_NATIVE_METHODS.some((method) => method === raw.method)) {
    throw new Error(`Browser Bridge 收到未知方法: ${raw.method}`);
  }
  if (typeof raw.generation_id !== "string" || typeof raw.nonce !== "string") {
    throw new Error("Browser Bridge envelope 缺少 generation 身份");
  }
  return raw as unknown as BridgeEnvelope;
}

function childWindow(): Window | null {
  return frame.contentWindow;
}

function setStatus(message: string, warning = false): void {
  status.textContent = message;
  status.closest(".lab-header__status")?.classList.toggle("is-warning", warning);
}

function setActiveButton(selector: string, active: HTMLButtonElement): void {
  document.querySelectorAll<HTMLButtonElement>(selector).forEach((button) => {
    button.classList.toggle("is-active", button === active);
    button.setAttribute("aria-pressed", String(button === active));
  });
}

function applyFocusMode(enabled: boolean): void {
  document.body.classList.toggle("is-focus-mode", enabled);
  focusButton.textContent = enabled ? "显示控制台" : "只看手机";
  const url = new URL(window.location.href);
  if (enabled) url.searchParams.set("focus", "1");
  else url.searchParams.delete("focus");
  window.history.replaceState(null, "", url);
}

function stopTimer(): void {
  if (streamTimer !== null) window.clearInterval(streamTimer);
  streamTimer = null;
}

function requireElement<T extends HTMLElement>(id: string): T {
  const element = document.getElementById(id);
  if (!element) throw new Error(`Mobile Lab 缺少 #${id}`);
  return element as T;
}

function requireScenario(value: string | undefined): LabScenarioId {
  if (value === "conversation" || value === "stream" || value === "long" || value === "reconnecting") return value;
  throw new Error(`未知场景: ${String(value)}`);
}

function scenarioLabel(value: LabScenarioId): string {
  return ({ conversation: "日常对话", stream: "流式生成", long: "长会话", reconnecting: "重连状态" })[value];
}

function summarizeArgs(args: unknown[]): string {
  if (args.length === 0) return "no arguments";
  const text = JSON.stringify(args);
  return text.length > 72 ? `${text.slice(0, 69)}…` : text;
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}
