import type { MobileSnapshot } from "./mobile-native";
import type { TimelineMessage } from "./message-timeline";

const BASE_TIME = Date.UTC(2026, 7, 26, 12, 0, 0);
const SESSION_ID = "browser-lab-session";

type LabMessage = TimelineMessage;
export type LabSnapshot = MobileSnapshot & { selectedSessionId: string };
let projectionGeneration = 0;

export type LabScenarioId = "conversation" | "stream" | "long" | "reconnecting";

export const LAB_STREAM_TEXT = "可以。浏览器现在运行的就是手机里那一份 React 界面。以后改颜色、间距、消息气泡或流式动画，都可以先在这里验收，不需要每次连接真机。";

export function createLabSnapshot(scenario: LabScenarioId): LabSnapshot {
  const streaming = scenario === "stream";
  const status: "ready" | "reconnecting" = scenario === "reconnecting" ? "reconnecting" : "ready";
  const messages = scenario === "long" ? longConversation() : dailyConversation();
  return {
    protocolVersion: 9,
    downloads: [],
    throughSeq: messages.at(-1)?.seq ?? -1,
    replyStatus: status === "reconnecting" ? null : replyStatus(streaming ? `lab-preview-${messages.length}` : undefined),
    connection: {
      label: "Browser Lab",
      status,
      ...(scenario === "reconnecting" ? { notice: "正在重新连接，现有内容仍可阅读" } : {}),
    },
    sessions: [{
      id: SESSION_ID,
      title: "浏览器验收实验室",
      lastMessagePreview: streaming ? "正在生成回答…" : "真实 Mobile WebUI",
      lastMessageAt: BASE_TIME + messages.length * 60_000,
      unreadCount: 0,
      isRunning: streaming,
      isAvailable: true,
      canRemove: false,
    }],
    selectedSessionId: SESSION_ID,
    projectionGeneration: ++projectionGeneration,
    messages,
    composer: {
      draft: { text: "" },
      attachments: [],
      pendingMessages: [],
      commands: [
        { command: "/status", description: "查看运行状态" },
        { command: "/new", description: "开始新会话" },
      ],
      isStreaming: streaming,
      isResyncing: false,
      canResync: true,
      isStopping: false,
      canStop: streaming,
      canSend: !streaming,
    },
    modelCatalog: {
      generationId: 1,
      defaultRuntime: "lab/companion",
      selectedRuntimeId: "lab/companion",
      selectedReasoningEffort: "medium",
      runtimes: [{
        id: "lab/companion",
        provider: "lab",
        model: "companion",
        sourceId: "browser-lab",
        sourceName: "Browser Lab",
        reasoningEffort: "medium",
        supportedReasoningEfforts: ["low", "medium", "high"],
        roles: ["default"],
        contextWindow: 128_000,
        inputModalities: ["text", "image"],
      }],
      loading: false,
    },
    runtimeInspection: {
      refreshing: false,
      detailLoading: false,
      documents: [],
      jobs: [],
      mcpServers: [],
      pluginCount: 3,
      skillCount: 8,
    },
  };
}

/** 活动预览只替换内存状态，不追加或改写 Message。 */
export function createPreviewEvent(snapshot: LabSnapshot, text: string) {
  const activity = snapshot.replyStatus?.items[0];
  if (!activity?.preview) throw new Error("Lab 缺少活动草稿");
  return { protocolVersion: 1, projectionGeneration: snapshot.projectionGeneration,
    event: { ...snapshot.replyStatus!, items: [{ ...activity, preview: { ...activity.preview, text } }] } };
}

export function completeLabReply(snapshot: LabSnapshot, content: string, interrupted = false): LabSnapshot {
  const preview = snapshot.replyStatus?.items[0]?.preview;
  if (!preview) throw new Error("Lab 缺少待提交草稿");
  const seq = snapshot.throughSeq + 1;
  const message: LabMessage = interrupted ? {
    id: `lab-pause-${seq}`, session_id: snapshot.selectedSessionId, seq,
    timestamp: new Date(BASE_TIME + seq * 60_000).toISOString(), author: "花月", source: "conversation",
    attachments: [], body: { kind: "control", action: "pause", through_seq: snapshot.throughSeq, reason: null },
  } : assistantMessage(preview.message_id, content, seq);
  return { ...snapshot, messages: [...snapshot.messages, message], throughSeq: seq,
    replyStatus: replyStatus(), sessions: snapshot.sessions.map((session) => ({ ...session, isRunning: false })),
    composer: { ...snapshot.composer, isStreaming: false, isStopping: false, canStop: false, canSend: true } };
}

export function appendLabInput(snapshot: LabSnapshot, text: string, id?: string): LabSnapshot {
  const seq = snapshot.throughSeq + 1;
  return { ...snapshot, messages: [...snapshot.messages, userMessage(id ?? `lab-input-${seq}`, text, seq)],
    throughSeq: seq, replyStatus: replyStatus(`lab-preview-${seq + 1}`),
    sessions: snapshot.sessions.map((session) => ({ ...session, lastMessagePreview: text, isRunning: true })),
    composer: { ...snapshot.composer, draft: { text: "" }, isStreaming: true, canStop: true, canSend: false } };
}

function replyStatus(previewId?: string): NonNullable<MobileSnapshot["replyStatus"]> {
  return { type: "reply.status", version: 2, session_id: SESSION_ID, snapshot_id: "lab-generation", available: true,
    items: previewId ? [{ session_id: SESSION_ID, source: "conversation", handle: previewId, active: true,
      preview: { message_id: previewId, text: "", thinking: "" } }] : [] };
}

function dailyConversation(): LabMessage[] {
  return [
    userMessage("lab-user-1", "我们以后改手机聊天界面的样式，还要每次打开模拟器吗？", 0),
    assistantMessage(
      "lab-assistant-1",
      "不用。这里直接运行手机 WebView 使用的同一份 **React + CSS**。\n\n只有改到相机、通知、文件选择或 Android 生命周期时，才需要真机。",
      1,
    ),
    userMessage("lab-user-2", "那流式回答也能看吗？", 2),
    { ...assistantMessage("lab-call", "先检查 Bridge。", 3), body: { kind: "output", finish: "continue", parts: [
      { kind: "model.facts", value: { call_record_id: "lab-call-record", thinking: "确认消息、排版和预览归 WebUI，系统能力归 Android。" } },
      { kind: "tool_call", binding_id: "lab-binding", name: "read", arguments: { path: "mobile-bridge.ts" } },
    ] } },
    { ...assistantMessage("lab-result", "", 4), author: "read", body: { kind: "tool_result",
      call_ref: { message_id: "lab-call", part_index: 1 }, outcome: "success",
      parts: [{ kind: "text", value: "Bridge 方法已核对" }] } },
    assistantMessage("lab-assistant-2", "能。选择流式生成，就能看到独立草稿在提交后变成正式消息。", 5),
  ];
}

function longConversation(): LabMessage[] {
  return Array.from({ length: 36 }, (_, index) => index % 2 === 0
    ? userMessage(`long-${index}`, `第 ${index / 2 + 1} 轮：检查长会话滚动、虚拟列表和消息间距。`, index)
    : assistantMessage(
      `long-${index}`,
      index % 6 === 1
        ? `### 长会话节点 ${index}\n\n- 消息身份保持稳定\n- 滚动位置不跳动\n- Markdown 在终态增强\n\n\`\`\`ts\nconst stableRow = ${index};\n\`\`\``
        : "这是一条稳定的长会话夹具，用来观察连续消息、留白和滚动到底部按钮。",
      index,
    ));
}

function userMessage(id: string, content: string, index: number): LabMessage {
  return message(id, content, index, "input");
}

function assistantMessage(id: string, content: string, index: number): LabMessage {
  return message(id, content, index, "output");
}

function message(id: string, content: string, seq: number, kind: "input" | "output"): LabMessage {
  const parts = [{ kind: "text" as const, value: content }];
  return { id, session_id: SESSION_ID, seq, timestamp: new Date(BASE_TIME + seq * 60_000).toISOString(),
    author: kind === "input" ? "花月" : "Akashic", source: "conversation", attachments: [],
    body: kind === "input" ? { kind, parts } : { kind, parts, finish: "complete" } };
}
