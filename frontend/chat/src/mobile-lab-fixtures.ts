const BASE_TIME = Date.UTC(2026, 7, 26, 12, 0, 0);
const SESSION_ID = "browser-lab-session";

export type LabConnectionStatus = "ready" | "reconnecting" | "degraded";

export interface LabMessage {
  id: string;
  sessionId: string;
  role: "user" | "assistant";
  content: string;
  createdAt: number;
  searchRevision: number;
  replyable: boolean;
  blocks: Array<Record<string, unknown>>;
  streaming: boolean;
  interrupted: boolean;
  attachments: Array<Record<string, unknown>>;
}

export interface LabSnapshot {
  protocolVersion: 8;
  connection: { label: string; status: LabConnectionStatus; notice?: string };
  sessions: Array<Record<string, unknown>>;
  selectedSessionId: string;
  projectionGeneration: number;
  messages: LabMessage[];
  composer: {
    draft: { text: string };
    attachments: Array<Record<string, unknown>>;
    pendingMessages: Array<Record<string, unknown>>;
    commands: Array<Record<string, unknown>>;
    isStreaming: boolean;
    isResyncing: boolean;
    canResync: boolean;
    isStopping: boolean;
    canStop: boolean;
    canSend: boolean;
  };
  modelCatalog: Record<string, unknown>;
  runtimeInspection: Record<string, unknown>;
}

export type LabScenarioId = "conversation" | "stream" | "long" | "reconnecting";

export const LAB_STREAM_TEXT = "可以。浏览器现在运行的就是手机里那一份 React 界面。以后改颜色、间距、消息气泡或流式动画，都可以先在这里验收，不需要每次连接真机。";

export function createLabSnapshot(scenario: LabScenarioId): LabSnapshot {
  const streaming = scenario === "stream";
  const status: LabConnectionStatus = scenario === "reconnecting" ? "reconnecting" : "ready";
  const messages = scenario === "long" ? longConversation() : dailyConversation();
  if (streaming) messages.push(assistantMessage("stream-assistant", "", messages.length, true));
  return {
    protocolVersion: 8,
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
    projectionGeneration: 1,
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

export function createStreamPatch(snapshot: LabSnapshot, revision: number, delta: string) {
  const messageIndex = snapshot.messages.length - 1;
  return {
    protocolVersion: 3,
    projectionGeneration: snapshot.projectionGeneration,
    selectedSessionId: snapshot.selectedSessionId,
    messageIndex,
    messageId: snapshot.messages[messageIndex].id,
    searchRevision: revision,
    contentAppend: delta,
  };
}

export function createTerminalPatch(snapshot: LabSnapshot, content: string, interrupted = false) {
  const messageIndex = snapshot.messages.length - 1;
  const previous = snapshot.messages[messageIndex];
  const message: LabMessage = {
    ...previous,
    content,
    searchRevision: Math.max(previous.searchRevision + 1, 10_000),
    streaming: false,
    interrupted,
  };
  const state = {
    protocolVersion: 1,
    connection: snapshot.connection,
    sessions: snapshot.sessions.map((session) => ({ ...session, isRunning: false })),
    selectedSessionId: snapshot.selectedSessionId,
    projectionGeneration: snapshot.projectionGeneration,
    composer: {
      ...snapshot.composer,
      isStreaming: false,
      isStopping: false,
      canStop: false,
      canSend: true,
    },
    modelCatalog: snapshot.modelCatalog,
    runtimeInspection: snapshot.runtimeInspection,
  };
  return {
    protocolVersion: 3,
    projectionGeneration: snapshot.projectionGeneration,
    selectedSessionId: snapshot.selectedSessionId,
    messageIndex,
    messageId: message.id,
    searchRevision: message.searchRevision,
    message,
    state,
  };
}

export function appendUserTurn(snapshot: LabSnapshot, text: string): LabSnapshot {
  const nextIndex = snapshot.messages.length;
  const user = userMessage(`lab-user-${nextIndex}`, text, nextIndex);
  const assistant = assistantMessage(`lab-assistant-${nextIndex + 1}`, "", nextIndex + 1, true);
  return {
    ...snapshot,
    projectionGeneration: snapshot.projectionGeneration + 1,
    sessions: snapshot.sessions.map((session) => ({
      ...session,
      lastMessagePreview: text,
      lastMessageAt: BASE_TIME + (nextIndex + 1) * 60_000,
      isRunning: true,
    })),
    messages: [...snapshot.messages, user, assistant],
    composer: {
      ...snapshot.composer,
      draft: { text: "" },
      isStreaming: true,
      canStop: true,
      canSend: false,
    },
  };
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
    {
      ...assistantMessage(
        "lab-assistant-2",
        "能。左边选择“流式生成”，就会像真实回复一样一段段长出来。Bridge 调用也会被记录下来，方便检查交互。",
        3,
      ),
      blocks: [{
        id: "lab-thinking-1",
        kind: "thinking",
        title: "已检查渲染边界",
        detail: "确认消息、排版和流式状态属于 WebUI；系统能力仍由 Android 拥有。",
        state: "completed",
        durationMillis: 420,
      }, {
        id: "lab-tool-1",
        kind: "tool",
        title: "读取 Mobile Bridge",
        detail: "核对浏览器适配器与 Android 使用相同的方法合同。",
        state: "completed",
        resultPreview: "42 个能力入口已载入",
        durationMillis: 180,
      }],
    },
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
  return message(id, "user", content, index, false);
}

function assistantMessage(id: string, content: string, index: number, streaming = false): LabMessage {
  return message(id, "assistant", content, index, streaming);
}

function message(
  id: string,
  role: "user" | "assistant",
  content: string,
  index: number,
  streaming: boolean,
): LabMessage {
  return {
    id,
    sessionId: SESSION_ID,
    role,
    content,
    createdAt: BASE_TIME + index * 60_000,
    searchRevision: index,
    replyable: true,
    blocks: [],
    streaming,
    interrupted: false,
    attachments: [],
  };
}
