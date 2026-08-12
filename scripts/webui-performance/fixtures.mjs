const BASE_TIME = Date.UTC(2026, 7, 12, 0, 0, 0);
const SESSION_ID = "perf-session";
const PLAIN_SESSION_ID = "perf-session-plain";

export function desktopSessions() {
  return {
    items: [
      {
        key: SESSION_ID,
        updated_at: new Date(BASE_TIME).toISOString(),
        message_count: 100,
        first_message_content: "性能基线会话",
      },
      {
        key: PLAIN_SESSION_ID,
        updated_at: new Date(BASE_TIME - 1_000).toISOString(),
        message_count: 100,
        first_message_content: "纯文本性能会话",
      },
    ],
  };
}

export function desktopMessages(count = 100, { profile = "rich" } = {}) {
  return {
    items: Array.from({ length: count }, (_, index) => ({
      id: `desktop-${profile}-${index}`,
      role: index % 2 === 0 ? "user" : "assistant",
      content: profile === "plain" ? plainFixtureContent(index) : fixtureContent(index),
      timestamp: new Date(BASE_TIME + index * 1_000).toISOString(),
      tool_chain: profile === "rich" && index % 10 === 9 ? [{
        call_id: `tool-${index}`,
        name: "performance_probe",
        status: "success",
        arguments: { index },
        result_preview: `完成 ${index}`,
      }] : [],
      reasoning_content: profile === "rich" && index % 10 === 9 ? `检查第 ${index} 个历史节点。` : "",
      reply_to_message_id: profile === "rich" && index === count - 1 ? `desktop-${profile}-10` : undefined,
      reply_role: profile === "rich" && index === count - 1 ? "user" : undefined,
      reply_preview: profile === "rich" && index === count - 1 ? "性能消息 10" : undefined,
      extra: {},
    })),
  };
}

export function desktopModels() {
  return {
    generationId: 1,
    defaultRuntime: "perf/runtime",
    sessionOverride: "",
    sessionSelection: { modelRef: "perf/runtime", reasoningEffort: "medium" },
    runtimes: [{
      id: "perf/runtime",
      provider: "fixture",
      model: "fixture",
      sourceId: "performance",
      sourceName: "性能夹具",
      reasoningEffort: "medium",
      supportedReasoningEfforts: ["medium"],
      roles: ["default"],
    }],
  };
}

export function mobileSnapshot(count = 300, { streaming = false } = {}) {
  const messages = Array.from({ length: count }, (_, index) => mobileMessage(index));
  if (streaming && messages.length > 0) {
    messages[messages.length - 1] = {
      ...messages[messages.length - 1],
      role: "assistant",
      content: "",
      streaming: true,
      blocks: [],
    };
  }
  return {
    protocolVersion: 8,
    connection: { label: "性能测试", status: "ready" },
    sessions: [{
      id: SESSION_ID,
      title: "性能基线会话",
      lastMessagePreview: "确定性历史夹具",
      lastMessageAt: BASE_TIME + count * 1_000,
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
      commands: [],
      isStreaming: streaming,
      isResyncing: false,
      canResync: true,
      isStopping: false,
      canStop: streaming,
      canSend: !streaming,
    },
    modelCatalog: {
      generationId: 1,
      defaultRuntime: "perf/runtime",
      selectedRuntimeId: "perf/runtime",
      selectedReasoningEffort: "medium",
      runtimes: [{
        id: "perf/runtime",
        provider: "fixture",
        model: "fixture",
        sourceId: "performance",
        sourceName: "性能夹具",
        reasoningEffort: "medium",
        supportedReasoningEfforts: ["medium"],
        roles: ["default"],
        contextWindow: 128_000,
        inputModalities: ["text"],
      }],
      loading: false,
    },
    runtimeInspection: {
      refreshing: false,
      detailLoading: false,
      documents: [],
      jobs: [],
      mcpServers: [],
      pluginCount: 0,
      skillCount: 0,
    },
  };
}

export function mobileStreamPatch(snapshot, index, delta) {
  const messageIndex = snapshot.messages.length - 1;
  const message = snapshot.messages[messageIndex];
  return {
    protocolVersion: 3,
    projectionGeneration: snapshot.projectionGeneration,
    selectedSessionId: snapshot.selectedSessionId,
    messageIndex,
    messageId: message.id,
    searchRevision: index + 1,
    contentAppend: delta,
  };
}

export function mobileTerminalPatch(snapshot, content) {
  const messageIndex = snapshot.messages.length - 1;
  const message = {
    ...snapshot.messages[messageIndex],
    content,
    searchRevision: 601,
    streaming: false,
  };
  const nextSnapshot = {
    ...snapshot,
    sessions: snapshot.sessions.map((session) => ({ ...session, isRunning: false })),
    messages: [...snapshot.messages.slice(0, -1), message],
    composer: {
      ...snapshot.composer,
      isStreaming: false,
      canStop: false,
      canSend: true,
    },
  };
  const { protocolVersion, messages, ...state } = nextSnapshot;
  void protocolVersion;
  void messages;
  return {
    protocolVersion: 3,
    projectionGeneration: snapshot.projectionGeneration,
    selectedSessionId: snapshot.selectedSessionId,
    messageIndex,
    messageId: message.id,
    searchRevision: message.searchRevision,
    message,
    state: { protocolVersion: 1, ...state },
  };
}

export const fixtureSessionId = SESSION_ID;
export const plainFixtureSessionId = PLAIN_SESSION_ID;

export function desktopMessagesForSession(sessionId, count = 100) {
  if (sessionId === SESSION_ID) return desktopMessages(count, { profile: "rich" });
  if (sessionId === PLAIN_SESSION_ID) return desktopMessages(count, { profile: "plain" });
  return undefined;
}

function mobileMessage(index) {
  return {
    id: `mobile-${index}`,
    sessionId: SESSION_ID,
    role: index % 2 === 0 ? "user" : "assistant",
    content: fixtureContent(index),
    createdAt: BASE_TIME + index * 1_000,
    searchRevision: index,
    replyable: true,
    blocks: index % 10 === 9 ? [{
      id: `block-${index}`,
      kind: "thinking",
      title: "已思考",
      detail: `检查第 ${index} 个历史节点。`,
      state: "completed",
      durationMillis: 12,
    }] : [],
    streaming: false,
    interrupted: false,
    attachments: [],
  };
}

function fixtureContent(index) {
  if (index % 10 === 9) {
    return `## 性能节点 ${index}\n\n- 保持消息身份稳定\n- 避免无关组件重绘\n\n\`\`\`ts\nconst sample = ${index};\n\`\`\``;
  }
  return `性能消息 ${index}：用于稳定覆盖中文段落、换行和连续历史渲染。`;
}

function plainFixtureContent(index) {
  return `纯文本消息 ${index}：稳定覆盖连续中文流与普通段落。`;
}
