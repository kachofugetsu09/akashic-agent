const BASE_TIME = Date.UTC(2026, 7, 12, 0, 0, 0);
const SESSION_ID = "perf-session";
const PLAIN_SESSION_ID = "perf-session-plain";

export function desktopSessions(messageCount = 100) {
  return {
    items: [
      {
        key: SESSION_ID,
        updated_at: new Date(BASE_TIME).toISOString(),
        message_count: messageCount,
        first_message_content: "性能基线会话",
      },
      {
        key: PLAIN_SESSION_ID,
        updated_at: new Date(BASE_TIME - 1_000).toISOString(),
        message_count: messageCount,
        first_message_content: "纯文本性能会话",
      },
    ],
    next_cursor: null,
    total: 2,
  };
}

export function desktopMessages(count = 100, { profile = "rich", sessionId = SESSION_ID } = {}) {
  return {
    items: Array.from({ length: count }, (_, index) => {
      const input = index % 2 === 0;
      const parts = [{ kind: "text", value: profile === "plain" ? plainFixtureContent(index) : fixtureContent(index) }];
      if (profile === "rich" && index === count - 1) parts.push({ kind: "reply_ref", value: `desktop-${profile}-10` });
      return {
        id: `desktop-${profile}-${index}`,
        seq: index,
        session_id: sessionId,
        timestamp: new Date(BASE_TIME + index * 1_000).toISOString(),
        author: input ? "user" : "assistant",
        source: "akashic",
        attachments: [],
        metadata: {},
        body: input
          ? { kind: "input", parts }
          : { kind: "output", parts, finish: "complete" },
      };
    }),
  };
}

export function desktopModels(count = 48) {
  const runtimes = Array.from({ length: count }, (_, index) => ({
    id: index === 0 ? "perf/runtime" : `perf/runtime-${index}`,
    provider: index % 2 === 0 ? "fixture" : "openrouter",
    model: index === 0 ? "fixture" : `fixture-${index}`,
    sourceId: index % 2 === 0 ? "performance" : "catalog",
    sourceName: index % 2 === 0 ? "性能夹具" : "OpenRouter",
    reasoningEffort: "medium",
    supportedReasoningEfforts: ["low", "medium", "high"],
    roles: ["default"],
  }));
  return {
    generationId: 1,
    defaultRuntime: "perf/runtime",
    sessionOverride: "",
    sessionSelection: { modelRef: "perf/runtime", reasoningEffort: "medium" },
    runtimes,
  };
}

export function desktopRuntimeOverview(pathname) {
  if (pathname === "/api/chat/runtime/documents") {
    return { items: [
      { id: "projectneed", title: "项目需求", relative_path: "docs/projectneed.md", group: "core", description: "长期产品合同", available: true },
      { id: "workflow", title: "工作流", relative_path: "docs/WORKFLOW.md", group: "core", description: "交付流程", available: true },
    ] };
  }
  if (pathname === "/api/chat/runtime/jobs") {
    return { items: [
      { id: "daily-review", name: "每日回顾", trigger: "schedule", tier: "routine", fire_at: new Date(BASE_TIME + 3_600_000).toISOString(), timezone: "Asia/Shanghai", enabled: true, run_count: 4 },
      { id: "weekly-backup", name: "每周备份", trigger: "schedule", tier: "maintenance", fire_at: new Date(BASE_TIME + 7_200_000).toISOString(), timezone: "Asia/Shanghai", enabled: false, run_count: 2 },
    ] };
  }
  if (pathname === "/api/chat/runtime/capabilities") {
    return {
      snapshot_id: "runtime-fixture",
      plugins: [{ id: "fixture-plugin" }],
      skills: [{ id: "fixture-skill" }],
      mcp_servers: [
        { owner_id: "core", name: "filesystem", tool_count: 4 },
        { owner_id: "plugin", name: "calendar", tool_count: 3 },
      ],
    };
  }
  return undefined;
}

export function desktopRuntimeDetail(url) {
  const documentMatch = url.pathname.match(/^\/api\/chat\/runtime\/documents\/([^/]+)$/u);
  if (documentMatch) {
    const id = decodeURIComponent(documentMatch[1]);
    return { title: id === "workflow" ? "工作流" : "项目需求", relative_path: id === "workflow" ? "docs/WORKFLOW.md" : "docs/projectneed.md", markdown: `## ${id}\n\n运行目录详情夹具。` };
  }
  const jobMatch = url.pathname.match(/^\/api\/chat\/runtime\/jobs\/([^/]+)$/u);
  if (jobMatch) {
    const id = decodeURIComponent(jobMatch[1]);
    return { id, name: id === "weekly-backup" ? "每周备份" : "每日回顾", timezone: "Asia/Shanghai", markdown: `## ${id}\n\n定时任务详情夹具。` };
  }
  if (url.pathname === "/api/chat/runtime/mcp") {
    const ownerId = url.searchParams.get("owner_id") ?? "";
    const name = url.searchParams.get("name") ?? "";
    return { owner_id: ownerId, name, markdown: `## ${name}\n\nMCP 详情夹具。` };
  }
  return undefined;
}

/** v11 快照：消息为 Message v2 行，流式草稿放在 replyStatus 预览而不是消息行里。 */
export function mobileSnapshot(count = 300, { streaming = false } = {}) {
  const messages = Array.from({ length: count }, (_, index) => mobileMessage(index));
  return {
    protocolVersion: 11,
    downloads: [],
    throughSeq: count - 1,
    replyStatus: streaming ? mobileReplyStatus(MOBILE_DRAFT_ID, "") : null,
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

export const MOBILE_DRAFT_ID = "mobile-stream-draft";

/** 流式更新走 receiveMessageEvent：reply.status 草稿预览逐帧增长。 */
export function mobileStreamPatch(snapshot, index, delta) {
  return {
    protocolVersion: 1,
    projectionGeneration: snapshot.projectionGeneration,
    event: mobileReplyStatus(MOBILE_DRAFT_ID, delta.repeat(index + 1)),
  };
}

/** 终态由 messages.appended 提交同 id 正式行，随后清空回复活动。 */
export function mobileTerminalPatch(snapshot, content) {
  const seq = snapshot.throughSeq + 1;
  return [
    {
      protocolVersion: 1,
      projectionGeneration: snapshot.projectionGeneration,
      event: {
        type: "messages.appended", version: 2, session_id: SESSION_ID,
        after_seq: snapshot.throughSeq, through_seq: seq, next_after_seq: seq, has_more: false,
        items: [{
          id: MOBILE_DRAFT_ID, seq, session_id: SESSION_ID,
          timestamp: new Date(BASE_TIME + seq * 1_000).toISOString(),
          author: "Akashic", source: "conversation", attachments: [], metadata: {},
          body: { kind: "output", parts: [{ kind: "text", value: content }], finish: "complete" },
        }],
      },
    },
    {
      protocolVersion: 1,
      projectionGeneration: snapshot.projectionGeneration,
      event: {
        type: "reply.status", version: 2, session_id: SESSION_ID, snapshot_id: "perf-final",
        available: true, items: [{ session_id: SESSION_ID, source: "conversation", handle: "perf-reply", active: false, preview: null }],
      },
    },
  ];
}

function mobileReplyStatus(draftId, text) {
  return {
    type: "reply.status", version: 2, session_id: SESSION_ID, snapshot_id: "perf-snap", available: true,
    items: [{
      session_id: SESSION_ID, source: "conversation", handle: "perf-reply", active: true,
      preview: { message_id: draftId, text, thinking: "" },
    }],
  };
}

export const fixtureSessionId = SESSION_ID;
export const plainFixtureSessionId = PLAIN_SESSION_ID;

export function desktopMessagesForSession(sessionId, count = 100) {
  if (sessionId === SESSION_ID) return desktopMessages(count, { profile: "rich", sessionId });
  if (sessionId === PLAIN_SESSION_ID) return desktopMessages(count, { profile: "plain", sessionId });
  return undefined;
}

function mobileMessage(index) {
  const input = index % 2 === 0;
  const parts = [{ kind: "text", value: fixtureContent(index) }];
  return {
    id: `mobile-${index}`,
    seq: index,
    session_id: SESSION_ID,
    timestamp: new Date(BASE_TIME + index * 1_000).toISOString(),
    author: input ? "花月" : "Akashic",
    source: "conversation",
    attachments: [],
    metadata: {},
    body: input
      ? { kind: "input", parts }
      : { kind: "output", parts, finish: "complete" },
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
