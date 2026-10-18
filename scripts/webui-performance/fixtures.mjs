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
  const sources = [
    { provider: "fixture", sourceId: "performance", sourceName: "性能夹具", prefix: "fixture", efforts: ["low", "medium", "high"] },
    { provider: "openrouter", sourceId: "catalog", sourceName: "OpenRouter", prefix: "openrouter", efforts: ["low", "medium", "high", "xhigh"] },
    { provider: "deepseek", sourceId: "deepseek-direct", sourceName: "DeepSeek 直连", prefix: "deepseek", efforts: ["medium", "high"] },
    { provider: "codex", sourceId: "codex-oauth", sourceName: "Codex", prefix: "codex", efforts: [] },
    { provider: "opencode-go", sourceId: "opencode-go", sourceName: "OpenCode Go", prefix: "opencode", efforts: ["low", "medium", "high", "max"] },
  ];
  const runtimes = Array.from({ length: count }, (_, index) => {
    const source = sources[index % sources.length];
    return {
      id: index === 0 ? "perf/runtime" : `perf/runtime-${index}`,
      provider: source.provider,
      model: index === 0 ? "fixture" : `${source.prefix}-model-${Math.floor(index / sources.length)}`,
      sourceId: source.sourceId,
      sourceName: source.sourceName,
      reasoningEffort: source.efforts[1] ?? "medium",
      supportedReasoningEfforts: source.efforts,
      contextWindow: index % 3 === 0 ? 1000000 : 128000 + index * 1000,
      maxOutputTokens: 8192,
      inputModalities: index % 4 === 1 ? ["text", "image"] : ["text"],
      roles: ["default"],
    };
  });
  return {
    generationId: 1,
    defaultRuntime: "perf/runtime",
    sessionOverride: "",
    sessionSelection: { modelRef: "perf/runtime", reasoningEffort: "medium" },
    runtimes,
    unavailableRuntimes: [],
  };
}

export const fixtureSessionId = SESSION_ID;
export const plainFixtureSessionId = PLAIN_SESSION_ID;

export function desktopMessagesForSession(sessionId, count = 100) {
  if (sessionId === SESSION_ID) return desktopMessages(count, { profile: "rich", sessionId });
  if (sessionId === PLAIN_SESSION_ID) return desktopMessages(count, { profile: "plain", sessionId });
  return undefined;
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
