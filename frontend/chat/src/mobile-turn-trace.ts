/**
 * Mobile WebView 端 turn 级低噪声观测：每 turn 每 kind 只记录一次的里程碑注册表。
 *
 * 1. 身份固定 session_id + turn_id + client_message_id，缺失显式 missing，不猜测。
 * 2. 输出单行 [akashic-trace] JSON（无正文）；身份冲突降级为一次性
 *    webui.identity_conflict 诊断，不阻断业务。
 */

export const MOBILE_TURN_MISSING = "missing";

/** 注册表有界上限：超限按插入序淘汰最旧 turn，避免长时间会话无界增长。 */
export const MOBILE_TURN_TRACE_MAX_TRACKED = 64;

export const MOBILE_TURN_TRACE_EVENTS = [
  "webui.patch_received",
  "webui.patch_applied",
  "webui.react_committed",
  "webui.next_frame_ready",
  "webui.identity_conflict",
  "webui.trace_sink_error",
] as const;

export type MobileTurnTraceEvent = (typeof MOBILE_TURN_TRACE_EVENTS)[number];

/** 可见 source 类型：thinking 先于 answer，终态兜底；同 patch 可同时引入多项。 */
export type MobileTurnSourceKind = "thinking" | "answer" | "terminal";

export interface MobileTurnTraceRecord {
  event: MobileTurnTraceEvent;
  session_id: string;
  turn_id: string;
  client_message_id: string;
  wall_ms: number;
  performance_ms: number;
  kind: string;
  origin: string;
  /** 仅身份冲突诊断携带：incoming 的结构化身份字段，不含正文。 */
  incoming_client_message_id?: string;
  /** 仅 sink 失败诊断携带：异常类型，不包含异常消息或业务正文。 */
  error_type?: string;
}

export type MobileTurnTraceEmit = (record: MobileTurnTraceRecord) => void;

/** 默认 sink：单行 [akashic-trace] 前缀 + 固定字段 JSON，便于 logcat join。 */
export const mobileTurnTraceEmit: MobileTurnTraceEmit = (record) => {
  // 1. 单行输出；字段固定，不含 message content / prompt / tool args
  console.log(`[akashic-trace] ${JSON.stringify(record)}`);
};

/** 从既有 assistant:<turn> messageId 合同解析 turn_id；非合同 ID 或空 turn 返回 undefined。 */
export function parseMobileTurnId(messageId: string): string | undefined {
  const prefix = "assistant:";
  return messageId.startsWith(prefix) && messageId.length > prefix.length
    ? messageId.slice(prefix.length)
    : undefined;
}

/** 可见性探针：只保留判定可见 source 所需的正文与 thinking 块文本。 */
export interface MobileTurnSourceProbe {
  content: string;
  thinking: string[];
}

/** pure helper 消费的结构化 patch 投影，不含业务字段与正文之外内容。 */
export interface MobileTurnPatchProbe {
  contentAppend?: string;
  thinkingAppend?: { blockIndex: number; delta: string };
  message?: { content: string; thinking: string[]; streaming: boolean };
  terminal?: boolean;
}

/**
 * 判定本 patch 首次引入的可见 source kinds 数组（顺序 thinking、answer、terminal）。
 */
export function mobileTurnFirstVisibleKinds(
  previous: MobileTurnSourceProbe | undefined,
  patch: MobileTurnPatchProbe,
): MobileTurnSourceKind[] {
  // 1. 计算本 patch 后的可见 thinking/answer 状态
  const nextThinkingVisible = patch.message !== undefined
    ? patch.message.thinking.some((detail) => detail !== "")
    : patch.thinkingAppend !== undefined && patch.thinkingAppend.delta !== "";
  const nextAnswerVisible = patch.message !== undefined
    ? patch.message.content !== ""
    : patch.contentAppend !== undefined && patch.contentAppend !== "";
  const previousThinkingVisible = previous !== undefined
    && previous.thinking.some((detail) => detail !== "");
  const previousAnswerVisible = previous !== undefined && previous.content !== "";
  const terminal = patch.terminal === true
    || (patch.message !== undefined && !patch.message.streaming);
  // 2. 独立比较 thinking / answer / terminal，顺序固定
  const kinds: MobileTurnSourceKind[] = [];
  if (!previousThinkingVisible && nextThinkingVisible) kinds.push("thinking");
  if (!previousAnswerVisible && nextAnswerVisible) kinds.push("answer");
  if (terminal) kinds.push("terminal");
  return kinds;
}

/** 规范化身份：session_id + turn_id + client_message_id，缺失部分显式 missing。 */
export interface MobileTurnTraceIdentity {
  sessionId: string;
  turnId: string;
  clientMessageId: string;
  key: string;
}

interface MobileTurnTraceEntry {
  /** 首注册 turn_id：markFirst 按 entry 发，别名键沿用同一 turn。 */
  turnId: string;
  clientMessageId: string;
  milestones: Set<string>;
  conflictReported: Set<string>;
  aliasConflictReported: Set<string>;
}

function mobileTurnTraceKey(sessionId: string, turnId: string | undefined): string {
  return `${sessionId}\u001f${turnId ?? MOBILE_TURN_MISSING}`;
}

function mobileTurnTraceAliasKey(sessionId: string, messageId: string): string {
  return `${sessionId}\u001f${messageId}`;
}

function mobileTurnMilestoneKey(event: MobileTurnTraceEvent, kind: string): string {
  return `${event}\u001f${kind}`;
}

/** 有界内存里程碑注册表：primary 键 (session, turn)，aliases 映射完整 messageId → primary。 */
export class MobileTurnTraceRegistry {
  private readonly entries = new Map<string, MobileTurnTraceEntry>();
  private readonly aliases = new Map<string, string>();
  private readonly degradedReported = new Set<string>();
  private readonly emit: MobileTurnTraceEmit;

  constructor(emit: MobileTurnTraceEmit = mobileTurnTraceEmit) {
    this.emit = emit;
  }

  /** 观测 sink 失败只写 content-free 降级诊断，绝不阻断消息投影与渲染。 */
  private safeEmit(record: MobileTurnTraceRecord): void {
    try {
      this.emit(record);
    } catch (error) {
      const errorType = error instanceof Error ? error.name : typeof error;
      console.error(`[akashic-trace] ${JSON.stringify({
        event: "webui.trace_sink_error",
        session_id: record.session_id,
        turn_id: record.turn_id,
        client_message_id: record.client_message_id,
        wall_ms: Date.now(),
        performance_ms: performance.now(),
        kind: record.kind,
        origin: "turn-trace-registry",
        error_type: errorType,
      })}`);
    }
  }

  /** 注册 (session, turn) primary 并返回规范化身份；冲突保留首次非缺失值，不抛。 */
  registerTurnIdentity(
    sessionId: string,
    turnId: string | undefined,
    clientMessageId: string | undefined,
  ): MobileTurnTraceIdentity {
    const key = mobileTurnTraceKey(sessionId, turnId);
    let entry = this.entries.get(key);
    if (entry === undefined) {
      // 1. 有界淘汰：达到上限时移除最旧 turn（连同指向它的别名）后再登记
      if (this.entries.size >= MOBILE_TURN_TRACE_MAX_TRACKED) {
        const oldest = this.entries.keys().next().value;
        if (oldest !== undefined) this.evictPrimary(oldest);
      }
      entry = {
        turnId: turnId ?? MOBILE_TURN_MISSING,
        clientMessageId: MOBILE_TURN_MISSING,
        milestones: new Set(),
        conflictReported: new Set(),
        aliasConflictReported: new Set(),
      };
      this.entries.set(key, entry);
    }
    // 2. 首次缺失可补齐；非缺失值彼此冲突保留已存值
    if (entry.clientMessageId === MOBILE_TURN_MISSING) {
      if (clientMessageId !== undefined) entry.clientMessageId = clientMessageId;
    } else if (clientMessageId !== undefined && clientMessageId !== entry.clientMessageId) {
      // 3. 降级观测：同一 turn+incoming 组合只发一次 content-free 诊断，不阻断业务
      if (!entry.conflictReported.has(clientMessageId)) {
        entry.conflictReported.add(clientMessageId);
        this.safeEmit({
          event: "webui.identity_conflict",
          session_id: sessionId,
          turn_id: turnId ?? MOBILE_TURN_MISSING,
          client_message_id: entry.clientMessageId,
          incoming_client_message_id: clientMessageId,
          wall_ms: Date.now(),
          performance_ms: performance.now(),
          kind: "identity",
          origin: "turn-trace-registry",
        });
      }
    }
    return {
      sessionId,
      turnId: turnId ?? MOBILE_TURN_MISSING,
      clientMessageId: entry.clientMessageId,
      key,
    };
  }

  /** 淘汰 primary 并清理指向它的 aliases，保持有界且不残留旧绑定。 */
  private evictPrimary(key: string): void {
    this.entries.delete(key);
    // 1. 清理指向被淘汰 primary 的别名；别名本身不计入 tracked-turn 上限
    for (const [aliasKey, primaryKey] of this.aliases) {
      if (primaryKey === key) this.aliases.delete(aliasKey);
    }
  }

  /** 把完整 messageId 绑定为 primary 的别名；与另一仍存 primary 冲突时发 content-free 诊断且不改绑。 */
  bindMessageIdentity(
    sessionId: string,
    messageId: string,
    identity: MobileTurnTraceIdentity,
  ): MobileTurnTraceIdentity | undefined {
    if (!this.entries.has(identity.key)) {
      // 1. 纯观测 source 已被淘汰：结构化降级，绝不阻断 terminal patch。
      this.reportDegraded(sessionId, messageId, identity, "stale_source");
      return undefined;
    }
    const aliasKey = mobileTurnTraceAliasKey(sessionId, messageId);
    const existing = this.aliases.get(aliasKey);
    if (existing !== undefined && existing !== identity.key) {
      // 2. 已指向另一仍存 primary：fail-loud 且绝不静默改绑
      if (this.entries.has(existing)) {
        this.reportAliasConflict(sessionId, messageId, existing);
        return this.primaryIdentity(sessionId, existing)!;
      }
      // 3. stale alias 只影响观测：清理后返回 undefined，不改写业务消息。
      this.aliases.delete(aliasKey);
      this.reportDegraded(sessionId, messageId, identity, "stale_alias");
      return undefined;
    }
    this.aliases.set(aliasKey, identity.key);
    return this.primaryIdentity(sessionId, identity.key)!;
  }

  /** 别名冲突诊断：同一 alias 已指向另一仍存 primary，每 (primary, alias) 组合只发一次。 */
  private reportAliasConflict(sessionId: string, messageId: string, primaryKey: string): void {
    const entry = this.entries.get(primaryKey);
    const aliasKey = mobileTurnTraceAliasKey(sessionId, messageId);
    if (entry === undefined || entry.aliasConflictReported.has(aliasKey)) return;
    entry.aliasConflictReported.add(aliasKey);
    this.safeEmit({
      event: "webui.identity_conflict",
      session_id: sessionId,
      turn_id: entry.turnId,
      client_message_id: entry.clientMessageId,
      incoming_client_message_id: messageId,
      wall_ms: Date.now(),
      performance_ms: performance.now(),
      kind: "alias",
      origin: "turn-trace-registry",
    });
  }

  /** 观测 registry 自身退化时打一条有界、无正文诊断。 */
  private reportDegraded(
    sessionId: string,
    messageId: string,
    identity: MobileTurnTraceIdentity,
    kind: "stale_source" | "stale_alias",
  ): void {
    const reportKey = `${kind}\u001f${identity.key}\u001f${messageId}`;
    if (this.degradedReported.has(reportKey)) return;
    if (this.degradedReported.size >= MOBILE_TURN_TRACE_MAX_TRACKED * 2) {
      const oldest = this.degradedReported.values().next().value;
      if (oldest !== undefined) this.degradedReported.delete(oldest);
    }
    this.degradedReported.add(reportKey);
    this.safeEmit({
      event: "webui.identity_conflict",
      session_id: sessionId,
      turn_id: identity.turnId,
      client_message_id: identity.clientMessageId,
      incoming_client_message_id: messageId,
      wall_ms: Date.now(),
      performance_ms: performance.now(),
      kind,
      origin: "turn-trace-registry",
    });
  }

  /** 行侧解析：临时 assistant id 解析 turn primary；canonical id 解析完整 id 别名。 */
  identityForMessage(sessionId: string, messageId: string): MobileTurnTraceIdentity | undefined {
    // 1. 临时 assistant id 优先走 turn primary
    const turnId = parseMobileTurnId(messageId);
    if (turnId !== undefined) return this.identityFor(sessionId, turnId);
    // 2. canonical id 通过完整 id 别名解析；未注册返回 undefined（不猜测）
    const aliasKey = mobileTurnTraceAliasKey(sessionId, messageId);
    const primaryKey = this.aliases.get(aliasKey);
    if (primaryKey === undefined) return undefined;
    // 3. stale alias 是纯观测退化：清理并返回 undefined，业务 render 继续。
    if (!this.entries.has(primaryKey)) {
      this.aliases.delete(aliasKey);
      this.reportDegraded(
        sessionId,
        messageId,
        {
          sessionId,
          turnId: MOBILE_TURN_MISSING,
          clientMessageId: MOBILE_TURN_MISSING,
          key: primaryKey,
        },
        "stale_alias",
      );
      return undefined;
    }
    return this.primaryIdentity(sessionId, primaryKey);
  }

  /** 行侧只读查询：返回该 turn 的规范化身份；未注册（含被有界淘汰）返回 undefined。 */
  identityFor(sessionId: string, turnId: string | undefined): MobileTurnTraceIdentity | undefined {
    return this.primaryIdentity(sessionId, mobileTurnTraceKey(sessionId, turnId));
  }

  /** 该身份是否仍被追踪（未注册或已被有界淘汰即 false）。 */
  tracks(key: string): boolean {
    return this.entries.has(key);
  }

  /**
   * 每身份每事件每 kind 只标记一次；首次标记返回 true 并发出观测记录。
   */
  markFirst(
    identity: MobileTurnTraceIdentity,
    event: MobileTurnTraceEvent,
    kind: string,
    origin: string,
  ): boolean {
    const entry = this.entries.get(identity.key);
    if (entry === undefined) return false;
    const milestoneKey = mobileTurnMilestoneKey(event, kind);
    if (entry.milestones.has(milestoneKey)) return false;
    entry.milestones.add(milestoneKey);
    this.safeEmit({
      event,
      session_id: identity.sessionId,
      // 发 entry 当前 clientMessageId，而非捕获 identity 的快照值：先渲染后补齐
      // （missing → real）或旧 rAF 闭包持有的旧 identity 都不带旧 id 上报；
      // turn_id 同理取 entry 首注册值，别名键（terminal 迁移）沿用同一 turn。
      turn_id: entry.turnId,
      client_message_id: entry.clientMessageId,
      wall_ms: Date.now(),
      performance_ms: performance.now(),
      kind,
      origin,
    });
    return true;
  }

  private primaryIdentity(sessionId: string, key: string): MobileTurnTraceIdentity | undefined {
    const entry = this.entries.get(key);
    if (entry === undefined) return undefined;
    return {
      sessionId,
      turnId: entry.turnId,
      clientMessageId: entry.clientMessageId,
      key,
    };
  }
}
