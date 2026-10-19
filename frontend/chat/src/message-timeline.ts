import { toolDataText } from "./message-rendering-policy";

export interface TimelineAttachment {
  artifact_id: string;
  kind: "file" | "image";
  filename: string | null;
  media_type: string | null;
  size_bytes: number;
  sha256: string;
}

export type TimelinePart =
  | { kind: "text" | "artifact_ref" | "reply_ref"; value: string }
  | { kind: "tool_call"; binding_id: string; name: string; owner?: string; arguments: Record<string, unknown> }
  | { kind: "context.notice"; value: { text: string } }
  | { kind: "history.provenance" | "history.transcript" | "history.record" | "history.turn_input"; archive: unknown }
  | { kind: string; display: "unavailable" | "hidden" }
  // owner 声明的思考过程：宿主只按通用思考行展示 text，不解读 owner 的 kind 与字段。
  | { kind: string; display: "thinking"; text: string | null }
  | { kind: string; display: "data"; value: unknown; rendered?: unknown };

export type TimelineBody =
  | { kind: "input"; parts: TimelinePart[] }
  | { kind: "output"; parts: TimelinePart[]; finish: "continue" | "complete" | "quiet" }
  | { kind: "tool_result"; parts: TimelinePart[]; call_ref: { message_id: string; part_index: number }; outcome: "success" | "denied" | "error" | "interrupted" }
  | { kind: "control"; action: "pause" | "resume" | "abandon" | "failure"; through_seq: number; reason: string | null };

/** 两端共用的展示合同；不把正文类型当作作者身份。 */
export interface TimelineMessage {
  id: string;
  session_id: string;
  seq: number;
  timestamp: string;
  author: string;
  source: string;
  attachments: TimelineAttachment[];
  body: TimelineBody;
  metadata: Record<string, unknown>;
}

export interface TimelineReply {
  id: string;
  author: string;
  preview: string;
}

/** 重试中的尝试进度：服务端时间戳为 Unix 秒，前端自己计时。 */
export interface ModelAttempt {
  attempt: number;
  limit: number | null;
  failing_since: number;
  last_failure: string;
  detail: string;
  next_attempt_at: number | null;
}

export interface ReplyActivity {
  session_id: string;
  source: string;
  handle: string;
  active: boolean;
  preview: { message_id: string; text: string; thinking: string; call_record_id?: string | null; retry_status?: string;
    attempt?: ModelAttempt | null; truncated?: boolean;
    /** 本次尝试的首字耗时（毫秒），服务端在首段增量时给出；未出字时为空。 */
    first_token_ms?: number | null;
    /** 本次尝试已到达的非空正文/思考增量数；生成中按它估算速度。 */
    output_deltas?: number;
    /** 从首段增量到最近一次增量的解码时长（毫秒）；未出字时为空。 */
    decode_ms?: number | null } | null;
}

export type MessageLogFrame =
  | { type: "session.following"; version: 2; session_id: string; through_seq: number; request_id: string }
  | { type: "messages.appended"; version: 2; session_id: string; after_seq: number; through_seq: number;
      next_after_seq: number; has_more: boolean; items: TimelineMessage[] }
  | { type: "reply.status"; version: 2; session_id: string; snapshot_id: string | null;
      available: boolean; items: ReplyActivity[] };

/** 实时事实与当前草稿分别验证；未识别的消息交给其他协议入口。 */
export function readMessageLogFrame(value: unknown): MessageLogFrame | null {
  const frame = object(value);
  if (!frame || !["session.following", "messages.appended", "reply.status"].includes(String(frame.type))) return null;
  if (frame.version !== 2 || !nonempty(frame.session_id)) throw new Error("实时消息协议版本或会话无效");
  if (frame.type === "session.following") {
    if (!cursor(frame.through_seq) || !nonempty(frame.request_id)) throw new Error("消息订阅确认无效");
  } else if (frame.type === "messages.appended") {
    if (!cursor(frame.after_seq) || !cursor(frame.through_seq) || !integer(frame.next_after_seq)
      || typeof frame.has_more !== "boolean" || !Array.isArray(frame.items) || !frame.items.length) {
      throw new Error("实时消息页无效");
    }
    const items = frame.items.map(readTimelineMessage);
    let seq = frame.after_seq as number;
    for (const row of items) {
      if (row.session_id !== frame.session_id || row.seq <= seq || row.seq > (frame.through_seq as number)) {
        throw new Error("实时消息页顺序或会话无效");
      }
      seq = row.seq;
    }
    if (seq !== frame.next_after_seq || frame.has_more !== (seq < (frame.through_seq as number))) {
      throw new Error("实时消息页游标无效");
    }
    return { ...frame, items } as unknown as MessageLogFrame;
  } else {
    if (typeof frame.available !== "boolean" || !Array.isArray(frame.items)
      || !(nonempty(frame.snapshot_id) || (frame.snapshot_id === null && !frame.available))
      || (!frame.available && frame.items.length)) throw new Error("回复状态无效");
    const handles = new Set<string>();
    const previews = new Set<string>();
    for (const value of frame.items) {
      const item = object(value);
      if (!item || item.session_id !== frame.session_id || !nonempty(item.handle) || !nonempty(item.source)
        || typeof item.active !== "boolean" || handles.has(item.handle)) throw new Error("回复活动无效");
      handles.add(item.handle);
      if (item.preview !== null) {
        const preview = object(item.preview);
        if (preview?.retry_status !== undefined && typeof preview.retry_status !== "string") throw new Error("模型重试状态无效");
        if (preview?.attempt !== undefined && preview.attempt !== null && !validAttempt(preview.attempt)) throw new Error("模型尝试进度无效");
        if (preview?.first_token_ms !== undefined && preview.first_token_ms !== null
          && !(typeof preview.first_token_ms === "number" && Number.isFinite(preview.first_token_ms) && preview.first_token_ms >= 0)) throw new Error("首字耗时无效");
        if (preview?.output_deltas !== undefined
          && !(typeof preview.output_deltas === "number" && Number.isSafeInteger(preview.output_deltas) && preview.output_deltas >= 0)) throw new Error("输出增量无效");
        if (preview?.decode_ms !== undefined && preview.decode_ms !== null
          && !(typeof preview.decode_ms === "number" && Number.isFinite(preview.decode_ms) && preview.decode_ms >= 0)) throw new Error("解码时长无效");
        if (!item.active || !preview || !nonempty(preview.message_id) || typeof preview.text !== "string"
          || typeof preview.thinking !== "string" || (preview.call_record_id !== undefined && preview.call_record_id !== null && !nonempty(preview.call_record_id)) || (preview.truncated !== undefined && typeof preview.truncated !== "boolean") || previews.has(preview.message_id)) throw new Error("回复草稿无效");
        previews.add(preview.message_id);
      }
    }
  }
  return frame as unknown as MessageLogFrame;
}

function validAttempt(value: unknown): boolean {
  const item = object(value);
  if (!item) return false;
  const finite = (field: unknown) => typeof field === "number" && Number.isFinite(field);
  return integer(item.attempt) && (item.limit === null || integer(item.limit))
    && finite(item.failing_since) && (item.next_attempt_at === null || finite(item.next_attempt_at))
    && typeof item.last_failure === "string" && typeof item.detail === "string";
}

/** 在 HTTP/桥接入口一次校验，组件只读取已验证的展示数据。 */
export function readTimelineMessage(value: unknown): TimelineMessage {
  // 1. 核对独立消息的身份、附件和正文。
  const row = object(value);
  if (!row || !nonempty(row.id) || !nonempty(row.session_id) || !integer(row.seq)
    || !nonempty(row.timestamp) || !Number.isFinite(Date.parse(row.timestamp))
    || !nonempty(row.author) || !nonempty(row.source)
    || !Array.isArray(row.attachments) || !row.attachments.every(validAttachment)) {
    throw new Error("历史消息身份或附件无效");
  }
  const body = object(row.body);
  if (row.metadata !== undefined && !object(row.metadata)) throw new Error("消息 metadata 必须是对象");
  if (!body) throw new Error("历史消息缺少正文");
  if (body.kind === "control") {
    if (!["pause", "resume", "abandon", "failure"].includes(String(body.action))
      || !integer(body.through_seq) || !nullableText(body.reason)) throw new Error("控制记录无效");
  } else {
    if (!["input", "output", "tool_result"].includes(String(body.kind))
      || !Array.isArray(body.parts) || !body.parts.every(validPart)) throw new Error("历史消息内容无效");
    if (body.kind === "output" && !["continue", "complete", "quiet"].includes(String(body.finish))) {
      throw new Error("输出结束状态无效");
    }
    if (body.parts.some((part) => object(part)?.kind === "tool_call")
      && (body.kind !== "output" || body.finish !== "continue")) throw new Error("工具调用所在消息无效");
    if (body.kind === "tool_result") {
      const ref = object(body.call_ref);
      if (!ref || !nonempty(ref.message_id) || !integer(ref.part_index)
        || !["success", "denied", "error", "interrupted"].includes(String(body.outcome))) throw new Error("工具结果引用或状态无效");
    }
    if (body.kind !== "tool_result" && body.parts.some((part) => object(part)?.display === "data")) {
      throw new Error("通用工具数据只能出现在工具结果中");
    }
    // 2. 附件引用必须能从本行元数据解析，不猜存储路径。
    const ids = new Set(row.attachments.map((item) => (item as TimelineAttachment).artifact_id));
    if (body.parts.some((part) => object(part)?.kind === "artifact_ref" && !ids.has(object(part)?.value as string))) {
      throw new Error("历史消息附件引用缺少元数据");
    }
  }
  return (row.metadata === undefined ? { ...row, metadata: {} } : row) as unknown as TimelineMessage;
}

/** 保持 seq 顺序并发现跨页身份冲突；正常重叠只保留一份。 */
export function mergeTimelineMessages(current: TimelineMessage[], incoming: TimelineMessage[]): TimelineMessage[] {
  const byId = new Map(current.map((row) => [row.id, row]));
  const bySeq = new Map(current.map((row) => [row.seq, row.id]));
  const sessionId = current[0]?.session_id ?? incoming[0]?.session_id;
  for (const row of incoming) {
    const prior = byId.get(row.id);
    if (row.session_id !== sessionId || (prior && prior.seq !== row.seq)
      || (bySeq.has(row.seq) && bySeq.get(row.seq) !== row.id)) throw new Error("历史消息分页身份冲突");
    if (prior) {
      const { metadata: before, ...oldFacts } = prior;
      const { metadata: after, ...newFacts } = row;
      if (JSON.stringify(messageFacts(oldFacts)) !== JSON.stringify(messageFacts(newFacts)) || JSON.stringify(before) !== JSON.stringify(after)) {
        throw new Error("历史消息正文发生变化");
      }
    }
    byId.set(row.id, row);
    bySeq.set(row.seq, row.id);
  }
  return [...byId.values()].sort((left, right) => left.seq - right.seq);
}

/** 引用展示可以随可用性刷新；原内容、身份与顺序仍必须一致。 */
function messageFacts(message: Omit<TimelineMessage, "metadata">) {
  if (message.body.kind !== "tool_result") return message;
  return { ...message, body: { ...message.body, parts: message.body.parts.map((part) => {
    if (!("display" in part) || part.display !== "data") return part;
    const { rendered, ...facts } = part;
    return facts;
  }) } };
}

export function timelineText(message: TimelineMessage): string {
  if (message.body.kind === "tool_result") return toolResultValues(message).map(toolDataText).join("\n");
  return message.body.kind === "control" ? message.body.reason ?? "" : message.body.parts
    .flatMap((part) => !("display" in part) && part.kind === "text" ? [part.value]
      : !("display" in part) && part.kind === "context.notice" ? [part.value.text] : []).join("\n");
}

/** 工具数据只交给通用结果视图，不把结构化内容伪装成聊天正文。 */
export function timelineToolOutput(message: TimelineMessage): unknown {
  if (message.body.kind !== "tool_result") return undefined;
  const values = toolResultValues(message);
  return values.length === 1 ? values[0] : values;
}

function toolResultValues(message: TimelineMessage): unknown[] {
  if (message.body.kind !== "tool_result") return [];
  return message.body.parts.flatMap((part): unknown[] => {
    if ("display" in part) return part.display === "data" ? ["rendered" in part ? part.rendered : part.value] : [];
    return part.kind === "text" ? [part.value] : [];
  });
}

export function timelineReply(message: TimelineMessage): TimelineReply {
  return { id: message.id, author: message.author === "legacy-attribution-unknown" ? "原消息" : message.author,
    preview: timelineText(message).replace(/\s+/gu, " ").trim().slice(0, 512)
      || (message.attachments.length ? "[附件]" : "[无文字消息]") };
}

const BUILTIN_PARTS: readonly unknown[] = ["text", "artifact_ref", "reply_ref", "tool_call"];

function validPart(value: unknown): boolean {
  const part = object(value);
  if (!part || !nonempty(part.kind)) return false;
  if (part.display === "data") return "value" in part && part.kind !== "tool_call";
  if (part.display === "thinking") return nullableText(part.text) && !BUILTIN_PARTS.includes(part.kind);
  if (part.display === "unavailable" || part.display === "hidden") return !BUILTIN_PARTS.includes(part.kind);
  switch (part.kind) {
    case "text": return typeof part.value === "string";
    case "artifact_ref": case "reply_ref": return nonempty(part.value);
    case "tool_call": return nonempty(part.binding_id) && nonempty(part.name) && object(part.arguments) !== null
      && (part.owner === undefined || nonempty(part.owner));
    case "context.notice": return nonempty(object(part.value)?.text);
    case "history.provenance": case "history.transcript": case "history.record": case "history.turn_input":
      return "archive" in part;
    default: return false;
  }
}

function validAttachment(value: unknown): boolean {
  const item = object(value);
  return item !== null && nonempty(item.artifact_id) && (item.kind === "file" || item.kind === "image")
    && nullableText(item.filename) && nullableText(item.media_type) && integer(item.size_bytes)
    && typeof item.sha256 === "string" && /^[0-9a-f]{64}$/u.test(item.sha256);
}
function object(value: unknown): Record<string, unknown> | null {
  return typeof value === "object" && value !== null && !Array.isArray(value) ? value as Record<string, unknown> : null;
}
function nonempty(value: unknown): value is string { return typeof value === "string" && value.length > 0; }
function nullableText(value: unknown): boolean { return value === null || typeof value === "string"; }
function integer(value: unknown): boolean { return typeof value === "number" && Number.isSafeInteger(value) && value >= 0; }
function cursor(value: unknown): boolean { return value === -1 || integer(value); }

export interface HistoryToolRecord {
  name: string;
  arguments: unknown;
  result: unknown;
}

export interface HistoryTranscriptGroup {
  text: string;
  thinking: string;
  calls: HistoryToolRecord[];
}

/** 只解释已知旧工具记录格式；未知记录保留在原 Message，不猜执行状态。 */
export function historyTranscript(archive: unknown): HistoryTranscriptGroup[] | null {
  const value = object(archive);
  if (!value || value.schema !== "sessions.messages.tool_chain.v0" || typeof value.raw !== "string") return null;
  let groups: unknown;
  try { groups = JSON.parse(value.raw); }
  catch (error) { if (error instanceof SyntaxError) return null; throw error; }
  if (!Array.isArray(groups)) return null;
  const result: HistoryTranscriptGroup[] = [];
  for (const item of groups) {
    const group = object(item);
    if (!group || !Array.isArray(group.calls)
      || (group.text != null && typeof group.text !== "string")
      || (group.reasoning_content != null && typeof group.reasoning_content !== "string")) return null;
    const calls: HistoryToolRecord[] = [];
    for (const item of group.calls) {
      const call = object(item);
      if (!call || typeof call.name !== "string" || !call.name) return null;
      calls.push({ name: call.name, arguments: call.arguments, result: call.result });
    }
    result.push({ text: group.text as string ?? "", thinking: group.reasoning_content as string ?? "", calls });
  }
  return result;
}

/** 聊天可见性只影响布局，原始 Message、part index 和同步 seq 不变。 */
export function isTimelinePartVisible(part: TimelinePart): boolean {
  // owner 声明为内部材料（hidden）的不显示；展示方缺席（unavailable）仍占一行“无法展示”，保留诊断线索。
  if ("display" in part) return part.display === "thinking" ? Boolean(part.text) : part.display !== "hidden";
  if ("archive" in part) {
    if (part.kind !== "history.transcript") return false;
    const groups = historyTranscript(part.archive);
    return groups === null || groups.some((group) => group.text || group.thinking || group.calls.length);
  }
  if (part.kind === "text") return part.value.length > 0;
  return true;
}

// 过程与指针类 part（工具调用、思考、上下文提示、回复引用）不算正文；附件与其它可见 part 都算。
const PROCESS_PARTS = new Set(["tool_call", "context.notice", "reply_ref"]);
function hasReplyBody(message: TimelineMessage): boolean {
  if (message.attachments.length > 0) return true;
  return message.body.kind !== "control" && message.body.parts.some((part) =>
    !PROCESS_PARTS.has(part.kind) && !("display" in part && part.display === "thinking") && isTimelinePartVisible(part));
}

// 工具结果里的图片是给模型看的观察（截图、读图），跟随所属调用；文件附件才是交给用户的回复产物。
export function timelineDeliversAttachment(message: TimelineMessage): boolean {
  return message.body.kind === "tool_result" && message.attachments.some((ref) => ref.kind === "file");
}

export function isTimelineMessageVisible(message: TimelineMessage): boolean {
  return message.attachments.length > 0 || message.body.kind === "control"
    || message.body.kind === "tool_result" || message.body.parts.some(isTimelinePartVisible);
}

/** 隐藏行沿原始顺序定位后续可见行；末尾隐藏行使用前一可见行。 */
export function timelineAnchorIndexes(messages: TimelineMessage[], groups = timelineReplyGroups(messages)): Map<string, number> {
  const visibleIds = new Set(timelineVisibleMessages(messages, groups).map((message) => message.id));
  const indexes = new Map<string, number>();
  const pending: string[] = [];
  let index = -1;
  for (const message of messages) {
    pending.push(message.id);
    if (!visibleIds.has(message.id)) continue;
    index += 1;
    for (const id of pending) indexes.set(id, index);
    pending.length = 0;
  }
  if (index >= 0) for (const id of pending) indexes.set(id, index);
  for (const [ending, members] of groups.completed) {
    const target = indexes.get(ending);
    if (target !== undefined) for (const member of members) indexes.set(member.id, target);
  }
  let activityIndex = visibleIds.size;
  for (const members of groups.active.values()) {
    for (const member of members) indexes.set(member.id, activityIndex);
    activityIndex += 1;
  }
  for (const message of messages) {
    if (message.body.kind !== "tool_result" || visibleIds.has(message.id)) continue;
    const callIndex = indexes.get(message.body.call_ref.message_id);
    if (callIndex !== undefined) indexes.set(message.id, callIndex);
  }
  return indexes;
}

/** 结果按原调用引用索引，展示层不推测工具是否完成。 */
export function timelineToolResults(messages: TimelineMessage[]): Map<string, TimelineMessage> {
  return new Map(messages.flatMap((message) => message.body.kind === "tool_result"
    ? [[`${message.body.call_ref.message_id}:${message.body.call_ref.part_index}`, message] as const] : []));
}

/** 实时召回槽位只由已接纳的用户 Input 持有。 */
export function timelineInputStarts(messages: TimelineMessage[]): Map<string, number> {
  const starts = new Map<string, number>();
  for (const message of messages) {
    const key = timelineSourceKey(message);
    if (message.body.kind === "input" && message.author === "user" && !starts.has(key)) starts.set(key, message.seq);
  }
  return starts;
}

export function timelineSourceKey(value: { session_id: string; source: string }): string {
  return JSON.stringify([value.session_id, value.source]);
}

/** 历史尾页可能缺少 Input；保留输出锚点，由服务端解析真实归属。 */
export function needsBeforeReasoningFallback(message: TimelineMessage, inputStarts: ReadonlyMap<string, number>): boolean {
  return message.body.kind === "output" && message.seq < (inputStarts.get(timelineSourceKey(message)) ?? Infinity);
}

/** 同来源提交进度与预览身份使事实失效；逐 token 正文不参与。 */
export function timelineSourceRefreshTokens(messages: TimelineMessage[], activities: ReplyActivity[], refresh: number): Map<string, string> {
  const heads = new Map<string, number>();
  for (const message of messages) heads.set(timelineSourceKey(message), message.seq);
  const previews = new Map<string, unknown[]>();
  for (const activity of activities) {
    const key = timelineSourceKey(activity);
    const items = previews.get(key) ?? [];
    items.push([activity.handle, activity.active, activity.preview?.message_id ?? null]);
    previews.set(key, items);
  }
  return new Map([...new Set([...heads.keys(), ...previews.keys()])].map((key) =>
    [key, JSON.stringify([heads.get(key) ?? -1, previews.get(key) ?? [], refresh])]));
}

/** 完成或停止展示时，把同来源的过程归到末条输出；不改变 Turn。 */
export function timelineReplyGroups(messages: TimelineMessage[], activities: ReplyActivity[] = [], replySynced = true) {
  const pending = new Map<string, TimelineMessage[]>();
  const completed = new Map<string, TimelineMessage[]>();
  const hiddenBodies = new Set<string>();
  // 被附件结果切开的同一回复只在末尾显示一次时间和操作。
  const joined = new Set<string>();
  for (const message of messages) {
    const key = `${message.session_id}:${message.source}`;
    const body = message.body;
    if (body.kind === "output") {
      const members = pending.get(key) ?? [];
      const prior = members.at(-1);
      if (prior) hiddenBodies.add(prior.id);
      members.push(message);
      if (body.finish === "continue") pending.set(key, members);
      else {
        if (members.length > 1) completed.set(message.id, members);
        pending.delete(key);
      }
    } else if ((body.kind === "input" && message.author === "user")
      || (body.kind === "control" && body.action !== "resume")
      || timelineDeliversAttachment(message)) {
      const throughSeq = body.kind === "control" ? body.through_seq : message.seq - 1;
      const members = pending.get(key) ?? [];
      const closed = members.filter((item) => item.seq <= throughSeq);
      const ending = closed.at(-1);
      if (ending) {
        hiddenBodies.delete(ending.id);
        if (closed.length > 1) completed.set(ending.id, closed);
      }
      if (body.kind === "tool_result") {
        joined.add(message.id);
        if (ending?.body.kind === "output" && ending.body.finish === "continue") joined.add(ending.id);
      }
      // 停止或暂停收尾、只有过程没有正文的输出：时间挂在紧随其后的"已暂停"上，空白处不再挂一行时间与指向空消息的操作。
      // 图片、artifact、插件数据都算正文：带着它们的输出照常保留时间与"引用此消息"。
      if (body.kind === "control" && ending?.body.kind === "output" && !hasReplyBody(ending)) joined.add(ending.id);
      pending.set(key, members.filter((item) => item.seq > throughSeq));
    }
  }
  const active = new Map<string, TimelineMessage[]>();
  for (const activity of activities) active.set(activity.handle, pending.get(`${activity.session_id}:${activity.source}`) ?? []);
  const moved = new Set([...completed.values()].flatMap((members) => members.slice(0, -1).map((message) => message.id)));
  for (const members of active.values()) for (const message of members) moved.add(message.id);
  // 附件把正文留在原位；整条图文回复的过程集中在首段上方。
  const inline = timelineInlineReplies(messages);
  // 回复状态未同步时，未结束的图文回复不按图文合并（合并会把前段过程折成摘要、文字按正文排），与普通未结束轮次一样逐行显示。
  const openInline = new Set<string>();
  if (!replySynced) for (const id of inline.open) {
    for (const item of inline.leading.get(id) ?? []) { inline.outputs.delete(item.id); openInline.add(item.id); }
    inline.leading.delete(id);
  }
  for (const members of inline.leading.values()) for (const item of members.slice(0, -1)) joined.add(item.id);
  for (const id of inline.outputs) {
    completed.delete(id); moved.delete(id); hiddenBodies.delete(id);
  }
  for (const [handle, members] of active) active.set(handle, members.filter(item => !inline.outputs.has(item.id)));
  // 轮次还没结束（末条输出 finish=continue，之后没有输入或控制收尾）的输出：本会话的回复状态还没同步到时按原位显示，
  // 不能当成已落定的一轮收进摘要。状态同步后，仍在运行的轮次由实时活动接管（moved）；没有活动说明它已不会再继续
  // （异常中断、没有收尾记录的旧轮次），照常收起并挂上时间与操作。
  const unsettled = replySynced ? new Set<string>() : new Set([...[...pending.values()].flat().map((message) => message.id), ...openInline]);
  return { completed, active, moved, hiddenBodies, joined, inline, unsettled };
}

export interface TimelineFailureRetries {
  /** 仍显示的失败 → 同一条输入上更早、已被重试取代的失败。 */
  earlier: Map<string, TimelineMessage[]>;
  /** 被重试取代或已被回答替换的失败，以及失败后的恢复记录：都不单独成行。 */
  superseded: Set<string>;
  /** 来源停在失败上：失败 id → 可以重跑的原用户输入 id。 */
  retryable: Map<string, string>;
}

// input 为 undefined 表示这一段的输入在已加载页之前（长回合只读了末尾一页）。
type FailureRun = { input: TimelineMessage | null | undefined; failures: TimelineMessage[]; tail: TimelineMessage };

// 输入不在已加载页时，重试引用来源的最新输入；服务端解析，重试本来也只允许最新输入。
export const RETRY_LATEST_INPUT = "latest";

// 同一条输入反复失败时只留最后一次，答出来后连这一次也收起；来源末尾停在失败上时，它指向可以重跑的原输入。
export function timelineFailureRetries(messages: TimelineMessage[]): TimelineFailureRetries {
  const earlier = new Map<string, TimelineMessage[]>();
  const superseded = new Set<string>();
  const runs = new Map<string, FailureRun>();
  for (const message of messages) {
    const key = timelineSourceKey(message), body = message.body;
    const run = runs.get(key) ?? { input: undefined, failures: [], tail: message };
    runs.set(key, run);
    run.tail = message;
    // 1. 新输入或放弃开始新的一段；自动输入不能由用户重跑。
    if (body.kind === "input") Object.assign(run, { input: message.author === "user" ? message : null, failures: [] });
    else if (body.kind === "control" && body.action === "abandon") run.failures = [];
    // 2. 重试最终答出来了：回答替换掉这一轮的失败。
    else if (body.kind === "output" && body.finish !== "continue") {
      for (const item of run.failures) superseded.add(item.id);
      run.failures = [];
    } else if (body.kind === "control" && body.action === "resume") {
      // 3. 失败之后的恢复就是重试本身，失败块已说明，不再单独成行。
      if (run.failures.length) superseded.add(message.id);
    } else if (body.kind === "control" && body.action === "failure") {
      // 4. 再次失败时，之前的失败收进这一次。
      if (run.failures.length) {
        for (const item of run.failures) superseded.add(item.id);
        earlier.set(message.id, [...run.failures]);
      }
      run.failures.push(message);
    }
  }
  // 5. 只有来源最后一条仍是失败时才可重试；正在恢复或已有新输入都不再提供。
  const retryable = new Map<string, string>();
  for (const run of runs.values()) {
    if (run.tail.body.kind !== "control" || run.tail.body.action !== "failure") continue;
    if (run.input) retryable.set(run.tail.id, run.input.id);
    else if (run.input === undefined && run.tail.source === "conversation") retryable.set(run.tail.id, RETRY_LATEST_INPUT);
  }
  return { earlier, superseded, retryable };
}

/** 从已提交消息识别图文回复；不依赖插件是否启用，也不改写消息顺序。 */
function timelineInlineReplies(messages: TimelineMessage[]) {
  const pending = new Map<string, { messages: TimelineMessage[]; attachment: boolean }>();
  const leading = new Map<string, TimelineMessage[]>();
  const outputs = new Set<string>();
  // 记录到末尾仍以 finish=continue 结尾、后面没有输入或控制收尾的图文回复（首段 id）。
  const open = new Set<string>();
  const save = (key: string, ended = true) => {
    const reply = pending.get(key);
    if (reply?.attachment && reply.messages.length) {
      leading.set(reply.messages[0].id, reply.messages);
      for (const item of reply.messages) outputs.add(item.id);
      const last = reply.messages.at(-1)?.body;
      if (!ended && last?.kind === "output" && last.finish === "continue") open.add(reply.messages[0].id);
    }
    pending.delete(key);
  };
  for (const message of messages) {
    const key = timelineSourceKey(message), body = message.body;
    if (body.kind === "input" || (body.kind === "control" && body.action !== "resume")) save(key);
    if (body.kind === "output") {
      const reply = pending.get(key) ?? { messages: [], attachment: false };
      reply.messages.push(message); pending.set(key, reply);
      if (body.finish !== "continue") save(key);
    } else if (timelineDeliversAttachment(message)) {
      const reply = pending.get(key);
      if (reply) reply.attachment = true;
    }
  }
  for (const key of [...pending.keys()]) save(key, false);
  return { leading, outputs, open };
}

/** 已加载的工具结果在原调用面板中展示；缺少调用的分页仍保留结果行。 */
export function timelineVisibleMessages(messages: TimelineMessage[], groups = timelineReplyGroups(messages)): TimelineMessage[] {
  const byId = new Map(messages.map((message) => [message.id, message]));
  const { superseded } = timelineFailureRetries(messages);
  return messages.filter((message) => {
    if (superseded.has(message.id)) return false;
    if (groups.moved.has(message.id) && !message.attachments.length) return false;
    if (!isTimelineMessageVisible(message) && !groups.completed.has(message.id)
      && !(message.body.kind === "input" && message.author === "user")) return false;
    if (groups.hiddenBodies.has(message.id) && !message.attachments.length && message.body.kind === "output"
      && !message.body.parts.some((part) => part.kind !== "text" && isTimelinePartVisible(part))) return false;
    if (message.body.kind !== "tool_result" || timelineDeliversAttachment(message)) return true;
    // 引用自身图片的 artifact_ref 随调用卡片展示，不单独成行。
    if (message.body.parts.some((part) => isTimelinePartVisible(part) && part.kind !== "text" && part.kind !== "artifact_ref"
      && !("display" in part && part.display === "data"))) return true;
    const call = byId.get(message.body.call_ref.message_id);
    return call?.body.kind !== "output" || call.body.parts[message.body.call_ref.part_index]?.kind !== "tool_call";
  });
}
