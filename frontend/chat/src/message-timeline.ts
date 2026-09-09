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
  | { kind: "tool_call"; binding_id: string; name: string; arguments: Record<string, unknown> }
  | { kind: "model.facts"; value: { call_record_id: string; thinking: string | null } }
  | { kind: "history.provenance" | "history.transcript" | "history.record" | "history.turn_input"; archive: unknown }
  | { kind: string; display: "unavailable" };

export type TimelineBody =
  | { kind: "input"; parts: TimelinePart[] }
  | { kind: "output"; parts: TimelinePart[]; finish: "continue" | "complete" | "quiet" }
  | { kind: "tool_result"; parts: TimelinePart[]; call_ref: { message_id: string; part_index: number }; outcome: "success" | "denied" | "error" | "unknown" | "interrupted" }
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

export interface ReplyActivity {
  session_id: string;
  source: string;
  handle: string;
  active: boolean;
  preview: { message_id: string; text: string; thinking: string; call_record_id?: string | null; truncated?: boolean } | null;
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
        if (!item.active || !preview || !nonempty(preview.message_id) || typeof preview.text !== "string"
          || typeof preview.thinking !== "string" || (preview.call_record_id !== undefined && preview.call_record_id !== null && !nonempty(preview.call_record_id)) || (preview.truncated !== undefined && typeof preview.truncated !== "boolean") || previews.has(preview.message_id)) throw new Error("回复草稿无效");
        previews.add(preview.message_id);
      }
    }
  }
  return frame as unknown as MessageLogFrame;
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
        || !["success", "denied", "error", "unknown", "interrupted"].includes(String(body.outcome))) throw new Error("工具结果引用或状态无效");
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
      if (JSON.stringify(oldFacts) !== JSON.stringify(newFacts) || JSON.stringify(before) !== JSON.stringify(after)) {
        throw new Error("历史消息正文发生变化");
      }
    }
    byId.set(row.id, row);
    bySeq.set(row.seq, row.id);
  }
  return [...byId.values()].sort((left, right) => left.seq - right.seq);
}

export function timelineText(message: TimelineMessage): string {
  return message.body.kind === "control" ? message.body.reason ?? "" : message.body.parts
    .flatMap((part) => !("display" in part) && part.kind === "text" ? [part.value] : []).join("\n");
}

export function timelineReply(message: TimelineMessage): TimelineReply {
  return { id: message.id, author: message.author === "legacy-attribution-unknown" ? "原消息" : message.author,
    preview: timelineText(message).replace(/\s+/gu, " ").trim().slice(0, 512)
      || (message.attachments.length ? "[附件]" : "[无文字消息]") };
}

function validPart(value: unknown): boolean {
  const part = object(value);
  if (!part || !nonempty(part.kind)) return false;
  if (part.display === "unavailable") return !["text", "artifact_ref", "reply_ref", "tool_call", "model.facts"].includes(part.kind);
  switch (part.kind) {
    case "text": return typeof part.value === "string";
    case "artifact_ref": case "reply_ref": return nonempty(part.value);
    case "tool_call": return nonempty(part.binding_id) && nonempty(part.name) && object(part.arguments) !== null;
    case "model.facts": {
      const facts = object(part.value);
      return facts !== null && nonempty(facts.call_record_id) && nullableText(facts.thinking);
    }
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
  if ("display" in part) return !["channel.origin", "context.summary", "tool.selection", "model.selection", "command.result", "akasha.recall", "akasha.feedback"].includes(part.kind);
  if ("archive" in part) {
    if (part.kind !== "history.transcript") return false;
    const groups = historyTranscript(part.archive);
    return groups === null || groups.some((group) => group.text || group.thinking || group.calls.length);
  }
  if (part.kind === "text") return part.value.length > 0;
  if (part.kind === "model.facts") return Boolean(part.value.thinking);
  return true;
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

/** 完成或停止展示时，把同来源的过程归到末条输出；不改变 Turn。 */
export function timelineReplyGroups(messages: TimelineMessage[], activities: ReplyActivity[] = []) {
  const pending = new Map<string, TimelineMessage[]>();
  const completed = new Map<string, TimelineMessage[]>();
  const hiddenBodies = new Set<string>();
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
    } else if (body.kind === "control" && body.action !== "resume") {
      const members = pending.get(key) ?? [];
      const closed = members.filter((item) => item.seq <= body.through_seq);
      const ending = closed.at(-1);
      if (ending) {
        hiddenBodies.delete(ending.id);
        if (closed.length > 1) completed.set(ending.id, closed);
      }
      pending.set(key, members.filter((item) => item.seq > body.through_seq));
    }
  }
  const active = new Map<string, TimelineMessage[]>();
  for (const activity of activities) active.set(activity.handle, pending.get(`${activity.session_id}:${activity.source}`) ?? []);
  const moved = new Set([...completed.values()].flatMap((members) => members.slice(0, -1).map((message) => message.id)));
  for (const members of active.values()) for (const message of members) moved.add(message.id);
  return { completed, active, moved, hiddenBodies };
}

/** 已加载的工具结果在原调用面板中展示；缺少调用的分页仍保留结果行。 */
export function timelineVisibleMessages(messages: TimelineMessage[], groups = timelineReplyGroups(messages)): TimelineMessage[] {
  const byId = new Map(messages.map((message) => [message.id, message]));
  return messages.filter((message) => {
    if (groups.moved.has(message.id) && !message.attachments.length) return false;
    if (!isTimelineMessageVisible(message) && !groups.completed.has(message.id)) return false;
    if (groups.hiddenBodies.has(message.id) && !message.attachments.length && message.body.kind === "output"
      && !message.body.parts.some((part) => part.kind !== "text" && isTimelinePartVisible(part))) return false;
    if (message.body.kind !== "tool_result" || message.attachments.length) return true;
    if (message.body.parts.some((part) => isTimelinePartVisible(part) && part.kind !== "text")) return true;
    const call = byId.get(message.body.call_ref.message_id);
    return call?.body.kind !== "output" || call.body.parts[message.body.call_ref.part_index]?.kind !== "tool_call";
  });
}
