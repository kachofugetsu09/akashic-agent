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
  | { kind: "tool_result"; parts: TimelinePart[]; call_ref: { message_id: string; part_index: number }; outcome: "success" | "denied" | "error" | "unknown" }
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
        || !["success", "denied", "error", "unknown"].includes(String(body.outcome))) throw new Error("工具结果引用或状态无效");
    }
    // 2. 附件引用必须能从本行元数据解析，不猜存储路径。
    const ids = new Set(row.attachments.map((item) => (item as TimelineAttachment).artifact_id));
    if (body.parts.some((part) => object(part)?.kind === "artifact_ref" && !ids.has(object(part)?.value as string))) {
      throw new Error("历史消息附件引用缺少元数据");
    }
  }
  return row as unknown as TimelineMessage;
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
    if (prior && JSON.stringify(prior) !== JSON.stringify(row)) throw new Error("历史消息正文发生变化");
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
  return { id: message.id, author: message.author,
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
