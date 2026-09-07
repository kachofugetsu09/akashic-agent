import { mergeTimelineMessages, readMessageLogFrame, readTimelineMessage, type MessageLogFrame, type TimelineMessage } from "./message-timeline.ts";

export type MobileReplyStatus = Extract<MessageLogFrame, { type: "reply.status" }>;

/** 原生只保存下载进度与设备 URL；附件身份和文件信息来自 Message。 */
export interface MobileDownload {
  artifactId: string;
  state: string;
  transferredBytes: number;
  contentUrl?: string;
}

export interface MobileMessageLog {
  selectedSessionId?: string;
  projectionGeneration: number;
  messages: TimelineMessage[];
  throughSeq: number;
  replyStatus: MobileReplyStatus | null;
}

/** v9 快照必须提供完整消息；未下载完的 message_ref 留在原生同步 owner。 */
export function readMobileMessageLog(raw: Record<string, unknown>): MobileMessageLog {
  // 1. 固定当前会话与同步代际，拒绝不完整或跨会话正文。
  const session = raw.selectedSessionId;
  if (session !== undefined && session !== null && (typeof session !== "string" || !session)) throw new Error("消息会话无效");
  if (!Number.isSafeInteger(raw.projectionGeneration) || (raw.projectionGeneration as number) < 0
    || !Number.isSafeInteger(raw.throughSeq) || (raw.throughSeq as number) < -1
    || !Array.isArray(raw.messages)) throw new Error("消息快照游标无效");
  const messages = raw.messages.map(readTimelineMessage);
  let seq = -1;
  for (const message of messages) {
    if (message.session_id !== session || message.seq <= seq || message.seq > (raw.throughSeq as number)) throw new Error("消息快照顺序或会话无效");
    seq = message.seq;
  }
  mergeTimelineMessages([], messages);
  if (!session && (messages.length || raw.throughSeq !== -1)) throw new Error("未选会话不能携带消息");

  // 2. 当前活动独立于日志；断线或尚未订阅时必须显式为空。
  const replyStatus = raw.replyStatus === null ? null : readMessageLogFrame(raw.replyStatus);
  if (raw.replyStatus !== null && (replyStatus?.type !== "reply.status" || replyStatus.session_id !== session)) {
    throw new Error("快照回复状态无效");
  }
  return { selectedSessionId: session === null ? undefined : session as string | undefined,
    projectionGeneration: raw.projectionGeneration as number, messages,
    throughSeq: raw.throughSeq as number, replyStatus: replyStatus as MobileReplyStatus | null };
}

/** 原生同步代际在切会话或重建时递增；迟到快照不能回退当前页面。 */
export function mergeMobileMessageSnapshot<T extends MobileMessageLog>(current: T, next: T): T | null {
  if (next.projectionGeneration < current.projectionGeneration) return null;
  if (next.projectionGeneration > current.projectionGeneration) return next;
  if (next.selectedSessionId !== current.selectedSessionId) throw new Error("切换会话必须递增同步代际");
  if (next.throughSeq < current.throughSeq) throw new Error("消息快照游标回退");
  return { ...next, messages: mergeTimelineMessages(current.messages, next.messages), replyStatus: current.replyStatus };
}

/** 同代际事件追加事实或替换活动；过期事件不改变当前页面。 */
export function applyMobileMessageEvent<T extends MobileMessageLog>(current: T, value: unknown): T | null {
  const raw = record(value);
  if (raw.protocolVersion !== 1 || !Number.isSafeInteger(raw.projectionGeneration)
    || (raw.projectionGeneration as number) < 0) throw new Error("消息事件协议无效");
  const frame = readMessageLogFrame(raw.event);
  if (!frame) throw new Error("消息事件类型无效");
  if (raw.projectionGeneration !== current.projectionGeneration || frame.session_id !== current.selectedSessionId) return null;
  if (frame.type === "session.following") return current;
  if (frame.type === "reply.status") return { ...current, replyStatus: frame };
  if (frame.after_seq !== current.throughSeq) throw new Error("消息事件游标不连续，请重新同步");
  return { ...current, messages: mergeTimelineMessages(current.messages, frame.items), throughSeq: frame.next_after_seq };
}

/** 状态 patch 只更新控制和下载字段，不能夹带日志、草稿或旧流式字段。 */
export function readMobileStateSnapshot(value: unknown): Record<string, unknown> {
  const raw = record(value);
  const fields = new Set(["protocolVersion", "connection", "sessions", "selectedSessionId", "readingPosition",
    "navigationTarget", "projectionGeneration", "downloads", "composer", "modelCatalog", "runtimeInspection"]);
  if (raw.protocolVersion !== 2 || Object.keys(raw).some((key) => !fields.has(key))) throw new Error("状态 patch 版本或字段无效");
  return { ...raw, protocolVersion: 9, messages: [], throughSeq: -1, replyStatus: null };
}

export function readMobileDownloads(value: unknown): MobileDownload[] {
  if (!Array.isArray(value)) throw new Error("附件下载状态不是数组");
  const ids = new Set<string>();
  return value.map((value) => {
    const raw = record(value);
    if (typeof raw.artifactId !== "string" || !raw.artifactId || ids.has(raw.artifactId)
      || !["remote", "pending", "downloading", "cached", "failed", "evicted"].includes(String(raw.state))
      || !Number.isSafeInteger(raw.transferredBytes) || (raw.transferredBytes as number) < 0
      || (raw.contentUrl !== undefined && typeof raw.contentUrl !== "string")) throw new Error("附件下载状态无效");
    ids.add(raw.artifactId);
    return raw as unknown as MobileDownload;
  });
}

function record(value: unknown): Record<string, unknown> {
  if (!value || typeof value !== "object" || Array.isArray(value)) throw new Error("消息事件不是对象");
  return value as Record<string, unknown>;
}
