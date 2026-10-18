import { readTimelineMessage, type TimelineMessage } from "./message-timeline.ts";
import type { ComposerFile } from "./desktop-composer";

export interface SessionRow {
  key: string;
  updated_at?: string;
  /** 服务端 session_row 提供；置顶解析补齐的目录外会话可能缺失。 */
  created_at?: string;
  message_count?: number;
  /** 日志当前最大 seq；用于判断某会话相对上次查看有无新消息。 */
  head_seq?: number;
  first_message_content?: string;
  /** 显式标题覆盖；null/缺失时按 first_message_content 推导。 */
  title?: string | null;
  /** Session 接纳时固定的宽键；缺失维度即 default。 */
  scope?: Record<string, string>;
}

export interface ChatHistoryPage {
  items: TimelineMessage[];
  throughSeq: number;
  hasMore: boolean;
  beforeSeq: number | null;
  /** 会话已软删时为 true；消息仍物理保留，按只读展示。 */
  deleted: boolean;
}

export interface WebShellState {
  status: "needs_setup" | "starting" | "unavailable" | "ready";
  configured: boolean;
  chatReady: boolean;
  /** 服务端当前前端的版本；与页面自带的版本不同时页面已过期。 */
  build: string | null;
}

export interface UploadedFile {
  filename: string;
  artifact_id: string;
  kind: "file" | "image";
  media_type?: string | null;
  size_bytes: number;
  sha256: string;
  upload_url?: string;
}

export function isAbortError(error: unknown): boolean {
  return error instanceof DOMException && error.name === "AbortError";
}

export function errorMessage(error: unknown): string {
  if (error instanceof TypeError && error.message === "Failed to fetch") {
    return "无法连接 Akashic。请确认服务仍在运行，然后重试。";
  }
  return error instanceof Error ? error.message : String(error);
}

/** HTTP 认证/权限错误与目录内容、远端模型调用失败分开。 */
export class ChatRequestError extends Error {
  constructor(readonly status: number, message: string) { super(message); }
}

export async function fetchChatJson<T>(url: string, options: RequestInit = {}): Promise<T> {
  const response = await fetch(url, options);
  const text = await response.text();
  let payload: unknown = null;
  if (text) {
    try {
      payload = JSON.parse(text);
    } catch {
      if (!response.ok) throw new ChatRequestError(response.status, `请求失败（${response.status}）`);
      throw new Error("服务器返回了无效 JSON");
    }
  }
  if (!response.ok) {
    const body = recordValue(payload);
    const detail = typeof body?.detail === "string" ? body.detail : typeof body?.message === "string" ? body.message : "";
    throw new ChatRequestError(response.status, detail || `请求失败（${response.status}）`);
  }
  if (payload === null) throw new Error("服务器返回空响应");
  return payload as T;
}

export function sessionPage(payload: unknown): { items: SessionRow[]; nextCursor: { updated_at: string; session_id: string } | null } {
  const items = responseItems(payload, "/api/chat/sessions");
  if (items.some((item) => (
    typeof item.key !== "string"
    || !item.key.trim()
    || (item.first_message_content !== undefined && typeof item.first_message_content !== "string")
    || (item.title !== undefined && item.title !== null && typeof item.title !== "string")
    || (item.updated_at !== undefined && typeof item.updated_at !== "string")
    || (item.created_at !== undefined && typeof item.created_at !== "string")
    || (item.message_count !== undefined && (typeof item.message_count !== "number" || !Number.isFinite(item.message_count)))
    || (item.head_seq !== undefined && !Number.isSafeInteger(item.head_seq))
    || (item.scope !== undefined && !isStringRecord(item.scope))
  ))) {
    throw new Error("/api/chat/sessions 返回了无效 session 行");
  }
  const cursor = recordValue(payload)?.next_cursor;
  const next = recordValue(cursor);
  if (cursor !== null && (!next || typeof next.updated_at !== "string"
    || !Number.isFinite(Date.parse(next.updated_at)) || typeof next.session_id !== "string" || !next.session_id)) {
    throw new Error("/api/chat/sessions 返回了无效目录游标");
  }
  return { items: items as unknown as SessionRow[], nextCursor: next as { updated_at: string; session_id: string } | null };
}

/** 历史页只接受 Message v2；分页上界与行顺序都来自服务端日志。 */
export function chatHistoryPage(payload: unknown, endpoint: string): ChatHistoryPage {
  const body = recordValue(payload);
  if (!body || body.version !== 2 || !Number.isSafeInteger(body.through_seq)
    || Number(body.through_seq) < -1 || typeof body.has_more !== "boolean") {
    throw new Error(`${endpoint} 返回了无效历史页`);
  }
  const items = responseItems(payload, endpoint).map(readTimelineMessage);
  const ids = new Set<string>();
  if (items.some((item, index) => {
    const invalid = ids.has(item.id) || item.session_id !== items[0].session_id
      || item.seq > Number(body.through_seq) || (index > 0 && item.seq <= items[index - 1].seq);
    ids.add(item.id);
    return invalid;
  }) || (body.has_more ? !items.length || body.before_seq !== items[0].seq : body.before_seq !== null)) {
    throw new Error(`${endpoint} 返回了不一致的历史游标`);
  }
  return { items, throughSeq: Number(body.through_seq), hasMore: body.has_more,
    beforeSeq: body.before_seq as number | null, deleted: body.deleted === true };
}

export function webShellState(payload: unknown): WebShellState {
  const body = recordValue(payload);
  if (!body
    || (body.status !== "needs_setup" && body.status !== "starting" && body.status !== "unavailable" && body.status !== "ready")
    || typeof body.configured !== "boolean"
    || typeof body.chatReady !== "boolean") {
    throw new Error("/api/shell/state 返回了无效状态");
  }
  return { status: body.status, configured: body.configured, chatReady: body.chatReady,
    build: typeof body.build === "string" ? body.build : null };
}

/** 软删/恢复是显式数据管理操作；返回值校验 key 与结果标记一致，防止对错会话生效。 */
export async function setChatSessionDeleted(sessionKey: string, deleted: boolean): Promise<void> {
  const endpoint = `/api/chat/sessions/${encodeURIComponent(sessionKey)}/${deleted ? "delete" : "undelete"}`;
  const payload = await fetchChatJson<unknown>(endpoint, { method: "POST" });
  const body = recordValue(payload);
  if (!body || body.key !== sessionKey || body.deleted !== deleted) {
    throw new Error(`${endpoint} 返回了不一致的删除结果`);
  }
}

/** 标题覆盖与软删同属显式数据管理操作；空标题清除覆盖，返回值即服务端落库结果。 */
export async function renameChatSession(sessionKey: string, title: string): Promise<string | null> {
  const endpoint = `/api/chat/sessions/${encodeURIComponent(sessionKey)}/rename`;
  const payload = await fetchChatJson<unknown>(endpoint, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ title }),
  });
  const body = recordValue(payload);
  if (!body || body.key !== sessionKey || (body.title !== null && typeof body.title !== "string")) {
    throw new Error(`${endpoint} 返回了不一致的标题结果`);
  }
  return typeof body.title === "string" ? body.title : null;
}

export async function uploadFiles(files: ComposerFile[], signal: AbortSignal): Promise<UploadedFile[]> {
  const result: UploadedFile[] = [];
  for (const file of files) {
    const label = file.filename || "未命名";
    if (!file.url) throw new Error(`附件 ${label} 缺少内容 URL`);
    let sourceResponse: Response;
    try {
      sourceResponse = await fetch(file.url, { signal });
    } catch (error) {
      if (isAbortError(error) || !(error instanceof TypeError)) throw error;
      throw new Error(`无法读取附件 ${label}，请移除后重新选择`, { cause: error });
    }
    if (!sourceResponse.ok) throw new Error(`读取附件失败: ${sourceResponse.status}`);
    const blob = await sourceResponse.blob();
    const filename = file.filename || "upload.bin";
    const payload = await fetchChatJson<unknown>(`/api/chat/uploads?filename=${encodeURIComponent(filename)}`, {
      method: "POST",
      body: blob,
      signal,
    });
    result.push(uploadedFileResponse(payload));
  }
  return result;
}

function responseItems(payload: unknown, endpoint: string): Record<string, unknown>[] {
  const body = recordValue(payload);
  if (!body || !Array.isArray(body.items) || body.items.some((item) => !recordValue(item))) {
    throw new Error(`${endpoint} 返回格式无效`);
  }
  return body.items as Record<string, unknown>[];
}

function uploadedFileResponse(payload: unknown): UploadedFile {
  const body = recordValue(payload);
  if (!body
    || typeof body.filename !== "string"
    || typeof body.artifact_id !== "string"
    || !body.artifact_id
    || (body.kind !== "file" && body.kind !== "image")
    || (body.media_type !== undefined && body.media_type !== null && typeof body.media_type !== "string")
    || typeof body.size_bytes !== "number"
    || !Number.isInteger(body.size_bytes)
    || body.size_bytes < 0
    || typeof body.sha256 !== "string"
    || !/^[0-9a-f]{64}$/.test(body.sha256)) {
    throw new Error("上传接口返回格式无效");
  }
  if (body.upload_url !== undefined
    && (typeof body.upload_url !== "string" || !body.upload_url.startsWith("/api/chat/artifacts/"))) {
    throw new Error("上传接口返回了无效 URL");
  }
  return body as unknown as UploadedFile;
}

function recordValue(value: unknown): Record<string, unknown> | null {
  return typeof value === "object" && value !== null && !Array.isArray(value)
    ? value as Record<string, unknown>
    : null;
}

function isStringRecord(value: unknown): value is Record<string, string> {
  const record = recordValue(value);
  return record !== null && Object.values(record).every((item) => typeof item === "string");
}
