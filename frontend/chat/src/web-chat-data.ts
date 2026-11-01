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
  if (error instanceof DOMException && error.name === "TimeoutError") return "网络太慢，读取超时，请重试。";
  return error instanceof Error ? error.message : String(error);
}

/** HTTP 认证/权限错误与目录内容、远端模型调用失败分开。 */
export class ChatRequestError extends Error {
  constructor(readonly status: number, message: string) { super(message); }
}

// 预热响应到达后只在这段时间内复用：首跑引导期间聊天迟迟未就绪，期间的变化会让早先的读取过时。
// 按到达时刻而不是页面打开时刻计：弱网下聊天挂载与就绪本身就要十几到二十秒，过早判过时会丢掉已到的响应，
// 再排在大文件后面重读一遍。会话目录随连接对账、插件目录有换代恢复，就绪状态不会倒退，放宽到 30 秒无碍。
const STARTUP_READ_MAX_AGE_MS = 30_000;

// index.html 内联脚本按 URL 预热的启动读取，一次性取用；失败、过时或没有预热时返回 null，由调用方正常请求。
// 预热请求不认识调用方的 signal：等待期间中止直接以 AbortError 结束，不退回成一次新请求；
// 读完正文后调用方还要再核一次 signal（见 fetchChatJson、loadWebPluginCatalog）。
export async function takeStartupResponse(url: string, signal?: AbortSignal | null): Promise<Response | null> {
  const reads = (window as { __akashicStartupReads?: Map<string, Promise<{ response: Response; at: number } | null>> }).__akashicStartupReads;
  const prefetched = reads?.get(url);
  reads?.delete(url);
  if (!prefetched) return null;
  signal?.throwIfAborted();
  const read = await new Promise<{ response: Response; at: number } | null>((resolve, reject) => {
    const abort = () => reject(signal?.reason);
    signal?.addEventListener("abort", abort, { once: true });
    void prefetched.then(resolve).finally(() => signal?.removeEventListener("abort", abort));
  });
  return read?.response.ok && performance.now() - read.at <= STARTUP_READ_MAX_AGE_MS ? read.response : null;
}

// 读取的期限：半开连接上的请求永远等不到结果，被复用的在途读取会让“重试”和重连也一起卡住。
// 超时以 TimeoutError 失败（不是 AbortError），调用方照常报错、重试。
const READ_DEADLINE_MS = 30_000;

export async function fetchChatJson<T>(url: string, options: RequestInit = {}): Promise<T> {
  const read = !options.method || options.method === "GET";
  const signal = read ? AbortSignal.any([options.signal ?? new AbortController().signal, AbortSignal.timeout(READ_DEADLINE_MS)])
    : options.signal;
  const response = await takeStartupResponse(url, signal) ?? await fetch(url, { ...options, signal });
  const text = await response.text();
  // 预热响应的正文读取不受 signal 约束：读完再核一次，中止后的结果不交给调用方。
  signal?.throwIfAborted();
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

// 上次读到的会话目录首段记在本机：打开页面先显示，不等目录往返；网络目录到达后整体替换。
// 首条消息只留够推导标题的长度，单份控制在几十 KB。
const SESSIONS_SNAPSHOT_KEY = "akashic.chat.sessions.v1";
const SESSIONS_SNAPSHOT_ROWS = 200;

export function readSessionsSnapshot(): SessionRow[] {
  try {
    return sessionPage({ items: JSON.parse(localStorage.getItem(SESSIONS_SNAPSHOT_KEY) ?? "[]"), next_cursor: null }).items;
  } catch { return []; }
}

export function writeSessionsSnapshot(rows: readonly SessionRow[]): void {
  const trimmed = rows.slice(0, SESSIONS_SNAPSHOT_ROWS).map((row) => row.first_message_content && row.first_message_content.length > 160
    ? { ...row, first_message_content: row.first_message_content.slice(0, 160) } : row);
  try { localStorage.setItem(SESSIONS_SNAPSHOT_KEY, JSON.stringify(trimmed)); }
  catch { /* 存储不可用时下次照常等网络 */ }
}

// 当前会话最近读到的尾页原文记在本机，只留一份：重新打开页面先显示它，同时照常读最新尾页。
const TAIL_SNAPSHOT_KEY = "akashic.chat.session-tail.v1";
const TAIL_SNAPSHOT_MAX_CHARS = 512 * 1024;

export function readTailSnapshot(): { sessionId: string; payload: unknown } | null {
  try {
    const value: unknown = JSON.parse(localStorage.getItem(TAIL_SNAPSHOT_KEY) ?? "null");
    const record = recordValue(value);
    return record && typeof record.sessionId === "string" ? { sessionId: record.sessionId, payload: record.payload } : null;
  } catch { return null; }
}

export function writeTailSnapshot(sessionId: string, payload: unknown): void {
  try {
    const text = JSON.stringify({ sessionId, payload });
    if (text.length <= TAIL_SNAPSHOT_MAX_CHARS) localStorage.setItem(TAIL_SNAPSHOT_KEY, text);
    else localStorage.removeItem(TAIL_SNAPSHOT_KEY);
  } catch { /* 存储不可用时下次照常等网络 */ }
}

// 实时追加的消息行与历史页同一格式：接到快照尾部，重新打开时连刚才这一轮一起显示；只留最近 50 条，游标随之前移。
export function appendTailSnapshot(sessionId: string, rows: readonly unknown[], throughSeq: number): void {
  const snapshot = readTailSnapshot();
  const body = recordValue(snapshot?.payload);
  if (!snapshot || snapshot.sessionId !== sessionId || !body || !Array.isArray(body.items)) return;
  const seq = (row: unknown) => Number(recordValue(row)?.seq);
  const merged = [...body.items, ...rows.filter((row) => seq(row) > Number(body.through_seq))];
  const items = merged.slice(-50);
  const hasMore = body.has_more === true || items.length < merged.length;
  writeTailSnapshot(sessionId, { ...body, items, through_seq: throughSeq, has_more: hasMore, before_seq: hasMore ? seq(items[0]) : null });
}

export function sessionPage(payload: unknown): {
  items: SessionRow[]; nextCursor: { updated_at: string; session_id: string } | null; catalogEpoch: string | null;
} {
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
  const epoch = recordValue(payload)?.catalog_epoch;
  return { items: items as unknown as SessionRow[], nextCursor: next as { updated_at: string; session_id: string } | null,
    catalogEpoch: typeof epoch === "string" ? epoch : null };
}

/** 目录刷新范围：全量，或已知发生变化的会话 key（追加、回复结束、新 head）。 */
export type SessionsRefresh = "full" | readonly string[];
/** 刷新触发：另可携带一份活动快照，到真正读取时再与当时的目录比对。 */
export type SessionsRefreshTrigger = SessionsRefresh | { readonly heads: Readonly<Record<string, number>> };

/** 活动快照与目录同前缀、同可见性：目录有而快照没有说明有移除（全量）；其余返回水位前进或目录未知的会话。 */
export function snapshotChanges(current: readonly SessionRow[], heads: Readonly<Record<string, number>>): SessionsRefresh {
  const known = new Map(current.map((session) => [session.key, session.head_seq]));
  if (current.some((session) => heads[session.key] === undefined)) return "full";
  return Object.keys(heads).filter((key) => {
    const head = known.get(key);
    return head === undefined || head < heads[key];
  });
}

/**
 * 第一页合并进现有目录。只有触发刷新的会话全部落在第一页、且第一页最旧一行仍是旧值时才合并；
 * 排序时间不是变更版本（时钟回拨、同时间并列都可能让变化行留在页外），否则返回 null 交给全量读取。
 */
export function mergeRecentSessions(
  current: readonly SessionRow[], page: { items: SessionRow[]; nextCursor: unknown }, changed: readonly string[],
): SessionRow[] | null {
  if (!page.nextCursor) return page.items;
  const listed = new Set(page.items.map((session) => session.key));
  if (changed.some((key) => !listed.has(key))) return null;
  const boundary = page.items.at(-1);
  const previous = boundary && current.find((session) => session.key === boundary.key);
  if (!boundary || !previous || previous.updated_at !== boundary.updated_at) return null;
  return [...page.items, ...current.filter((session) => !listed.has(session.key))];
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

// 附件对象由草稿持有；移除或发送后可回收，不把回执留在持久缓存里。
const uploadedFiles = new WeakMap<ComposerFile, UploadedFile>();

/** 最多并行上传三个附件；失败停止新上传，回执按原附件顺序返回。 */
export async function uploadFiles(files: ComposerFile[], signal: AbortSignal): Promise<UploadedFile[]> {
  // 1. 本次失败只中止本批在途读取，不撤销已经落盘的附件。
  const batch = new AbortController();
  const activeSignal = AbortSignal.any([signal, batch.signal]);
  const result = new Array<UploadedFile>(files.length);
  let next = 0;
  let failure: { error: unknown } | undefined;
  const uploadNext = async () => {
    while (next < files.length && !activeSignal.aborted) {
      const index = next++;
      try {
        result[index] = await uploadFile(files[index], activeSignal);
      } catch (error) {
        if (!failure) {
          failure = { error };
          batch.abort();
        }
        return;
      }
    }
  };
  // 2. 等已开始的上传结束，才让输入框恢复重试；旧请求不能与重试混在一起。
  await Promise.all(Array.from({ length: Math.min(3, files.length) }, uploadNext));
  if (failure) throw failure.error;
  signal.throwIfAborted();
  return result;
}

/** 只有收到并校验成功回执后，才记住这份草稿附件的服务端身份。 */
async function uploadFile(file: ComposerFile, signal: AbortSignal): Promise<UploadedFile> {
  const cached = uploadedFiles.get(file);
  if (cached) return cached;
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
  const uploaded = uploadedFileResponse(payload);
  uploadedFiles.set(file, uploaded);
  return uploaded;
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
