import { toolDataText } from "./message-rendering-policy";
import type { TimelineMessage, TimelinePart } from "./message-timeline";

/**
 * 会话内搜索的索引与命中。
 *
 * 为什么不用浏览器查找：过程折叠（`CollapsedToolGroup`）收起的步骤在展开前
 * 不在 DOM 里，`window.find` 与 Ctrl+F 都只看到当前渲染出来的节点，会对着
 * 「N 个更早步骤」那一行说没有匹配，而要找的内容就在里面。所以索引建在
 * 时间线数据上，折叠只是视图状态，搜索穿得过去。
 */

/** 命中所在的 part 类别：面板按它显示类型圆点，也按它决定跳转时要不要展开折叠。 */
export type SearchHitKind = "input" | "text" | "thinking" | "tool" | "tool-result" | "notice";

export interface SearchHit {
  message_id: string;
  /** part 在该消息 body.parts 里的下标，与 DOM 上的 data-part-index 同一个值。 */
  part_index: number;
  kind: SearchHitKind;
  /** 命中的字段：工具行分动作名与目标，正文类只有 text。 */
  field: "text" | "name" | "target";
  /** 命中在该字段文本里的起点与长度。 */
  start: number;
  length: number;
  /** 该 part 的完整可搜文本（按字段拼接后的那份），面板用来截片段。 */
  text: string;
  /** 消息在时间线里的位置，用来分组与排序。 */
  order: number;
}

export interface SearchResult {
  hits: SearchHit[];
  /** 命中涉及的消息数，面板显示「N 处 · M 条消息」。 */
  messages: number;
}

/** 只搜「你说的话」时排掉过程项；只搜「工具」时反过来。 */
export type SearchScope = "all" | "text" | "tool";

/** 一个 part 里可被搜到的字段。工具行拆成动作与目标两段，其余只有正文。 */
function partFields(part: TimelinePart): { field: SearchHit["field"]; value: string }[] {
  if ("display" in part) {
    if (part.display === "thinking") {
      return typeof part.text === "string" && part.text ? [{ field: "text", value: part.text }] : [];
    }
    if (part.display === "data") {
      // 工具结果的 data part：展示文本由渲染策略统一给出，与页面上看到的一致。
      const text = toolDataText(part.value);
      return text ? [{ field: "text", value: text }] : [];
    }
    return [];
  }
  if (part.kind === "tool_call") {
    const target = toolCallTarget(part.arguments);
    return [
      ...(part.name ? [{ field: "name" as const, value: part.name }] : []),
      ...(target ? [{ field: "target" as const, value: target }] : []),
    ];
  }
  if (part.kind === "session.reference") {
    // 引用的会话标题也是可搜内容：搜标题能找回「我当时引用了哪一段」。
    const value = part.value.sessions.map((session) => session.title).join("\n");
    return value ? [{ field: "text", value }] : [];
  }
  if (part.kind === "text" || part.kind === "reply_ref") {
    return part.value ? [{ field: "text", value: part.value }] : [];
  }
  if (part.kind === "context.notice") {
    return part.value.text ? [{ field: "text", value: part.value.text }] : [];
  }
  return [];
}

/** 工具调用的目标：命令、路径、URL 这些用户在搜的东西。 */
function toolCallTarget(args: Record<string, unknown>): string {
  for (const key of ["command", "path", "file_path", "url", "query", "pattern", "name"]) {
    const value = args[key];
    if (typeof value === "string" && value) return value;
  }
  return "";
}

function partKind(message: TimelineMessage, part: TimelinePart): SearchHitKind {
  if (message.body.kind === "input") return "input";
  if (message.body.kind === "tool_result") return "tool-result";
  if ("display" in part) return part.display === "thinking" ? "thinking" : "tool-result";
  if (part.kind === "tool_call") return "tool";
  if (part.kind === "context.notice") return "notice";
  return "text";
}

function inScope(kind: SearchHitKind, scope: SearchScope): boolean {
  if (scope === "text") return kind === "input" || kind === "text";
  if (scope === "tool") return kind === "tool" || kind === "tool-result";
  return true;
}

/**
 * 在一段文本里找出全部命中。大小写不敏感，但保留原文位置，
 * 这样高亮与截片段都能用回原字符串。
 */
function matchAll(text: string, needle: string): number[] {
  const haystack = text.toLowerCase();
  const starts: number[] = [];
  let from = 0;
  while (from <= haystack.length - needle.length) {
    const at = haystack.indexOf(needle, from);
    if (at < 0) break;
    starts.push(at);
    from = at + needle.length;
  }
  return starts;
}

/**
 * 从时间线里找出全部命中。`needle` 已经由调用方 trim 过；
 * 空串返回空结果，面板据此显示初始态而不是「全部命中」。
 */
export function searchTimeline(
  messages: readonly TimelineMessage[],
  rawNeedle: string,
  scope: SearchScope = "all",
): SearchResult {
  const needle = rawNeedle.trim().toLowerCase();
  if (!needle) return { hits: [], messages: 0 };
  const hits: SearchHit[] = [];
  const touched = new Set<string>();
  messages.forEach((message, order) => {
    if (message.body.kind === "control") return;
    message.body.parts.forEach((part, part_index) => {
      const kind = partKind(message, part);
      if (!inScope(kind, scope)) return;
      for (const { field, value } of partFields(part)) {
        for (const start of matchAll(value, needle)) {
          hits.push({
            message_id: message.id, part_index, kind, field,
            start, length: needle.length, text: value, order,
          });
          touched.add(message.id);
        }
      }
    });
  });
  return { hits, messages: touched.size };
}

/**
 * 面板里同一 part 的多个命中合成一行：它们指向同一个位置，拆开只会让列表
 * 看起来在重复自己。返回每个 part 的第一条命中，并带上该 part 的命中数。
 */
export interface SearchGroup {
  key: string;
  hit: SearchHit;
  count: number;
}

export function groupByPart(hits: readonly SearchHit[]): SearchGroup[] {
  const groups = new Map<string, SearchGroup>();
  for (const hit of hits) {
    const key = `${hit.message_id}:${hit.part_index}`;
    const existing = groups.get(key);
    if (existing) existing.count += 1;
    else groups.set(key, { key, hit, count: 1 });
  }
  return [...groups.values()];
}

/** 面板按消息分组：先按消息归拢，消息内按 part 顺序。 */
export function groupByMessage(groups: readonly SearchGroup[]): { order: number; message_id: string; groups: SearchGroup[] }[] {
  const byMessage = new Map<string, { order: number; message_id: string; groups: SearchGroup[] }>();
  for (const group of groups) {
    const existing = byMessage.get(group.hit.message_id);
    if (existing) existing.groups.push(group);
    else byMessage.set(group.hit.message_id, { order: group.hit.order, message_id: group.hit.message_id, groups: [group] });
  }
  return [...byMessage.values()].sort((left, right) => left.order - right.order);
}

/** 命中片段：前后各留一点上下文，片段里保留原文换行供面板压成一行。 */
export function snippet(text: string, start: number, length: number, before = 24, after = 26): {
  pre: string; hit: string; post: string;
} {
  const from = Math.max(0, start - before);
  const to = Math.min(text.length, start + length + after);
  return {
    pre: (from > 0 ? "…" : "") + text.slice(from, start),
    hit: text.slice(start, start + length),
    post: text.slice(start + length, to) + (to < text.length ? "…" : ""),
  };
}
