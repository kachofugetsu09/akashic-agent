import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { createPortal } from "react-dom";
import { chatPortalRoot } from "./chat-layout";
import type { TimelineMessage } from "./message-timeline";
import {
  groupByMessage, groupByPart, searchTimeline, snippet,
  type SearchGroup, type SearchHit, type SearchScope,
} from "./session-search";

/**
 * 会话内搜索浮层。
 *
 * 位置：入口在会话标题行右侧（与「新会话」同一排、同尺寸），浮层本身挂进
 * 聊天根节点（`chatPortalRoot()`），不是 `document.body`：聊天断点读的是根
 * 节点自身的宽度，挂到外面后作用域样式与 `data-chat-width` 全部命中不了。
 * 遮罩只压主区，侧栏仍然可见可点——本来也只在当前会话里找东西。
 *
 * 快捷键用 Ctrl/⌘+F：Ctrl+K 已被侧栏的会话搜索占用（紧凑断点下会打开导航
 * 抽屉并聚焦其中的搜索框），两者作用域不同，不能共用一个组合。
 */

const SCOPE_LABEL: Record<SearchScope, string> = { all: "全部", text: "正文", tool: "工具" };

const KIND_LABEL: Record<SearchHit["kind"], string> = {
  input: "我", text: "正文", thinking: "思考",
  tool: "工具", "tool-result": "输出", notice: "提示",
};

/** 消息在时间线里的日期，分组标题用它区分「今天 / 昨天 / 更早」。 */
function dayLabel(timestamp: string): string {
  const date = new Date(timestamp);
  if (Number.isNaN(date.getTime())) return "";
  const today = new Date();
  const sameDay = (a: Date, b: Date) => a.getFullYear() === b.getFullYear()
    && a.getMonth() === b.getMonth() && a.getDate() === b.getDate();
  if (sameDay(date, today)) return "今天";
  const yesterday = new Date(today);
  yesterday.setDate(today.getDate() - 1);
  if (sameDay(date, yesterday)) return "昨天";
  return `${date.getMonth() + 1} 月 ${date.getDate()} 日`;
}

export interface SessionSearchProps {
  messages: readonly TimelineMessage[];
  open: boolean;
  onClose: () => void;
  /** 跳到某个 part：由视图负责滚动与展开折叠。 */
  onJump: (hit: SearchHit) => void;
}

export function SessionSearch({ messages, open, onClose, onJump }: SessionSearchProps) {
  const [query, setQuery] = useState("");
  const [scope, setScope] = useState<SearchScope>("all");
  const [active, setActive] = useState(0);
  const inputRef = useRef<HTMLInputElement>(null);
  const listRef = useRef<HTMLDivElement>(null);

  const result = useMemo(() => searchTimeline(messages, query, scope), [messages, query, scope]);
  const groups = useMemo(() => groupByMessage(groupByPart(result.hits)), [result.hits]);
  const flat = useMemo(() => groups.flatMap((group) => group.groups), [groups]);
  // 按天分组：同一天的消息合成一组，标题只在换天时出现一次。
  const dayGroups = useMemo(() => {
    const buckets: { label: string; count: number; items: SearchGroup[] }[] = [];
    for (const message of groups) {
      const label = dayLabel(messages[message.order]?.timestamp ?? "");
      const last = buckets.at(-1);
      if (last && last.label === label) {
        last.items.push(...message.groups);
        last.count += message.groups.reduce((total, group) => total + group.count, 0);
      } else {
        buckets.push({
          label, items: [...message.groups],
          count: message.groups.reduce((total, group) => total + group.count, 0),
        });
      }
    }
    return buckets;
  }, [groups, messages]);

  // 换词或换范围后回到第一条，否则选中的那行可能已经不存在了。
  useEffect(() => { setActive(0); }, [query, scope]);

  // 打开时聚焦并选中已有词：接着上一次的搜索比从零开始更常见。
  useEffect(() => {
    if (!open) return;
    const input = inputRef.current;
    if (!input) return;
    // 触屏不自动 select：会整段反蓝，且软键盘行为也不受控。
    const coarse = window.matchMedia("(pointer: coarse)").matches;
    input.focus();
    if (!coarse) input.select();
  }, [open]);

  const jump = useCallback((index: number) => {
    const group = flat[index];
    if (!group) return;
    setActive(index);
    onJump(group.hit);
  }, [flat, onJump]);

  // 键盘：↑↓ 选，Enter 跳，Esc 关。焦点在输入框里也照常工作。
  useEffect(() => {
    if (!open) return;
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape") { event.preventDefault(); onClose(); return; }
      if (event.key === "ArrowDown" || event.key === "ArrowUp") {
        event.preventDefault();
        if (!flat.length) return;
        const delta = event.key === "ArrowDown" ? 1 : -1;
        setActive((current) => (current + delta + flat.length) % flat.length);
        return;
      }
      if (event.key === "Enter") {
        event.preventDefault();
        jump(active);
      }
    };
    document.addEventListener("keydown", onKeyDown, true);
    return () => document.removeEventListener("keydown", onKeyDown, true);
  }, [open, flat.length, active, jump, onClose]);

  // 选中行跟着键盘滚进视野。
  useEffect(() => {
    const node = listRef.current?.querySelector<HTMLElement>("[data-active='true']");
    node?.scrollIntoView({ block: "nearest" });
  }, [active]);

  if (!open) return null;

  // 挂进 chat-main 而不是聊天根节点：遮罩只压主区，侧栏仍然可见可点——
  // 本来也只在当前会话里找东西。断点属性在根节点上，chat-main 是它的后代，
  // 作用域样式照样命中。
  const root = document.querySelector<HTMLElement>(".chat-main") ?? chatPortalRoot();
  const content = (
    <div className="session-search" role="dialog" aria-modal="true" aria-label="在这个会话里搜索"
      onMouseDown={(event) => { if (event.target === event.currentTarget) onClose(); }}>
      <div className="session-search__panel">
        <div className="session-search__field">
          <SearchGlyph />
          <input
            ref={inputRef}
            type="text"
            value={query}
            placeholder="在这个会话里找…"
            aria-label="搜索关键词"
            autoComplete="off"
            spellCheck={false}
            onChange={(event) => setQuery(event.target.value)}
          />
          <button type="button" className="session-search__cancel" onClick={onClose}>取消</button>
        </div>

        <div className="session-search__bar">
          <span className="session-search__count">
            {!query.trim() ? "—"
              : result.hits.length === 0 ? "没有匹配"
              : <><b>{result.hits.length}</b> 处 · {result.messages} 条消息</>}
          </span>
          <span className="session-search__scopes" role="group" aria-label="搜索范围">
            {(Object.keys(SCOPE_LABEL) as SearchScope[]).map((key) => (
              <button key={key} type="button" data-scope={key}
                aria-pressed={scope === key}
                onClick={() => setScope(key)}>{SCOPE_LABEL[key]}</button>
            ))}
          </span>
        </div>

        <div className="session-search__list" ref={listRef}>
          {!query.trim() ? (
            <p className="session-search__empty">输入关键词。<br />收起的过程也会被搜到。</p>
          ) : flat.length === 0 ? (
            <p className="session-search__empty"><b>这个会话里没有</b><br />换个词，或者把范围切到「全部」</p>
          ) : dayGroups.map((day, dayIndex) => (
            <div key={`${day.label}:${dayIndex}`} className="session-search__group">
              {/* 一天一个标题：每条消息都写一次「今天」只是在重复。 */}
              <p className="session-search__group-head">
                <span>{day.label}</span>
                <span className="session-search__group-count">{day.count} 处</span>
              </p>
              {day.items.map((group) => {
                const parts = snippet(group.hit.text, group.hit.start, group.hit.length);
                const index = flat.indexOf(group);
                return (
                  <button key={group.key} type="button"
                    className="session-search__hit"
                    data-active={index === active ? "true" : undefined}
                    onClick={() => jump(index)}>
                    <span className={`session-search__kind session-search__kind--${group.hit.kind}`}>
                      <i aria-hidden="true" />{KIND_LABEL[group.hit.kind]}
                      {group.hit.kind === "tool" ? ` · ${group.hit.field === "target" ? "目标" : "动作"}` : ""}
                    </span>
                    <span className="session-search__snippet">
                      {parts.pre}<mark>{parts.hit}</mark>{parts.post}
                    </span>
                    {group.count > 1 ? (
                      <span className="session-search__more">这段里有 {group.count} 处</span>
                    ) : null}
                  </button>
                );
              })}
            </div>
          ))}
        </div>

        <div className="session-search__foot">
          <span className="session-search__keys"><kbd>↑</kbd><kbd>↓</kbd> 选</span>
          <span className="session-search__keys"><kbd>Enter</kbd> 跳过去</span>
          <span className="session-search__hint">点结果直接跳过去</span>
        </div>
      </div>
    </div>
  );
  // 浮层挂进聊天根节点：断点属性与作用域样式都写在那个根节点上。
  return createPortal(content, root);
}

function SearchGlyph() {
  return (
    <svg width="15" height="15" viewBox="0 0 16 16" fill="none" aria-hidden="true">
      <circle cx="6.6" cy="6.6" r="4.4" stroke="currentColor" strokeWidth="1.6" />
      <path d="M10 10l3.4 3.4" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" />
    </svg>
  );
}
