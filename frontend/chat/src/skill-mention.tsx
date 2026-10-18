import { FileText } from "lucide-react";
import { useCallback, useEffect, useId, useLayoutEffect, useMemo, useRef, useState, useSyncExternalStore, type KeyboardEvent as ReactKeyboardEvent, type ReactNode, type RefObject } from "react";
import { fetchChatJson } from "./web-chat-data";

/** 运行时技能目录里聊天菜单用到的字段（来自 /api/chat/runtime/capabilities）。 */
export interface SkillOption {
  name: string;
  displayName: string;
  description: string;
  source: string;
  available: boolean;
  missing: string;
}

/** 光标前正在输入的触发词：sigil 在行首或空白后，查询不含空白。 */
export interface MentionTrigger {
  sigil: "/" | "@";
  query: string;
  start: number;
  end: number;
}

// 与服务端 standard_tools.skills.mentioned_skills 同一规则：行首或空白后的 / 接已知名称，
// 名称后是结尾、空白或标点；同一位置取最长名称。
const MENTION_START = /(^|\s)\//gu;
const MENTION_END = new Set([..."，。；：！？、)）\"',;:!?"]);
const WORD = /[\p{L}\p{N}_]/u;
const TRIGGER = /(^|\s)([/@])([^\s/@]{0,50})$/u;
const SOURCE_LABEL: Record<string, string> = { builtin: "内置", workspace: "工作区", user: "我的" };

// 页面共享一份技能目录；每次打开菜单重新读取，热安装与依赖变化随之反映。
interface SkillCatalog {
  skills: readonly SkillOption[];
  status: "idle" | "loading" | "ready" | "failed";
}
let catalog: SkillCatalog = { skills: [], status: "idle" };
const listeners = new Set<() => void>();
function publish(next: SkillCatalog) {
  catalog = next;
  listeners.forEach((listener) => listener());
}

function readSkill(row: unknown): SkillOption | null {
  if (!row || typeof row !== "object") return null;
  const value = row as Record<string, unknown>;
  if (typeof value.name !== "string" || !value.name) return null;
  const text = (key: string) => typeof value[key] === "string" ? value[key] as string : "";
  return { name: value.name, displayName: text("display_name") || value.name, description: text("description"),
    source: text("source"), available: value.available !== false, missing: text("missing") };
}

// revalidate 为 false 时只在从未读过时读取（消息气泡）；菜单打开时总是重新读取。
export function loadSkills(revalidate = true): void {
  if (catalog.status === "loading" || (!revalidate && catalog.status !== "idle")) return;
  publish({ ...catalog, status: "loading" });
  fetchChatJson<{ skills?: unknown }>("/api/chat/runtime/capabilities")
    .then((payload) => publish({ status: "ready", skills: (Array.isArray(payload.skills) ? payload.skills : [])
      .map(readSkill).filter((skill): skill is SkillOption => skill !== null) }))
    .catch(() => publish({ ...catalog, status: "failed" }));
}

function useSkillCatalog(): SkillCatalog {
  return useSyncExternalStore((listener) => {
    listeners.add(listener);
    return () => listeners.delete(listener);
  }, () => catalog);
}

export function useSkills(): readonly SkillOption[] {
  return useSkillCatalog().skills;
}

// 文本里所有已知技能提及的位置，start 指向 "/"。
export function findMentions(text: string, skills: readonly SkillOption[]): { start: number; end: number; skill: SkillOption }[] {
  const ordered = [...skills].sort((a, b) => b.name.length - a.name.length);
  const ends = (end: number) => end === text.length || /\s/u.test(text[end]) || MENTION_END.has(text[end])
    || (text[end] === "." && (end + 1 === text.length || !WORD.test(text[end + 1])));
  const found: { start: number; end: number; skill: SkillOption }[] = [];
  for (const match of text.matchAll(MENTION_START)) {
    const start = (match.index ?? 0) + match[1].length;
    const skill = ordered.find((item) => text.startsWith(item.name, start + 1) && ends(start + 1 + item.name.length));
    if (skill) found.push({ start, end: start + 1 + skill.name.length, skill });
  }
  return found;
}

export function findTrigger(value: string, caret: number): MentionTrigger | null {
  const match = TRIGGER.exec(value.slice(0, caret));
  if (!match) return null;
  return { sigil: match[2] as "/" | "@", query: match[3], start: caret - match[3].length - 1, end: caret };
}

// 名称或显示名前缀 > 包含 > 描述包含；不可用的技能照样列出，排在同分的后面。
function rank(skills: readonly SkillOption[], query: string): SkillOption[] {
  const q = query.toLowerCase();
  const score = (skill: SkillOption) => {
    const name = skill.name.toLowerCase(), display = skill.displayName.toLowerCase();
    if (!q) return 1;
    if (name.startsWith(q) || display.startsWith(q)) return 3;
    if (name.includes(q) || display.includes(q)) return 2;
    return skill.description.toLowerCase().includes(q) ? 1 : 0;
  };
  return skills.map((skill) => ({ skill, score: score(skill) - (skill.available ? 0 : 0.5) }))
    .filter((row) => row.score > 0).sort((a, b) => b.score - a.score).map((row) => row.skill);
}

function Highlight({ text, query }: { text: string; query: string }) {
  const index = query ? text.toLowerCase().indexOf(query.toLowerCase()) : -1;
  if (index < 0) return <>{text}</>;
  return <>{text.slice(0, index)}<b>{text.slice(index, index + query.length)}</b>{text.slice(index + query.length)}</>;
}

/** 输入框里的 / 与 @ 技能菜单：选中后把 "/名称 " 写进草稿，发送时由服务端识别并注入技能正文。 */
export function useSkillMention({ value, textareaRef, setValue }: {
  value: string;
  textareaRef: RefObject<HTMLTextAreaElement | null>;
  setValue: (next: string) => void;
}) {
  const { skills, status } = useSkillCatalog();
  const listId = useId();
  const [caret, setCaret] = useState<number | null>(null);
  // 高亮项跟着触发词与查询走，换词即回到第一项。
  const [highlight, setHighlight] = useState({ query: "", index: 0 });
  // Esc 按触发位置记住；之后继续输入都当普通文字，这次触发消失（删掉 sigil、换会话、发送）即失效。
  const [dismissed, setDismissed] = useState<MentionTrigger | null>(null);
  const trigger = caret === null ? null : findTrigger(value, caret);
  if (dismissed && caret !== null && (trigger?.start !== dismissed.start || trigger.sigil !== dismissed.sigil)) {
    setDismissed(null);
  }
  const query = trigger ? `${trigger.sigil}${trigger.start}:${trigger.query}` : "";
  const active = highlight.query === query ? highlight.index : 0;
  const setActive = useCallback((index: number) => setHighlight({ query, index }), [query]);
  const filter = trigger?.query ?? null;
  const items = useMemo(() => filter === null ? [] : rank(skills, filter), [skills, filter]);
  const closed = dismissed !== null && trigger?.start === dismissed.start && trigger.sigil === dismissed.sigil;
  // 光标停在一个完整提及之后（刚插入或退格回来）不再弹出，退格改为整段选中。
  const complete = trigger?.sigil === "/" && skills.some((skill) => skill.name === trigger.query);
  const open = trigger !== null && !closed && !complete;
  const triggered = trigger !== null;
  useEffect(() => { if (triggered) loadSkills(); }, [triggered]);

  const sync = useCallback(() => {
    const textarea = textareaRef.current;
    setCaret(textarea && textarea.selectionStart === textarea.selectionEnd ? textarea.selectionStart : null);
  }, [textareaRef]);

  const pending = useRef<{ value: string; position: number } | null>(null);
  const pick = useCallback((skill: SkillOption | undefined) => {
    const textarea = textareaRef.current;
    if (!skill || !skill.available || !trigger || !textarea) return;
    const rest = value.slice(trigger.end);
    const inserted = `/${skill.name}${rest.startsWith(" ") ? "" : " "}`;
    const next = value.slice(0, trigger.start) + inserted + rest;
    const position = trigger.start + inserted.length + (rest.startsWith(" ") ? 1 : 0);
    pending.current = { value: next, position };
    setValue(next);
    setCaret(position);
  }, [textareaRef, trigger, value, setValue]);
  // 光标在这次草稿提交后同步归位；放到下一帧会和紧接着的键入抢位置。
  useLayoutEffect(() => {
    const textarea = textareaRef.current;
    const target = pending.current;
    if (!textarea || !target || textarea.value !== target.value) return;
    pending.current = null;
    textarea.focus();
    textarea.setSelectionRange(target.position, target.position);
  }, [value, textareaRef]);

  // 菜单打开时接管方向键、回车、Tab 与 Esc；返回 true 表示事件已处理。
  const onKeyDown = useCallback((event: ReactKeyboardEvent<HTMLTextAreaElement>): boolean => {
    const native = event.nativeEvent;
    if (native.isComposing || native.keyCode === 229) return false;
    if (open && items.length) {
      const step = (delta: number) => {
        let next = active;
        for (let index = 0; index < items.length; index += 1) {
          next = (next + delta + items.length) % items.length;
          if (items[next].available) break;
        }
        setActive(next);
      };
      if (event.key === "ArrowDown" || event.key === "ArrowUp") {
        event.preventDefault();
        step(event.key === "ArrowDown" ? 1 : -1);
        return true;
      }
      if ((event.key === "Enter" && !event.shiftKey) || event.key === "Tab") {
        event.preventDefault();
        pick(items[active]?.available ? items[active] : items.find((skill) => skill.available));
        return true;
      }
    }
    // 目录还在读取时回车不把半截的 "/" 当消息发出去。
    if (open && !items.length && status === "loading" && event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      return true;
    }
    if (open && event.key === "Escape") {
      event.preventDefault();
      setDismissed(trigger);
      return true;
    }
    // 紧贴提及的退格先选中整个 /名称，再按一次删除，提及像一个整体。
    if (event.key === "Backspace" && !open) {
      const textarea = event.currentTarget;
      const end = textarea.selectionStart;
      if (end !== textarea.selectionEnd) return false;
      const mention = mentionBefore(value, end, skills);
      if (mention !== null) {
        event.preventDefault();
        textarea.setSelectionRange(mention, end);
        return true;
      }
    }
    return false;
  }, [open, items, status, active, setActive, trigger, pick, value, skills]);

  const menu = open && trigger ? <SkillMenu id={listId} items={items} query={trigger.query} active={active}
    status={status} onHover={setActive} onPick={pick} /> : null;
  const close = useCallback(() => setCaret(null), []);
  // 焦点留在输入框，读屏通过 activedescendant 跟随高亮项。
  const inputProps = {
    "aria-autocomplete": "list" as const,
    "aria-expanded": menu !== null,
    "aria-controls": menu ? listId : undefined,
    "aria-activedescendant": menu && items[active] ? `${listId}-${active}` : undefined,
  };
  return { menu, sync, close, onKeyDown, inputProps };
}

// 光标正好落在一个已知技能提及之后时返回它的起点。
function mentionBefore(value: string, caret: number, skills: readonly SkillOption[]): number | null {
  return findMentions(value.slice(0, caret), skills).find((mention) => mention.end === caret)?.start ?? null;
}

function SkillMenu({ id, items, query, active, status, onHover, onPick }: {
  id: string;
  items: SkillOption[];
  query: string;
  active: number;
  status: SkillCatalog["status"];
  onHover: (index: number) => void;
  onPick: (skill: SkillOption) => void;
}) {
  const counts = new Map<string, number>();
  items.forEach((skill) => counts.set(skill.name, (counts.get(skill.name) ?? 0) + 1));
  return <div id={id} className="skill-menu" role="listbox" aria-label="技能" data-escape-layer
    onMouseDown={(event) => event.preventDefault()}>
    {items.length ? <div className="skill-menu__section">技能</div> : null}
    {items.map((skill, index) => <div key={`${skill.source}:${skill.name}`} id={`${id}-${index}`} role="option" aria-selected={index === active}
      aria-disabled={!skill.available} className="skill-menu__row" data-active={index === active || undefined}
      ref={index === active ? (node) => node?.scrollIntoView({ block: "nearest" }) : undefined}
      onMouseMove={() => skill.available && index !== active && onHover(index)}
      onClick={() => onPick(skill)}>
      <FileText aria-hidden="true" />
      <span className="skill-menu__label"><Highlight text={skill.displayName} query={query} />
        <code><Highlight text={skill.name} query={query} /></code></span>
      <span className="skill-menu__desc">{skill.available ? skill.description : `不可用：${skill.missing}`}</span>
      {(counts.get(skill.name) ?? 0) > 1 ? <span className="skill-menu__source">{SOURCE_LABEL[skill.source] ?? skill.source}</span> : null}
    </div>)}
    {!items.length ? <div className="skill-menu__empty">{status === "failed" ? "读取技能失败，请稍后再试"
      : status === "ready" ? "没有匹配的技能" : "正在读取技能…"}</div> : null}
  </div>;
}

/** 用户消息里的已知技能提及显示成强调色文字；未知的 /xxx 保持原样。 */
export function MentionText({ text }: { text: string }) {
  const skills = useSkills();
  useEffect(() => { if (text.includes("/")) loadSkills(false); }, [text]);
  const nodes: ReactNode[] = [];
  let last = 0;
  for (const { start, end, skill } of findMentions(text, skills)) {
    nodes.push(text.slice(last, start), <span key={start} className="skill-mention" title={skill.description}>
      <span aria-hidden="true">/</span>{skill.displayName}</span>);
    last = end;
  }
  if (!nodes.length) return <>{text}</>;
  nodes.push(text.slice(last));
  return <>{nodes}</>;
}
