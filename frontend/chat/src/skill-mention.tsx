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

// 与服务端 standard_tools.skills.SKILL_MENTION 同一规则：名称后不能紧跟路径字符。
const MENTION = /(^|\s)\/([\p{L}\p{N}_][\p{L}\p{N}_-]*(?:\.[\p{L}\p{N}_-]+)*)(?![\p{L}\p{N}_/-]|\.[\p{L}\p{N}_])/gu;
const TRIGGER = /(^|\s)([/@])([^\s/@]{0,50})$/u;
const SOURCE_LABEL: Record<string, string> = { builtin: "内置", workspace: "工作区", user: "我的" };

// 历史消息共享目录；每次重新打开菜单都刷新当前可用性。
let catalog: { skills: readonly SkillOption[]; status: "idle" | "loading" | "ready" | "error"; error: string } = {
  skills: [], status: "idle", error: "",
};
let loading: Promise<void> | null = null;
const listeners = new Set<() => void>();
const subscribe = (listener: () => void) => { listeners.add(listener); return () => { listeners.delete(listener); }; };

function readSkill(row: unknown): SkillOption {
  if (!row || typeof row !== "object") throw new Error("技能目录格式错误");
  const value = row as Record<string, unknown>;
  if (typeof value.name !== "string" || !value.name || typeof value.available !== "boolean"
    || !["display_name", "description", "source", "missing"].every((key) => typeof value[key] === "string")) {
    throw new Error("技能目录格式错误");
  }
  return { name: value.name, displayName: value.display_name as string || value.name,
    description: value.description as string, source: value.source as string,
    available: value.available, missing: value.missing as string };
}

function loadSkills(refresh = false): void {
  if (loading || (!refresh && catalog.status !== "idle")) return;
  catalog = { skills: [], status: "loading", error: "" };
  loading = fetchChatJson<{ skills: unknown }>("/api/chat/runtime/capabilities")
    .then((payload) => {
      if (!Array.isArray(payload.skills)) throw new Error("技能目录格式错误");
      catalog = { skills: payload.skills.map(readSkill), status: "ready", error: "" };
    })
    .catch((error: unknown) => {
      catalog = { skills: [], status: "error", error: error instanceof Error ? error.message : String(error) };
    })
    .finally(() => { loading = null; listeners.forEach((listener) => listener()); });
  listeners.forEach((listener) => listener());
}

function useSkills() {
  return useSyncExternalStore(subscribe, () => catalog);
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
  const { skills, status, error } = useSkills();
  const [caret, setCaret] = useState<number | null>(null);
  // 高亮项跟着触发词与查询走，换词即回到第一项。
  const [highlight, setHighlight] = useState({ query: "", index: 0 });
  // Esc 按触发位置记住；之后继续输入都当普通文字，该位置的 sigil 被删掉才失效。
  const [dismissed, setDismissed] = useState<MentionTrigger | null>(null);
  const trigger = caret === null ? null : findTrigger(value, caret);
  const query = trigger ? `${trigger.sigil}${trigger.start}:${trigger.query}` : "";
  const active = highlight.query === query ? highlight.index : 0;
  const setActive = useCallback((index: number) => setHighlight({ query, index }), [query]);
  const filter = trigger?.query ?? null;
  const items = useMemo(() => filter === null ? [] : rank(skills, filter), [skills, filter]);
  const closed = dismissed !== null && value[dismissed.start] === dismissed.sigil
    && trigger?.start === dismissed.start && trigger.sigil === dismissed.sigil;
  // 光标停在一个完整提及之后（刚插入或退格回来）不再弹出，退格改为整段选中。
  const complete = trigger?.sigil === "/" && skills.some((skill) => skill.name === trigger.query);
  const open = trigger !== null && !closed && !complete;
  const triggerStart = trigger?.start;
  const triggerSigil = trigger?.sigil;
  useEffect(() => { if (triggerStart !== undefined) loadSkills(true); }, [triggerStart, triggerSigil]);
  const menuId = useId();

  const sync = useCallback(() => {
    const textarea = textareaRef.current;
    setCaret(textarea && textarea.selectionStart === textarea.selectionEnd ? textarea.selectionStart : null);
    setDismissed((previous) => previous && textarea?.value[previous.start] === previous.sigil ? previous : null);
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
  }, [open, items, active, setActive, trigger, pick, value, skills]);

  const menu = open && trigger ? <SkillMenu id={menuId} items={items} query={trigger.query} active={active}
    status={status} error={error} onHover={setActive} onPick={pick} /> : null;
  const close = useCallback(() => setCaret(null), []);
  return { menu, sync, close, onKeyDown, inputProps: {
    "aria-autocomplete": "list" as const, "aria-controls": open ? menuId : undefined,
    "aria-activedescendant": open && items[active] ? `${menuId}-${active}` : undefined,
  } };
}

// 光标正好落在一个已知技能提及之后时返回它的起点。
function mentionBefore(value: string, caret: number, skills: readonly SkillOption[]): number | null {
  const names = new Set(skills.map((skill) => skill.name));
  for (const match of value.slice(0, caret).matchAll(MENTION)) {
    const start = (match.index ?? 0) + match[1].length;
    if (start + 1 + match[2].length === caret && names.has(match[2])) return start;
  }
  return null;
}

function SkillMenu({ id, items, query, active, status, error, onHover, onPick }: {
  id: string;
  items: SkillOption[];
  query: string;
  active: number;
  status: typeof catalog.status;
  error: string;
  onHover: (index: number) => void;
  onPick: (skill: SkillOption) => void;
}) {
  const counts = new Map<string, number>();
  items.forEach((skill) => counts.set(skill.name, (counts.get(skill.name) ?? 0) + 1));
  return <div id={id} className="skill-menu" role="listbox" aria-label="技能" data-escape-layer
    onMouseDown={(event) => event.preventDefault()}>
    {items.length ? <div className="skill-menu__section">技能</div> : null}
    {items.map((skill, index) => <div id={`${id}-${index}`} key={`${skill.source}:${skill.name}`} role="option" aria-selected={index === active}
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
    {!items.length ? <div className="skill-menu__empty">{status === "error" ? `没能读取技能：${error}。重新输入 / 可重试。`
      : status === "loading" || status === "idle" ? "正在读取技能…" : query ? "没有匹配的技能" : "没有已安装的技能"}</div> : null}
  </div>;
}

/** 用户消息里的已知技能提及显示成强调色文字；未知的 /xxx 保持原样。 */
export function MentionText({ text }: { text: string }) {
  const { skills } = useSkills();
  useEffect(() => { if (text.includes("/")) loadSkills(); }, [text]);
  const byName = new Map(skills.map((skill) => [skill.name, skill]));
  const nodes: ReactNode[] = [];
  let last = 0;
  for (const match of text.matchAll(MENTION)) {
    const skill = byName.get(match[2]);
    if (!skill) continue;
    const start = (match.index ?? 0) + match[1].length;
    nodes.push(text.slice(last, start), <span key={start} className="skill-mention" title={skill.description}>
      <span aria-hidden="true">/</span>{skill.displayName}</span>);
    last = start + 1 + match[2].length;
  }
  if (!nodes.length) return <>{text}</>;
  nodes.push(text.slice(last));
  return <>{nodes}</>;
}
