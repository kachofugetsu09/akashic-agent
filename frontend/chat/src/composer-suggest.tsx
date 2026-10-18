import { FileText } from "lucide-react";
import { Fragment, useCallback, useEffect, useId, useLayoutEffect, useMemo, useRef, useState, useSyncExternalStore, type KeyboardEvent as ReactKeyboardEvent, type ReactNode, type RefObject } from "react";
import { composerSuggestSources, subscribePluginUiCatalog, usePluginUiCatalogVersion, type ComposerSuggestSource, type ComposerSuggestion } from "./plugin-ui-runtime";

/** 菜单里的一项：插件给的建议加上宿主需要的来源身份与分组。 */
interface SuggestItem extends ComposerSuggestion {
  key: string;
  source: string;
  prefix: "/" | "@";
  group: string;
}

/** 光标前正在输入的触发词：sigil 在行首或空白后，查询不含空白。 */
export interface MentionTrigger {
  sigil: "/" | "@";
  query: string;
  start: number;
  end: number;
}

// 提及写法：前缀接名称，名称后不能紧跟路径字符（"/usr/bin" 不算）；插件后端按同一写法识别。
const MENTION = /(^|\s)([/@])([\p{L}\p{N}_][\p{L}\p{N}_-]*(?:\.[\p{L}\p{N}_-]+)*)(?![\p{L}\p{N}_/-]|\.[\p{L}\p{N}_])/gu;
const TRIGGER = /(^|\s)([/@])([^\s/@]{0,50})$/u;

// 每个数据源单独加载、单独超时：一个源悬挂或出错不拖住其他源；加载中再要求刷新排到这一轮之后。
const LOAD_TIMEOUT_MS = 8_000;
interface SourceState { status: "loading" | "ready" | "error"; items: readonly SuggestItem[]; error: string; again: boolean; abort: () => void }
let states: ReadonlyMap<string, SourceState> = new Map();
const listeners = new Set<() => void>();
const subscribe = (listener: () => void) => { listeners.add(listener); return () => { listeners.delete(listener); }; };
const setState = (key: string, next: SourceState | null) => {
  const copy = new Map(states);
  if (next) copy.set(key, next); else copy.delete(key);
  states = copy;
  listeners.forEach((listener) => listener());
};

// 外部数据边界：可选字段也要是字符串，格式错误只算这个源加载失败。
function readItem(value: ComposerSuggestion, source: ComposerSuggestSource): SuggestItem {
  const optional = (field: unknown) => field === undefined || typeof field === "string";
  if (!value || typeof value.token !== "string" || !value.token || typeof value.label !== "string"
    || typeof value.description !== "string" || !optional(value.unavailable) || !optional(value.badge)) throw new Error("建议条目格式错误");
  return { token: value.token, label: value.label, description: value.description, unavailable: value.unavailable || undefined,
    badge: value.badge, key: `${source.key}:${value.badge ?? ""}:${value.token}`, source: source.key,
    prefix: source.provider.prefix, group: source.provider.label };
}

function loadSource(source: ComposerSuggestSource, refresh: boolean): void {
  const current = states.get(source.key);
  if (current?.status === "loading") { if (refresh && !current.again) setState(source.key, { ...current, again: true }); return; }
  if (current && !refresh) return;
  const controller = new AbortController();
  const loading: SourceState = { status: "loading", items: current?.items ?? [], error: "", again: false, abort: () => controller.abort() };
  setState(source.key, loading);
  void (async () => {
    // 调用、等待、条目校验在同一个边界里：同步抛错、超时、格式错误都只算这个源失败。
    const timer = window.setTimeout(() => controller.abort(new Error("读取超时")), LOAD_TIMEOUT_MS);
    const aborted = new Promise<never>((_, reject) => controller.signal.addEventListener("abort",
      () => reject(controller.signal.reason instanceof Error ? controller.signal.reason : new Error("读取已取消")), { once: true }));
    let settled: Pick<SourceState, "status" | "items" | "error">;
    try {
      const rows = await Promise.race([source.provider.load(source.context(controller.signal)), aborted]);
      settled = { status: "ready", items: rows.map((row) => readItem(row, source)), error: "" };
    } catch (error) {
      settled = { status: "error", items: [], error: error instanceof Error ? error.message : String(error) };
    } finally {
      window.clearTimeout(timer);
    }
    const latest = states.get(source.key);
    // 撤下或被新一轮取代的加载不写回。
    if (latest?.abort !== loading.abort) return;
    setState(source.key, { ...settled, again: false, abort: () => undefined });
    if (latest.again) loadSource(source, true);
  })();
}

// 目录一变就对齐：撤下的源中止在途加载并丢掉旧条目，不等菜单或消息再用到建议。
function pruneSources(): void {
  const keys = new Set(composerSuggestSources().map((source) => source.key));
  for (const [key, state] of states) if (!keys.has(key)) { state.abort(); setState(key, null); }
}
subscribePluginUiCatalog(pruneSources);

function loadSuggestions(refresh = false): void {
  pruneSources();
  composerSuggestSources().forEach((source) => loadSource(source, refresh));
}

function useSuggestions() {
  const { version } = usePluginUiCatalogVersion();
  const current = useSyncExternalStore(subscribe, () => states);
  const sources = useMemo(() => { void version; return composerSuggestSources(); }, [version]);
  return useMemo(() => {
    // 目录更新中或源已缺席时，旧缓存不参与菜单、提及与退格。
    const live = sources.map((source) => current.get(source.key));
    const items = live.flatMap((state) => state?.items ?? []);
    const failed = live.find((state) => state?.status === "error");
    const status = live.some((state) => !state) ? "idle" as const : live.some((state) => state?.status === "loading") ? "loading" as const
      : failed && !items.length ? "error" as const : "ready" as const;
    return { items, status, error: failed?.error ?? "", sources };
  }, [current, sources]);
}

export function findTrigger(value: string, caret: number, sigils: ReadonlySet<string>): MentionTrigger | null {
  const match = TRIGGER.exec(value.slice(0, caret));
  if (!match || !sigils.has(match[2])) return null;
  return { sigil: match[2] as "/" | "@", query: match[3], start: caret - match[3].length - 1, end: caret };
}

// 名称或显示名前缀 > 包含 > 描述包含；不可用的照样列出，排在同分的后面；只取声明了该触发符的源，按源顺序分组。
function rank(items: readonly SuggestItem[], query: string, sources: readonly string[]): SuggestItem[] {
  const q = query.toLowerCase();
  const score = (item: SuggestItem) => {
    const token = item.token.toLowerCase(), label = item.label.toLowerCase();
    if (!q) return 1;
    if (token.startsWith(q) || label.startsWith(q)) return 3;
    if (token.includes(q) || label.includes(q)) return 2;
    return item.description.toLowerCase().includes(q) ? 1 : 0;
  };
  return items.map((item) => ({ item, score: score(item) - (item.unavailable ? 0.5 : 0) }))
    .filter((row) => row.score > 0 && sources.includes(row.item.source))
    .sort((a, b) => sources.indexOf(a.item.source) - sources.indexOf(b.item.source) || b.score - a.score)
    .map((row) => row.item);
}

function Highlight({ text, query }: { text: string; query: string }) {
  const index = query ? text.toLowerCase().indexOf(query.toLowerCase()) : -1;
  if (index < 0) return <>{text}</>;
  return <>{text.slice(0, index)}<b>{text.slice(index, index + query.length)}</b>{text.slice(index + query.length)}</>;
}

/** 输入框里的建议菜单：选中后把 "前缀名称 " 写进草稿；条目由插件的 composer.suggest 数据源提供。 */
export function useComposerSuggest({ value, scope, textareaRef, setValue }: {
  value: string;
  /** 草稿身份；切换会话后 Esc 记下的位置不再生效。 */
  scope: string;
  textareaRef: RefObject<HTMLTextAreaElement | null>;
  setValue: (next: string) => void;
}) {
  const { items: all, status, error, sources } = useSuggestions();
  const sigils = useMemo(() => new Set(sources.flatMap((source) => source.provider.sigils)), [sources]);
  const [caret, setCaret] = useState<number | null>(null);
  // 高亮按条目身份记住；目录刷新后条目换位也不会选到别的。
  const [highlight, setHighlight] = useState({ query: "", key: "" });
  // Esc 按草稿与触发位置记住；之后继续输入都当普通文字，该位置的 sigil 被删掉才失效。
  const [dismissed, setDismissed] = useState<(MentionTrigger & { scope: string }) | null>(null);
  const trigger = caret === null ? null : findTrigger(value, caret, sigils);
  const query = trigger ? `${trigger.sigil}${trigger.start}:${trigger.query}` : "";
  const filter = trigger?.query ?? null;
  const items = useMemo(() => filter === null || !trigger ? [] : rank(all, filter,
    sources.filter((source) => source.provider.sigils.includes(trigger.sigil)).map((source) => source.key)), [all, filter, trigger, sources]);
  const found = highlight.query === query ? items.findIndex((item) => item.key === highlight.key) : -1;
  const active = found >= 0 ? found : 0;
  const setActive = useCallback((index: number) => setHighlight({ query, key: items[index]?.key ?? "" }), [query, items]);
  const closed = dismissed !== null && dismissed.scope === scope && value[dismissed.start] === dismissed.sigil
    && trigger?.start === dismissed.start && trigger.sigil === dismissed.sigil;
  // 光标停在一个完整提及之后（刚插入或退格回来）不再弹出，退格改为整段选中。
  const complete = trigger !== null && all.some((item) => item.prefix === trigger.sigil && item.token === trigger.query);
  const open = trigger !== null && !closed && !complete;
  const triggerStart = trigger?.start;
  const triggerSigil = trigger?.sigil;
  // 每次菜单打开都重新读取当前可用性。
  useEffect(() => { if (open) loadSuggestions(true); }, [open, triggerStart, triggerSigil]);
  const menuId = useId();

  const sync = useCallback(() => {
    const textarea = textareaRef.current;
    setCaret(textarea && textarea.selectionStart === textarea.selectionEnd ? textarea.selectionStart : null);
    setDismissed((previous) => previous && textarea?.value[previous.start] === previous.sigil ? previous : null);
  }, [textareaRef]);

  const pending = useRef<{ value: string; position: number } | null>(null);
  const pick = useCallback((item: SuggestItem | undefined) => {
    const textarea = textareaRef.current;
    if (!item || item.unavailable || !trigger || !textarea) return;
    const rest = value.slice(trigger.end);
    const inserted = `${item.prefix}${item.token}${rest.startsWith(" ") ? "" : " "}`;
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
          if (!items[next].unavailable) break;
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
        pick(items[active] && !items[active].unavailable ? items[active] : items.find((item) => !item.unavailable));
        return true;
      }
    }
    if (open && event.key === "Escape" && trigger) {
      event.preventDefault();
      setDismissed({ ...trigger, scope });
      return true;
    }
    // 紧贴提及的退格先选中整个提及，再按一次删除，提及像一个整体。
    if (event.key === "Backspace" && !open) {
      const textarea = event.currentTarget;
      const end = textarea.selectionStart;
      if (end !== textarea.selectionEnd) return false;
      const mention = mentionBefore(value, end, all);
      if (mention !== null) {
        event.preventDefault();
        textarea.setSelectionRange(mention, end);
        return true;
      }
    }
    return false;
  }, [open, items, active, setActive, trigger, scope, pick, value, all]);

  const groups = sources.map((source) => source.provider).filter((provider) => trigger && provider.sigils.includes(trigger.sigil));
  const menu = open && trigger ? <SuggestMenu id={menuId} items={items} query={trigger.query} sigil={trigger.sigil} active={active}
    title={groups.map((provider) => provider.label).join("、")} empty={groups.map((provider) => provider.empty).join("；")}
    status={status} error={error} onHover={setActive} onPick={pick} /> : null;
  const close = useCallback(() => setCaret(null), []);
  return { menu, sync, close, onKeyDown, inputProps: {
    "aria-autocomplete": "list" as const, "aria-controls": open ? menuId : undefined,
    "aria-activedescendant": open && items[active] ? `${menuId}-${active}` : undefined,
  } };
}

// 光标正好落在一个已知提及之后时返回它的起点。
function mentionBefore(value: string, caret: number, items: readonly SuggestItem[]): number | null {
  const names = new Set(items.map((item) => `${item.prefix}${item.token}`));
  for (const match of value.matchAll(MENTION)) {
    const start = (match.index ?? 0) + match[1].length;
    if (start + 1 + match[3].length === caret && names.has(match[2] + match[3])) return start;
  }
  return null;
}

function SuggestMenu({ id, items, query, sigil, active, title, empty, status, error, onHover, onPick }: {
  id: string;
  items: SuggestItem[];
  query: string;
  sigil: string;
  active: number;
  title: string;
  empty: string;
  status: "idle" | "loading" | "ready" | "error";
  error: string;
  onHover: (index: number) => void;
  onPick: (item: SuggestItem) => void;
}) {
  const counts = new Map<string, number>();
  items.forEach((item) => counts.set(item.token, (counts.get(item.token) ?? 0) + 1));
  return <div id={id} className="composer-suggest" role="listbox" aria-label={title} data-escape-layer
    onMouseDown={(event) => event.preventDefault()}>
    {items.map((item, index) => <Fragment key={item.key}>
      {item.source !== items[index - 1]?.source ? <div className="composer-suggest__section">{item.group}</div> : null}
      <div id={`${id}-${index}`} role="option" aria-selected={index === active}
        aria-disabled={!!item.unavailable} className="composer-suggest__row" data-active={index === active || undefined}
        ref={index === active ? (node) => node?.scrollIntoView({ block: "nearest" }) : undefined}
        onMouseMove={() => !item.unavailable && index !== active && onHover(index)}
        onClick={() => onPick(item)}>
        <FileText aria-hidden="true" />
        <span className="composer-suggest__label"><Highlight text={item.label} query={query} />
          <code><Highlight text={item.token} query={query} /></code></span>
        <span className="composer-suggest__desc">{item.unavailable ? `不可用：${item.unavailable}` : item.description}</span>
        {(counts.get(item.token) ?? 0) > 1 && item.badge ? <span className="composer-suggest__source">{item.badge}</span> : null}
      </div>
    </Fragment>)}
    {!items.length ? <div className="composer-suggest__empty">{status === "error" ? `没能读取${title}：${error}。重新输入 ${sigil} 可重试。`
      : status === "loading" || status === "idle" ? `正在读取${title}…` : query ? `没有匹配的${title}` : empty}</div> : null}
  </div>;
}

/** 用户消息里的已知提及显示成强调色文字；未知的 /xxx 保持原样。 */
export function MentionText({ text }: { text: string }) {
  const { items, sources } = useSuggestions();
  const prefixes = sources.map((source) => source.provider.prefix);
  const watched = prefixes.some((prefix) => text.includes(prefix));
  useEffect(() => { if (watched) loadSuggestions(); }, [watched, sources]);
  const byName = new Map(items.map((item) => [`${item.prefix}${item.token}`, item]));
  const nodes: ReactNode[] = [];
  let last = 0;
  for (const match of text.matchAll(MENTION)) {
    const item = byName.get(match[2] + match[3]);
    if (!item) continue;
    const start = (match.index ?? 0) + match[1].length;
    nodes.push(text.slice(last, start), <span key={start} className="composer-mention" title={item.description}>
      <span aria-hidden="true">{item.prefix}</span>{item.label}</span>);
    last = start + 1 + match[3].length;
  }
  if (!nodes.length) return <>{text}</>;
  nodes.push(text.slice(last));
  return <>{nodes}</>;
}
