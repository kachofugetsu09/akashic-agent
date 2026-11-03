import React, { useEffect, useLayoutEffect, useMemo, useRef, useState, useSyncExternalStore } from "react";
import { createPortal } from "react-dom";
import { AlertCircle } from "lucide-react";

import type { HeaderMetaControls, HeaderMetaItem } from "@akashic/chat-ui-v1";
import { chatLayout, chatPortalRoot, subscribeChatLayout } from "./chat-layout";
import { HeaderMetaSourceMount, headerMetaSources, usePluginUiCatalogVersion } from "./plugin-ui-runtime";
import "./header-meta.css";

// 标题下的说明行：插件经 header.meta 登记说明项，宿主建树、冒泡、沿树收起，点开是整棵树的会话信息面板。
type Tone = HeaderMetaItem["tone"];
const TONE_RANK: Record<Tone, number> = { alert: 0, live: 1, info: 2 };
const SCOPES = new Set(["project", "session", "account"]);
const MAX_ITEMS = 32;

interface MetaNode {
  key: string;
  item: HeaderMetaItem;
  kids: MetaNode[];
  /** 子树里最紧急的一档。 */
  subtree: Tone;
}

interface LineLevel {
  path: MetaNode[];
  roots: MetaNode[];
  hidden: number;
  account: MetaNode[];
  aggregate?: { count: number; focus: string };
}

function validItem(value: unknown): value is HeaderMetaItem {
  if (typeof value !== "object" || value === null) return false;
  const item = value as Record<string, unknown>;
  const optional = (key: string, type: string) => item[key] === undefined || typeof item[key] === type;
  return typeof item.id === "string" && item.id !== "" && typeof item.label === "string"
    && typeof item.scope === "string" && SCOPES.has(item.scope) && typeof item.tone === "string" && item.tone in TONE_RANK
    && optional("parent", "string") && optional("text", "string") && optional("value", "string") && optional("path", "boolean")
    && optional("detail", "function") && (item.weight === undefined || Number.isFinite(item.weight))
    && (item.iconSvg === undefined || (typeof item.iconSvg === "string" && item.iconSvg.startsWith("<svg")));
}

const byWeight = (a: MetaNode, b: MetaNode) => (b.item.weight ?? 0) - (a.item.weight ?? 0);
const byUrgency = (a: MetaNode, b: MetaNode) => TONE_RANK[a.subtree] - TONE_RANK[b.subtree] || byWeight(a, b);

// 建树：parent 先找本插件的 id，再找「插件名/id」；跨范围、找不到或成环的挂回范围根。
function buildTree(published: ReadonlyMap<string, readonly HeaderMetaItem[]>): MetaNode[] {
  const nodes = new Map<string, MetaNode>();
  const named = new Map<string, MetaNode>();
  for (const [pluginId, items] of published) {
    for (const item of items) {
      const node = { key: `${pluginId}:${item.id}`, item, kids: [], subtree: item.tone };
      nodes.set(node.key, node);
      named.set(`${pluginId.split("@")[0]}/${item.id}`, node);
    }
  }
  const parentOf = new Map<MetaNode, MetaNode>();
  const roots: MetaNode[] = [];
  for (const [pluginId, items] of published) {
    for (const item of items) {
      const node = nodes.get(`${pluginId}:${item.id}`)!;
      const parent = item.parent === undefined ? undefined : nodes.get(`${pluginId}:${item.parent}`) ?? named.get(item.parent);
      let cursor = parent;
      while (cursor && cursor !== node) cursor = parentOf.get(cursor);
      if (parent && parent !== node && !cursor && parent.item.scope === item.scope) {
        parentOf.set(node, parent);
        parent.kids.push(node);
      } else roots.push(node);
    }
  }
  const roll = (node: MetaNode): Tone => {
    node.kids.sort(byWeight);
    node.subtree = node.kids.reduce<Tone>((tone, kid) => {
      const next = roll(kid);
      return TONE_RANK[next] < TONE_RANK[tone] ? next : tone;
    }, node.item.tone);
    return node.subtree;
  };
  roots.forEach(roll);
  return roots;
}

// 由最丰富到最省的排法：先缩短路径，再把顶层项收进 +N；需要注意的不收，最后一档才合并。
function lineLevels(roots: MetaNode[], compact: boolean, project: boolean): LineLevel[] {
  const path: MetaNode[] = [];
  for (let node = roots.find((root) => root.item.scope === "project" && root.item.path); node; node = node.kids.find((kid) => kid.item.path)) {
    path.push(node);
  }
  const activity = roots.filter((root) => root.item.scope === "session" && (root.item.text || root.subtree !== "info")).sort(byUrgency);
  const account = roots.filter((root) => root.item.scope === "account" && root.subtree === "alert");
  const pinned = activity.filter((root) => root.subtree === "alert");
  // 只剩面板里的项（例如都已结束的代理任务）又没有项目名时，留一个 +N 作入口，否则面板打不开。
  if (!project && !path.length && !activity.length && !account.length) {
    return roots.length ? [{ path: [], roots: [], hidden: roots.length, account: [] }] : [];
  }
  const levels: LineLevel[] = [];
  const depths = compact ? [0] : [...new Set([path.length, Math.min(1, path.length), 0])];
  for (const depth of depths) {
    const kept = path.filter((node, index) => index < depth || node.subtree === "alert");
    for (let shown = Math.max(pinned.length, Math.min(compact ? 1 : 3, activity.length)); shown >= pinned.length; shown -= 1) {
      const visible = [...pinned, ...activity.filter((root) => root.subtree !== "alert").slice(0, shown - pinned.length)];
      levels.push({ path: kept, roots: visible, hidden: activity.length - visible.length, account });
    }
  }
  const alerts = [...path, ...activity, ...account].filter((node) => node.subtree === "alert");
  if (alerts.length > 1) {
    levels.push({ path: [], roots: [], hidden: activity.length - pinned.length, account: [], aggregate: { count: alerts.length, focus: alerts[0].key } });
  }
  return levels;
}

function MetaIcon({ node, small = true }: { node: MetaNode; small?: boolean }) {
  if (node.item.tone === "alert") return <AlertCircle size={small ? 13 : 15} strokeWidth={1.75} aria-hidden="true" />;
  if (node.item.tone === "live" && !node.item.iconSvg) return <span className="header-meta__live" aria-hidden="true" />;
  return node.item.iconSvg ? <span className="header-meta__icon" aria-hidden="true" dangerouslySetInnerHTML={{ __html: node.item.iconSvg }} /> : null;
}

function useCompact(): boolean {
  return useSyncExternalStore(subscribeChatLayout, () => chatLayout().compact);
}

/** 标题下的说明行与会话信息面板；没有任何说明项也没有项目时不占位置。 */
export function HeaderMeta({ sessionId, project, refreshToken }: {
  sessionId: string;
  /** 会话所在的分组名，作为「在哪」路径的第一段。 */
  project?: string;
  refreshToken?: string | number;
}) {
  const { version } = usePluginUiCatalogVersion();
  const sources = useMemo(() => { void version; return headerMetaSources(); }, [version]);
  const [published, setPublished] = useState<ReadonlyMap<string, readonly HeaderMetaItem[]>>(new Map());
  const [open, setOpen] = useState<string | null>(null);
  const lineRef = useRef<HTMLDivElement>(null);
  const compact = useCompact();
  // 每个插件一份稳定的宿主能力：身份变化会让插件重新挂载。
  const controls = useMemo(() => new Map(sources.map((source) => [source.pluginId, {
    publish(items: readonly HeaderMetaItem[]) {
      const valid = Array.isArray(items) ? items.filter(validItem).slice(0, MAX_ITEMS) : [];
      setPublished((current) => new Map(current).set(source.pluginId, valid));
    },
    closePanel() { setOpen(null); },
  } satisfies HeaderMetaControls])), [sources]);
  // 插件卸载或换代后撤回它登记的说明项。
  useEffect(() => {
    setPublished((current) => {
      const live = [...current].filter(([pluginId]) => controls.has(pluginId));
      return live.length === current.size ? current : new Map(live);
    });
  }, [controls]);
  const roots = useMemo(() => buildTree(published), [published]);
  const levels = useMemo(() => lineLevels(roots, compact, !!project), [roots, compact, project]);
  return <>
    <div className="header-meta__sources" hidden>{sources.map((source) => <HeaderMetaSourceMount key={source.key} source={source}
      sessionId={sessionId} meta={controls.get(source.pluginId)!} refreshToken={refreshToken} />)}</div>
    {project || levels.length
      ? <MetaLine lineRef={lineRef} project={project} levels={levels} onOpen={setOpen} open={open !== null} /> : null}
    {open !== null ? createPortal(<SessionInfoPanel anchor={lineRef} compact={compact} roots={roots} project={project} focus={open}
      onFocus={setOpen} onClose={() => setOpen(null)} />, chatPortalRoot()) : null}
  </>;
}

// 渲染后再量宽：放不下就换下一档，宽度或内容变化时从最丰富的一档重新开始。
function MetaLine({ lineRef: ref, project, levels, open, onOpen }: {
  lineRef: React.RefObject<HTMLDivElement | null>;
  project?: string;
  levels: LineLevel[];
  open: boolean;
  onOpen(focus: string): void;
}) {
  const [width, setWidth] = useState(0);
  const [fit, setFit] = useState({ levels, width, index: 0 });
  const index = fit.levels === levels && fit.width === width ? fit.index : 0;
  useEffect(() => {
    const element = ref.current;
    if (!element) return;
    const observer = new ResizeObserver(() => setWidth(element.clientWidth));
    observer.observe(element);
    return () => observer.disconnect();
  }, [ref]);
  useLayoutEffect(() => {
    const element = ref.current;
    const next = element && element.scrollWidth > element.clientWidth + 1 && index < levels.length - 1 ? index + 1 : index;
    if (fit.levels !== levels || fit.width !== width || fit.index !== next) setFit({ levels, width, index: next });
  }, [fit, index, levels, ref, width]);
  const level = levels[index] ?? { path: [], roots: [], hidden: 0, account: [] };
  const item = (node: MetaNode, kind: "path" | "root") => <button key={node.key} type="button" data-tone={node.subtree}
    aria-label={`${node.item.label}：${node.item.text ?? node.item.value ?? ""}`} onClick={() => onOpen(node.key)}>
    {kind === "path" && node.item.tone !== "alert" ? null : <MetaIcon node={node} />}
    {node.item.text ?? node.item.label}
    {node.subtree !== node.item.tone ? <span className="header-meta__flag" data-tone={node.subtree} aria-hidden="true" /> : null}
  </button>;
  const more = level.hidden ? <button type="button" className="header-meta__more" aria-label={`另有 ${level.hidden} 项`}
    onClick={() => onOpen("")}>+{level.hidden}</button> : null;
  return <div ref={ref} className="header-meta" aria-expanded={open}>
    <div className="header-meta__group header-meta__path">
      {project ? <button type="button" className="header-meta__project" onClick={() => onOpen("project")}>{project}</button> : null}
      {level.path.map((node) => item(node, "path"))}
    </div>
    {level.aggregate ? <div className="header-meta__group">
      <button type="button" data-tone="alert" onClick={() => onOpen(level.aggregate!.focus)}>
        <AlertCircle size={13} strokeWidth={1.75} aria-hidden="true" />{level.aggregate.count} 项需要注意</button>{more}
    </div> : level.roots.length || more ? <div className="header-meta__group">{level.roots.map((node) => item(node, "root"))}{more}</div> : null}
    {level.account.length ? <div className="header-meta__group">{level.account.map((node) => item(node, "root"))}</div> : null}
  </div>;
}

// 会话信息面板：按项目、本会话、账号三个范围列出整棵树；点中的一项展开插件给的详细内容。
function SessionInfoPanel({ anchor, compact, roots, project, focus, onFocus, onClose }: {
  anchor: React.RefObject<HTMLDivElement | null>;
  /** 紧凑外壳里是底部抽屉，否则锚在说明行下方。 */
  compact: boolean;
  roots: MetaNode[];
  project?: string;
  focus: string;
  onFocus(key: string): void;
  onClose(): void;
}) {
  const ref = useRef<HTMLDivElement>(null);
  const [place, setPlace] = useState<React.CSSProperties>({});
  // 浮层挂在聊天根节点上，按说明行当前的位置摆放；手机上顶栏有位移动画，不能跟着它定位。
  useLayoutEffect(() => {
    const rect = anchor.current?.getBoundingClientRect();
    if (compact || !rect) { setPlace({}); return; }
    setPlace({ insetBlockStart: rect.bottom + 6, insetInlineStart: Math.max(16, Math.min(rect.left - 10, window.innerWidth - 16 - 416)) });
  }, [anchor, compact]);
  useEffect(() => {
    const onPointer = (event: PointerEvent) => {
      const target = event.target as Node;
      if (!ref.current?.contains(target) && !anchor.current?.contains(target)) onClose();
    };
    const onKey = (event: KeyboardEvent) => { if (event.key === "Escape" && !event.defaultPrevented) onClose(); };
    document.addEventListener("pointerdown", onPointer);
    document.addEventListener("keydown", onKey);
    return () => { document.removeEventListener("pointerdown", onPointer); document.removeEventListener("keydown", onKey); };
  }, [anchor, onClose]);
  useEffect(() => { ref.current?.querySelector<HTMLElement>("[data-focus='true']")?.scrollIntoView({ block: "nearest" }); }, [focus]);
  const row = (node: MetaNode, depth: number): React.ReactNode => <div key={node.key} className="header-meta-node" data-tone={node.item.tone}
    style={{ "--depth": depth } as React.CSSProperties}>
    <button type="button" className="header-meta-node__row" data-focus={node.key === focus} aria-expanded={node.item.detail ? node.key === focus : undefined}
      onClick={() => onFocus(node.key)}>
      <MetaIcon node={node} small={false} /><span className="header-meta-node__label">{node.item.label}</span>
      <span className="header-meta-node__value">{node.item.value ?? node.item.text ?? ""}</span>
      {node.kids.length && node.subtree !== "info" ? <span className="header-meta__flag" data-tone={node.subtree} aria-hidden="true" /> : <span />}
    </button>
    {node.key === focus && node.item.detail ? <MetaDetail key={node.key} detail={node.item.detail} /> : null}
    {node.kids.length ? <div className="header-meta-node__kids">{node.kids.map((kid) => row(kid, depth + 1))}</div> : null}
  </div>;
  const scope = (name: HeaderMetaItem["scope"], label: string, title?: string) => {
    const nodes = roots.filter((root) => root.item.scope === name).sort(name === "project" ? byWeight : byUrgency);
    if (!nodes.length && !title) return null;
    return <section aria-label={label}>
      <h2 className="header-meta-panel__scope" data-focus={name === "project" && focus === "project"}>{label}{title ? <b>{title}</b> : null}</h2>
      {nodes.map((node) => row(node, 0))}
    </section>;
  };
  return <>
    {compact ? <div className="header-meta-scrim" onClick={onClose} /> : null}
    <div ref={ref} className="header-meta-panel" data-sheet={compact || undefined} style={place} role="group" aria-label="会话信息" data-escape-layer="">
      {scope("project", "项目", project)}{scope("session", "本会话")}{scope("account", "账号")}
      {roots.length ? null : <p className="header-meta-panel__empty">工作目录、代理任务、后台任务等状态出现后会列在这里。</p>}
    </div>
  </>;
}

// 详细内容只在展开时挂载一次；插件自己保持内容最新。
function MetaDetail({ detail }: { detail: NonNullable<HeaderMetaItem["detail"]> }) {
  const ref = useRef<HTMLDivElement>(null);
  const latest = useRef(detail);
  useEffect(() => {
    const host = ref.current;
    if (!host) return;
    let cleanup: void | (() => void);
    try { cleanup = latest.current(host); } catch (error) { console.error("[header-meta] detail failed", error); }
    return () => { try { cleanup?.(); } finally { host.replaceChildren(); } };
  }, []);
  return <div ref={ref} className="header-meta-node__detail" />;
}
