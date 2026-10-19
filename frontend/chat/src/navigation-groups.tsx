import { ChevronRight, FolderPlus, Plus, SlidersHorizontal } from "lucide-react";
import { Fragment, useCallback, useEffect, useMemo, useRef, useState, type FormEvent, type ReactNode } from "react";
import type { NavigationControls, NavigationForm, NavigationFormControls, NavigationGroup } from "@akashic/chat-ui-v1";
import { Dialog, DialogContent, DialogFooter, DialogHeader, DialogTitle } from "@/components/ui/dialog";
import { NavigationRowMenu, type NavigationRowAction } from "./navigation-row-menu";
import { SessionStatusMark, summarizeStatus, type SessionStatus } from "./session-status-mark";
import { MarqueeText } from "./marquee-text";
import { NavigationGroupMount, PluginUiSlot, navigationGroupSources, usePluginUiCatalogVersion, type NavigationGroupSource } from "./plugin-ui-runtime";
import { createScopeFields } from "./scope-fields";

/** 会话所在的分组：会话维度名与它的一个取值。 */
export interface GroupRef {
  dimension: string;
  id: string;
}

export const groupKey = (ref: GroupRef): string => `${ref.dimension}:${ref.id}`;

/** 一个插件提供的侧栏分区：数据源与最近一次读到的分组。 */
export interface GroupSection {
  source: NavigationGroupSource;
  groups: readonly NavigationGroup[];
}

/** 打开一个分组对话框：新建取值，或某个分组的动作。 */
export interface GroupFormRequest {
  source: NavigationGroupSource;
  form: NavigationForm;
  group?: NavigationGroup;
}

export interface GroupSessionItem {
  id: string;
  title: string;
  active: boolean;
  status?: SessionStatus;
  pinned?: boolean;
}

/** 由侧栏提供的分组内会话行（带行菜单）；label 是分区名，例如“项目”。 */
export type GroupSessionRenderer = (item: GroupSessionItem, label: string) => ReactNode;

/** 侧栏分组的宿主状态与动作；分组数据全部来自插件的 navigation.groups。 */
export interface NavigationGroupsState {
  sections: readonly GroupSection[];
  /** 每次重读递增；插件的提示区据此重新读取自己的状态。 */
  revision: number;
  active: (GroupRef & { label: string }) | null;
  reload: () => void;
  startChat: (ref: GroupRef) => void;
}

function checkGroups(value: unknown): NavigationGroup[] {
  if (!Array.isArray(value)) throw new Error("分组列表无效");
  return value.map((item: unknown) => {
    const row = typeof item === "object" && item !== null ? item as Partial<Record<keyof NavigationGroup, unknown>> : null;
    if (!row || typeof row.id !== "string" || !row.id || typeof row.label !== "string"
      || (row.detail !== undefined && typeof row.detail !== "string")) throw new Error("分组记录无效");
    return { id: row.id, label: row.label, ...(row.detail ? { detail: row.detail } : {}) };
  });
}

// 单个分区的读取期限；插件永不返回时只影响它自己的分区。
const LOAD_TIMEOUT_MS = 20_000;

const abortion = (signal: AbortSignal) => new Promise<never>((_, reject) => {
  signal.addEventListener("abort", () => reject(signal.reason instanceof Error ? signal.reason : new Error("读取已取消")), { once: true });
});

// 目录或重读请求变化时按目录顺序读取各分区；每个分区读完即显示，失败只报错并保留它上次的分组。
export function useNavigationGroupSections(enabled: boolean, catalogVersion: number, reportError: (error: unknown) => void) {
  const [sources, setSources] = useState<NavigationGroupSource[]>([]);
  const [loaded, setLoaded] = useState<ReadonlyMap<string, readonly NavigationGroup[]>>(new Map());
  const [revision, setRevision] = useState(0);
  useEffect(() => {
    const next = enabled ? navigationGroupSources() : [];
    const keys = new Set(next.map((source) => source.key));
    setSources(next);
    setLoaded((current) => new Map([...current].filter(([key]) => keys.has(key))));
    const controller = new AbortController();
    const publish = (key: string, groups: readonly NavigationGroup[] | null) => {
      if (!controller.signal.aborted) setLoaded((current) => new Map(current).set(key, groups ?? current.get(key) ?? []));
    };
    for (const source of next) {
      const own = new AbortController();
      const stop = () => own.abort();
      controller.signal.addEventListener("abort", stop, { once: true });
      const timer = window.setTimeout(() => own.abort(new Error("读取超时")), LOAD_TIMEOUT_MS);
      void Promise.race([source.provider.load(source.context(own.signal)), abortion(own.signal)])
        .then((groups) => publish(source.key, checkGroups(groups)))
        .catch((error: unknown) => {
          if (controller.signal.aborted) return;
          reportError(new Error(`${source.provider.label}读取失败：${error instanceof Error ? error.message : String(error)}`));
          publish(source.key, null);
        })
        .finally(() => { window.clearTimeout(timer); controller.signal.removeEventListener("abort", stop); });
    }
    return () => controller.abort();
  }, [enabled, catalogVersion, revision, reportError]);
  const sections = useMemo(() => sources.flatMap((source) => {
    const groups = loaded.get(source.key);
    return groups ? [{ source, groups }] : [];
  }), [sources, loaded]);
  const reload = useCallback(() => setRevision((value) => value + 1), []);
  return { sections, revision, reload };
}

/** 一个分区：标题与新建入口、插件提示、分组行；分组行与会话、展开、置顶都由宿主负责。 */
export function GroupNavigation({
  section, groups, revision, sessionsByGroup, active, pendingSessionId, expanded, searching, heading,
  onToggle, onSelectSession, onPrefetchSession, onOpenForm, onNewChat, onReload, actions, renderSession,
}: {
  section: GroupSection;
  groups: readonly NavigationGroup[];
  revision: number;
  sessionsByGroup: ReadonlyMap<string, GroupSessionItem[]>;
  active: GroupRef | null;
  pendingSessionId: string;
  expanded: ReadonlySet<string>;
  searching: boolean;
  heading: string;
  onToggle: (key: string) => void;
  onSelectSession: (sessionId: string) => void;
  onPrefetchSession?: (sessionId: string) => void;
  onOpenForm: (request: GroupFormRequest) => void;
  onNewChat: (ref: GroupRef) => void;
  onReload: () => void;
  actions: (group: NavigationGroup) => NavigationRowAction[];
  renderSession?: GroupSessionRenderer;
}) {
  const { provider } = section.source;
  const create = provider.create;
  const latest = useRef({ onNewChat, onReload });
  latest.current = { onNewChat, onReload };
  const extra = useMemo((): { navigation: NavigationControls } => ({ navigation: {
    reload: () => latest.current.onReload(),
    startChat: (id) => latest.current.onNewChat({ dimension: provider.dimension, id }),
  } }), [provider.dimension]);
  // 搜索没有命中任何分组时整节收起；插件提示只在非搜索时显示。
  if (searching && groups.length === 0) return null;
  const openCreate = create ? () => onOpenForm({ source: section.source, form: create }) : undefined;
  return <section className="navigation-section" aria-label={provider.label}>
    <header className="navigation-section__header">
      <span>{heading}</span>
      {openCreate ? <button type="button" className="navigation-section__icon" aria-label={`新建${provider.label}`}
        data-tip={`新建${provider.label}`} onClick={openCreate}>
        <Plus size={15} aria-hidden="true" />
      </button> : null}
    </header>
    {!searching && provider.notice ? <NavigationGroupMount source={section.source} renderer={provider.notice}
      extra={extra} scope={{ dimension: provider.dimension }} refreshToken={revision} /> : null}
    {groups.length === 0 && !searching && openCreate && heading === provider.label ? (
      <button type="button" className="navigation-section__empty" onClick={openCreate}>
        <FolderPlus size={16} aria-hidden="true" />
        <span>新建{provider.label}</span>
      </button>
    ) : null}
    {groups.map((group) => {
      const ref = { dimension: provider.dimension, id: group.id };
      return <GroupNavigationRow key={group.id} groupRef={ref} group={group}
        items={sessionsByGroup.get(groupKey(ref)) ?? []}
        open={searching || expanded.has(groupKey(ref))}
        active={active?.dimension === ref.dimension && active.id === ref.id}
        pendingSessionId={pendingSessionId} searching={searching}
        onToggle={() => onToggle(groupKey(ref))} onNewChat={() => onNewChat(ref)}
        onSelectSession={onSelectSession} onPrefetchSession={onPrefetchSession} actions={actions(group)}
        renderSession={renderSession && ((item) => renderSession(item, provider.label))} />;
    })}
  </section>;
}

/** 分组图标。scope.badge 的标签带 data-scope-line（double / dashed / dotted）时，样式把线型换到这枚图标上，标签文字只留给读屏。 */
export function FolderMark() {
  return <svg className="folder-mark" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor"
    strokeWidth="1.75" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
    <path d="M20 20a2 2 0 0 0 2-2V8a2 2 0 0 0-2-2h-7.9a2 2 0 0 1-1.69-.9L9.6 3.9A2 2 0 0 0 7.93 3H4a2 2 0 0 0-2 2v13a2 2 0 0 0 2 2Z" />
    <rect className="folder-mark__inner" x="5.5" y="9.5" width="13" height="7" rx="1" />
  </svg>;
}

/** 插件给分组行的动作，每项打开一个宿主对话框。 */
export function groupFormActions(source: NavigationGroupSource, group: NavigationGroup,
  onOpenForm: (request: GroupFormRequest) => void): NavigationRowAction[] {
  return (source.provider.actions?.(group) ?? []).map((action) => ({
    label: action.label,
    icon: <SlidersHorizontal size={18} aria-hidden="true" />,
    onSelect: () => onOpenForm({ source, form: action.form, group }),
  }));
}

/** 同一个分组行用于置顶和分区，展开状态由导航层按 groupKey 持有。 */
export function GroupNavigationRow({
  groupRef, group, items, open, active, pendingSessionId, onToggle, onNewChat,
  onSelectSession, onPrefetchSession, actions, searching = false, renderSession,
}: {
  groupRef: GroupRef;
  group: NavigationGroup;
  items: GroupSessionItem[];
  open: boolean;
  active: boolean;
  pendingSessionId: string;
  onToggle: () => void;
  onNewChat: () => void;
  onSelectSession: (sessionId: string) => void;
  onPrefetchSession?: (sessionId: string) => void;
  actions?: NavigationRowAction[];
  searching?: boolean;
  renderSession?: (item: GroupSessionItem) => ReactNode;
}) {
  // 折叠时分组行汇总会话状态，展开后状态回到各自的会话前。
  const collapsedStatus = open ? undefined : summarizeStatus(items.map((item) => item.status));
  return <div className={`navigation-group ${active ? "active" : ""}`} data-group-key={groupKey(groupRef)}>
    <NavigationRowMenu title={group.label} className="navigation-group__row" actions={[
      { label: "新建对话", icon: <Plus size={18} aria-hidden="true" />, onSelect: onNewChat },
      ...(actions ?? []),
    ]}>
      {/* 与右侧名称按钮做同一件事，键盘只停名称一处。 */}
      <button type="button" className="navigation-group__toggle" tabIndex={-1}
        aria-label={`${open ? "收起" : "展开"} ${group.label} 的对话`} aria-expanded={open}
        onClick={onToggle} disabled={searching}>
        <ChevronRight size={14} aria-hidden="true" className="navigation-group__chevron" />
      </button>
      {/* 搜索时只停用折叠，不禁用按钮：它是这一行唯一的键盘入口，行菜单（含"新建对话"）要靠它 Shift+F10 打开。 */}
      <button type="button" className="navigation-group__open" data-row-primary="" onClick={searching ? undefined : onToggle}
        aria-expanded={open} aria-disabled={searching || undefined}
        title={`${open ? "收起" : "展开"} ${group.label}${group.detail ? ` · ${group.detail}` : ""}`}>
        {collapsedStatus
          ? <span className="navigation-group__icon"><SessionStatusMark status={collapsedStatus} /></span>
          : <FolderMark />}
        <span className="navigation-group__name"><MarqueeText>{group.label}</MarqueeText></span>
        {/* 其他插件给这个取值的短标签，例如记忆范围；只放文字，没有插件时不占位。 */}
        <PluginUiSlot name="scope.badge" scope={{ dimension: groupRef.dimension, value: groupRef.id }} quiet />
      </button>
      {/* 触屏没有悬停，项目行常驻这一枚；宽屏改用展开后的“新对话”行。键盘走行菜单（Shift+F10），不多占 Tab 停靠点。 */}
      <button type="button" className="navigation-section__icon navigation-group__new-chat" aria-label={`在 ${group.label} 中新建对话`}
        data-tip="新建对话" tabIndex={-1} onClick={onNewChat}>
        <Plus size={14} aria-hidden="true" />
      </button>
    </NavigationRowMenu>
    {open ? <nav className="navigation-group__sessions" aria-label={`${group.label} 的对话`}>
      {/* 新对话是列表里的第一行，与会话标题左对齐，不浮在项目行上。 */}
      {searching ? null : <button type="button" className="group-session group-session--new" onClick={onNewChat}>
        <Plus size={14} aria-hidden="true" />
        <span>新对话</span>
      </button>}
      {searching && items.length === 0 ? <small className="navigation-group__hint">没有匹配的对话</small> : null}
      {items.map((session) => renderSession ? <Fragment key={session.id}>{renderSession(session)}</Fragment> : <button key={session.id} type="button"
        className={`group-session ${session.active ? "active" : ""}`}
        aria-current={session.active ? "true" : undefined}
        aria-busy={pendingSessionId === session.id || undefined}
        title={session.title}
        onClick={() => onSelectSession(session.id)}
        onPointerEnter={() => onPrefetchSession?.(session.id)}
        onFocus={() => onPrefetchSession?.(session.id)}>
        <SessionStatusMark status={session.status} />
        <span className="group-session__title"><MarqueeText>{session.title}</MarqueeText></span>
      </button>)}
    </nav> : null}
  </div>;
}

type Submit = Parameters<NavigationFormControls["onSubmit"]>[0];

/** 分组对话框外框：标题、错误与按钮归宿主，表单内容由插件挂载；新建时其他插件经 scope.fields 附加设置。 */
export function GroupFormDialog({ request, groups, onClose, onCloseFocus }: {
  request: GroupFormRequest;
  groups: NavigationGroupsState;
  onClose: () => void;
  onCloseFocus?: () => void;
}) {
  const { source, form, group } = request;
  const dimension = source.provider.dimension;
  const [busy, setBusy] = useState(false);
  const [ready, setReady] = useState(false);
  const [error, setError] = useState("");
  const busyRef = useRef(false);
  const submitRef = useRef<Submit | null>(null);
  const closeRef = useRef(onClose);
  closeRef.current = onClose;
  const fields = useMemo(() => createScopeFields(() => busyRef.current), []);
  // 所属插件换代或撤下后，旧表单不再挂载、不能提交，避免先写入其他插件的附加设置再失败。
  const { version } = usePluginUiCatalogVersion();
  const isLive = () => navigationGroupSources().some((item) => item.key === source.key);
  const live = useMemo(isLive, [version, source.key]); // eslint-disable-line react-hooks/exhaustive-deps
  const extra = useMemo((): { form: NavigationFormControls } => ({ form: {
    onSubmit(submit) {
      submitRef.current = submit;
      return () => { if (submitRef.current === submit) submitRef.current = null; };
    },
    setReady: (value) => setReady(Boolean(value)),
    busy: () => busyRef.current,
    close: () => { if (!busyRef.current) closeRef.current(); },
  } }), []);
  const settle = (value: boolean) => { busyRef.current = value; setBusy(value); };
  const submit = async (event: FormEvent) => {
    event.preventDefault();
    const handler = submitRef.current;
    if (!handler || !ready || busyRef.current) return;
    if (!isLive()) {
      setError("这个表单所属的插件已更新或撤下，请关闭后重新打开。");
      return;
    }
    settle(true);
    setError("");
    try {
      const created = await handler((value) => fields.commit(value));
      settle(false);
      onClose();
      // 新建表单返回新取值：宿主随即在其中开始新对话。
      if (!group && typeof created === "string" && created) groups.startChat({ dimension, id: created });
    } catch (reason: unknown) {
      settle(false);
      setError(reason instanceof Error ? reason.message : "提交失败");
    } finally {
      // 失败也可能已部分提交（例如插件记下的待确认请求），分组与提示都重读。
      groups.reload();
    }
  };
  return <Dialog open onOpenChange={(open) => { if (!open && !busyRef.current) onClose(); }}>
    <DialogContent className="group-dialog" overlayClassName="group-dialog-overlay" aria-describedby={undefined}
      onCloseAutoFocus={onCloseFocus ? (event) => { event.preventDefault(); onCloseFocus(); } : undefined}>
      <form onSubmit={(event) => void submit(event)}>
        <DialogHeader className="group-dialog__header"><DialogTitle>{form.title}</DialogTitle></DialogHeader>
        {live ? <NavigationGroupMount source={source} renderer={form} extra={extra}
          scope={{ dimension, ...(group ? { value: group.id } : {}) }} refreshToken={busy ? 1 : 0} />
          : <p role="status">这个表单所属的插件已更新或撤下，请关闭后重新打开。</p>}
        {error ? <p className="group-dialog__error" role="alert">{error}</p> : null}
        <DialogFooter className="group-dialog__footer">
          {/* 其他插件的附加设置（例如记忆范围）；没有插件时这里为空，按钮仍靠右。 */}
          {group || !live ? <span /> : <PluginUiSlot name="scope.fields" scope={{ dimension }} fields={fields}
            refreshToken={busy ? 1 : 0} quiet fallback={<span />} />}
          {form.submitLabel && live ? <button type="submit" className="group-dialog__button primary" disabled={!ready || busy}>
            {busy ? "正在提交…" : form.submitLabel}
          </button> : <button type="button" className="group-dialog__button" onClick={onClose}>关闭</button>}
        </DialogFooter>
      </form>
    </DialogContent>
  </Dialog>;
}
