import { ChevronRight, Folder, FolderPlus, Plus, SlidersHorizontal } from "lucide-react";
import { useCallback, useEffect, useMemo, useRef, useState, type FormEvent } from "react";
import type { NavigationControls, NavigationForm, NavigationFormControls, NavigationGroup } from "@akashic/chat-ui-v1";
import { Dialog, DialogContent, DialogFooter, DialogHeader, DialogTitle } from "@/components/ui/dialog";
import { NavigationRowMenu, type NavigationRowAction } from "./navigation-row-menu";
import { SessionStatusMark, summarizeStatus, type SessionStatus } from "./session-status-mark";
import { MarqueeText } from "./marquee-text";
import { NavigationGroupMount, PluginUiSlot, navigationGroupSources, type NavigationGroupSource } from "./plugin-ui-runtime";
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
}

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

// 目录或重读请求变化时按目录顺序读取各分区；单个分区失败只报错并保留它上次的分组。
export function useNavigationGroupSections(enabled: boolean, catalogVersion: number, reportError: (error: unknown) => void) {
  const [sections, setSections] = useState<GroupSection[]>([]);
  const [revision, setRevision] = useState(0);
  useEffect(() => {
    const sources = enabled ? navigationGroupSources() : [];
    const controller = new AbortController();
    void Promise.all(sources.map(async (source) => {
      try {
        return { source, groups: checkGroups(await source.provider.load(source.context(controller.signal))) };
      } catch (error) {
        if (!controller.signal.aborted) reportError(new Error(`${source.provider.label}读取失败：${error instanceof Error ? error.message : String(error)}`));
        return { source, groups: null };
      }
    })).then((next) => {
      if (controller.signal.aborted) return;
      setSections((current) => next.map(({ source, groups }) => ({
        source, groups: groups ?? current.find((section) => section.source.key === source.key)?.groups ?? [],
      })));
    });
    return () => controller.abort();
  }, [enabled, catalogVersion, revision, reportError]);
  const reload = useCallback(() => setRevision((value) => value + 1), []);
  return { sections, revision, reload };
}

/** 一个分区：标题与新建入口、插件提示、分组行；分组行与会话、展开、置顶都由宿主负责。 */
export function GroupNavigation({
  section, groups, revision, sessionsByGroup, active, pendingSessionId, expanded, searching, heading,
  onToggle, onSelectSession, onPrefetchSession, onOpenForm, onNewChat, onReload, actions,
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
        onSelectSession={onSelectSession} onPrefetchSession={onPrefetchSession} actions={actions(group)} />;
    })}
  </section>;
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
  onSelectSession, onPrefetchSession, actions, searching = false,
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
          : <Folder size={18} strokeWidth={1.75} aria-hidden="true" />}
        <span className="navigation-group__name"><MarqueeText>{group.label}</MarqueeText></span>
        {/* 其他插件给这个取值的短标签，例如记忆范围；只放文字，没有插件时不占位。 */}
        <PluginUiSlot name="scope.badge" scope={{ dimension: groupRef.dimension, value: groupRef.id }} quiet />
      </button>
      {/* 行菜单里有同一个"新建对话"：键盘用 Shift+F10 打开，不再每个分组多占一个 Tab 停靠点。 */}
      <button type="button" className="navigation-section__icon navigation-group__new-chat" aria-label={`在 ${group.label} 中新建对话`}
        data-tip="新建对话" tabIndex={-1} onClick={onNewChat}>
        <Plus size={14} aria-hidden="true" />
      </button>
    </NavigationRowMenu>
    {open ? <nav className="navigation-group__sessions" aria-label={`${group.label} 的对话`}>
      {items.length === 0 ? <small className="navigation-group__hint">{searching ? "没有匹配的对话" : "还没有对话"}</small> : null}
      {items.map((session) => <button key={session.id} type="button"
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
        <NavigationGroupMount source={source} renderer={form} extra={extra}
          scope={{ dimension, ...(group ? { value: group.id } : {}) }} refreshToken={busy ? 1 : 0} />
        {error ? <p className="group-dialog__error" role="alert">{error}</p> : null}
        <DialogFooter className="group-dialog__footer">
          {/* 其他插件的附加设置（例如记忆范围）；没有插件时这里为空，按钮仍靠右。 */}
          {group ? <span /> : <PluginUiSlot name="scope.fields" scope={{ dimension }} fields={fields}
            refreshToken={busy ? 1 : 0} quiet fallback={<span />} />}
          {form.submitLabel ? <button type="submit" className="group-dialog__button primary" disabled={!ready || busy}>
            {busy ? "正在提交…" : form.submitLabel}
          </button> : <button type="button" className="group-dialog__button" onClick={onClose}>关闭</button>}
        </DialogFooter>
      </form>
    </DialogContent>
  </Dialog>;
}
