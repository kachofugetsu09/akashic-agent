import { ChevronRight, PenLine, Pin, Trash2 } from "lucide-react";
import { Fragment, memo, useEffect, useMemo, useRef, useState, type ReactNode } from "react";
import "./conversation-navigation.css";
import { SessionStatusMark, type SessionStatus } from "./session-status-mark";
import { MarqueeText } from "./marquee-text";
import { NavigationRowMenu, type NavigationRowAction } from "./navigation-row-menu";

export interface ConversationDestination {
  id: string;
  icon: ReactNode;
  label: string;
  description?: string;
  badge?: ReactNode;
  href?: string;
  featured?: boolean;
  active?: boolean;
  disabled?: boolean;
  onActivate?: () => void;
}

export interface ConversationSession {
  id: string;
  title: string;
  preview: string;
  active: boolean;
  status?: SessionStatus;
  unavailable?: boolean;
  /** 在所在分组内置顶：行尾一枚小图钉，悬停时让位给行菜单。 */
  pinned?: boolean;
}

export interface ConversationAction {
  id: string;
  icon: ReactNode;
  label: string;
  primary?: boolean;
  disabled?: boolean;
  onActivate: () => void;
}

/** 会话行原生拖拽排序的逐行状态：由列表 owner 按行给值；marker 是本行的上/下半插入指示。 */
export interface ConversationRowDragState {
  /** 有同族行拖拽在进行。 */
  active: boolean;
  /** 本行是被拖源。 */
  source: boolean;
  /** 当前插入指示落在本行的哪一半。 */
  marker?: "before" | "after";
}

/** 拖拽处理函数全列表共享一份稳定实例（按会话 id 寻址），行组件才能靠 memo 跳出无关重渲染。 */
export interface ConversationRowDragHandlers {
  /** 只有严格返回 false 才算当下拒绝拖拽（如搜索中），行会把 dragstart preventDefault 掉；void 放行。 */
  start: (id: string) => boolean | void;
  end: () => void;
  hover: (id: string, half: "before" | "after") => void;
  drop: (id: string, half: "before" | "after") => void;
}

/** 指针纵向位置 → 行内上/下半。 */
function rowHalf(event: { clientY: number; currentTarget: HTMLElement }): "before" | "after" {
  const rect = event.currentTarget.getBoundingClientRect();
  return event.clientY < rect.top + rect.height / 2 ? "before" : "after";
}

/** Render the shared navigation language while adapters provide platform capabilities. */
export function ConversationNavigation({
  destinations,
  sessions,
  onSessionActivate,
  onSessionPrefetch,
  pendingSessionId,
  actions,
  closeAction,
  panelRef,
  dialog,
  destinationHeading,
  sessionHeading,
  sessionHeadingAction,
  className = "",
  sessionActions,
  sessionDrag,
  dragHandlers,
  onSessionDelete,
  onSessionRename,
  sessionGroup,
  hiddenSessions,
}: {
  destinations: ConversationDestination[];
  sessions: ConversationSession[];
  onSessionActivate: (sessionId: string) => void;
  onSessionPrefetch?: (sessionId: string) => void;
  pendingSessionId?: string;
  actions: ConversationAction[];
  closeAction?: ReactNode;
  panelRef?: React.Ref<HTMLElement>;
  dialog?: boolean;
  destinationHeading?: string | false;
  sessionHeading?: string;
  /** 分组标题行右侧的展示态操作（如会话排序），不参与标题语义。 */
  sessionHeadingAction?: ReactNode;
  className?: string;
  sessionActions?: (session: ConversationSession) => NavigationRowAction[];
  /** 提供后会话行可拖拽排序；行只表达拖拽态（进行中/来源/落点），当下能否拖拽由 dragHandlers.start 决定。 */
  sessionDrag?: (session: ConversationSession) => ConversationRowDragState | undefined;
  /** sessionDrag 的处理函数：必须是渲染之间稳定的同一实例（如 useMemo([])）。 */
  dragHandlers?: ConversationRowDragHandlers;
  /** 提供后会话行获得就地两步删除；Promise 拒绝时行保持原位。 */
  onSessionDelete?: (session: ConversationSession) => Promise<void>;
  /** 提供后会话行获得双击/菜单行内重命名；Promise 拒绝时编辑态保留。 */
  onSessionRename?: (session: ConversationSession, title: string) => Promise<void>;
  /** 提供后这些 id 的行保留挂载但 hidden：搜索收窄不再成批卸载行。 */
  hiddenSessions?: ReadonlySet<string>;
  /** 提供后在组名变化处插入一行安静的分组标签（如按时间排序时的"今天""昨天"）。 */
  sessionGroup?: (session: ConversationSession) => string | undefined;
}) {
  const featuredDestinations = destinations.filter((destination) => destination.featured);
  const standardDestinations = destinations.filter((destination) => !destination.featured);
  // 分组标签只贴在可见行的边界上：hidden 行不参与组名比较，
  // 否则整组被隐藏时标签悬空、组内首行被隐藏时标签又跟着消失。
  const groupHeaderIds = useMemo(() => {
    if (!sessionGroup) return undefined;
    const ids = new Set<string>();
    let last: string | undefined;
    for (const session of sessions) {
      if (hiddenSessions?.has(session.id)) continue;
      const label = sessionGroup(session);
      if (label && label !== last) ids.add(session.id);
      last = label;
    }
    return ids;
  }, [sessions, sessionGroup, hiddenSessions]);

  return (
    <aside
      ref={panelRef}
      className={`conversation-navigation ${className}`}
      role={dialog ? "dialog" : undefined}
      aria-modal={dialog || undefined}
      aria-label="会话列表"
      tabIndex={dialog ? -1 : undefined}
    >
      {featuredDestinations.length === 0 && destinationHeading !== false ? (
        <header className="conversation-navigation__header">
          <div className="conversation-navigation__heading">{destinationHeading || "会话"}</div>
          {closeAction}
        </header>
      ) : closeAction ? (
        <header className="conversation-navigation__header">{closeAction}</header>
      ) : null}

      {featuredDestinations.length > 0 ? (
        <DestinationList destinations={featuredDestinations} featured />
      ) : null}
      <DestinationList destinations={standardDestinations} />
      {sessionHeading || featuredDestinations.length > 0 ? <div
        className={`conversation-navigation__heading conversation-navigation__heading--section${sessionHeadingAction ? " conversation-navigation__heading--row" : ""}`}>
        <span>{sessionHeading || "会话"}</span>
        {sessionHeadingAction}
      </div> : null}

      <section className="conversation-navigation__sessions">
        <nav className="conversation-session-list" aria-label="最近会话">
          {sessions.map((session) => <Fragment key={session.id}>
            {groupHeaderIds?.has(session.id)
              ? <div className="conversation-session-group">{sessionGroup?.(session)}</div> : null}
            <ConversationSessionRow
            session={session}
            hidden={hiddenSessions?.has(session.id) || undefined}
            pendingSessionId={pendingSessionId}
            onActivate={onSessionActivate}
            onPrefetch={onSessionPrefetch}
            actions={sessionActions}
            drag={sessionDrag?.(session)}
            dragHandlers={dragHandlers}
            onDelete={onSessionDelete}
            onRename={onSessionRename}
          /></Fragment>)}
        </nav>
      </section>

      {actions.length ? <div className="conversation-navigation__actions">
        {actions.map((action) => (
          <button
            className={`conversation-navigation__action ${action.primary ? "primary" : ""}`}
            type="button"
            key={action.id}
            disabled={action.disabled}
            onClick={action.onActivate}
          >
            {action.icon}
            <span>{action.label}</span>
          </button>
        ))}
      </div> : null}
    </aside>
  );
}

function sameSessionView(a: ConversationSession, b: ConversationSession): boolean {
  return a === b || (a.id === b.id && a.title === b.title && a.preview === b.preview
    && a.active === b.active && a.status === b.status && a.unavailable === b.unavailable && a.pinned === b.pinned);
}

function sameDragState(a?: ConversationRowDragState, b?: ConversationRowDragState): boolean {
  if (a === b) return true;
  if (!a || !b) return false;
  return a.active === b.active && a.source === b.source && a.marker === b.marker;
}

/**
 * 会话行按值记忆：流式帧、目录全量重拉都不该让未变化的行重渲染（2000 行目录实测每帧约 480ms）。
 * 调用方须保证会话对象结构共享（字段不变则身份不变）、回调与 dragHandlers 身份稳定。
 */
export const ConversationSessionRow = memo(function ConversationSessionRow({ session, pendingSessionId, onActivate, onPrefetch, actions, drag, dragHandlers, onDelete, onRename, hidden }: {
  session: ConversationSession;
  hidden?: boolean;
  pendingSessionId?: string;
  onActivate: (sessionId: string) => void;
  onPrefetch?: (sessionId: string) => void;
  /** 行动数组，或按会话求解的工厂；工厂身份变化时才重算（pending/pinned 等输入变化应换工厂身份）。 */
  actions?: NavigationRowAction[] | ((session: ConversationSession) => NavigationRowAction[]);
  /** 提供后行可拖拽排序并在落点上/下半给出插入指示。 */
  drag?: ConversationRowDragState;
  dragHandlers?: ConversationRowDragHandlers;
  onDelete?: (session: ConversationSession) => Promise<void>;
  onRename?: (session: ConversationSession, title: string) => Promise<void>;
}) {
  const [armed, setArmed] = useState(false);
  const [renaming, setRenaming] = useState(false);
  const [renameValue, setRenameValue] = useState("");
  const [renameBusy, setRenameBusy] = useState(false);
  const armTimer = useRef<ReturnType<typeof setTimeout> | null>(null);
  const renameInputRef = useRef<HTMLInputElement>(null);
  const renameCancelledRef = useRef(false);
  // 回车提交后输入框转为 disabled 会触发 blur，blur 也提交：同一次编辑只发一次请求。
  const renameInFlightRef = useRef(false);
  const resolvedActions = useMemo(
    () => (typeof actions === "function" ? actions(session) : actions),
    [actions, session],
  );
  useEffect(() => () => {
    if (armTimer.current !== null) clearTimeout(armTimer.current);
  }, []);

  const startRename = () => {
    if (!onRename || session.unavailable) return;
    renameCancelledRef.current = false;
    setRenameValue(session.title);
    setRenaming(true);
    window.setTimeout(() => {
      renameInputRef.current?.focus();
      renameInputRef.current?.select();
    }, 0);
  };
  const cancelRename = () => {
    renameCancelledRef.current = true;
    setRenaming(false);
  };
  const commitRename = async () => {
    if (renameCancelledRef.current || renameInFlightRef.current || !renaming || !onRename) return;
    const title = renameValue.trim();
    if (title === session.title.trim()) {
      setRenaming(false);
      return;
    }
    renameInFlightRef.current = true;
    setRenameBusy(true);
    try {
      await onRename(session, title);
      setRenaming(false);
    } finally {
      renameInFlightRef.current = false;
      setRenameBusy(false);
    }
  };

  // 3 秒未确认自动还原；确认提交由 onDelete 的拒绝与否决定是否保留原位。
  const disarm = () => {
    if (armTimer.current !== null) clearTimeout(armTimer.current);
    armTimer.current = null;
    setArmed(false);
  };
  const arm = () => {
    disarm();
    setArmed(true);
    armTimer.current = setTimeout(() => {
      armTimer.current = null;
      setArmed(false);
    }, 3000);
  };
  const fire = async () => {
    if (!onDelete) return;
    disarm();
    try {
      await onDelete(session);
    } finally {
      setArmed(false);
    }
  };

  const renameAction: NavigationRowAction[] = onRename && !session.unavailable ? [{
    label: "重命名",
    icon: <PenLine size={18} aria-hidden="true" />,
    onSelect: startRename,
  }] : [];
  // 删除只存在于行菜单内：第一次选择保持菜单开启、该项就地变为「确认删除」，
  // 第二次选择才执行；菜单关闭或 3 秒未确认自动还原。行上不再有独立的删除方块。
  const deleteAction: NavigationRowAction[] = onDelete ? [{
    label: armed ? "确认删除" : "删除",
    danger: true,
    icon: <Trash2 size={18} aria-hidden="true" />,
    onSelect: (event) => {
      if (!armed) {
        event.preventDefault();
        arm();
        return;
      }
      void fire();
    },
  }] : [];
  const rowActions = [...renameAction, ...(resolvedActions ?? []), ...deleteAction];

  const body = renaming ? <div
    className={`conversation-session editing ${session.active ? "active" : ""}`}
    onClick={(event) => event.stopPropagation()}>
    <SessionStatusMark status={session.status} />
    <span className="conversation-session__copy">
      <span className="conversation-session__title">
        <input
          ref={renameInputRef}
          className="conversation-session-title-input"
          value={renameValue}
          maxLength={200}
          aria-label={`重命名 ${session.title}`}
          disabled={renameBusy}
          onChange={(event) => setRenameValue(event.target.value)}
          onKeyDown={(event) => {
            // 输入法组合中的回车是确认候选词、Esc 是取消候选，都不属于编辑框（Safari 只剩 keyCode 229 可辨认）。
            if (event.nativeEvent.isComposing || event.nativeEvent.keyCode === 229) return;
            if (event.key === "Enter") {
              event.preventDefault();
              void commitRename();
            } else if (event.key === "Escape") {
              event.preventDefault();
              cancelRename();
            }
          }}
          onBlur={() => void commitRename()}
        />
      </span>
      <small>{session.preview}</small>
    </span>
  </div> : <button
      className={`conversation-session ${session.active ? "active" : ""} ${session.unavailable ? "unavailable" : ""}`}
      type="button"
      aria-current={session.active ? "true" : undefined}
      aria-busy={pendingSessionId === session.id || undefined}
      data-row-primary=""
      disabled={session.unavailable}
      title={session.preview ? `${session.title} · ${session.preview}` : session.title}
      draggable={drag !== undefined}
      onDragStart={drag === undefined ? undefined : (event) => {
        if (dragHandlers?.start(session.id) === false) {
          event.preventDefault();
          return;
        }
        event.dataTransfer.setData("text/plain", session.id);
        event.dataTransfer.effectAllowed = "move";
      }}
      onDragEnd={drag === undefined ? undefined : () => dragHandlers?.end()}
      onDragOver={drag === undefined ? undefined : (event) => {
        if (!drag.active) return;
        event.preventDefault();
        event.dataTransfer.dropEffect = "move";
        dragHandlers?.hover(session.id, rowHalf(event));
      }}
      onDrop={drag === undefined ? undefined : (event) => {
        if (!drag.active) return;
        event.preventDefault();
        dragHandlers?.drop(session.id, rowHalf(event));
      }}
      onClick={() => onActivate(session.id)}
      onPointerEnter={() => { if (!session.unavailable) onPrefetch?.(session.id); }}
      onFocus={() => { if (!session.unavailable) onPrefetch?.(session.id); }}
      onDoubleClick={startRename}>
      <SessionStatusMark status={session.status} />
      <span className="conversation-session__copy">
        <span className="conversation-session__title">
          <strong><MarqueeText>{session.title}</MarqueeText></strong>
        </span>
        <small><MarqueeText still>{session.preview}</MarqueeText></small>
      </span>
      {session.pinned ? <Pin size={13} className="conversation-session__pin" aria-label="已置顶" /> : null}
    </button>;

  return <NavigationRowMenu title={session.title} actions={rowActions} hidden={hidden}
    className={`conversation-session-row ${session.active ? "active" : ""}${drag?.source ? " session-row-drag-source" : ""}${drag?.marker ? ` session-row-drop-${drag.marker}` : ""}`}
    onOpenChange={(open) => { if (!open) disarm(); }}>
    {body}
  </NavigationRowMenu>;
}, (prev, next) => sameSessionView(prev.session, next.session)
  && prev.pendingSessionId === next.pendingSessionId
  && prev.onActivate === next.onActivate
  && prev.onPrefetch === next.onPrefetch
  && prev.actions === next.actions
  && sameDragState(prev.drag, next.drag)
  && prev.dragHandlers === next.dragHandlers
  && prev.onDelete === next.onDelete
  && prev.onRename === next.onRename
  && prev.hidden === next.hidden);

function DestinationList({ destinations, featured = false }: { destinations: ConversationDestination[]; featured?: boolean }) {
  if (destinations.length === 0) return null;
  return (
    <nav className={`conversation-destinations ${featured ? "featured" : ""}`} aria-label={featured ? "重点功能入口" : "功能入口"}>
      {destinations.map((destination) => {
        const content = (
          <>
            <span className="conversation-destination__icon" aria-hidden="true">{destination.icon}</span>
            <span className="conversation-destination__copy">
              <strong>{destination.label}</strong>
              {destination.description ? <small>{destination.description}</small> : null}
            </span>
            <span className="conversation-destination__trail">
              {destination.badge ? (
                <span className="conversation-destination__badge">{destination.badge}</span>
              ) : null}
              <ChevronRight size={18} aria-hidden="true" />
            </span>
          </>
        );
        const className = `conversation-destination ${featured ? "featured" : ""} ${destination.active ? "active" : ""}`;
        return destination.href && !destination.disabled ? (
          <a className={className} href={destination.href} aria-current={destination.active ? "page" : undefined} key={destination.id}>{content}</a>
        ) : (
          <button className={className} type="button" aria-current={destination.active ? "page" : undefined} disabled={destination.disabled} onClick={destination.onActivate} key={destination.id}>{content}</button>
        );
      })}
    </nav>
  );
}
