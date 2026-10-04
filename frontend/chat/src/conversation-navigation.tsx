import { ChevronRight, PenLine, Trash2 } from "lucide-react";
import { useEffect, useRef, useState, type ReactNode } from "react";
import "./conversation-navigation.css";
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
  updatedLabel?: string;
  active: boolean;
  unavailable?: boolean;
  state?: ReactNode;
}

export interface ConversationAction {
  id: string;
  icon: ReactNode;
  label: string;
  primary?: boolean;
  disabled?: boolean;
  onActivate: () => void;
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
  sessionAfterContent,
  panelRef,
  dialog,
  destinationHeading,
  sessionHeading,
  sessionHeadingAction,
  className = "",
  sessionActions,
  onSessionDelete,
  onSessionRename,
}: {
  destinations: ConversationDestination[];
  sessions: ConversationSession[];
  onSessionActivate: (sessionId: string) => void;
  onSessionPrefetch?: (sessionId: string) => void;
  pendingSessionId?: string;
  actions: ConversationAction[];
  closeAction?: ReactNode;
  sessionAfterContent?: ReactNode;
  panelRef?: React.Ref<HTMLElement>;
  dialog?: boolean;
  destinationHeading?: string | false;
  sessionHeading?: string;
  /** 分组标题行右侧的展示态操作（如会话排序），不参与标题语义。 */
  sessionHeadingAction?: ReactNode;
  className?: string;
  sessionActions?: (session: ConversationSession) => NavigationRowAction[];
  /** 提供后会话行获得就地两步删除；Promise 拒绝时行保持原位。 */
  onSessionDelete?: (session: ConversationSession) => Promise<void>;
  /** 提供后会话行获得双击/菜单行内重命名；Promise 拒绝时编辑态保留。 */
  onSessionRename?: (session: ConversationSession, title: string) => Promise<void>;
}) {
  const featuredDestinations = destinations.filter((destination) => destination.featured);
  const standardDestinations = destinations.filter((destination) => !destination.featured);

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
          {sessions.map((session) => <ConversationSessionRow
            key={session.id}
            session={session}
            pendingSessionId={pendingSessionId}
            onActivate={onSessionActivate}
            onPrefetch={onSessionPrefetch}
            actions={sessionActions?.(session)}
            onDelete={onSessionDelete ? () => onSessionDelete(session) : undefined}
            onRename={onSessionRename ? (title: string) => onSessionRename(session, title) : undefined}
          />)}
        </nav>
      </section>

      {sessionAfterContent ? (
        <div className="conversation-navigation__auxiliary">
          {sessionAfterContent}
        </div>
      ) : null}

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

export function ConversationSessionRow({ session, pendingSessionId, onActivate, onPrefetch, actions, onDelete, onRename }: {
  session: ConversationSession;
  pendingSessionId?: string;
  onActivate: (sessionId: string) => void;
  onPrefetch?: (sessionId: string) => void;
  actions?: NavigationRowAction[];
  onDelete?: () => Promise<void>;
  onRename?: (title: string) => Promise<void>;
}) {
  const [armed, setArmed] = useState(false);
  const [renaming, setRenaming] = useState(false);
  const [renameValue, setRenameValue] = useState("");
  const [renameBusy, setRenameBusy] = useState(false);
  const armTimer = useRef<ReturnType<typeof setTimeout> | null>(null);
  const renameInputRef = useRef<HTMLInputElement>(null);
  const renameCancelledRef = useRef(false);
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
    if (renameCancelledRef.current || !renaming || !onRename) return;
    const title = renameValue.trim();
    if (title === session.title.trim()) {
      setRenaming(false);
      return;
    }
    setRenameBusy(true);
    try {
      await onRename(title);
      setRenaming(false);
    } finally {
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
      await onDelete();
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
  const rowActions = [...renameAction, ...(actions ?? []), ...deleteAction];

  const body = renaming ? <div
    className={`conversation-session editing ${session.active ? "active" : ""}`}
    onClick={(event) => event.stopPropagation()}>
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
      disabled={session.unavailable}
      title={session.preview ? `${session.title} · ${session.preview}` : session.title}
      onClick={() => onActivate(session.id)}
      onPointerEnter={() => { if (!session.unavailable) onPrefetch?.(session.id); }}
      onFocus={() => { if (!session.unavailable) onPrefetch?.(session.id); }}
      onDoubleClick={startRename}>
      <span className="conversation-session__copy">
        <span className="conversation-session__title">
          <strong>{session.title}</strong>
        </span>
        <small>{session.preview}</small>
      </span>
      {session.state ? <span className="conversation-session__state">{session.state}</span> : null}
    </button>;

  return <NavigationRowMenu title={session.title} actions={rowActions}
    className={`conversation-session-row ${session.active ? "active" : ""}`}
    onOpenChange={(open) => { if (!open) disarm(); }}>
    {body}
  </NavigationRowMenu>;
}

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
