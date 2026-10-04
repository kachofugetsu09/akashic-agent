import { ChevronRight, Trash2, X } from "lucide-react";
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
          />)}
        </nav>
      </section>

      {sessionAfterContent ? (
        <div className="conversation-navigation__auxiliary">
          {sessionAfterContent}
        </div>
      ) : null}

      <div className="conversation-navigation__actions">
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
      </div>
    </aside>
  );
}

export function ConversationSessionRow({ session, pendingSessionId, onActivate, onPrefetch, actions, onDelete }: {
  session: ConversationSession;
  pendingSessionId?: string;
  onActivate: (sessionId: string) => void;
  onPrefetch?: (sessionId: string) => void;
  actions?: NavigationRowAction[];
  onDelete?: () => Promise<void>;
}) {
  const [armed, setArmed] = useState(false);
  const [deleting, setDeleting] = useState(false);
  const armTimer = useRef<ReturnType<typeof setTimeout> | null>(null);
  const deleteRef = useRef<HTMLButtonElement>(null);
  useEffect(() => () => {
    if (armTimer.current !== null) clearTimeout(armTimer.current);
  }, []);

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
    setDeleting(true);
    try {
      await onDelete();
    } finally {
      setDeleting(false);
      setArmed(false);
    }
  };

  const rowActions = onDelete
    ? [...(actions ?? []), {
      label: "删除", danger: true,
      icon: <Trash2 size={18} aria-hidden="true" />,
      // 菜单删除不直接执行：就地亮起确认药丸，并把焦点送过去完成第二次点击。
      onSelect: () => {
        arm();
        window.setTimeout(() => deleteRef.current?.focus(), 0);
      },
    } satisfies NavigationRowAction]
    : actions;

  return <NavigationRowMenu title={session.title} actions={rowActions}
    className={`conversation-session-row ${session.active ? "active" : ""}`}>
    <button
      className={`conversation-session ${session.active ? "active" : ""} ${session.unavailable ? "unavailable" : ""}`}
      type="button"
      aria-current={session.active ? "true" : undefined}
      aria-busy={pendingSessionId === session.id || undefined}
      disabled={session.unavailable}
      title={session.preview ? `${session.title} · ${session.preview}` : session.title}
      onClick={() => onActivate(session.id)}
      onPointerEnter={() => { if (!session.unavailable) onPrefetch?.(session.id); }}
      onFocus={() => { if (!session.unavailable) onPrefetch?.(session.id); }}>
      <span className="conversation-session__copy">
        <span className="conversation-session__title">
          <strong>{session.title}</strong>
        </span>
        <small>{session.preview}</small>
      </span>
      {session.state ? <span className="conversation-session__state">{session.state}</span> : null}
    </button>
    {onDelete ? <button
      ref={deleteRef}
      type="button"
      className={`conversation-session-delete${armed ? " is-armed" : ""}`}
      aria-label={armed ? `确认删除 ${session.title}` : `删除 ${session.title}`}
      title={armed ? "再次点击确认删除" : "删除会话"}
      disabled={deleting}
      aria-busy={deleting || undefined}
      onClick={(event) => {
        event.stopPropagation();
        if (!armed) arm();
        else void fire();
      }}
    >{armed ? <span>确认删除</span> : <X size={16} strokeWidth={1.75} aria-hidden="true" />}</button> : null}
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
