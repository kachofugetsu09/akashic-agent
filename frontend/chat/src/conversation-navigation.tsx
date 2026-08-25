import { ChevronRight } from "lucide-react";
import type { ReactNode } from "react";
import "./conversation-navigation.css";

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
  pendingSessionId,
  actions,
  closeAction,
  sessionAfterContent,
  panelRef,
  dialog,
  destinationHeading,
  sessionHeading,
  className = "",
}: {
  destinations: ConversationDestination[];
  sessions: ConversationSession[];
  onSessionActivate: (sessionId: string) => void;
  pendingSessionId?: string;
  actions: ConversationAction[];
  closeAction?: ReactNode;
  sessionAfterContent?: ReactNode;
  panelRef?: React.Ref<HTMLElement>;
  dialog?: boolean;
  destinationHeading?: string | false;
  sessionHeading?: string;
  className?: string;
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
        <>
          <DestinationList destinations={featuredDestinations} featured />
          <div className="conversation-navigation__heading conversation-navigation__heading--section">会话</div>
        </>
      ) : null}
      <DestinationList destinations={standardDestinations} />
      {sessionHeading ? <div className="conversation-navigation__heading conversation-navigation__heading--section">{sessionHeading}</div> : null}

      <section className="conversation-navigation__sessions">
        <nav className="conversation-session-list" aria-label="最近会话">
          {sessions.map((session) => (
            <button
              className={`conversation-session ${session.active ? "active" : ""} ${session.unavailable ? "unavailable" : ""}`}
              type="button"
              key={session.id}
              aria-current={session.active ? "true" : undefined}
              aria-busy={pendingSessionId === session.id || undefined}
              onClick={() => onSessionActivate(session.id)}
            >
              <span className="conversation-session__copy">
                <span className="conversation-session__title">
                  <strong>{session.title}</strong>
                  {session.updatedLabel ? <time>{session.updatedLabel}</time> : null}
                </span>
                <small>{session.preview}</small>
              </span>
              {session.state ? <span className="conversation-session__state">{session.state}</span> : null}
            </button>
          ))}
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
            {destination.badge ? <span className="conversation-destination__badge">{destination.badge}</span> : null}
            <ChevronRight size={18} aria-hidden="true" />
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
