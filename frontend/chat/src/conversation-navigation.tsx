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
  onActivate: () => void;
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
  actions,
  closeAction,
  sessionAfterContent,
  panelRef,
  dialog,
  className = "",
}: {
  destinations: ConversationDestination[];
  sessions: ConversationSession[];
  actions: ConversationAction[];
  closeAction?: ReactNode;
  sessionAfterContent?: ReactNode;
  panelRef?: React.Ref<HTMLElement>;
  dialog?: boolean;
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
      <header className="conversation-navigation__header">
        {featuredDestinations.length === 0 ? <div className="conversation-navigation__heading">会话</div> : null}
        {closeAction}
      </header>

      {featuredDestinations.length > 0 ? (
        <>
          <DestinationList destinations={featuredDestinations} featured />
          <div className="conversation-navigation__heading conversation-navigation__heading--section">会话</div>
        </>
      ) : null}
      <DestinationList destinations={standardDestinations} />

      <section className="conversation-navigation__sessions">
        <nav className="conversation-session-list" aria-label="最近会话">
          {sessions.map((session) => (
            <button
              className={`conversation-session ${session.active ? "active" : ""} ${session.unavailable ? "unavailable" : ""}`}
              type="button"
              key={session.id}
              onClick={session.onActivate}
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
        {sessionAfterContent}
      </section>

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
        const className = `conversation-destination ${featured ? "featured" : ""}`;
        return destination.href && !destination.disabled ? (
          <a className={className} href={destination.href} key={destination.id}>{content}</a>
        ) : (
          <button className={className} type="button" disabled={destination.disabled} onClick={destination.onActivate} key={destination.id}>{content}</button>
        );
      })}
    </nav>
  );
}
