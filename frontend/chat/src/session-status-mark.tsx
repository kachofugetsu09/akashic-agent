import "./session-status-mark.css";

/** 侧栏会话的状态：正在回复，或有尚未查看的新回复；空闲不显示任何标记。 */
export type SessionStatus = "running" | "unread";

const STATUS_LABEL: Record<SessionStatus, string> = { running: "正在回复", unread: "有新回复" };

// 3x3 环去掉中心：每个格子在环上的次序，决定明暗依次传递的方向。
const SWIRL_RING: readonly (number | null)[] = [0, 1, 2, 7, null, 3, 6, 5, 4];

/** 行首固定宽度的状态槽位；无状态时留白，状态出现或消失都不会让标题位移。 */
export function SessionStatusMark({ status }: { status?: SessionStatus }) {
  return <span className="session-status-slot">
    {status === "running" ? <SessionSwirl /> : null}
    {status === "unread" ? <i className="session-status-dot" aria-hidden="true" /> : null}
    {status ? <span className="session-status-label">{STATUS_LABEL[status]}</span> : null}
  </span>;
}

function SessionSwirl() {
  return <span className="session-swirl" aria-hidden="true">
    {SWIRL_RING.map((place, index) => place === null
      ? <b key={index} />
      : <i key={index} style={{ "--swirl-step": place } as React.CSSProperties} />)}
  </span>;
}

/** 折叠的项目汇总其会话：有会话在回复优先，其次是未读。 */
export function summarizeStatus(statuses: readonly (SessionStatus | undefined)[]): SessionStatus | undefined {
  if (statuses.includes("running")) return "running";
  return statuses.includes("unread") ? "unread" : undefined;
}
