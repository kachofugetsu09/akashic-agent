import type { ProactiveTick } from "./types";

export function escapeHtml(value: unknown): string {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}

export function encodePath(value: string): string {
  return encodeURIComponent(value).replaceAll("%2F", "/");
}

export function stripMarkdown(text: unknown): string {
  return String(text ?? "")
    .replace(/\*\*(.+?)\*\*/g, "$1")
    .replace(/\*(.+?)\*/g, "$1")
    .replace(/__(.+?)__/g, "$1")
    .replace(/_(.+?)_/g, "$1")
    .replace(/~~(.+?)~~/g, "$1")
    .replace(/`{1,3}[\s\S]*?`{1,3}/g, "")
    .replace(/\[(.+?)\]\(.+?\)/g, "$1")
    .replace(/^#{1,6}\s+/gm, "")
    .replace(/^>\s*/gm, "")
    .replace(/\n+/g, " ")
    .trim();
}

export function renderMarkdown(text: unknown): string {
  const raw = String(text ?? "").trim();
  if (!raw) {
    return '<span class="detail-subtext">empty</span>';
  }
  return `<span class="pre-wrap">${escapeHtml(raw).replaceAll("\n", "<br>")}</span>`;
}

export function formatSessionKeyForTable(key: unknown): string {
  const raw = String(key || "");
  const parts = raw.split(":");
  if (parts.length < 2) {
    return raw;
  }
  const channel = parts[0];
  const tail = parts.slice(1).join(":");
  if (tail.length <= 10) {
    return `${channel}:${tail}`;
  }
  return `${channel}:${tail.slice(0, 6)}...${tail.slice(-4)}`;
}

export function shortTs(value: unknown): string {
  if (!value) {
    return "-";
  }
  const date = new Date(String(value));
  if (Number.isNaN(date.getTime())) {
    return String(value);
  }
  return `${date.getMonth() + 1}-${String(date.getDate()).padStart(2, "0")} ${String(
    date.getHours(),
  ).padStart(2, "0")}:${String(date.getMinutes()).padStart(2, "0")}`;
}

export function relativeTime(value: unknown): string {
  if (!value) {
    return "未更新";
  }
  const time = new Date(String(value)).getTime();
  if (Number.isNaN(time)) {
    return String(value);
  }
  const diff = Date.now() - time;
  const minute = 60 * 1000;
  const hour = 60 * minute;
  const day = 24 * hour;
  if (diff < hour) {
    return `${Math.max(1, Math.round(diff / minute))} 分钟前`;
  }
  if (diff < day) {
    return `${Math.round(diff / hour)} 小时前`;
  }
  return `${Math.round(diff / day)} 天前`;
}

export function formatNumber(value: unknown): string {
  return new Intl.NumberFormat("zh-CN").format(Number(value || 0));
}

export function roleClass(role: string): string {
  return `role-${role || "unknown"}`;
}

export function memoryTypeClass(memoryType: string): string {
  return `memory-type-${memoryType || "unknown"}`;
}

export function proactiveResultLabel(item: ProactiveTick): string {
  return item.terminal_action || item.gate_exit || "unknown";
}

export function proactiveFlowLabel(item: ProactiveTick | null): string {
  return item?.drift_entered ? "Drift" : "Proactive";
}

export function proactiveSectionLabel(section: string): string {
  const labels: Record<string, string> = {
    all: "Tick Logs",
    drift: "Drift",
    proactive: "Proactive",
    reply: "Reply",
    skip: "Skip",
    busy: "Busy",
    cooldown: "Cooldown",
    presence: "Presence",
  };
  return labels[section] || section;
}

export function proactiveTickPreview(item: ProactiveTick): string {
  const parts: string[] = [];
  if (item.skip_reason) {
    parts.push(item.skip_reason);
  }
  if (item.final_message) {
    parts.push(stripMarkdown(item.final_message));
  }
  if (!parts.length) {
    parts.push(
      `alerts ${item.alert_count || 0} · content ${item.content_count || 0} · context ${item.context_count || 0}`,
    );
  }
  return parts.join(" · ");
}


