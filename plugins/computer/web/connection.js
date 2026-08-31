export const BACKGROUND_HOLD_MS = 30_000;

export function reconnectDelay(attempt) {
  const checked = Number.isInteger(attempt) && attempt > 0 ? attempt : 0;
  return Math.min(10_000, 500 * (2 ** Math.min(checked, 5)));
}

export function shouldOpenForActivity(lastNotice, noticeId, active) {
  if (noticeId === lastNotice) return false;
  return lastNotice !== null || active;
}
