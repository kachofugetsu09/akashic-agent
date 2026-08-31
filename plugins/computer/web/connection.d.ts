export const BACKGROUND_HOLD_MS: number;
export function reconnectDelay(attempt: number): number;
export function shouldOpenForActivity(
  lastNotice: number | null,
  noticeId: number,
  active: boolean,
): boolean;
