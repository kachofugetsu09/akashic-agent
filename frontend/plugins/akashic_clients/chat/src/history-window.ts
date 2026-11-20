import { useCallback, useEffect, useRef, useState, type RefObject } from "react";
import { mergeTimelineMessages, type TimelineMessage } from "./message-timeline";
import { chatHistoryPage, chatNewerPage, fetchChatJson } from "./web-chat-data";

// 查找跳到很早的位置时不把中间几千条都渲染出来：只读命中前后一段，作为独立窗口显示，
// 时间线（实时尾部）照常在后台接收新消息；窗口往下读到与尾部相接时并回时间线，或一键回到最新。

// 离已加载的最早一条不超过这么多条时直接往前连续补齐，不开窗口。
const CONTIGUOUS_GAP = 150;
const WINDOW_BEFORE = 40;
const WINDOW_SIZE = 80;
const PAGE_SIZE = 50;

export interface HistoryWindow {
  items: TimelineMessage[];
  /** 再往前读的游标；null 表示已到对话开头。 */
  olderSeq: number | null;
  loadingOlder: boolean;
  loadingNewer: boolean;
}

export interface HistoryWindowDeps {
  sessionId: string;
  throughSeq: number | null;
  beforeSeq: number | null;
  timelineRef: RefObject<TimelineMessage[]>;
  /** 把一段与时间线相接的更早消息并进时间线，并换上新的「更早」游标。 */
  adopt: (items: TimelineMessage[], beforeSeq: number | null) => void;
}

export interface HistoryWindowApi {
  window: HistoryWindow | null;
  /** 确保这条消息已经渲染：在时间线里就不动，近处连续补齐，远处开窗口。 */
  reveal: (seq: number) => Promise<void>;
  loadOlder: () => Promise<void>;
  loadNewer: () => Promise<void>;
  close: () => void;
}

export function useHistoryWindow({ sessionId, throughSeq, beforeSeq, timelineRef, adopt }: HistoryWindowDeps): HistoryWindowApi {
  const [view, setWindowState] = useState<HistoryWindow | null>(null);
  const windowRef = useRef<HistoryWindow | null>(null);
  const setWindow = useCallback((next: HistoryWindow | null) => { windowRef.current = next; setWindowState(next); }, []);
  const latest = useRef({ sessionId, throughSeq, beforeSeq, adopt });
  latest.current = { sessionId, throughSeq, beforeSeq, adopt };
  const endpoint = `/api/chat/sessions/${encodeURIComponent(sessionId)}/messages`;

  // 换会话即丢弃窗口；迟到的读取按会话核对后丢弃。
  useEffect(() => () => setWindow(null), [sessionId, setWindow]);

  const tail = useCallback(async (before: number, size: number) => {
    const { throughSeq: through } = latest.current;
    const page = chatHistoryPage(await fetchChatJson<unknown>(
      `${endpoint}?page_size=${size}&before_seq=${before}${through === null ? "" : `&through_seq=${through}`}`), endpoint);
    if (latest.current.sessionId !== sessionId) throw new DOMException("会话已切换", "AbortError");
    return page;
  }, [endpoint, sessionId]);

  const close = useCallback(() => setWindow(null), [setWindow]);

  const reveal = useCallback(async (seq: number) => {
    const current = windowRef.current;
    if (current && current.items[0].seq <= seq && seq <= current.items.at(-1)!.seq) return;
    const first = timelineRef.current[0]?.seq ?? Infinity;
    const { beforeSeq: older, throughSeq: through } = latest.current;
    // 1. 已在时间线里：离开窗口回到时间线
    if (seq >= first || older === null) { setWindow(null); return; }
    // 2. 近处：一次读齐到这条之前几条，并进时间线
    if (first - seq <= CONTIGUOUS_GAP) {
      setWindow(null);
      const page = await tail(older, Math.min(200, older - seq + WINDOW_BEFORE));
      latest.current.adopt(page.items, page.hasMore ? page.beforeSeq : null);
      return;
    }
    // 3. 远处：读命中前后一段作为窗口
    const before = Math.min(seq + WINDOW_SIZE - WINDOW_BEFORE, (through ?? seq) + 1);
    const page = await tail(before, WINDOW_SIZE);
    if (!page.items.length) return;
    setWindow({ items: page.items, olderSeq: page.hasMore ? page.beforeSeq : null, loadingOlder: false, loadingNewer: false });
  }, [setWindow, tail, timelineRef]);

  const loadOlder = useCallback(async () => {
    const current = windowRef.current;
    if (!current || current.olderSeq === null || current.loadingOlder) return;
    setWindow({ ...current, loadingOlder: true });
    try {
      const page = await tail(current.olderSeq, PAGE_SIZE);
      const now = windowRef.current;
      if (!now) return;
      setWindow({ ...now, items: mergeTimelineMessages(now.items, page.items),
        olderSeq: page.hasMore ? page.beforeSeq : null, loadingOlder: false });
    } catch (error) {
      if (windowRef.current) setWindow({ ...windowRef.current, loadingOlder: false });
      throw error;
    }
  }, [setWindow, tail]);

  // 往下读一页；读到与时间线相接（或已读到上界）时把窗口并回时间线。
  const loadNewer = useCallback(async () => {
    const current = windowRef.current;
    if (!current || current.loadingNewer) return;
    setWindow({ ...current, loadingNewer: true });
    try {
      const { throughSeq: through } = latest.current;
      const after = current.items.at(-1)!.seq;
      const page = chatNewerPage(await fetchChatJson<unknown>(
        `${endpoint}?page_size=${PAGE_SIZE}&after_seq=${after}${through === null ? "" : `&through_seq=${through}`}`), endpoint);
      const now = windowRef.current;
      if (latest.current.sessionId !== sessionId || !now) return;
      const items = mergeTimelineMessages(now.items, page.items);
      const first = timelineRef.current[0]?.seq ?? Infinity;
      if (!page.hasMore || items.at(-1)!.seq + 1 >= first) {
        latest.current.adopt(items, now.olderSeq);
        setWindow(null);
        return;
      }
      setWindow({ ...now, items, loadingNewer: false });
    } catch (error) {
      if (windowRef.current) setWindow({ ...windowRef.current, loadingNewer: false });
      throw error;
    }
  }, [endpoint, sessionId, setWindow, timelineRef]);

  return { window: view, reveal, loadOlder, loadNewer, close };
}
