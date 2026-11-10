import { useCallback, useEffect, useRef, useState } from "react";
import { createUuidV7 } from "./browser-uuid.ts";
import type { TimelineMessage } from "./message-timeline";

export interface StagedSteer {
  id: string;
  sessionId: string;
  text: string;
}

export interface SteerResult {
  staged?: boolean;
  withdrawn?: boolean;
}

const STEER_ACK_TIMEOUT_MS = 10_000;

// 插话帧的请求与回执：按 request_id 配对，超时视为没有送达。
function useSteerRequests(send: (payload: Record<string, unknown>) => Promise<void>) {
  const waiters = useRef(new Map<string, (result: SteerResult) => void>());
  const request = useCallback((requestId: string, payload: Record<string, unknown>) => new Promise<SteerResult>((resolve, reject) => {
    const finish = () => { window.clearTimeout(timer); waiters.current.delete(requestId); };
    const timer = window.setTimeout(() => {
      finish();
      reject(new Error("网络不稳定，插话没有送达，已放回输入框。"));
    }, STEER_ACK_TIMEOUT_MS);
    waiters.current.set(requestId, (result) => { finish(); resolve(result); });
    send(payload).catch((error: unknown) => { finish(); reject(error); });
  }), [send]);
  const onResult = useCallback((requestId: string, result: SteerResult) => {
    waiters.current.get(requestId)?.(result);
  }, []);
  return { request, onResult };
}

// 生成中的插话：服务端暂存到本轮下一步，写进对话后卡片消失；不接纳时返回 false，由输入框走普通发送。
export function useSteers({ sessionId, timeline, send }: {
  sessionId: string;
  timeline: TimelineMessage[];
  send: (payload: Record<string, unknown>) => Promise<void>;
}) {
  const [steers, setSteers] = useState<StagedSteer[]>([]);
  const latest = useRef(steers);
  useEffect(() => { latest.current = steers; }, [steers]);
  const { request, onResult } = useSteerRequests(send);

  const steer = useCallback(async (text: string): Promise<boolean> => {
    if (!sessionId) return false;
    const id = createUuidV7();
    const drop = () => setSteers((current) => current.filter((item) => item.id !== id));
    setSteers((current) => [...current, { id, sessionId, text }]);
    try {
      const result = await request(id, { type: "message.steer", request_id: id, session_id: sessionId, text });
      if (!result.staged) drop();
      return Boolean(result.staged);
    } catch (error) {
      drop();
      throw error;
    }
  }, [sessionId, request]);

  // 撤回尚未交给模型的插话；已写入时返回 null，卡片随消息进入对话而消失。
  const withdraw = useCallback(async (id: string): Promise<string | null> => {
    const item = steers.find((candidate) => candidate.id === id);
    if (!item) return null;
    const requestId = createUuidV7();
    const result = await request(requestId, { type: "message.steer.withdraw", request_id: requestId, session_id: item.sessionId, message_id: id });
    if (!result.withdrawn) return null;
    setSteers((current) => current.filter((candidate) => candidate.id !== id));
    return item.text;
  }, [steers, request]);

  // 本轮停下后仍未进入对话的插话：服务端仍暂存的撤回后退回输入框；已被取走的再等 graceMs，
  // 仍未进入对话（重启丢失）才退回。只按撤回回执与时间线判断，避免抢走停止后排队的插话或重复成草稿。
  const reclaim = useCallback(async (graceMs: number): Promise<string[]> => {
    const own = latest.current.filter((item) => item.sessionId === sessionId);
    const results = await Promise.all(own.map((item) => {
      const requestId = createUuidV7();
      return request(requestId, { type: "message.steer.withdraw", request_id: requestId, session_id: item.sessionId, message_id: item.id })
        .then((result) => Boolean(result.withdrawn), () => false);
    }));
    const taken = own.filter((_, index) => !results[index]);
    if (taken.length) await new Promise((resolve) => window.setTimeout(resolve, graceMs));
    const back = own.filter((item, index) => results[index] || latest.current.some((current) => current.id === item.id));
    if (!back.length) return [];
    setSteers((current) => current.filter((item) => !back.some((other) => other.id === item.id)));
    return back.map((item) => item.text);
  }, [sessionId, request]);

  useEffect(() => {
    const ids = new Set(timeline.map((message) => message.id));
    setSteers((current) => current.some((item) => ids.has(item.id)) ? current.filter((item) => !ids.has(item.id)) : current);
  }, [timeline]);

  return { items: steers.filter((item) => item.sessionId === sessionId), onResult, steer, withdraw, reclaim };
}
