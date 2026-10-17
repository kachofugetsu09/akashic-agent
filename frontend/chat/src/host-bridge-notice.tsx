import { useEffect, useState } from "react";

const POLL_INTERVAL_MS = 5000;
const REQUEST_TIMEOUT_MS = 5000;
// 单次查询最多 5 秒超时，15 秒宽限保证一次偶发失败永远不报警，持续失败约 15 秒后才提示未知。
const UNCONFIRMED_GRACE_MS = 15_000;
const UNCONFIRMED_NOTICE = "暂时无法确认宿主执行状态，正在重新检查。";

/** 只显示宿主执行连接状态；恢复连接不会恢复已经失效的命令句柄。 */
/** paused：聊天连接本身断开时宿主状态无从确认，同源提示交给连接状态行或发送错误行，这里不再重复。 */
export function HostBridgeNotice({ paused = false }: { paused?: boolean }) {
  const [notice, setNotice] = useState("");
  useEffect(() => {
    const abort = new AbortController();
    let timer: ReturnType<typeof setTimeout>;
    // 连续失败从第一次失败请求发出时起算；任何一次确认到状态都清零。
    let failingSince: number | null = null;
    async function refresh() {
      const startedAt = Date.now();
      try {
        setNotice(await readHostBridgeNotice(abort.signal));
        failingSince = null;
      } catch {
        if (abort.signal.aborted) return;
        failingSince ??= startedAt;
        // 宽限期内保留上一次确认到的提示，避免共享事件循环偶发停顿造成闪烁。
        if (Date.now() - failingSince >= UNCONFIRMED_GRACE_MS) setNotice(UNCONFIRMED_NOTICE);
      }
      if (!abort.signal.aborted) timer = setTimeout(refresh, POLL_INTERVAL_MS);
    }
    void refresh();
    return () => { abort.abort(); clearTimeout(timer); };
  }, []);
  return notice && !paused ? <p role="status" className="reply-unavailable">{notice}</p> : null;
}

// 读取服务端明确给出的宿主状态并换成提示文案；请求失败或响应无效时抛错。
async function readHostBridgeNotice(signal: AbortSignal): Promise<string> {
  const response = await fetch("/api/runtime/host-bridge", {
    signal: AbortSignal.any([signal, AbortSignal.timeout(REQUEST_TIMEOUT_MS)]), cache: "no-store",
  });
  if (!response.ok) throw new Error(`HTTP ${response.status}`);
  const body: unknown = await response.json();
  if (typeof body !== "object" || body === null || !("state" in body)
    || !["disabled", "checking", "healthy", "degraded"].includes(String(body.state))) {
    throw new Error("宿主状态响应无效");
  }
  return body.state === "degraded"
    ? "宿主执行暂时不可用，正在恢复连接。依赖宿主的命令和文件操作可能失败；已有对话仍可阅读。"
    : body.state === "checking" ? "正在检查宿主执行连接…" : "";
}
