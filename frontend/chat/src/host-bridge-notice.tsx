import { useEffect, useState } from "react";

/** 只显示宿主执行连接状态；恢复连接不会恢复已经失效的命令句柄。 */
export function HostBridgeNotice() {
  const [notice, setNotice] = useState("");
  useEffect(() => {
    const abort = new AbortController();
    let timer: ReturnType<typeof setTimeout>;
    async function refresh() {
      try {
        const response = await fetch("/api/runtime/host-bridge", {
          signal: AbortSignal.any([abort.signal, AbortSignal.timeout(5000)]), cache: "no-store",
        });
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        const body: unknown = await response.json();
        if (typeof body !== "object" || body === null || !("state" in body)
          || !["disabled", "checking", "healthy", "degraded"].includes(String(body.state))) {
          throw new Error("宿主状态响应无效");
        }
        setNotice(body.state === "degraded"
          ? "宿主执行暂时不可用，正在恢复连接。依赖宿主的命令和文件操作可能失败；已有对话仍可阅读。"
          : body.state === "checking" ? "正在检查宿主执行连接…" : "");
      } catch {
        if (abort.signal.aborted) return;
        setNotice("暂时无法确认宿主执行状态，正在重新检查。");
      }
      if (!abort.signal.aborted) timer = setTimeout(refresh, 5000);
    }
    void refresh();
    return () => { abort.abort(); clearTimeout(timer); };
  }, []);
  return notice ? <p role="status" className="reply-unavailable">{notice}</p> : null;
}
