import { useEffect, useState } from "react";
import { Notice, render, type PluginUiContext, type PluginUiDefinition } from "@akashic/chat-ui-v1";

const POLL_INTERVAL_MS = 5000;
// 单次查询由宿主限时；15 秒宽限保证一次偶发失败不报警，持续失败约 15 秒后才提示未知。
const UNCONFIRMED_GRACE_MS = 15_000;
const UNCONFIRMED_NOTICE = "暂时无法确认宿主执行状态，正在重新检查。";

// 只显示宿主执行连接状态；恢复连接不会恢复已经失效的命令句柄。
function HostBridgeNotice({ context }: { context: PluginUiContext<"chat.notice"> }) {
  const [notice, setNotice] = useState("");
  useEffect(() => {
    let active = true;
    let timer: ReturnType<typeof setTimeout>;
    // 连续失败从第一次失败请求发出时起算；任何一次确认到状态都清零。
    let failingSince: number | null = null;
    async function refresh() {
      const startedAt = Date.now();
      try {
        const next = noticeFor(await context.query("bridge.state", {}, { cache: "none" }));
        if (active) setNotice(next);
        failingSince = null;
      } catch {
        if (!active) return;
        failingSince ??= startedAt;
        // 宽限期内保留上一次确认到的提示，避免共享事件循环偶发停顿造成闪烁。
        if (Date.now() - failingSince >= UNCONFIRMED_GRACE_MS) setNotice(UNCONFIRMED_NOTICE);
      }
      if (active) timer = setTimeout(refresh, POLL_INTERVAL_MS);
    }
    void refresh();
    return () => { active = false; clearTimeout(timer); };
  }, [context]);
  return notice ? <Notice>{notice}</Notice> : null;
}

// 把服务端明确给出的状态换成提示文案；响应无效时抛错，按查询失败处理。
function noticeFor(body: Record<string, unknown>): string {
  if (!["disabled", "checking", "healthy", "degraded"].includes(String(body.state))) throw new Error("宿主状态响应无效");
  return body.state === "degraded"
    ? "宿主执行暂时不可用，正在恢复连接。依赖宿主的命令和文件操作可能失败；已有对话仍可阅读。"
    : body.state === "checking" ? "正在检查宿主执行连接…" : "";
}

const definition: PluginUiDefinition = {
  slots: { "chat.notice": { mount: (host, context) => render(host, <HostBridgeNotice context={context} />) } },
};
export default definition;
