import type { WebHostContextV1 } from "@akashic/web-ui-v1";

type Http = WebHostContextV1["http"];
interface NavigateDetail { go?: () => void; wait?: () => void; reason?: string }

/**
 * 扩展包安装进行中暂缓界面换代：每装好一个包目录就变一次，装完只换一次，外框不反复重建。
 * 期间只读请求照常（宿主按当前目录重绑），写操作会被宿主以“界面刚更新过”拒绝，装完换代后恢复。
 */
let current: { kick: () => void } | null = null;

// 提交安装后调用：立即开始跟踪任务。
export function kickInstallHold(): void {
  current?.kick();
}

// 外框挂载时启用：先读一次任务状态（页面重开时接上在装的任务），装着就每 2 秒再读。
export function holdWhileInstalling(http: Http): () => void {
  let running = false;
  let held: (() => void) | null = null;
  let timer = 0;
  let reading = false;
  // 1. 不在装或读不到状态都放行，避免一直拦着；2. 放行时补发被拦下的那次换代；
  // 3. 期间宿主没核对过（窗口一直在前台）就请它核对一次，装好的功能才会出现。
  const release = () => {
    const was = running;
    running = false;
    window.clearInterval(timer); timer = 0;
    const go = held; held = null;
    if (go) go();
    else if (was) window.dispatchEvent(new CustomEvent("akashic:configuration-submitted"));
  };
  const read = async () => {
    if (reading) return;
    reading = true;
    try {
      const response = await http.request("/api/dashboard/plugins/install");
      const body = response.ok ? await response.json() as { task: { state: string } | null } : null;
      if (body?.task?.state === "running") {
        running = true;
        if (!timer) timer = window.setInterval(() => void read(), 2000);
      } else release();
    } catch { release(); } finally { reading = false; }
  };
  const onNavigate = (event: Event) => {
    const detail = (event as CustomEvent<NavigateDetail>).detail;
    if (!running || detail?.reason !== "catalog" || !detail.go) return;
    event.preventDefault();
    detail.wait?.();
    held = detail.go;
  };
  window.addEventListener("akashic:before-navigate", onNavigate);
  current = { kick: () => void read() };
  void read();
  return () => {
    window.removeEventListener("akashic:before-navigate", onNavigate);
    window.clearInterval(timer);
    current = null;
  };
}
