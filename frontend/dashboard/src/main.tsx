import "./styles.css";
import { initializeTheme, startCrossPortThemeSync } from "../../theme/src/theme-runtime";
import { startWebHost } from "./webHost";

initializeTheme();
startCrossPortThemeSync();

const root = document.getElementById("root");
if (!(root instanceof HTMLElement)) throw new Error("Dashboard root is missing");

// 目录暂不可用时允许原页面重新读取，不清除路由或已保存的操作 ID。
async function open(): Promise<void> {
  if (!(root instanceof HTMLElement)) return;
  try {
    const session = await startWebHost(root);
    window.addEventListener("pagehide", () => session.close(), { once: true });
  } catch (reason) {
    console.error("[web-host] Web UI bootstrap unavailable", reason);
    const notice = document.createElement("p");
    notice.className = "web-host-entry-error";
    notice.setAttribute("role", "alert");
    notice.textContent = "设置界面正在更新或暂时不可用。请稍后重新加载界面；已提交的操作会继续执行。";
    const retry = document.createElement("button");
    retry.type = "button";
    retry.textContent = "重新加载界面";
    retry.onclick = () => { retry.disabled = true; void open(); };
    root.replaceChildren(notice, retry);
  }
}

void open();
