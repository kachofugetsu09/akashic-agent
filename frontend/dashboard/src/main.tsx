import "./styles.css";
import { initializeTheme, startCrossPortThemeSync } from "../../theme/src/theme-runtime";
import { exposeRuntime } from "./design/runtime";
import { startWebHost } from "./webHost";

initializeTheme();
startCrossPortThemeSync();
exposeRuntime();

const root = document.getElementById("root");
if (!(root instanceof HTMLElement)) throw new Error("Dashboard root is missing");

void startWebHost(root).then((session) => {
  window.addEventListener("pagehide", () => session.close(), { once: true });
}).catch((reason) => {
  console.error("[web-host] Web UI bootstrap unavailable", reason);
  const notice = document.createElement("p");
  notice.className = "web-host-entry-error";
  notice.setAttribute("role", "alert");
  notice.textContent = "Web 界面暂时不可用，请刷新重试。";
  root.replaceChildren(notice);
});
