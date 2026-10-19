import { initializeTheme, startCrossPortThemeSync } from "../../theme/src/theme-runtime";
import { mountChat } from "../src/mount";

// 夹具页面没有外壳：主题由本页自己初始化，宿主接口全部空实现（没有底栏动作、工作台与覆盖层）。
initializeTheme();
startCrossPortThemeSync();
mountChat(document.getElementById("chat")!, {
  activateRailAction() {},
  reportSession() {},
  reportOverlay() {},
  requestIntent: async () => "unhandled",
});
