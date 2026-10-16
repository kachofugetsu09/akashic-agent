// 与 use-desktop-chat-controller 的 requestedSessionId 同一判断：URL 的非空 session 参数或标签页记住的会话。
// 壳不知道外壳状态：外壳仍在启动或不可用时，应用会把首帧让给引导页，这一次版式切换是有意的。
try {
  if (!new URLSearchParams(location.search).get("session") && !sessionStorage.getItem("akashic.chat.active-session")) {
    document.documentElement.setAttribute("data-boot-home", "");
  }
} catch (error) {}
