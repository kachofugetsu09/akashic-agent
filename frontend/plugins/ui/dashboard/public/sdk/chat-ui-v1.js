// 聊天插件界面的公开运行时：挂载 React 根与宿主统一呈现的基础件。
const C = window.__akashicRuntime.ChatUi;
export const {
  render, Notice, HeaderMarker, HeaderMarkerActions, MarqueeText, ChoiceMenu, copyText, ToolText, ToolData, formatDuration,
} = C;
