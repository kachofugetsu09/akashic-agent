// 实际聊天组件读取真实工具产生的消息；控制按钮只驱动可重复的追加/目录事件。
import React, { useRef, useState } from "react";
import { createRoot } from "react-dom/client";
import { StickToBottom } from "use-stick-to-bottom";
import { DesktopTimelineMessages } from "../../frontend/chat/src/desktop-conversation";
import { loadWebPluginCatalog, receivePluginUiCatalog } from "../../frontend/chat/src/plugin-ui-runtime";
import { readTimelineMessage, type ReplyActivity } from "../../frontend/chat/src/message-timeline";
import { cycleTheme, initializeTheme } from "../../frontend/theme/src/theme-runtime";
import { TooltipProvider } from "../../frontend/chat/src/components/ui/tooltip";
import "../../frontend/chat/src/styles.css";
import "../../frontend/chat/src/message-view.css";

initializeTheme();
const messages = (await (await fetch("/messages.json")).json()).slice(0, 4).map(readTimelineMessage);
await loadWebPluginCatalog();
function App() {
  const [phase, setPhase] = useState(0);
  const [disabled, setDisabled] = useState(false);
  const anchors = useRef(new Map());
  const activities: ReplyActivity[] = phase === 2 ? [] : [{ session_id: messages[0].session_id,
    source: messages[0].source, handle: "live", active: true,
    preview: phase === 1 ? { message_id: "first:end", text: "参数越大，同一横坐标对应的数值越高。", thinking: "" } : null }];
  return <TooltipProvider><nav>
    <button onClick={() => setPhase(1)}>追加流式文字</button>
    <button onClick={() => setPhase(2)}>完成回复</button>
    <button onClick={() => cycleTheme()}>切换主题</button>
    <button onClick={async () => {
      if (disabled) await loadWebPluginCatalog();
      else await receivePluginUiCatalog({ catalogRevision: "disabled", updating: false, plugins: [] });
      setDisabled(!disabled);
    }}>{disabled ? "启用插件" : "禁用插件"}</button>
  </nav><main><StickToBottom><DesktopTimelineMessages messages={messages.slice(0, phase === 2 ? 4 : 3)}
    activities={activities} status={phase === 2 ? "idle" : "streaming"} copiedMessageId="" messageElementsRef={anchors}
    onReply={() => {}} onCopied={() => {}} onError={error => { throw error; }} />
  </StickToBottom></main></TooltipProvider>;
}
createRoot(document.getElementById("root")!).render(<App />);
