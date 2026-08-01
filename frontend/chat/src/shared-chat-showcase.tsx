import { RotateCcw } from "lucide-react";
import { useEffect, useMemo, useState } from "react";
import type { ChatMessage, ToolBlock } from "./main";
import { ChatMessageView } from "./message-view";
import "./shared-chat-showcase.css";

const PHASE_DELAYS = [500, 1_100, 1_700, 2_700, 3_300, 3_900, 4_500, 5_100, 5_700];
const FINAL_CONTENT = [
  "## WebUI 统一结果\n\n",
  "桌面 Web 与 Android WebView 正在使用同一套消息组件。\n\n",
  "- 浅蓝主题一致\n",
  "- thinking 与工具轨迹一致\n",
  "- 平台能力由各自 adapter 提供",
].join("");

export function SharedChatShowcase() {
  const [run, setRun] = useState(0);
  const [phase, setPhase] = useState(0);

  useEffect(() => {
    setPhase(0);
    const timers = PHASE_DELAYS.map((delay, index) => (
      window.setTimeout(() => setPhase(index + 1), delay)
    ));
    return () => timers.forEach(window.clearTimeout);
  }, [run]);

  const assistantMessage = useMemo<ChatMessage>(() => ({
    id: `shared-preview-assistant-${run}`,
    role: "assistant",
    content: previewContent(phase),
    blocks: previewBlocks(phase),
    streaming: phase < PHASE_DELAYS.length,
    durationMs: phase < PHASE_DELAYS.length ? undefined : 5_700,
  }), [phase, run]);

  const userMessage: ChatMessage = {
    id: "shared-preview-user",
    role: "user",
    content: "验证共享 WebUI 的 thinking、工具调用与正文生长",
    blocks: [],
  };

  return (
    <main className="shared-chat-showcase">
      <header className="shared-chat-showcase-header">
        <div>
          <span>AKASHIC · SHARED WEBUI</span>
          <h1>生产消息组件离线验收</h1>
          <p>不连接 Runtime，不读取正式会话；下面直接渲染桌面与 Android 共用的 ChatMessageView。</p>
        </div>
        <button type="button" onClick={() => setRun((current) => current + 1)}>
          <RotateCcw size={16} />
          重新播放
        </button>
      </header>

      <section className="shared-chat-showcase-thread" aria-label="共享消息组件预览">
        <div className="shared-chat-showcase-user">
          <ChatMessageView message={userMessage} />
        </div>
        <div className="shared-chat-showcase-assistant">
          <ChatMessageView message={assistantMessage} />
        </div>
      </section>
    </main>
  );
}

function previewBlocks(phase: number): ChatMessage["blocks"] {
  if (phase === 0) return [];
  const blocks: ChatMessage["blocks"] = [
    {
      kind: "thinking",
      content: phase === 1
        ? "先确认共享主题。"
        : "先确认共享主题，再核对桌面和 Android 的消息渲染入口。",
    },
  ];
  if (phase >= 3) blocks.push(previewTool(phase));
  if (phase >= 5) {
    blocks.push({
      kind: "thinking",
      content: "工具结果确认两端使用同一组件，现在组织最终结论。",
    });
  }
  return blocks;
}

function previewTool(phase: number): ToolBlock {
  const completed = phase >= 4;
  return {
    kind: "tool",
    callId: "shared-preview-tool",
    name: "inspect_shared_webui",
    status: completed ? "output-available" : "input-available",
    input: {
      description: "检查两端是否共用主题与消息组件",
      source: "frontend/chat/src/theme.css",
      targets: ["desktop", "android-webview"],
    },
    output: completed ? "共享主题、ProcessTrace 和 ToolStep 均来自当前仓库。" : undefined,
    errorText: undefined,
    durationMs: completed ? 1_000 : undefined,
  };
}

function previewContent(phase: number) {
  if (phase < 6) return "";
  if (phase === 6) return "## WebUI 统一结果\n\n";
  if (phase === 7) return FINAL_CONTENT.slice(0, 58);
  return FINAL_CONTENT;
}
