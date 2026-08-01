import { RotateCcw } from "lucide-react";
import { useEffect, useMemo, useState } from "react";
import type { ChatMessage, MessageAttachment, ToolBlock } from "./main";
import { ChatMessageView } from "./message-view";
import "./shared-chat-showcase.css";

const PHASE_DELAYS = [500, 1_100, 1_700, 2_700, 3_300, 3_900, 4_500, 5_100, 5_700];
const LONG_REPORT_URL = "https://preview.akashic.local/validation/shared-webui/desktop-and-android/message-rendering?fixture=thinking-tool-markdown-gif&viewport=1440x900&mode=offline-production-component";
const PREVIEW_GIF_URL = "data:image/gif;base64,R0lGODlhoABaAPMAABk4bkwweR5Fd0RAf1pimEFLhJtf7aN074J6sW2MtJ+p1rXJ6svg/Mba+ImJugAAACH/C05FVFNDQVBFMi4wAwEAAAAh+QQARgAAACwAAAAAoABaAAAE/5DJSau9OOvNu/9gKI5kaZ5oqq5s675wLM90bd94ru987//AoHBILBqPyKRyyWw6n9CodEqtWq/YrHbL7Xq/4LB4TC6bz+i0es1uu9/wuHxOr6sWUQW+CegDBAR7EgsEAQEEDhIOBBIAggwKBSILAYMLlA2DAQgMAoCcFgqMEwsAAQoUBYmREgQJFQmjRQB6C5ECma2BtqcMi42PCgKTlQyxDAOovgQFkQW9obIMpdASqpCSGMdGtBMNAokNhQIJmQ4FAgMFhb3MzwED0/CHDQiG6oEOhvoSxwSgBBQcSpCgmD5DAlQRYEfOlANAApx5KuZqGgEA66QN6TZhIQNmC//XfXxlz0EDAAOYBbD1bIGAQ8IWEgDkqkCBBeICeOrjSVKDAOIELHAwYOg6mgsPgTvkEEACnPT0vaq4sIFLjUI46mLQTZQhmwp+ndR5ydeAAA5MoRIo6ulKtJouxboEVNTHSjIRufp5zubUtGgBZKK0asCwvY76YQ2idWSpPzsJocPYYGxEewPOPlwJyZDnt4kY/FTcSlSiZy8nfkZ7FBUih8M6rx0AwLFgRYuBaP2GCoC+ynj4FXrlMGUDYadM4WG7l25oSqR96bP16Wa6ttMiMfsLYGFsSrMPl0tsLPcPWpec6UrJTGCBBlK5Viow4PjZSCuvegdJH2ehUDZ9hMjuAJwkwF4C+pwTQAIRMaWPUF4RAN+CIwloFSBH+PGHhBIEdRY8zKTznnydBUAfWpR4Uo49CEl4ED8WbHKSHr2Uwk4g2hFwVkN3GRLQTC9lUpE4G5pnRkV2HIFkkkUsyeSTUEYp5ZRUVmnllVhmqeWWXHbp5ZdghinmmD4cd4ABByiQC5koLGDAm3Aa8AibI7gZZ5xz0vlBA3f2uaaeHyjQ553KAPrBmYPCeYChICR6J6OHOvrmopB2IKikhVaqAZ+S/qlpBnYOmuenGISKJ6l7KnBmmp6i6uqrsMYq66y01mrrrbjmquuuvPbqa68RAAAh+QQARgAAACwAAAAAoABaAINQMHprUJJZO4KKc62bX+2nc++bh7yvns3GsujXyvDl2/wAAAAAAAAAAAAAAAAAAAAE/1DJSau9OOvNu/9gKI5kaZ5oqq5s675wLM90bd94ru987//AoHBILBqPyKRyyWw6n9CodEqtWq/YrHbL7Xq/4LB4TC6bz+i0es1uu9/wuHxOr9ujiQAAEDhIDAEsCQAUCAAJEoYDEwGLFgeBFQB+EwJ+BwJdjQgJB5MKgIKEFJZ/AZkSAAgXkBaflZeoW6oTBwagjQCluLq3CgADALcDAgKOoMUGoxMDjn0Cq4Z/xQGIkMQBq7+UEqWYXQMBBtp/n8QKhtGHvwMJCeGIjQqY8csSrYMKA7eh9PqBnm55QvSqWywvB4gBQxQK3ShECdL90rYrn7hE9hTka9WqjwKPqf86yfJYUIE3WV7yBGqYT1+xYKvW/dpDM0HFjB8T3hrE0yTNPQhaSbi4y+A8lFg8FSLEkmm1VDERmSQXz5e0CgYMCJAqLhJIjfMi5fzoC+rRLgL6dJLX9J8EmL+kwnN5dJUeCwiKTdDqCBDDTJ6ioVJ2IOKpe0ixvBOgq2+kfHkak5Tqko9UrQKUXQA2QeI0PqsgBRhNTquup2fvqF7NurXr17Bjy55Nu7bt27hz697Nu7fv38CD70ZQgEABcsJNICDAvDkB5MlDLHfuHHp0D9SzX5eenbr17RmKd29eADz28c7NdxCPvrz6DdPRf39fAT1z+hziZ5+Pv1B3/v1VQJwccQAGaOCBCCao4IIMNujggxBGKOGEFFZoIW0RAAA7";
const FINAL_CONTENT = [
  "## WebUI 统一结果\n\n",
  "桌面 Web 与 Android WebView 正在使用同一套消息组件。\n\n",
  "- 浅蓝主题一致\n",
  "- thinking 与工具轨迹一致\n",
  "- 平台能力由各自 adapter 提供\n\n",
  "### 边界样例\n\n",
  `[打开超长验证地址](${LONG_REPORT_URL})\n\n`,
  "```tsx\n",
  "<ChatMessageView message={message} />\n",
  "```",
].join("");
const FINAL_ATTACHMENTS: MessageAttachment[] = [{
  id: "shared-preview-gif",
  type: "file",
  mediaType: "image/gif",
  filename: "shared-webui-two-frame.gif",
  url: PREVIEW_GIF_URL,
}];

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
    attachments: phase >= PHASE_DELAYS.length ? FINAL_ATTACHMENTS : undefined,
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
    <main
      className="shared-chat-showcase"
      data-preview-phase={phase}
      data-preview-state={phase < PHASE_DELAYS.length ? "streaming" : "complete"}
    >
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
      report: {
        url: LONG_REPORT_URL,
        checksum: "sha256:cf96bb78ce809f1b6c80b68023733d0cdd35b86f3d4794eb84c6fe14720b5108",
        evidence: [
          "browser-p0-special/06-final-1440x900.png",
          "browser-p0-special/07-tool-parameters-expanded-1440x900.png",
        ],
      },
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
