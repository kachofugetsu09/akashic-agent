import { AnimatePresence, motion } from "motion/react";
import {
  BatteryMedium,
  Check,
  ChevronDown,
  CircleCheck,
  Menu,
  MoreVertical,
  Pencil,
  Plus,
  RefreshCw,
  SendHorizontal,
  SignalLow,
  Wifi,
  WifiOff,
  Wrench,
} from "lucide-react";
import { FormEvent, useEffect, useState } from "react";
import "./mobile-showcase.css";

interface MobileSessionPreview {
  id: string;
  title: string;
}

const MOBILE_SESSIONS: MobileSessionPreview[] = [
  {
    id: "mobile:7f2a8c1d-3e91-4a74-b981-82a7f1e54009",
    title: "Android 会话设计",
  },
  {
    id: "mobile:15e290af-97ce-49f9-b58b-f5f49dece104",
    title: "Cloudflare Tunnel",
  },
  {
    id: "mobile:893e8af4-fb7a-4bd0-9f05-e4c2ddaf4f8a",
    title: "主动推送协议",
  },
];

const TURN_STAGES = [
  { kind: "thinking", text: "我先确认手机端当前会话与 WebChat 的 session 命名空间。" },
  { kind: "tool", name: "codegraph_explore", text: "定位 session 选择与消息渲染调用链" },
  { kind: "thinking", text: "移动端保持 mobile: 前缀，抽屉只展示本设备的独立会话。" },
] as const;

const CONNECTION_STATES = [
  { key: "ready", label: "连接正常", icon: CircleCheck },
  { key: "degraded", label: "网络不稳 · 正在续传", icon: SignalLow },
  { key: "reconnecting", label: "正在重连", icon: RefreshCw },
  { key: "offline", label: "连接已断开", icon: WifiOff },
] as const;

export function MobileShowcase() {
  const [drawerOpen, setDrawerOpen] = useState(false);
  const [thinkingOpen, setThinkingOpen] = useState(true);
  const [selectedSessionId, setSelectedSessionId] = useState(MOBILE_SESSIONS[0].id);
  const [draft, setDraft] = useState("把手机端 session 选择和 thinking 动效做得像 WebChat");
  const [submittedText, setSubmittedText] = useState(draft);
  const [streamStep, setStreamStep] = useState<number>(TURN_STAGES.length);
  const [streaming, setStreaming] = useState(false);
  const [finalVisible, setFinalVisible] = useState(true);
  const [runId, setRunId] = useState(0);
  const [connectionIndex, setConnectionIndex] = useState(0);
  const connection = CONNECTION_STATES[connectionIndex];
  useEffect(() => {
    if (runId === 0) return;

    const timers = [
      window.setTimeout(() => setStreamStep(1), 420),
      window.setTimeout(() => setStreamStep(2), 980),
      window.setTimeout(() => setStreamStep(3), 1_520),
      window.setTimeout(() => {
        setStreaming(false);
        setFinalVisible(true);
      }, 2_100),
      window.setTimeout(() => setThinkingOpen(false), 3_100),
    ];
    return () => timers.forEach(window.clearTimeout);
  }, [runId]);

  useEffect(() => {
    if (!drawerOpen) return;
    const closeOnEscape = (event: KeyboardEvent) => {
      if (event.key === "Escape") setDrawerOpen(false);
    };
    window.addEventListener("keydown", closeOnEscape);
    return () => window.removeEventListener("keydown", closeOnEscape);
  }, [drawerOpen]);

  const runDemoTurn = (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    const message = draft.trim();
    if (!message) return;

    setSubmittedText(message);
    setDraft("");
    setStreamStep(0);
    setStreaming(true);
    setFinalVisible(false);
    setThinkingOpen(true);
    setRunId((current) => current + 1);
  };

  const selectSession = (sessionId: string) => {
    setSelectedSessionId(sessionId);
    setDrawerOpen(false);
  };

  return (
    <main className="mobile-showcase-page">
      <section className="mobile-showcase-copy" aria-labelledby="mobile-showcase-title">
        <span className="mobile-showcase-eyebrow">AKASHIC · ANDROID</span>
        <h1 id="mobile-showcase-title">会话入口属于导航，<br />thinking 属于过程。</h1>
        <p>页面只为有明确结构或状态含义的内容增加容器，颜色用于区分连接、用户输入与执行过程。</p>
        <div className="mobile-showcase-guide" aria-label="设计说明">
          <span><i className="guide-mark navigation" />左上角展开 mobile session</span>
          <span><i className="guide-mark process" />thinking 与 tool 共用过程时间线</span>
          <span><i className="guide-mark realtime" />连接状态固定出现在标题栏</span>
        </div>
        <p className="mobile-showcase-hint">可点击菜单、thinking 和发送按钮体验交互。</p>
      </section>

      <section className="mobile-device" aria-label="Akashic Android 界面预览">
        <div className="mobile-device-screen">
          <header className="mobile-status-bar" aria-label="设备状态">
            <time>9:41</time>
            <span className="mobile-status-icons"><Wifi size={15} /><BatteryMedium size={17} /></span>
          </header>

          <motion.button
            className="mobile-icon-button mobile-drawer-toggle"
            type="button"
            whileTap={{ scale: 0.96 }}
            aria-label={drawerOpen ? "收起手机会话列表" : "打开手机会话列表"}
            aria-expanded={drawerOpen}
            aria-controls="mobile-session-drawer"
            onClick={() => setDrawerOpen((open) => !open)}
          >
            <Menu size={22} />
          </motion.button>

          <div className="mobile-app-bar">
            <span aria-hidden="true" />
            <button
              className={`mobile-connection-status ${connection.key}`}
              type="button"
              title="点击切换连接状态预览"
              aria-label={`${connection.label}，点击切换状态预览`}
              onClick={() => setConnectionIndex((current) => (current + 1) % CONNECTION_STATES.length)}
            >
              <AnimatePresence initial={false} mode="popLayout">
                <motion.span
                  className="mobile-connection-icon"
                  key={connection.key}
                  initial={{ opacity: 0, scale: 0.25, filter: "blur(4px)" }}
                  animate={{ opacity: 1, scale: 1, filter: "blur(0px)" }}
                  exit={{ opacity: 0, scale: 0.25, filter: "blur(4px)" }}
                  transition={{ type: "spring", duration: 0.3, bounce: 0 }}
                >
                  <connection.icon size={15} />
                </motion.span>
              </AnimatePresence>
              <span>{connection.label}</span>
            </button>
            <motion.button className="mobile-icon-button" type="button" whileTap={{ scale: 0.96 }} aria-label="更多操作">
              <MoreVertical size={21} />
            </motion.button>
          </div>

          <div className="mobile-conversation" aria-live="polite">
            <div className="mobile-day-divider"><span>今天</span></div>
            <div className="mobile-user-row">
              <div className="mobile-user-bubble">{submittedText}</div>
            </div>

            <section className="mobile-assistant-turn" aria-label="Akashic 回复">
              <button
                className="mobile-thinking-trigger"
                type="button"
                aria-expanded={thinkingOpen}
                onClick={() => setThinkingOpen((open) => !open)}
              >
                <span>{streaming ? "正在思考" : "已思考 4s"}</span>
                <ChevronDown className={thinkingOpen ? "open" : ""} size={16} />
              </button>

              <AnimatePresence initial={false}>
                {thinkingOpen ? (
                  <motion.div
                    className="mobile-process-clip"
                    initial={{ opacity: 0, height: 0, filter: "blur(4px)" }}
                    animate={{ opacity: 1, height: "auto", filter: "blur(0px)" }}
                    exit={{ opacity: 0, height: 0, filter: "blur(4px)" }}
                    transition={{ type: "spring", duration: 0.3, bounce: 0 }}
                  >
                    <div className="mobile-process">
                      <div className="mobile-process-line" aria-hidden="true" />
                      <div className="mobile-process-items">
                        {streamStep === 0 ? <StreamingPlaceholder /> : null}
                        {TURN_STAGES.slice(0, streamStep).map((stage, index) => (
                          <motion.div
                            className={`mobile-process-item ${stage.kind} ${streaming && index === streamStep - 1 ? "active" : ""}`}
                            key={`${stage.kind}-${index}`}
                            initial={{ opacity: 0, y: 8, filter: "blur(4px)" }}
                            animate={{ opacity: 1, y: 0, filter: "blur(0px)" }}
                            transition={{ type: "spring", duration: 0.3, bounce: 0 }}
                          >
                            <span className={`mobile-process-node ${stage.kind}`} />
                            {stage.kind === "tool" ? (
                              <div className="mobile-tool-body">
                                <strong><Wrench size={14} />{stage.name}</strong>
                                <span>{stage.text}</span>
                              </div>
                            ) : <p>{stage.text}</p>}
                          </motion.div>
                        ))}
                      </div>
                    </div>
                  </motion.div>
                ) : null}
              </AnimatePresence>

              <AnimatePresence initial={false}>
                {finalVisible ? (
                  <motion.div
                    className="mobile-final-answer"
                    initial={{ opacity: 0, y: 10, filter: "blur(4px)" }}
                    animate={{ opacity: 1, y: 0, filter: "blur(0px)" }}
                    exit={{ opacity: 0, y: -8, filter: "blur(4px)" }}
                    transition={{ type: "spring", duration: 0.3, bounce: 0 }}
                  >
                    <p>可以。手机端会话使用独立的 <code>mobile:</code> 标识，左上角抽屉只展示当前设备的会话。</p>
                    <p>thinking 和 tool 保留 WebChat 的圆点、菱形节点与扳手图标，但按手机宽度重新排版。</p>
                  </motion.div>
                ) : null}
              </AnimatePresence>
            </section>
          </div>

          <form className="mobile-composer" onSubmit={runDemoTurn}>
            <motion.button className="mobile-composer-action" type="button" whileTap={{ scale: 0.96 }} aria-label="添加附件">
              <Plus size={21} />
            </motion.button>
            <input value={draft} onChange={(event) => setDraft(event.target.value)} aria-label="消息" placeholder="给 Akashic 发消息" />
            <motion.button className="mobile-send-button" type="submit" whileTap={{ scale: 0.96 }} aria-label="发送消息" disabled={!draft.trim()}>
              <AnimatePresence initial={false} mode="popLayout">
                <motion.span
                  key={streaming ? "done" : "send"}
                  initial={{ opacity: 0, scale: 0.25, filter: "blur(4px)" }}
                  animate={{ opacity: 1, scale: 1, filter: "blur(0px)" }}
                  exit={{ opacity: 0, scale: 0.25, filter: "blur(4px)" }}
                  transition={{ type: "spring", duration: 0.3, bounce: 0 }}
                >
                  {streaming ? <Check size={19} /> : <SendHorizontal size={19} />}
                </motion.span>
              </AnimatePresence>
            </motion.button>
          </form>

          <AnimatePresence initial={false}>
            {drawerOpen ? (
              <>
                <motion.button
                  className="mobile-drawer-scrim"
                  type="button"
                  aria-label="关闭手机会话列表"
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  exit={{ opacity: 0 }}
                  transition={{ duration: 0.18, ease: "easeOut" }}
                  onClick={() => setDrawerOpen(false)}
                />
                <motion.aside
                  className="mobile-session-drawer"
                  id="mobile-session-drawer"
                  role="dialog"
                  aria-modal="true"
                  aria-label="手机会话列表"
                  initial={{ x: "-100%" }}
                  animate={{ x: 0 }}
                  exit={{ x: "-100%" }}
                  transition={{ type: "spring", duration: 0.3, bounce: 0 }}
                >
                  <div className="mobile-drawer-status" aria-hidden="true">
                    <time>9:41</time>
                    <span><Wifi size={15} /><BatteryMedium size={17} /></span>
                  </div>
                  <div className="mobile-drawer-header">
                    <div><span>手机对话</span><small>仅显示 Android 独立会话</small></div>
                  </div>
                  <div className="mobile-session-list">
                    <div className="mobile-session-group-label">最近</div>
                    {MOBILE_SESSIONS.map((session) => {
                      const selected = session.id === selectedSessionId;
                      return (
                        <motion.button
                          className={`mobile-session-item ${selected ? "selected" : ""}`}
                          type="button"
                          key={session.id}
                          whileTap={{ scale: 0.96 }}
                          aria-current={selected ? "page" : undefined}
                          onClick={() => selectSession(session.id)}
                        >
                          <span className="mobile-session-item-title">{session.title}</span>
                        </motion.button>
                      );
                    })}
                  </div>
                  <div className="mobile-drawer-actions">
                    <motion.button
                      className="mobile-new-chat"
                      type="button"
                      whileTap={{ scale: 0.96 }}
                      onClick={() => selectSession(MOBILE_SESSIONS[0].id)}
                    >
                      <Pencil size={18} />
                      <span>聊天</span>
                    </motion.button>
                  </div>
                </motion.aside>
              </>
            ) : null}
          </AnimatePresence>
        </div>
      </section>
    </main>
  );
}

function StreamingPlaceholder() {
  return (
    <div className="mobile-process-item thinking active">
      <span className="mobile-process-node thinking" />
      <p className="mobile-streaming-text"><span />正在理解请求</p>
    </div>
  );
}
