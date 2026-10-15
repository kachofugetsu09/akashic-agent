import { timelineReplyGroups, timelineToolResults, timelineInputStarts, timelineSourceKey, timelineSourceRefreshTokens } from "./message-timeline";
import React, { useCallback, useEffect, useLayoutEffect, useMemo, useRef, useState } from "react";
import { useStickToBottomContext } from "use-stick-to-bottom";
import { cycleTheme, useTheme } from "../../theme/src/theme-runtime";
import { MaterialButton } from "../../theme/src/material-react";
import {
  Conversation,
  ConversationContent,
  ConversationEmptyState,
} from "@/components/ai-elements/conversation";
import { HostBridgeNotice } from "./host-bridge-notice";
import { DesktopAutoScroll, NarrowReadingChrome, ReadingReturnButton, TurnRail } from "./desktop-auto-scroll";
import { ComposerStatsLine } from "./composer-stats-line";
import { ThinkingPlaceholder } from "./thinking-placeholder";
import { DesktopComposer, type ComposerApi } from "./desktop-composer";
import { JobListAction } from "./job-list-action";
import { DesktopConversationMessages, DesktopTimelineMessages, messageDayKey } from "./desktop-conversation";
import { ReplyActivityView } from "./message-view";
import { MessageSquarePlus } from "lucide-react";
import { CompactNavigation } from "./compact-navigation";
import { DesktopSidebar } from "./desktop-sidebar";
import { useSidebarRail } from "./use-sidebar-rail";
import type { DesktopChatController } from "./use-desktop-chat-controller";
import { SessionDirectory } from "./session-directory";
import { isGeneratingChatStatus } from "./web-chat-status";

interface DesktopChatViewProps {
  embeddedShell: boolean;
  controller: DesktopChatController;
}

export function DesktopChatView({ embeddedShell, controller }: DesktopChatViewProps) {
  const theme = useTheme();
  const rail = useSidebarRail();
  const composerApi = useRef<ComposerApi | null>(null);
  // 改一改再问：原文接在已有草稿之后，不覆盖用户正在写的内容。
  const editInput = useCallback((text: string) => composerApi.current?.insertDraft(text, { append: true }), []);
  const replyGroups = useMemo(() => timelineReplyGroups(controller.timelineMessages, controller.replyActivities), [controller.timelineMessages, controller.replyActivities]);
  const inputStarts = useMemo(() => timelineInputStarts(controller.timelineMessages), [controller.timelineMessages]);
  const refreshTokens = timelineSourceRefreshTokens(controller.timelineMessages, controller.replyActivities, controller.timelineRefresh);
  const toolResults = useMemo(() => timelineToolResults(controller.timelineMessages), [controller.timelineMessages]);
  const {
    surface, sidebarSessions, activeSessionId, pendingSessionId, chatReady, messages, timelineMessages, replyActivities, replyAvailable, status,
    streamStore, messageElementsRef, copiedMessageId, shellState, stopPending, modelState,
    selectedRuntimeId, selectedReasoningEffort, replyTarget, error,
    canSend, modelProblem, modelsError, retryModels, draftKey,
    historyHasMore, historyLoading, historyLoadingOlder, loadOlderMessages, sessionRequested,
    activeSessionDeleted, deletedNotice, deleteSession, restoreSession, dismissDeletedNotice, renameSession,
    activateSession, prefetchSessionTail, startNewChat, handleReplyMessage, handleCopiedMessage,
    reportError, handleModelChange, cancelReply, sendMessage, stopTurn, retry,
    projects, pendingProjects, pendingProjectsError, projectsInstalled, memoryInstalled, activeProject,
    startProjectChat, createProject, continueProject, stopProject, bindDirectory,
  } = controller;
  const sidebarProjects = useMemo(() => projectsInstalled ? {
    items: projects, pending: pendingProjects, pendingError: pendingProjectsError,
    activeProjectId: activeProject?.id ?? "", memoryInstalled,
    onNewChat: startProjectChat, onCreate: createProject, onContinue: continueProject, onStop: stopProject,
    onBindDirectory: bindDirectory,
  } : undefined, [activeProject?.id, bindDirectory, createProject, continueProject, memoryInstalled, pendingProjects,
    pendingProjectsError, projects, projectsInstalled, startProjectChat, stopProject]);
  const activeTitle = sidebarSessions.find((session) => session.active)?.title || "新会话";
  const headingTitle = activeProject ? `${activeProject.name} / ${activeTitle}` : activeTitle;
  const committed = new Set(timelineMessages.map((message) => message.id));
  const hasMessages = messages.length + timelineMessages.length + replyActivities.length > 0;
  // 读取已有会话期间就按会话版式摆放：输入区停在底部，消息到达时不从屏幕中央跳下来。
  // 刷新或通知入口带着会话打开时，会话要等聊天就绪才开始读取；外壳需要配置、启动中或不可用时让给引导页。
  const shellStatus = shellState?.status ?? null;
  const sessionOpening = historyLoading || (sessionRequested && (shellStatus === null || shellStatus === "ready"));
  const conversationLayout = hasMessages || sessionOpening;

  const shellClass = embeddedShell ? "chat-shell is-embedded" : "chat-shell";

  return (
    <main className={shellClass}>
      <div className="chat-shell-body" style={rail.style}>
        <DesktopSidebar
            embeddedShell={embeddedShell} surface={surface} sessions={sidebarSessions}
            activeSessionId={activeSessionId} pendingSessionId={pendingSessionId} chatReady={chatReady}
            themeLabel={theme.label} projects={sidebarProjects} navigationPins={controller.navigationPins} onSelectSession={activateSession}
            onPrefetchSession={prefetchSessionTail}
            onCycleTheme={cycleTheme} onNewChat={startNewChat} rail={rail}
            onDeleteSession={deleteSession} deletedNotice={deletedNotice}
            onRestoreSession={(key) => { void restoreSession(key); }} onDismissDeletedNotice={dismissDeletedNotice}
            onRenameSession={renameSession}
          />

        <section className={`chat-main${conversationLayout ? "" : " is-empty"}`}>
        <header className="conversation-heading">
          <CompactNavigation
            embeddedShell={embeddedShell} surface={surface} sessions={sidebarSessions}
            activeSessionId={activeSessionId} pendingSessionId={pendingSessionId} chatReady={chatReady}
            themeLabel={theme.label} projects={sidebarProjects} navigationPins={controller.navigationPins} onSelectSession={activateSession}
            onPrefetchSession={prefetchSessionTail}
            onCycleTheme={cycleTheme} onNewChat={startNewChat}
            onDeleteSession={deleteSession} deletedNotice={deletedNotice}
            onRestoreSession={(key) => { void restoreSession(key); }} onDismissDeletedNotice={dismissDeletedNotice}
            onRenameSession={renameSession}
          />
          {/* 同一父节点下的兄弟 key 必须互不相同；共用会话 ID 会让 React 留下旧标题节点。 */}
          <SessionHeadingTitle key={`title:${activeSessionId}`} heading={headingTitle} value={activeTitle}
            onRename={activeSessionId && !activeSessionDeleted ? (title) => renameSession(activeSessionId, title) : undefined} />
          {activeSessionId ? <SessionDirectory key={`directory:${activeSessionId}`} sessionId={activeSessionId}
            refreshKey={Array.from(toolResults.keys()).join("|")} /> : null}
          <JobListAction key={`jobs:${activeSessionId}`} jobs={controller.activeSessionJobs} onStop={controller.stopJob} />
          {/* 窄屏没有常驻侧栏：新会话留在拇指可达的标题行，不必先打开抽屉。 */}
          {conversationLayout ? <button type="button" className="conversation-heading__new" aria-label="新会话" title="新会话"
            onClick={() => startNewChat()}>
            <MessageSquarePlus size={20} strokeWidth={1.75} aria-hidden="true" />
          </button> : null}
        </header>
        <Conversation className="conversation" resize="instant">
          <ConversationContent className={conversationLayout ? "conversation-content" : "conversation-content empty"}>
            {!hasMessages && sessionOpening ? <p className="history-switch-loading" role="status">正在读取消息</p>
              : !hasMessages ? <DesktopEmptyState shellStatus={shellStatus} modelProblem={modelProblem} /> : (
              <MessageRendererErrorBoundary>
                <DesktopHistoryLoader
                  firstMessageId={timelineMessages[0]?.id ?? messages[0]?.id}
                  hasMore={historyHasMore}
                  loading={historyLoadingOlder}
                  onLoadOlder={() => loadOlderMessages().catch(reportError)}
                />
                <DesktopTimelineMessages messages={timelineMessages} activities={replyActivities} refresh={controller.timelineRefresh} status={status}
                  messageElementsRef={messageElementsRef} copiedMessageId={copiedMessageId}
                  onReply={handleReplyMessage} onEditInput={editInput} onCopied={handleCopiedMessage} onError={reportError} />
                <DesktopConversationMessages
                  messages={messages} status={status}
                  carryDayKey={timelineMessages.length
                    ? messageDayKey(timelineMessages[timelineMessages.length - 1].timestamp)
                    : undefined}
                  copiedMessageId={copiedMessageId} streamStore={streamStore}
                  messageElementsRef={messageElementsRef}
                  onCopied={handleCopiedMessage} onError={reportError}
                />
                {replyActivities.map((activity) => <ReplyActivityView key={activity.handle}
                  activity={activity} committed={committed} inputStarts={inputStarts} refreshToken={refreshTokens.get(timelineSourceKey(activity))} processMessages={replyGroups.active.get(activity.handle)} toolResults={toolResults} onError={reportError} onStop={stopTurn} />)}
              </MessageRendererErrorBoundary>
            )}
            {status === "submitted" && !replyActivities.some((activity) => activity.active) ? <ThinkingPlaceholder /> : null}
          </ConversationContent>
          <DesktopAutoScroll messages={messages} status={status} streamStore={streamStore}
            timelineMessages={timelineMessages} replyActivities={replyActivities} />
          <NarrowReadingChrome />
          <ReadingReturnButton generating={isGeneratingChatStatus(status)} />
          <TurnRail revision={`${activeSessionId}:${timelineMessages.length}:${messages.length}:${timelineMessages[0]?.id ?? ""}`} />
        </Conversation>

        <div className={`composer-wrap ${!conversationLayout ? "home" : ""}`}>
          {status === "uploading" ? <p className="reply-unavailable" role="status">正在上传附件…</p> : null}
          {activeSessionDeleted ? <p className="reply-unavailable" role="status">
            此会话已删除，内容只读保留。
            <button type="button" className="session-restore-button" onClick={() => { void restoreSession(activeSessionId); }}>恢复会话</button>
          </p> : null}
          {chatReady && (modelProblem || modelsError) ? <div className="chat-model-notice" role="status">
            {modelProblem ? <p id="chat-model-reason">{modelProblem}</p> : null}
            {modelsError ? <p>{modelsError}</p> : null}
            <div>
              {modelProblem ? <a href="/#models" target="_blank" rel="noopener">打开模型设置</a> : null}
              {modelsError ? <button type="button" onClick={retryModels}>重试加载模型列表</button> : null}
            </div>
          </div> : null}
          <DesktopComposer
            ref={composerApi}
            chatReady={chatReady} canSend={canSend} modelProblem={modelProblem} draftKey={draftKey} autoFocus={!conversationLayout && chatReady} fresh={!conversationLayout} status={status} stopPending={stopPending} modelState={modelState}
            selectedRuntimeId={selectedRuntimeId} selectedEffort={selectedReasoningEffort}
            replyTarget={replyTarget} onModelChange={handleModelChange} onCancelReply={cancelReply}
            onSend={sendMessage} onStop={stopTurn}
          />
          <ComposerStatsLine messages={timelineMessages} activities={replyActivities} connected={replyAvailable !== null} />
          {replyAvailable === false ? <p className="reply-unavailable" role="status">当前未加载回复插件</p> : null}
          <HostBridgeNotice />
          {error ? <div className="error-line" role="alert"><span>{error}</span>
            <MaterialButton variant="danger" onClick={retry}>重试</MaterialButton>
          </div> : null}
        </div>
      </section>
      </div>
    </main>
  );
}

// 滚动到顶部附近时自动补载更早消息；文字按钮保留给键盘与不触发滚动的场景。
// 隐藏行（如被归组的工具结果）没有 DOM，取第一条实际渲染的消息作锚。
function pinFirstMessage(scroller: HTMLElement): { id: string; top: number } | null {
  const anchor = scroller.querySelector<HTMLElement>("[data-message-id]");
  const id = anchor?.dataset.messageId;
  return anchor && id ? { id, top: anchor.getBoundingClientRect().top - scroller.getBoundingClientRect().top } : null;
}

function DesktopHistoryLoader({
  firstMessageId,
  hasMore,
  loading,
  onLoadOlder,
}: {
  firstMessageId: string | undefined;
  hasMore: boolean;
  loading: boolean;
  onLoadOlder: () => Promise<void>;
}) {
  const { scrollRef } = useStickToBottomContext();
  const rowRef = useRef<HTMLDivElement>(null);
  const busyRef = useRef(false);
  const latest = useRef({ onLoadOlder, loading });
  latest.current = { onLoadOlder, loading };
  // 首条消息在滚动区里的最新位置；请求期间用户继续滚动也随之更新。
  const pinRef = useRef<{ id: string; top: number } | null>(null);

  const load = useCallback(async () => {
    if (busyRef.current) return;
    busyRef.current = true;
    const scroller = scrollRef.current;
    if (scroller) pinRef.current = pinFirstMessage(scroller);
    try {
      await latest.current.onLoadOlder();
    } finally {
      busyRef.current = false;
    }
  }, [scrollRef]);

  // 1. 请求期间每次滚动都更新原首条消息离滚动区顶部的距离，而不是只在请求发出时记一次。
  //    没有请求时不读：滚动事件里同步读几何会逐次强制布局，长会话滚动时是主要的脚本开销。
  useEffect(() => {
    const scroller = scrollRef.current;
    if (!scroller || !firstMessageId) return;
    const record = () => {
      if (busyRef.current || latest.current.loading) pinRef.current = pinFirstMessage(scroller);
    };
    scroller.addEventListener("scroll", record, { passive: true });
    return () => scroller.removeEventListener("scroll", record);
  }, [firstMessageId, scrollRef]);

  // 2. 更早的消息插入后、浏览器绘制前，把原首条放回最后一次滚动时的位置；原生滚动锚定已生效时差值为 0。
  useLayoutEffect(() => {
    const scroller = scrollRef.current;
    const pin = pinRef.current;
    if (!scroller || !pin) return;
    const anchor = scroller.querySelector<HTMLElement>(`[data-message-id="${CSS.escape(pin.id)}"]`);
    if (anchor && anchor !== scroller.querySelector("[data-message-id]")) scroller.scrollTop += anchor.getBoundingClientRect().top - scroller.getBoundingClientRect().top - pin.top;
  }, [firstMessageId, scrollRef]);

  // 3. 行进入顶部 240px 预读范围时自动加载；只在相交状态变化时触发，失败不会循环重试。
  useEffect(() => {
    const row = rowRef.current;
    const root = scrollRef.current;
    if (!hasMore || !row || !root || typeof IntersectionObserver === "undefined") return;
    const observer = new IntersectionObserver((entries) => {
      if (entries.some((entry) => entry.isIntersecting)) void load();
    }, { root, rootMargin: "240px 0px 0px 0px" });
    observer.observe(row);
    return () => observer.disconnect();
  }, [hasMore, load, scrollRef]);

  // 4. 已到对话开头时给出安静的终点标记，而不是让入口凭空消失。
  if (!hasMore) return <p className="desktop-history-start">对话开头</p>;
  return <div ref={rowRef} className="desktop-history-loader" data-loading={loading || undefined}>
    {loading
      ? <span role="status"><i className="desktop-history-loader__spinner" aria-hidden="true" />正在加载更早消息</span>
      : <button type="button" onClick={() => void load()}>↑ 更早的消息</button>}
  </div>;
}

/** 时段问候只做一行小字；主标题保持"布置下一件事"的任务口吻。 */
function greetingFor(hour: number): string {
  if (hour < 5) return "夜深了";
  if (hour < 11) return "早上好";
  if (hour < 14) return "中午好";
  if (hour < 18) return "下午好";
  return "晚上好";
}

function DesktopEmptyState({ shellStatus, modelProblem }: { shellStatus: string | null; modelProblem: string }) {
  return <ConversationEmptyState className="home-state">
    {shellStatus === "needs_setup" ? <div className="model-connection-state">
      <span>对话尚未就绪</span><h1>请完成所需配置</h1>
      <p>绑定 Codex、OpenCode 或自己的 API Key 后，就可以在这里直接对话。</p>
      <a href="/#models">连接模型</a>
    </div> : shellStatus === "unavailable" ? <div className="model-connection-state">
      <span>对话暂不可用</span><h1>尚未连接到聊天服务</h1>
      <p>服务可能正在启动、已停用或启动失败。连接恢复后，这个页面会自动更新。</p>
      <a href="/">查看管理面板</a>
    </div> : shellStatus === "starting" ? <div className="model-connection-state">
      <span>正在启动</span><h1>Akashic 正在准备对话</h1>
      <p>这个页面会自动恢复，不需要切换端口或刷新浏览器。</p>
      <a href="/#models">查看模型设置</a>
    </div> : shellStatus === null ? (
      <div className="home-state__ready">
        <strong>正在连接</strong>
        <span>稍等，工作区马上就绪</span>
      </div>
    ) : modelProblem ? (
      <div className="home-state__ready"><strong>准备对话</strong><span>下方会说明模型状态；你可以先写下想说的话</span></div>
    ) : (
      <div className="home-hero">
        <p className="home-hero__greeting">{greetingFor(new Date().getHours())}</p>
        <h1 className="home-hero__title">布置下一件事</h1>
      </div>
    )}
  </ConversationEmptyState>;
}

/** 会话标题：双击进入行内编辑；只编辑会话名，项目前缀是展示拼接、不参与提交。 */
function SessionHeadingTitle({ heading, value, onRename }: {
  heading: string;
  value: string;
  onRename?: (title: string) => Promise<void>;
}) {
  const [editing, setEditing] = useState(false);
  const [draft, setDraft] = useState("");
  const inputRef = useRef<HTMLInputElement>(null);
  const cancelledRef = useRef(false);
  if (!onRename) return <h1 title={heading}>{heading}</h1>;
  const start = () => {
    cancelledRef.current = false;
    setDraft(value);
    setEditing(true);
    window.setTimeout(() => {
      inputRef.current?.focus();
      inputRef.current?.select();
    }, 0);
  };
  const commit = async () => {
    if (cancelledRef.current) return;
    const title = draft.trim();
    if (title === value.trim()) {
      setEditing(false);
      return;
    }
    try {
      await onRename(title);
      setEditing(false);
    } catch {
      // 错误已由 controller 上报；编辑态保留，用户可再试或按 Esc 放弃。
    }
  };
  if (!editing) return <h1 title={heading} onDoubleClick={start}>{heading}</h1>;
  return <input
    ref={inputRef}
    className="conversation-heading-input"
    value={draft}
    maxLength={200}
    aria-label="重命名会话"
    onChange={(event) => setDraft(event.target.value)}
    onKeyDown={(event) => {
      if (event.key === "Enter") {
        event.preventDefault();
        void commit();
      } else if (event.key === "Escape") {
        event.preventDefault();
        cancelledRef.current = true;
        setEditing(false);
      }
    }}
    onBlur={() => void commit()}
  />;
}

class MessageRendererErrorBoundary extends React.Component<
  { children: React.ReactNode },
  { error: Error | null }
> {
  state: { error: Error | null } = { error: null };

  static getDerivedStateFromError(error: Error) {
    return { error };
  }

  componentDidCatch(error: Error, info: React.ErrorInfo) {
    console.error("消息渲染器加载失败", error, info.componentStack);
  }

  render() {
    if (this.state.error) {
      return <div className="message-row message-renderer-error" role="alert">
        <span>消息渲染器加载失败</span>
        <button type="button" onClick={() => window.location.reload()}>重新加载页面</button>
      </div>;
    }
    return this.props.children;
  }
}
