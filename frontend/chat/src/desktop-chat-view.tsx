import { timelineReplyGroups, timelineToolResults, timelineInputStarts, timelineSourceKey, timelineSourceRefreshTokens } from "./message-timeline";
import React, { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useStickToBottomContext } from "use-stick-to-bottom";
import { cycleTheme, useTheme } from "../../theme/src/theme-runtime";
import { MaterialButton } from "../../theme/src/material-react";
import {
  Conversation,
  ConversationContent,
  ConversationEmptyState,
  ConversationScrollButton,
} from "@/components/ai-elements/conversation";
import { HostBridgeNotice } from "./host-bridge-notice";
import { DesktopAutoScroll } from "./desktop-auto-scroll";
import { ComposerStatsLine } from "./composer-stats-line";
import { ThinkingPlaceholder } from "./thinking-placeholder";
import { DesktopComposer, type ComposerApi } from "./desktop-composer";
import { DesktopConversationMessages, DesktopTimelineMessages, messageDayKey } from "./desktop-conversation";
import { ReplyActivityView } from "./message-view";
import { MessageSquarePlus } from "lucide-react";
import { CompactNavigation } from "./compact-navigation";
import { DesktopSidebar } from "./desktop-sidebar";
import { useSidebarRail } from "./use-sidebar-rail";
import type { DesktopChatController } from "./use-desktop-chat-controller";
import { SessionDirectory } from "./session-directory";

interface DesktopChatViewProps {
  embeddedShell: boolean;
  controller: DesktopChatController;
}

export function DesktopChatView({ embeddedShell, controller }: DesktopChatViewProps) {
  const theme = useTheme();
  const rail = useSidebarRail();
  const composerApi = useRef<ComposerApi | null>(null);
  const replyGroups = useMemo(() => timelineReplyGroups(controller.timelineMessages, controller.replyActivities), [controller.timelineMessages, controller.replyActivities]);
  const inputStarts = useMemo(() => timelineInputStarts(controller.timelineMessages), [controller.timelineMessages]);
  const refreshTokens = timelineSourceRefreshTokens(controller.timelineMessages, controller.replyActivities, controller.timelineRefresh);
  const toolResults = useMemo(() => timelineToolResults(controller.timelineMessages), [controller.timelineMessages]);
  const {
    surface, sidebarSessions, activeSessionId, pendingSessionId, chatReady, messages, timelineMessages, replyActivities, replyAvailable, status,
    streamStore, messageElementsRef, copiedMessageId, shellState, stopPending, modelState,
    selectedRuntimeId, selectedReasoningEffort, replyTarget, error,
    canSend, modelProblem, modelsError, retryModels, draftKey,
    historyHasMore, historyLoading, historyLoadingOlder, loadOlderMessages,
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

        <section className={`chat-main${hasMessages ? "" : " is-empty"}`}>
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
          <SessionHeadingTitle key={activeSessionId} heading={headingTitle} value={activeTitle}
            onRename={activeSessionId && !activeSessionDeleted ? (title) => renameSession(activeSessionId, title) : undefined} />
          {activeSessionId ? <SessionDirectory key={activeSessionId} sessionId={activeSessionId}
            refreshKey={Array.from(toolResults.keys()).join("|")} /> : null}
          {/* 窄屏没有常驻侧栏：新会话留在拇指可达的标题行，不必先打开抽屉。 */}
          {hasMessages ? <button type="button" className="conversation-heading__new" aria-label="新会话" title="新会话"
            onClick={() => startNewChat()}>
            <MessageSquarePlus size={20} strokeWidth={1.75} aria-hidden="true" />
          </button> : null}
        </header>
        <Conversation className="conversation" resize="instant">
          <ConversationContent className={hasMessages ? "conversation-content" : "conversation-content empty"}>
            {!hasMessages ? <DesktopEmptyState shellStatus={shellState?.status ?? null} loadingSession={historyLoading} modelProblem={modelProblem} /> : (
              <MessageRendererErrorBoundary>
                <DesktopHistoryLoader
                  firstMessageId={timelineMessages[0]?.id ?? messages[0]?.id}
                  hasMore={historyHasMore}
                  loading={historyLoadingOlder}
                  onLoadOlder={() => loadOlderMessages().catch(reportError)}
                />
                <DesktopTimelineMessages messages={timelineMessages} activities={replyActivities} refresh={controller.timelineRefresh} status={status}
                  messageElementsRef={messageElementsRef} copiedMessageId={copiedMessageId}
                  onReply={handleReplyMessage} onCopied={handleCopiedMessage} onError={reportError} />
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
                  activity={activity} committed={committed} inputStarts={inputStarts} refreshToken={refreshTokens.get(timelineSourceKey(activity))} processMessages={replyGroups.active.get(activity.handle)} toolResults={toolResults} onError={reportError} />)}
              </MessageRendererErrorBoundary>
            )}
            {status === "submitted" && !replyActivities.some((activity) => activity.active) ? <ThinkingPlaceholder /> : null}
          </ConversationContent>
          <DesktopAutoScroll messages={messages} status={status} streamStore={streamStore}
            timelineMessages={timelineMessages} replyActivities={replyActivities} />
          <ConversationScrollButton className="desktop-scroll-return" />
        </Conversation>

        <div className={`composer-wrap ${!hasMessages ? "home" : ""}`}>
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
            chatReady={chatReady} canSend={canSend} modelProblem={modelProblem} draftKey={draftKey} autoFocus={!hasMessages && chatReady} status={status} stopPending={stopPending} modelState={modelState}
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
  const latest = useRef({ firstMessageId, onLoadOlder });
  latest.current = { firstMessageId, onLoadOlder };

  // 1. 记录首条消息位置，加载完成后把它放回原处，避免视图跳动。
  const load = useCallback(async () => {
    if (busyRef.current) return;
    busyRef.current = true;
    try {
      const scrollElement = scrollRef.current;
      const { firstMessageId: id, onLoadOlder: loadOlder } = latest.current;
      const escapedId = id ? CSS.escape(id) : "";
      const anchor = escapedId
        ? scrollElement?.querySelector<HTMLElement>(`[data-message-id="${escapedId}"]`)
        : null;
      const anchorTop = anchor?.getBoundingClientRect().top;
      await loadOlder();
      if (!scrollElement || anchorTop === undefined || !escapedId) return;
      await new Promise<void>((resolve) => window.requestAnimationFrame(() => resolve()));
      const restoredAnchor = scrollElement.querySelector<HTMLElement>(`[data-message-id="${escapedId}"]`);
      if (restoredAnchor) scrollElement.scrollTop += restoredAnchor.getBoundingClientRect().top - anchorTop;
    } finally {
      busyRef.current = false;
    }
  }, [scrollRef]);

  // 2. 行进入顶部 240px 预读范围时自动加载；只在相交状态变化时触发，失败不会循环重试。
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

  // 3. 已到对话开头时给出安静的终点标记，而不是让入口凭空消失。
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

function DesktopEmptyState({ shellStatus, loadingSession, modelProblem }: { shellStatus: string | null; loadingSession: boolean; modelProblem: string }) {
  return <ConversationEmptyState className="home-state">
    {loadingSession ? <div className="home-state__ready" role="status"><strong>正在读取消息</strong></div> : shellStatus === "needs_setup" ? <div className="model-connection-state">
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
