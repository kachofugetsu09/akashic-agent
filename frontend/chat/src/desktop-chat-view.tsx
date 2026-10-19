import { type TimelineMessage, type TimelineReply, timelineReplyGroups, timelineText, timelineToolResults, timelineInputStarts, timelineSourceKey, timelineSourceRefreshTokens } from "./message-timeline";
import React, { useCallback, useEffect, useLayoutEffect, useMemo, useRef, useState } from "react";
import { useStickToBottomContext } from "use-stick-to-bottom";
import { cycleTheme, useTheme } from "../../theme/src/theme-runtime";
import { MaterialButton } from "../../theme/src/material-react";
import {
  Conversation,
  ConversationContent,
  ConversationEmptyState,
} from "@/components/ai-elements/conversation";
import { DesktopAutoScroll, ComposerPageKeys, NarrowReadingChrome, ReadingReturnButton, SessionReadingMemory, TurnRail } from "./desktop-auto-scroll";
import { ComposerStatsLine } from "./composer-stats-line";
import { ThinkingPlaceholder } from "./thinking-placeholder";
import { DesktopComposer, type ComposerApi } from "./desktop-composer";
import { onWorkbenchQuote } from "./shell-workbench";
import { DesktopConversationMessages, DesktopTimelineMessages, messageDayKey, withoutLeadingOrphanResults } from "./desktop-conversation";
import { MediaGalleryViewer, ReplyActivityView } from "./message-view";
import { MessageSquarePlus } from "lucide-react";
import { CompactNavigation } from "./compact-navigation";
import { NewChatScope } from "./new-chat-scope";
import { DesktopSidebar } from "./desktop-sidebar";
import { useSidebarRail } from "./use-sidebar-rail";
import type { DesktopChatController } from "./use-desktop-chat-controller";
import { PluginUiSlot } from "./plugin-ui-runtime";
import { SelectionQuote } from "./selection-quote";
import { MarqueeText } from "./marquee-text";
import { CopyNotice, useCopyLastCode } from "./copy-last-code";
import { isGeneratingChatStatus } from "./web-chat-status";
import { SteerCards } from "./steer-cards";

interface DesktopChatViewProps {
  controller: DesktopChatController;
}

// 当前会话最后一条用户提问的正文。
// 本轮停下后等这么久，服务端仍没把插话写进对话就退回输入框（正常结束时它会转成新一轮输入）。
const STEER_SETTLE_MS = 4000;

function lastUserInput(messages: readonly TimelineMessage[]): string {
  for (let index = messages.length - 1; index >= 0; index -= 1) {
    const message = messages[index];
    if (message.body.kind === "input" && message.author === "user") return timelineText(message);
  }
  return "";
}

export function DesktopChatView({ controller }: DesktopChatViewProps) {
  const theme = useTheme();
  const rail = useSidebarRail();
  const composerApi = useRef<ComposerApi | null>(null);
  // 空输入框按 ↑ 取回的上一条提问。
  const lastInput = useMemo(() => lastUserInput(controller.timelineMessages), [controller.timelineMessages]);
  // 改一改再问：原文接在已有草稿之后，不覆盖用户正在写的内容。
  const editInput = useCallback((text: string) => composerApi.current?.insertDraft(text, { append: true }), []);
  // 工具区的“引用到对话”：宿主已核对过会话，这里再核对一次，切走后迟到的引用不写进别的会话草稿。
  useEffect(() => onWorkbenchQuote(({ session, text, source }) => {
    if (session !== controller.activeSessionId) return;
    composerApi.current?.insertDraft(`> ${source.replace(/\s+/g, " ")}\n\n${text}`, { append: true });
  }), [controller.activeSessionId]);
  const replyGroups = useMemo(() => timelineReplyGroups(controller.timelineMessages, controller.replyActivities), [controller.timelineMessages, controller.replyActivities]);
  const inputStarts = useMemo(() => timelineInputStarts(controller.timelineMessages), [controller.timelineMessages]);
  const refreshTokens = timelineSourceRefreshTokens(controller.timelineMessages, controller.replyActivities, controller.timelineRefresh);
  const toolResults = useMemo(() => timelineToolResults(controller.timelineMessages), [controller.timelineMessages]);
  const {
    surface, sidebarSessions, activeSessionId, pendingSessionId, chatReady, messages, timelineMessages, replyActivities, replyAvailable, status,
    streamStore, messageElementsRef, copiedMessageId, shellState, stopPending,
    replyTarget, error, steers, steerMessage, withdrawSteer, reclaimSteers,
    canSend, sendProblem, draftKey,
    historyHasMore, historyLoading, historyLoadingOlder, loadOlderMessages, sessionRequested,
    activeSessionDeleted, deletedNotice, deleteSession, restoreSession, dismissDeletedNotice, renameSession,
    activateSession, prefetchSessionTail, startNewChat, handleReplyMessage, handleCopiedMessage,
    reportError, cancelReply, sendMessage, retryTurn, stopTurn, retry,
    navigationGroups,
  } = controller;
  // 改的是这条未发送草稿的归属，而不是换一份草稿：切过去后把原文接到目标位置已有草稿之后。
  const moveDraft = useCallback((switchScope: () => void) => {
    const text = composerApi.current?.takeDraft() ?? "";
    switchScope();
    if (text) window.setTimeout(() => composerApi.current?.insertDraft(text, { append: true }), 0);
  }, []);

  // 用户停止时插话由服务端作为下一轮输入送出；本轮停下一段时间仍没进入对话的才退回输入框。
  const returnSteers = useCallback(() => {
    reclaimSteers(STEER_SETTLE_MS).then((texts) => {
      if (texts.length) composerApi.current?.insertDraft(texts.join("\n\n"), { append: true });
    }).catch((error: unknown) => reportError(error));
  }, [reclaimSteers, reportError]);
  const withdrawToDraft = useCallback((id: string) => {
    withdrawSteer(id).then((text) => {
      if (text !== null) composerApi.current?.insertDraft(text, { append: true });
    }).catch((error: unknown) => reportError(error));
  }, [withdrawSteer, reportError]);
  // 本轮停下后定时核对仍挂着的插话：服务端还暂存（等不到下一步）或重启丢失的才退回输入框。
  const generating = isGeneratingChatStatus(status);
  useEffect(() => {
    if (generating || !steers.length) return;
    const timer = window.setTimeout(returnSteers, STEER_SETTLE_MS);
    return () => window.clearTimeout(timer);
  }, [generating, steers.length, returnSteers]);
  const pagedTimeline = useMemo(() => withoutLeadingOrphanResults(timelineMessages, historyHasMore),
    [historyHasMore, timelineMessages]);
  // 已落定的回复原文，按时间顺序：时间线在前，本地仍在补齐的消息在后（与渲染顺序一致）。
  const replyTexts = useMemo(() => [
    ...timelineMessages.filter((message) => message.body.kind === "output").map(timelineText),
    ...messages.filter((message) => message.role === "assistant" && !message.streaming).map((message) => message.content),
  ], [messages, timelineMessages]);
  const copyNotice = useCopyLastCode(replyTexts, reportError);
  // 选了回复引用就是要开始写：焦点直接进输入框（触屏会弹出键盘），不必再点一次。
  const replyTo = useCallback((reply: TimelineReply) => {
    handleReplyMessage(reply);
    composerApi.current?.focus();
  }, [handleReplyMessage]);
  const activeTitle = sidebarSessions.find((session) => session.active)?.title || "新会话";
  const activeGroup = navigationGroups.active;
  const headingTitle = activeGroup ? `${activeGroup.label} / ${activeTitle}` : activeTitle;
  useDocumentTitle(activeSessionId || activeGroup ? headingTitle : "", isGeneratingChatStatus(status));
  const composerWrapRef = useRef<HTMLDivElement>(null);
  useComposerClearance(composerWrapRef);
  const committed = new Set(timelineMessages.map((message) => message.id));
  const hasMessages = messages.length + timelineMessages.length + replyActivities.length > 0;
  // 读取已有会话期间就按会话版式摆放：输入区停在底部，消息到达时不从屏幕中央跳下来。
  // 刷新或通知入口带着会话打开时，会话要等聊天就绪才开始读取；外壳需要配置、启动中或不可用时让给引导页。
  const shellStatus = shellState?.status ?? null;
  const sessionOpening = historyLoading || (sessionRequested && (shellStatus === null || shellStatus === "ready"));
  const historyFailed = controller.historyFailure !== null && !hasMessages;
  const conversationLayout = hasMessages || sessionOpening || historyFailed;

  return (
    <main className="chat-shell">
      <div className="chat-shell-body" style={rail.style}>
        <DesktopSidebar
            surface={surface} sessions={sidebarSessions}
            activeSessionId={activeSessionId} pendingSessionId={pendingSessionId} chatReady={chatReady}
            themeLabel={theme.label} groups={navigationGroups} navigationPins={controller.navigationPins} onSelectSession={activateSession}
            onPrefetchSession={prefetchSessionTail}
            onCycleTheme={cycleTheme} onNewChat={startNewChat} rail={rail}
            onDeleteSession={deleteSession} deletedNotice={deletedNotice}
            onRestoreSession={(key) => { void restoreSession(key); }} onDismissDeletedNotice={dismissDeletedNotice}
            onRenameSession={renameSession}
          />

        <section className={`chat-main${conversationLayout ? "" : " is-empty"}`}>
        <header className={`conversation-heading${!activeSessionId && !activeGroup ? " is-blank" : ""}`}>
          <CompactNavigation
            surface={surface} sessions={sidebarSessions}
            activeSessionId={activeSessionId} pendingSessionId={pendingSessionId} chatReady={chatReady}
            themeLabel={theme.label} groups={navigationGroups} navigationPins={controller.navigationPins} onSelectSession={activateSession}
            onPrefetchSession={prefetchSessionTail}
            onCycleTheme={cycleTheme} onNewChat={startNewChat}
            onDeleteSession={deleteSession} deletedNotice={deletedNotice}
            onRestoreSession={(key) => { void restoreSession(key); }} onDismissDeletedNotice={dismissDeletedNotice}
            onRenameSession={renameSession}
          />
          {/* 同一父节点下的兄弟 key 必须互不相同；共用会话 ID 会让 React 留下旧标题节点。 */}
          <SessionHeadingTitle key={`title:${activeSessionId}`} heading={headingTitle} value={activeTitle}
            onRename={activeSessionId && !activeSessionDeleted ? (title) => renameSession(activeSessionId, title) : undefined} />
          {/* 插件在标题行右侧的标记；新的工具结果到达时通知插件重新读取会话状态。 */}
          {activeSessionId ? <PluginUiSlot key={`header:${activeSessionId}`} name="header.end" sessionId={activeSessionId}
            refreshToken={Array.from(toolResults.keys()).join("|")} quiet /> : null}
          {/* 窄屏没有常驻侧栏：新会话留在拇指可达的标题行，不必先打开抽屉；继承当前会话所在的分组。 */}
          {conversationLayout ? <button type="button" className="conversation-heading__new"
            aria-label={activeGroup ? `在 ${activeGroup.label} 中新建会话` : "新会话"} data-tip="新会话"
            onClick={() => activeGroup ? navigationGroups.startChat(activeGroup) : startNewChat()}>
            <MessageSquarePlus size={20} strokeWidth={1.75} aria-hidden="true" />
          </button> : null}
        </header>
        <Conversation className="conversation" resize="instant">
          <ConversationContent className={conversationLayout ? "conversation-content" : "conversation-content empty"}>
            {historyFailed ? <div className="history-switch-loading is-failed" role="alert">
              <p>没能读取这段对话的记录{controller.historyFailure ? `：${controller.historyFailure}` : "。"}</p>
              <button type="button" onClick={controller.retryHistory}>重试</button>
            </div>
              : !hasMessages && sessionOpening ? <p className="history-switch-loading" role="status">正在读取消息</p>
              : !hasMessages ? <DesktopEmptyState shellStatus={shellStatus} sendProblem={sendProblem} /> : (
              <MessageRendererErrorBoundary>
                <DesktopHistoryLoader
                  firstMessageId={timelineMessages[0]?.id ?? messages[0]?.id}
                  hasMore={historyHasMore}
                  loading={historyLoadingOlder}
                  onLoadOlder={() => loadOlderMessages().catch(reportError)}
                />
                <DesktopTimelineMessages messages={pagedTimeline} activities={replyActivities} replySynced={replyAvailable !== null} refresh={controller.timelineRefresh} status={status}
                  messageElementsRef={messageElementsRef} copiedMessageId={copiedMessageId}
                  onReply={replyTo} onEditInput={editInput} onRetry={retryTurn} onCopied={handleCopiedMessage} onError={reportError} />
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
          <ComposerPageKeys />
          <ReadingReturnButton generating={isGeneratingChatStatus(status)} />
          {hasMessages ? <SelectionQuote onQuote={editInput} /> : null}
          {hasMessages ? <MediaGalleryViewer /> : null}
          <SessionReadingMemory sessionId={activeSessionId}
            ready={!historyLoading && timelineMessages.at(-1)?.session_id === activeSessionId
              ? `${timelineMessages.length}:${timelineMessages[0]?.id ?? ""}` : ""} />
          <TurnRail revision={`${activeSessionId}:${timelineMessages.length}:${messages.length}:${timelineMessages[0]?.id ?? ""}`} />
        </Conversation>

        <div ref={composerWrapRef} className={`composer-wrap ${!conversationLayout ? "home" : ""}`}>
          {status === "uploading" ? <p className="reply-unavailable" role="status">正在上传附件…</p> : null}
          {activeSessionDeleted ? <p className="reply-unavailable" role="status">
            此会话已删除，内容只读保留。
            <button type="button" className="session-restore-button" onClick={() => { void restoreSession(activeSessionId); }}>恢复会话</button>
          </p> : null}
          {/* 输入框工具位的插件说明为什么现在不能发；处理入口在插件自己的工具里。 */}
          {chatReady && sendProblem ? <p className="chat-send-notice" id="chat-send-reason" role="status">{sendProblem}</p> : null}
          {!conversationLayout && !activeSessionId ? <NewChatScope groups={navigationGroups}
            onStart={(ref) => moveDraft(() => navigationGroups.startChat(ref))} onUngrouped={() => moveDraft(startNewChat)} /> : null}

          <SteerCards items={steers} onWithdraw={withdrawToDraft} />
          <DesktopComposer
            ref={composerApi}
            chatReady={chatReady} canSend={canSend} sendProblem={sendProblem} draftKey={draftKey} autoFocus={!conversationLayout && chatReady} fresh={!conversationLayout} status={status} stopPending={stopPending}
            replyTarget={replyTarget} onCancelReply={cancelReply}
            onSend={sendMessage} onSteer={steerMessage} onStop={stopTurn} lastInput={lastInput}
          />
          <CopyNotice text={copyNotice} />
          <ComposerStatsLine sessionId={activeSessionId} messages={timelineMessages} activities={replyActivities} connected={replyAvailable !== null} status={status} stopping={stopPending} />
          {replyAvailable === false ? <p className="reply-unavailable" role="status">当前未加载回复插件</p> : null}
          {controller.reconnecting ? <p className="reply-unavailable connection-status" role="status">
            <i aria-hidden="true" />连接已断开，正在重新连接…
            <button type="button" onClick={retry}>立即重试</button>
          </p> : null}
          {/* 插件的状态提示；聊天连接断开时由上面的连接行统一说明，插件提示让位。 */}
          {controller.connectionLost ? null : <PluginUiSlot name="chat.notice" sessionId={activeSessionId ?? undefined} quiet />}
          {/* 已有消息（旧记录或先到的实时回复）时读取失败不占阅读区：正文保留，失败与重试放在输入框下的错误行。 */}
          {controller.historyFailure !== null && hasMessages ? <div className="error-line" role="alert">
            <span>没能读取这段对话的完整记录：{controller.historyFailure}</span>
            <MaterialButton variant="danger" onClick={controller.retryHistory}>重试</MaterialButton>
          </div> : null}
          {controller.pageOutdated ? <div className="error-line" role="alert">
            <span>聊天页面已更新，刷新后才能完整显示新的回复。</span>
            <MaterialButton variant="danger" onClick={() => window.location.reload()}>刷新页面</MaterialButton>
          </div> : null}
          {error ? <div className="error-line" role="alert"><span>{error}</span>
            {controller.errorRetry ? <MaterialButton variant="danger" onClick={retry}>重试</MaterialButton> : null}
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

// 输入卡浮在消息区底部，展开（多行、附件、引用、排队、触屏聚焦）时会长高；消息区底部留白只按紧凑态预留，
// 多出来的部分写进 --composer-grown，留白与回到底部按钮一起让开，最新一条回复不会被盖在卡片下面。
function useComposerClearance(wrapRef: React.RefObject<HTMLDivElement | null>): void {
  // 必要 effect：输入卡高度只能在布局后量到。
  useEffect(() => {
    const wrap = wrapRef.current;
    const main = wrap?.closest<HTMLElement>(".chat-main");
    const scroller = main?.querySelector<HTMLElement>(".conversation-scroll");
    if (!wrap || !main || !scroller) return;
    // 1. 记下最近一次滚动时是否停在底部；留白变大时只有贴底的读者需要跟下去，往上翻着读的不动。
    let pinned = true;
    const onScroll = () => { pinned = scroller.scrollHeight - scroller.scrollTop - scroller.clientHeight <= 4; };
    scroller.addEventListener("scroll", onScroll, { passive: true });
    const sync = () => {
      const style = getComputedStyle(main);
      const rem = parseFloat(getComputedStyle(document.documentElement).fontSize) || 16;
      const length = (name: string) => {
        const value = style.getPropertyValue(name).trim();
        return value.endsWith("rem") ? parseFloat(value) * rem : parseFloat(value) || 0;
      };
      // 生成中的统计行本来就在预算里，不重复计入。
      const grown = wrap.offsetHeight - length("--composer-compact-h") - length("--composer-stats-line");
      main.style.setProperty("--composer-grown", `${Math.max(0, Math.round(grown))}px`);
      // 2. 紧凑高度本身也会随聚焦态变化（窄屏），所以无论哪种长高，贴底时都把底部重新露出来。
      if (pinned) scroller.scrollTop = scroller.scrollHeight;
    };
    const observer = new ResizeObserver(sync);
    observer.observe(wrap);
    sync();
    return () => {
      observer.disconnect();
      scroller.removeEventListener("scroll", onScroll);
      main.style.removeProperty("--composer-grown");
    };
  }, [wrapRef]);
}

// 标签页标题跟随当前会话，开着几个标签也分得清；页面在后台时本轮答完，标题前加"●"，回到页面即清除。
function useDocumentTitle(title: string, generating: boolean): void {
  const unseen = useRef(false);
  const wasGenerating = useRef(generating);
  // 必要 effect：document.title 与页面可见性都在 React 之外。
  useEffect(() => {
    if (wasGenerating.current && !generating && document.hidden) unseen.current = true;
    wasGenerating.current = generating;
    const apply = () => {
      if (!document.hidden) unseen.current = false;
      document.title = `${unseen.current ? "● " : ""}${title ? `${title} · ` : ""}Akashic`;
    };
    apply();
    document.addEventListener("visibilitychange", apply);
    return () => document.removeEventListener("visibilitychange", apply);
  }, [title, generating]);
}

/** 时段问候只做一行小字；主标题保持"布置下一件事"的任务口吻。 */
function greetingFor(hour: number): string {
  if (hour < 5) return "夜深了";
  if (hour < 11) return "早上好";
  if (hour < 14) return "中午好";
  if (hour < 18) return "下午好";
  return "晚上好";
}

function DesktopEmptyState({ shellStatus, sendProblem }: { shellStatus: string | null; sendProblem: string }) {
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
    ) : sendProblem ? (
      <div className="home-state__ready"><strong>准备对话</strong><span>下方会说明还差什么；你可以先写下想说的话</span></div>
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
  // 回车提交后紧接着失焦会再触发一次 commit；请求未回时只认第一次。
  const inFlightRef = useRef(false);
  if (!onRename) return <h1 title={heading}><MarqueeText>{heading}</MarqueeText></h1>;
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
    if (cancelledRef.current || inFlightRef.current) return;
    const title = draft.trim();
    if (title === value.trim()) {
      setEditing(false);
      return;
    }
    inFlightRef.current = true;
    try {
      await onRename(title);
      setEditing(false);
    } catch {
      // 错误已由 controller 上报；编辑态保留，用户可再试或按 Esc 放弃。
    } finally {
      inFlightRef.current = false;
    }
  };
  if (!editing) return <h1 title={heading} onDoubleClick={start}><MarqueeText>{heading}</MarqueeText></h1>;
  return <input
    ref={inputRef}
    className="conversation-heading-input"
    value={draft}
    maxLength={200}
    aria-label="重命名会话"
    onChange={(event) => setDraft(event.target.value)}
    onKeyDown={(event) => {
      // 输入法组合中的回车、Esc 属于输入法，不提交也不放弃编辑。
      if (event.nativeEvent.isComposing || event.nativeEvent.keyCode === 229) return;
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
