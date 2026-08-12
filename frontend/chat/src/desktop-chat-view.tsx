import React, { lazy, Suspense } from "react";
import { cycleTheme, useTheme } from "../../theme/src/theme-runtime";
import { MaterialButton } from "../../theme/src/material-react";
import {
  Conversation,
  ConversationContent,
  ConversationEmptyState,
  ConversationScrollButton,
} from "@/components/ai-elements/conversation";
import { DesktopAutoScroll } from "./desktop-auto-scroll";
import { DesktopComposer } from "./desktop-composer";
import { DesktopConversationMessages } from "./desktop-conversation";
import { DesktopMobileNavigation } from "./desktop-mobile-navigation";
import { DesktopSidebar } from "./desktop-sidebar";
import type { DesktopChatController } from "./use-desktop-chat-controller";

const LazyMobilePairingDialog = lazy(() =>
  import("./mobile-pairing-dialog").then(({ MobilePairingDialog }) => ({ default: MobilePairingDialog })),
);
const LazyRuntimeDashboard = lazy(() =>
  import("./runtime-dashboard").then(({ RuntimeDashboard }) => ({ default: RuntimeDashboard })),
);

interface DesktopChatViewProps {
  embeddedShell: boolean;
  embeddedRuntime: boolean;
  controller: DesktopChatController;
}

export function DesktopChatView({ embeddedShell, embeddedRuntime, controller }: DesktopChatViewProps) {
  const theme = useTheme();
  const {
    surface, sidebarSessions, activeSessionId, pendingSessionId, chatReady, messages, status,
    streamStore, messageElementsRef, copiedMessageId, shellState, stopPending, modelState,
    selectedRuntimeId, selectedReasoningEffort, replyTarget, error, mobilePairingOpen,
    activateSession, openRuntime, startNewChat, handleReplyMessage, handleCopiedMessage,
    reportError, handleModelChange, cancelReply, sendMessage, stopTurn, retry,
    setMobilePairingOpen,
  } = controller;
  const openPairing = () => setMobilePairingOpen(true);

  return (
    <main className={`chat-shell ${embeddedRuntime ? "embedded-runtime" : ""}`}>
      {!embeddedRuntime ? <>
        <DesktopSidebar
          embeddedShell={embeddedShell} surface={surface} sessions={sidebarSessions}
          activeSessionId={activeSessionId} pendingSessionId={pendingSessionId} chatReady={chatReady}
          themeLabel={theme.label} onSelectSession={activateSession} onOpenRuntime={openRuntime}
          onCycleTheme={cycleTheme} onOpenPairing={openPairing} onNewChat={startNewChat}
        />
        <DesktopMobileNavigation
          embeddedShell={embeddedShell} surface={surface} sessions={sidebarSessions}
          activeSessionId={activeSessionId} pendingSessionId={pendingSessionId} chatReady={chatReady}
          themeLabel={theme.label} onSelectSession={activateSession} onOpenRuntime={openRuntime}
          onCycleTheme={cycleTheme} onOpenPairing={openPairing} onNewChat={startNewChat}
        />
      </> : null}

      {surface === "runtime" ? (
        <Suspense fallback={<section className="runtime-dashboard" aria-busy="true">正在加载知识与运行…</section>}>
          <LazyRuntimeDashboard />
        </Suspense>
      ) : <section className="chat-main">
        <Conversation className="conversation" resize={status === "streaming" ? "smooth" : "instant"}>
          <ConversationContent className={messages.length ? "conversation-content" : "conversation-content empty"}>
            {messages.length === 0 ? <DesktopEmptyState shellStatus={shellState?.status ?? null} /> : (
              <MessageRendererErrorBoundary>
                <DesktopConversationMessages
                  messages={messages} activeSessionId={activeSessionId} status={status}
                  copiedMessageId={copiedMessageId} streamStore={streamStore}
                  messageElementsRef={messageElementsRef} onReply={handleReplyMessage}
                  onCopied={handleCopiedMessage} onError={reportError}
                />
              </MessageRendererErrorBoundary>
            )}
          </ConversationContent>
          <DesktopAutoScroll messages={messages} status={status} streamStore={streamStore} />
          <ConversationScrollButton className="desktop-scroll-return" />
        </Conversation>

        <div className={`composer-wrap ${messages.length === 0 ? "home" : ""}`}>
          <DesktopComposer
            chatReady={chatReady} status={status} stopPending={stopPending} modelState={modelState}
            selectedRuntimeId={selectedRuntimeId} selectedEffort={selectedReasoningEffort}
            replyTarget={replyTarget} onModelChange={handleModelChange} onCancelReply={cancelReply}
            onSend={sendMessage} onStop={stopTurn}
          />
          {error ? <div className="error-line" role="alert"><span>{error}</span>
            <MaterialButton variant="danger" onClick={retry}>重试</MaterialButton>
          </div> : null}
        </div>
      </section>}
      {mobilePairingOpen ? <Suspense fallback={null}>
        <LazyMobilePairingDialog open onOpenChange={setMobilePairingOpen} />
      </Suspense> : null}
    </main>
  );
}

function DesktopEmptyState({ shellStatus }: { shellStatus: string | null }) {
  return <ConversationEmptyState className="home-state">
    {shellStatus === "needs_setup" ? <div className="model-connection-state">
      <span>首次使用</span><h1>先连接一个模型</h1>
      <p>绑定 Codex、OpenCode 或自己的 API Key 后，就可以在这里直接对话。</p>
      <a href="/settings">连接模型</a>
    </div> : shellStatus === "starting" ? <div className="model-connection-state">
      <span>正在启动</span><h1>模型已保存，Akashic 正在准备对话</h1>
      <p>这个页面会自动恢复，不需要切换端口或刷新浏览器。</p>
      <a href="/settings">查看模型设置</a>
    </div> : shellStatus === null ? <h1>正在连接 Akashic…</h1> : <h1>今天有什么计划?</h1>}
  </ConversationEmptyState>;
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
