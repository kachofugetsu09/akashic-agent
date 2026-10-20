import { PluginUiSlot, pluginSlotAccepts, usePluginUiCatalogVersion } from "./plugin-ui-runtime";
import { ThinkingPlaceholder } from "./thinking-placeholder";
import { MentionText } from "./composer-suggest";
import { usePacedText } from "./stream-pace";
import { SentAttachments, type MediaKind, type SentAttachmentItem } from "./attachment-card";
import {
  Message,
  MessageContent,
} from "@/components/ai-elements/message";
import { detectMessageRenderingFeatures, messageNeedsMarkdown, toolDataText } from "@/message-rendering-policy";
import {
  Reasoning,
  ReasoningTrigger,
  useReasoning,
} from "@/components/ai-elements/reasoning";
import { CollapsibleContent } from "@/components/ui/collapsible";
import {
  Dialog,
  DialogClose,
  DialogContent,
  DialogTitle,
} from "@/components/ui/dialog";
import { Brain, Check, ChevronDown, ChevronLeft, ChevronRight, Copy, Download, ExternalLink, Gauge, Wrench, X, ZoomIn, ZoomOut } from "lucide-react";
import { useStickToBottomContext } from "use-stick-to-bottom";
import { MarqueeText } from "./marquee-text";
import {
  Fragment,
  lazy,
  memo,
  Suspense,
  type MouseEvent as ReactMouseEvent,
  type PointerEvent as ReactPointerEvent,
  type ReactNode,
  useEffect,
  useLayoutEffect,
  useRef,
  useMemo,
  useState,
} from "react";
import type {
  AgentBlock,
  ChatMessage,
  MessageAttachment,
  ToolBlock,
} from "./chat-message";
import type { ModelAttempt, ReplyActivity, TimelineAttachment, TimelineMessage, TimelinePart } from "./message-timeline";
import { timelineReply, timelineToolOutput, historyTranscript, isTimelinePartVisible, needsBeforeReasoningFallback } from "./message-timeline";
import { MessageReplyReference } from "./message-actions";
import { StaticMessageResponse } from "./static-message-response";
import { Disclosure, formatToolDuration, ToolResultContent, useStickyDisclosure } from "./tool-result-view";

const LazyMessageResponse = lazy(() =>
  import("@/components/ai-elements/message-response").then(({ MessageResponse }) => ({ default: MessageResponse })),
);

// 富文本渲染器是懒加载分块(950KB / 307KB gzip):首个含代码或数学的消息出现时才取,
// 慢链路上打开或切入富会话要在关键路径上付这笔传输与求值。挂载后空闲预热一次;
// 省流模式不预热；其他会话会在空闲时下载这份分块。
export function warmMessageRenderer(): void {
  const connection = (navigator as { connection?: { saveData?: boolean } }).connection;
  if (connection?.saveData) return;
  void import("@/components/ai-elements/message-response").catch((error: unknown) => {
    console.warn("Message renderer preload failed; rendering will report any loading failure.", error);
  });
}

const MessageBody = memo(function MessageBody({
  content,
  streaming,
  paced = false,
  deferRichContent,
  onError,
}: {
  content: string;
  streaming: boolean;
  /** 外层已按节奏放字：渲染器只做贴身的淡入，不再叠一层延迟。 */
  paced?: boolean;
  deferRichContent: boolean;
  onError?: (error: unknown) => void;
}) {
  if (!streaming && !messageNeedsMarkdown(content)) {
    return <p className="plain-message-response">{content}</p>;
  }
  if (!streaming && deferRichContent) {
    const features = detectMessageRenderingFeatures(content);
    if (features.math || features.code) {
      return (
        <Suspense fallback={<p className="plain-message-response">{content}</p>}>
          <LazyMessageResponse isAnimating={false}>{content}</LazyMessageResponse>
        </Suspense>
      );
    }
    return <StaticMessageResponse onError={onError}>{content}</StaticMessageResponse>;
  }
  return (
    <Suspense fallback={<p className="plain-message-response">{content}</p>}>
      <LazyMessageResponse isAnimating={streaming} paced={paced}>{content}</LazyMessageResponse>
    </Suspense>
  );
});

/** 草稿复用聊天的等待和思考组件，提交后由相同样式的历史行接替。 */
export function ReplyActivityView({ activity, committed, onError, onStop, processMessages = [], toolResults = new Map(), inputStarts = new Map(), refreshToken }: {
  activity: ReplyActivity;
  onStop?: () => void;
  inputStarts?: ReadonlyMap<string, number>;
  refreshToken?: string;
  committed: ReadonlySet<string>;
  processMessages?: TimelineMessage[];
  toolResults?: ReadonlyMap<string, TimelineMessage>;
  onError?: (error: unknown) => void;
}) {
  // 展示保留最后草稿直到同 ID 落库；provider 退出预览 scope 不让正文倒退。
  const [lastPreview, setLastPreview] = useState(activity.preview);
  if (activity.preview && activity.preview !== lastPreview) setLastPreview(activity.preview);
  const preview = activity.preview ?? (activity.active ? lastPreview : null);
  const draft = preview && !committed.has(preview.message_id) ? preview : null;
  if (preview && !draft && !processMessages.length) return null;
  const flow = timelineFlow(processMessages, undefined, toolResults);
  const text = draft?.text ?? "";
  // 重试中的整次尝试都显示一行进度；退避等待仍由状态卡负责。
  const attemptLine = activity.active && draft?.attempt && !draft.retry_status
    ? <AttemptStatus attempt={draft.attempt} onStop={onStop} /> : null;
  const beforeReasoning = (message: TimelineMessage) => needsBeforeReasoningFallback(message, inputStarts)
    ? <PluginUiSlot name="turn.before_reasoning" sessionId={message.session_id} messageId={message.id}
      block={{ source: message.source }} refreshToken={refreshToken} /> : null;
  return <div className="message-row agent-row reply-activity" data-reply-handle={activity.handle}
    data-preview-message-id={draft?.message_id} aria-busy={activity.active}>
    <div className="agent-content">
      <TimelineProcess flow={flow} draftThinking={draft?.thinking} draftThinkingActive={activity.active && !text}
        draftLead={attemptLine} beforeReasoning={beforeReasoning} />
      {activity.active && draft?.retry_status ? <RetryStatus text={draft.retry_status} attempt={draft.attempt} onStop={onStop} /> : null}
      {activity.active && !draft?.retry_status && !text && !draft?.thinking && !flow.length ? <ThinkingPlaceholder /> : null}
      {text ? <PacedMessageBody content={text} streaming={Boolean(draft?.text) && activity.active} onError={onError} />
        : activity.active ? <RichMarkdownWarmup /> : null}
    </div>
  </div>;
}

/** 正文出字前先把富文本渲染器加载好：否则首段先以纯文本占位显示，渲染器就绪后又从头重打一遍。 */
function RichMarkdownWarmup() {
  return <div hidden><Suspense fallback={null}><LazyMessageResponse>{""}</LazyMessageResponse></Suspense></div>;
}

/** 草稿正文按到达节奏放字，结束时由 usePacedText 一次刷出剩余。 */
function PacedMessageBody({ content, streaming, onError }: { content: string; streaming: boolean; onError?: (error: unknown) => void }) {
  const paced = usePacedText(content, streaming);
  return <MessageBody content={paced} streaming={streaming} paced deferRichContent onError={onError} />;
}

/** 重试与压缩进度共用一张状态卡：首行是状态，其余是原始原因，停止就在旁边。 */
function RetryStatus({ text, attempt, onStop }: { text: string; attempt?: ModelAttempt | null; onStop?: () => void }) {
  const [serverHeadline, ...detail] = text.split("\n");
  const now = useClock(attempt ? 1000 : null);
  // 退避等待按服务端允许时间自己倒数，断线重连后也不显示过期的秒数。
  const remaining = attempt?.next_attempt_at ? Math.max(0, Math.ceil(attempt.next_attempt_at - now / 1000)) : null;
  const headline = attempt && remaining !== null
    ? `模型暂时不可用，${remaining ? `约 ${remaining} 秒后` : "即将"}进行第 ${attemptLabel(attempt)} 次尝试`
    : serverHeadline;
  return <div className="retry-status" role="status">
    <div className="retry-status__text">
      <strong>{headline}</strong>
      {attempt ? <span>{elapsedLabel(now / 1000 - attempt.failing_since)} · 上次：{attempt.last_failure}</span> : null}
      {detail.length ? <code>{detail.join("\n")}</code> : null}
    </div>
    {onStop ? <button type="button" onClick={onStop}>停止</button> : null}
  </div>;
}

/** 重试中的尝试：草稿上方一行小字，持续到这次尝试结束；原始错误折叠起来。 */
function AttemptStatus({ attempt, onStop }: { attempt: ModelAttempt; onStop?: () => void }) {
  const now = useClock(15000);
  return <div className="attempt-status" role="status">
    <details>
      <summary>上游不稳定 · 第 {attemptLabel(attempt)} 次尝试 · {elapsedLabel(now / 1000 - attempt.failing_since)} · 上次：{attempt.last_failure}</summary>
      {attempt.detail ? <code>{attempt.detail}</code> : null}
    </details>
    {onStop ? <button type="button" onClick={onStop}>停止</button> : null}
  </div>;
}

function attemptLabel(attempt: ModelAttempt): string {
  return attempt.limit === null ? String(attempt.attempt) : `${attempt.attempt}/${attempt.limit}`;
}

// 面向用户的持续时长只精确到分钟，避免每秒改写状态文字。
function elapsedLabel(seconds: number): string {
  const minutes = Math.floor(Math.max(0, seconds) / 60);
  if (minutes < 1) return "已持续不到 1 分钟";
  if (minutes < 60) return `已持续 ${minutes} 分钟`;
  return `已持续 ${Math.floor(minutes / 60)} 小时${minutes % 60 ? ` ${minutes % 60} 分钟` : ""}`;
}

// 按给定间隔刷新的当前时间；间隔为 null 时不挂定时器。
function useClock(intervalMs: number | null): number {
  const [now, setNow] = useState(() => Date.now());
  useEffect(() => {
    if (intervalMs === null) return;
    const timer = window.setInterval(() => setNow(Date.now()), intervalMs);
    return () => window.clearInterval(timer);
  }, [intervalMs]);
  return now;
}

export function ChatMessageView({
  message,
  leadingContent,
  attachmentContent,
  processStartContent,
  beforeProcessBlock,
  answerEndContent,
  onCopyToolDetail,
  onError,
  deferRichContent = false,
  waitingForResponse = false,
}: {
  message: ChatMessage;
  waitingForResponse?: boolean;
  leadingContent?: ReactNode;
  attachmentContent?: ReactNode;
  processStartContent?: ReactNode;
  beforeProcessBlock?: (block: AgentBlock, index: number) => ReactNode;
  answerEndContent?: ReactNode;
  onCopyToolDetail?: (text: string) => void;
  onError?: (error: unknown) => void;
  deferRichContent?: boolean;
}) {
  const attachments = attachmentContent !== undefined
    ? attachmentContent
    : message.attachments?.length
      ? <MessageAttachments attachments={message.attachments} pending={message.delivery === "sending"} />
      : null;
  if (message.role === "user") {
    // 附件在气泡上方、气泡外（Desktop 同款）：气泡只包文字，只发附件时不留一个空气泡。
    return (
      <Message from="user" className="message-row user-row">
        {attachments}
        {leadingContent || message.content ? <MessageContent className="user-bubble">
          {leadingContent}
          {message.content ? (
            <UserTextClamp enabled>
              {message.streaming !== true && !messageNeedsMarkdown(message.content)
                ? <UserPlainText text={message.content} />
                : <MessageBody
                  content={message.content}
                  streaming={message.streaming === true}
                  deferRichContent={deferRichContent}
                  onError={onError}
                />}
            </UserTextClamp>
          ) : null}
        </MessageContent> : null}
      </Message>
    );
  }

  return (
    <Message from="assistant" className="message-row agent-row">
      <MessageContent className="agent-content">
        {leadingContent}
        {message.blocks.length ? (
          <ProcessTrace
            blocks={message.blocks}
            streaming={message.streaming === true}
            interrupted={message.interrupted === true}
            durationMs={message.durationMs}
            startContent={processStartContent}
            beforeBlock={beforeProcessBlock}
            onCopyToolDetail={onCopyToolDetail}
          />
        ) : null}
        {waitingForResponse && message.streaming && !message.content && message.blocks.length === 0 ? <ThinkingPlaceholder /> : null}
        {attachments}
        {message.content ? (
          <MessageBody
            content={message.content}
            streaming={message.streaming === true}
            deferRichContent={deferRichContent}
            onError={onError}
          />
        ) : null}
        {answerEndContent}
      </MessageContent>
    </Message>
  );
}

type FlowItem =
  | { kind: "thinking"; key: string; origin: TimelineMessage; index: number; part: TimelinePart; content: string }
  | { kind: "tool"; key: string; origin: TimelineMessage; index: number; part: TimelinePart; block: ToolBlock }
  | { kind: "narration"; key: string; origin: TimelineMessage; index: number; part: TimelinePart; text: string }
  | { kind: "notice"; key: string; origin: TimelineMessage; index: number; part: TimelinePart; text: string }
  | { kind: "part"; key: string; origin: TimelineMessage; index: number; part: TimelinePart };

/** 组内全部成员按 part 顺序展开成独立流项：思考、工具、中途正文、末条正文各归原位。 */
function timelineFlow(messages: TimelineMessage[], finalId: string | undefined, toolResults: ReadonlyMap<string, TimelineMessage>): FlowItem[] {
  return messages.flatMap((origin) => origin.body.kind !== "output" ? [] :
    origin.body.parts.map((part, index) => ({ part, index }))
    // 上下文提示发生在生成之前，排在思考之前；思考再排在正文之前。
    .sort((left, right) => flowRank(right.part) - flowRank(left.part))
    .flatMap(({ part, index }): FlowItem[] => {
      const key = `${origin.id}:${index}`;
      if ("display" in part) {
        return part.display === "thinking" && part.text
          ? [{ kind: "thinking", key, origin, index, part, content: part.text }] : [];
      }
      if ("archive" in part) {
        return part.kind === "history.transcript"
          ? historyBlocks(part.archive).flatMap((block, blockIndex): FlowItem[] => block.kind === "thinking"
            ? [{ kind: "thinking", key: `${key}:h${blockIndex}`, origin, index, part, content: block.content }]
            : [{ kind: "tool", key: `${key}:h${blockIndex}`, origin, index, part, block }])
          : [];
      }
      if (part.kind === "tool_call") {
        const result = toolResults.get(key);
        const outcome = result?.body.kind === "tool_result" ? result.body.outcome : null;
        return [{ kind: "tool", key, origin, index, part, block: {
          kind: "tool", callId: key, name: part.name, owner: part.owner, input: part.arguments,
          output: result ? timelineToolOutput(result) : undefined,
          attachments: toolImages(result),
          status: outcome === null ? "input-available" : outcome === "success" ? "output-available" : "output-error",
          errorText: outcome && outcome !== "success" ? outcomeLabels[outcome] : undefined } }];
      }
      if (part.kind === "context.notice") return [{ kind: "notice", key, origin, index, part, text: part.value.text }];
      if (origin.id === finalId) {
        return isTimelinePartVisible(part) ? [{ kind: "part", key, origin, index, part }] : [];
      }
      if (part.kind === "text" && isTimelinePartVisible(part)) {
        return [{ kind: "narration", key, origin, index, part, text: part.value }];
      }
      return [];
    }));
}

// 只有图片观察进调用卡片；文件附件仍作为回复插图展示，不在卡片里重复。
function toolImages(result: TimelineMessage | undefined): SentAttachmentItem[] | undefined {
  const images = result?.attachments.filter((ref) => ref.kind === "image") ?? [];
  return images.length ? images.map(timelineAttachmentItem) : undefined;
}

function flowRank(part: TimelinePart): number {
  if ("display" in part) return part.display === "thinking" ? 1 : 0;
  return part.kind === "context.notice" ? 2 : 0;
}

/** 上下文提示是本轮请求的诊断，与思考行同级：收起时一行摘要，展开看全文。 */
function ContextNoticeRow({ text }: { text: string }) {
  return <details className="context-notice">
    <summary className="process-trigger">
      <Gauge className="process-trigger-icon" size={13} aria-hidden="true" />
      <span className="process-trigger-summary">{text}</span>
      <ChevronDown className="process-chevron" size={13} aria-hidden="true" />
    </summary>
    <p className="process-content process-text" role="status">{text}</p>
  </details>;
}

interface FlowPartViewProps {
  attachment: (id: string) => ReactNode;
  lookupMessage: (id: string) => TimelineMessage | undefined;
  onNavigate: (id: string, partIndex?: number) => void;
  onError?: (error: unknown) => void;
  canLoadReferences: boolean;
}

function FlowItemView({ item, onCopyToolDetail, partView }: {
  item: FlowItem;
  onCopyToolDetail?: (text: string) => void;
  partView?: FlowPartViewProps;
}) {
  const content = item.kind === "thinking" ? <ThinkingRow content={item.content} streaming={false} />
    : item.kind === "tool" ? <ToolStep block={item.block} active={item.block.status === "input-available"} onCopyDetail={onCopyToolDetail} />
    : item.kind === "notice" ? <ContextNoticeRow text={item.text} />
    : item.kind === "narration" ? <div className="timeline-narration">
      <MessageBody content={item.text} streaming={false} deferRichContent onError={partView?.onError} />
    </div>
    : partView ? <TimelinePartView part={item.part} {...partView} /> : null;
  return (
    <div className={`flow-item flow-${item.kind}`} data-process-message-id={item.origin.id}
      data-part-index={item.index} tabIndex={-1}>
      {content}
    </div>
  );
}

/** 连续 ≥3 个已完成工具收成一行摘要；运行中的工具不参与折叠，保持逐行可见。 */
function groupToolRuns<T>(items: T[], settledTool: (item: T) => ToolBlock | null): (T | { members: T[]; durationMs?: number })[] {
  const result: (T | { members: T[]; durationMs?: number })[] = [];
  let run: T[] = [];
  let runMs = 0;
  let runCounted = false;
  const flush = () => {
    if (run.length >= 3) result.push({ members: run, durationMs: runCounted ? runMs : undefined });
    else result.push(...run);
    run = [];
    runMs = 0;
    runCounted = false;
  };
  for (const item of items) {
    const block = settledTool(item);
    if (block === null) { flush(); result.push(item); continue; }
    run.push(item);
    if (block.durationMs !== undefined) { runMs += block.durationMs; runCounted = true; }
  }
  flush();
  return result;
}

/** 收起的工具组摘要行复用工具行栅格；展开后逐行恢复原行和详情。 */
function CollapsedToolGroup({ count, durationMs, children }: {
  count: number;
  durationMs?: number;
  children: ReactNode;
}) {
  const { open, instant, rowRef, toggle } = useStickyDisclosure();
  // 成员首次展开后保持挂载，收起时才有高度过渡可走。
  const [mounted, setMounted] = useState(false);
  return (
    <div className={`tool-group${open ? " is-open" : ""}`}>
      <div ref={rowRef} className={`tool-step-sticky${open ? " is-open" : ""}`}>
        <button
          type="button"
          className="tool-step-summary tool-group-summary"
          aria-expanded={open}
          onClick={() => { setMounted(true); toggle(); }}
        >
          <span className="tool-step-heading">
            <span className="tool-step-title">
              <Wrench className="tool-step-icon" size={13} aria-hidden="true" />
              <span>已执行 {count} 个工具</span>
            </span>
            <span className="tool-step-description" />
            <span className="tool-step-state">
              {durationMs !== undefined ? formatToolDuration(durationMs) : ""}
              <CollapseHint open={open} lead={durationMs !== undefined} />
            </span>
            <ChevronDown className={`tool-step-chevron ${open ? "open" : ""}`} size={13} aria-hidden="true" />
          </span>
        </button>
      </div>
      <Disclosure open={open} instant={instant}>
        {mounted ? <div className="tool-group-items">{children}</div> : null}
      </Disclosure>
    </div>
  );
}

/** 历史与实时回复共用一条流，过程项继续引用原消息和 part。 */
function TimelineProcess({ flow, draftThinking = "", draftThinkingActive = false, draftLead, groupTools = true, beforeReasoning, onCopyToolDetail, partView }: {
  flow: FlowItem[];
  draftThinking?: string;
  /** 紧贴在草稿思考之前的一行（重试进度），属于当前草稿而不是已提交的过程。 */
  draftLead?: ReactNode;
  /** 草稿思考是否仍在进行：正文开始输出（或本轮已停）时思考已经结束，标题改成"已思考"，不再与正文同时显示"正在思考"。 */
  draftThinkingActive?: boolean;
  /** 本轮摘要里已经收过一次：展开后直接逐行，不再套一层"已执行 N 个工具"。 */
  groupTools?: boolean;
  beforeReasoning?: (message: TimelineMessage) => ReactNode;
  onCopyToolDetail?: (text: string) => void;
  partView?: FlowPartViewProps;
}) {
  // 槽位锚在首个过程项之前；纯文本输出没有过程项时退到流开头，仍由提交消息提供上下文。
  const slotIndex = flow.findIndex((item) => item.kind === "thinking" || item.kind === "tool");
  const slotKey = (slotIndex >= 0 ? flow[slotIndex] : flow[0])?.key ?? null;
  const grouped = groupTools ? groupToolRuns(flow,
    (item) => item.kind === "tool" && item.block.status !== "input-available" ? item.block : null) : flow;
  if (!flow.length && !draftThinking && !draftLead) return null;
  return <>
    <div className="process-trace">
      {grouped.map((item) => "members" in item ? (
        <Fragment key={`toolgroup-${item.members[0].key}`}>
          {item.members.some((member) => member.key === slotKey) ? beforeReasoning?.(item.members[0].origin) : null}
          <CollapsedToolGroup count={item.members.length} durationMs={item.durationMs}>
            {item.members.map((member) => <FlowItemView key={member.key} item={member} onCopyToolDetail={onCopyToolDetail} partView={partView} />)}
          </CollapsedToolGroup>
        </Fragment>
      ) : (
        <Fragment key={item.key}>
          {item.key === slotKey ? beforeReasoning?.(item.origin) : null}
          <FlowItemView item={item} onCopyToolDetail={onCopyToolDetail} partView={partView} />
        </Fragment>
      ))}
      {draftLead}
      {draftThinking ? <ThinkingRow content={draftThinking} streaming={draftThinkingActive} /> : null}
    </div>
  </>;
}

/** 一轮结束后，思考、工具与中途旁白收进一行摘要；正文、附件和仍在运行的工具留在外面。 */
function SettledTurnFlow({ flow, startedAt, endedAt, beforeReasoning, partView }: {
  flow: FlowItem[];
  startedAt?: string;
  endedAt: string;
  beforeReasoning?: (message: TimelineMessage) => ReactNode;
  partView: FlowPartViewProps;
}) {
  const process = flow.filter((item) => item.kind !== "part");
  const running = process.some((item) => item.kind === "tool" && item.block.status === "input-available");
  const worthSummary = process.some((item) => item.kind === "tool") || process.length >= 2;
  if (running || !worthSummary) {
    return <TimelineProcess flow={flow} beforeReasoning={beforeReasoning} partView={partView} />;
  }
  // 1. 召回等插件槽位属于本轮开头，留在摘要外保持可见。
  // 2. 摘要内复用逐行过程视图，展开后与运行时看到的一致。
  return <>
    {beforeReasoning?.(process[0].origin)}
    <TurnProcessSummary items={process} durationMs={turnDurationMs(startedAt, endedAt)}>
      <TimelineProcess flow={process} groupTools={false} />
    </TurnProcessSummary>
    <TimelineProcess flow={flow.filter((item) => item.kind === "part")} partView={partView} />
  </>;
}

function turnDurationMs(startedAt: string | undefined, endedAt: string): number | undefined {
  const elapsed = startedAt ? Date.parse(endedAt) - Date.parse(startedAt) : Number.NaN;
  return Number.isFinite(elapsed) && elapsed >= 1_000 ? elapsed : undefined;
}

function formatTurnDuration(durationMs: number): string {
  const seconds = Math.round(durationMs / 1_000);
  return seconds < 60 ? `${seconds}s` : `${Math.floor(seconds / 60)}m ${seconds % 60}s`;
}

function TurnProcessSummary({ items, durationMs, children }: {
  items: FlowItem[];
  durationMs?: number;
  children: ReactNode;
}) {
  const { open, instant, rowRef, toggle } = useStickyDisclosure();
  const [mounted, setMounted] = useState(false);
  const thinking = items.filter((item) => item.kind === "thinking").length;
  const tools = items.filter((item) => item.kind === "tool");
  const failed = tools.filter((item) => item.kind === "tool" && item.block.status === "output-error").length;
  const label = [thinking ? `思考 ${thinking} 次` : "", tools.length ? `调用 ${tools.length} 个工具` : ""]
    .filter(Boolean).join("，") || "过程记录";
  const Icon = tools.length ? Wrench : Brain;
  return (
    <div className={`turn-process${open ? " is-open" : ""}`}>
      <div ref={rowRef} className={`tool-step-sticky${open ? " is-open" : ""}`}>
        <button type="button" className="tool-step-summary turn-process-summary" aria-expanded={open}
          onClick={() => { setMounted(true); toggle(); }}>
          <span className="tool-step-heading">
            <span className="tool-step-title">
              <Icon className="tool-step-icon" size={13} aria-hidden="true" />
              <span>{label}</span>
            </span>
            <span className="tool-step-description">{failed ? <em className="turn-process-failed">{failed} 个失败</em> : null}</span>
            <span className="tool-step-state">
              {durationMs !== undefined ? formatTurnDuration(durationMs) : ""}
              <CollapseHint open={open} lead={durationMs !== undefined} />
            </span>
            <ChevronDown className={`tool-step-chevron ${open ? "open" : ""}`} size={13} aria-hidden="true" />
          </span>
        </button>
      </div>
      <Disclosure open={open} instant={instant}>
        {mounted ? <div className="turn-process-items">{children}</div> : null}
      </Disclosure>
    </div>
  );
}

/** 图文回复只收纳思考与已完成工具；正文和仍在执行的工具保持可见。 */
function InlineReplyProcess({ messages, toolResults, beforeReasoning }: {
  messages: TimelineMessage[]; toolResults: ReadonlyMap<string, TimelineMessage>;
  beforeReasoning?: (message: TimelineMessage) => ReactNode;
}) {
  const flow = timelineFlow(messages, undefined, toolResults).filter(item => item.kind === "tool" || item.kind === "thinking");
  const running = flow.filter(item => item.kind === "tool" && item.block.status === "input-available");
  const settled = flow.filter(item => !running.includes(item));
  const count = settled.filter(item => item.kind === "tool").length;
  return <div className="inline-reply-process">
    {settled.length ? <details className="context-notice"><summary className="process-trigger">
      {count ? <Wrench className="process-trigger-icon" size={13} aria-hidden="true" />
        : <Brain className="process-trigger-icon" size={13} aria-hidden="true" />}
      <span className="process-trigger-summary">{count ? `思考并调用 ${count} 个工具` : "思考过程"}</span>
      <ChevronDown className="process-chevron" size={13} aria-hidden="true" />
    </summary>
      <TimelineProcess flow={settled} beforeReasoning={beforeReasoning} />
    </details> : null}
    <TimelineProcess flow={running} beforeReasoning={beforeReasoning} />
  </div>;
}

/** 保留消息引用与 part 位置，复用原聊天的过程和正文组件。 */
export function TimelineMessageView({ message, lookupMessage, toolResults, onNavigate, onError, beforeReasoning, afterBody, renderAttachment, hideBody = false, processMessages = [message], hideProcess = false, inlineReply = false, leadingProcess, canLoadReferences = false, turnStartedAt, turnOpen = false }: {
  message: TimelineMessage;
  /** 本轮用户输入的时间；摘要行据此给出整轮用时。 */
  turnStartedAt?: string;
  /** 本轮还没结束：过程逐行显示，不收摘要。 */
  turnOpen?: boolean;
  hideBody?: boolean;
  canLoadReferences?: boolean;
  toolResults: ReadonlyMap<string, TimelineMessage>;
  renderAttachment?: (attachment: TimelineAttachment) => ReactNode;
  beforeReasoning?: (message: TimelineMessage) => ReactNode;
  processMessages?: TimelineMessage[];
  hideProcess?: boolean;
  inlineReply?: boolean;
  leadingProcess?: TimelineMessage[];
  afterBody?: ReactNode;
  lookupMessage: (id: string) => TimelineMessage | undefined;
  onNavigate: (id: string, partIndex?: number) => void;
  onError?: (error: unknown) => void;
}) {
  const body = message.body;
  // 未结束轮次里的输出都是 finish=continue，文字后面还有工具调用，是旁白而不是最终回答。
  const flow = hideProcess || inlineReply ? [] : timelineFlow(processMessages, turnOpen ? "" : message.id, toolResults);
  const hasFlow = body.kind === "output" && flow.length > 0;
  const leadingContent = hideProcess || hasFlow || inlineReply ? null : beforeReasoning?.(message);
  const referencedArtifacts = new Set(body.kind === "control" ? [] : body.parts.flatMap((part) =>
    !("display" in part) && part.kind === "artifact_ref" ? [part.value] : []));
  const origin = attachmentOrigin(message, lookupMessage);
  // 单文件成功结果作为回复插图；原始工具数据仍可从过程行查看。
  // 同一结果里的图片是给模型自检的观察（如渲染截图），已在调用卡片里展示，不再进正文。
  const deliveredFiles = message.attachments.filter(ref => ref.kind === "file");
  const attachmentResult = body.kind === "tool_result" && body.outcome === "success" && deliveredFiles.length === 1
    && body.parts.every(part => !("display" in part) && (part.kind === "text" || part.kind === "artifact_ref"));
  const observation = (ref: TimelineAttachment) => attachmentResult && ref.kind === "image";
  const fallback = attachmentResult ? <div className="attachment-description">{body.parts.map((part, index) =>
    part.kind === "text" && !("display" in part) ? <MessageBody key={index} content={part.value}
      streaming={false} deferRichContent onError={onError} /> : null)}
    {!body.parts.some(part => part.kind === "text" && !("display" in part) && part.value.trim())
      ? <p>{deliveredFiles[0].filename ?? "附件"}</p> : null}</div> : null;
  // 插件认领的文件附件（例如网页）在任何消息里都作为插图展示，插件缺席时退回普通附件卡片。
  usePluginUiCatalogVersion();
  const figure = (ref: TimelineAttachment) => !renderAttachment && ref.kind === "file"
    && pluginSlotAccepts("message.attachment", { ...ref, origin });
  const attachment = (id: string) => {
    const ref = message.attachments.find(item => item.artifact_id === id)!;
    if (renderAttachment) return renderAttachment(ref);
    if (observation(ref)) return null;
    if (attachmentResult || figure(ref)) return <MessageAttachmentFigure attachment={ref} origin={origin}
      sessionId={message.session_id} messageId={message.id}
      fallback={attachmentResult ? fallback : <SentAttachments items={[timelineAttachmentItem(ref)]} />} />;
    return <SentAttachments items={[timelineAttachmentItem(ref)]} />;
  };
  // 与实时消息同一个容器；用户输入的附件（含正文里的 artifact_ref）全部收进气泡上方的附件条，发出时与刷新后位置一致。
  const input = body.kind === "input";
  const looseAttachments = input ? message.attachments
    : message.attachments.filter((ref) => !referencedArtifacts.has(ref.artifact_id) && !observation(ref));
  const figures = attachmentResult ? looseAttachments : looseAttachments.filter(figure);
  const strip = looseAttachments.filter(ref => !figures.includes(ref));
  // 插图沿正文宽度展示，不进附件条。
  const attachmentList = !looseAttachments.length ? null : <>
    {strip.length ? <SentAttachments items={strip.map(timelineAttachmentItem)} /> : null}
    {figures.map((ref, index) => <Fragment key={`${ref.artifact_id}:${index}`}>{attachment(ref.artifact_id)}</Fragment>)}
  </>;
  const emptyBubble = input && !leadingContent && !leadingProcess && !afterBody && body.parts.every((part) => !isTimelinePartVisible(part)
    || (!("display" in part) && (part.kind === "artifact_ref" || (part.kind === "text" && !part.value.trim()))));
  // 不是用户写下的输入（定时任务、唤醒、子代理等任何插件发起）靠左显示成一条事件行。
  const automatic = input && message.author !== "user" && message.author !== "legacy-attribution-unknown";
  return <div className={`message-row timeline-message timeline-${body.kind}${automatic ? " timeline-automatic" : ""}`}>
    {input ? attachmentList : null}
    {emptyBubble ? null : <div className={automatic ? "automatic-input" : input ? "user-bubble" : "agent-content"}>
      {leadingContent}
      {leadingProcess ? <InlineReplyProcess messages={leadingProcess} toolResults={toolResults}
        beforeReasoning={beforeReasoning} /> : null}
      {hasFlow ? turnOpen
        ? <TimelineProcess flow={flow} beforeReasoning={beforeReasoning}
          partView={{ attachment, lookupMessage, onNavigate, onError, canLoadReferences }} />
        : <SettledTurnFlow flow={flow} startedAt={turnStartedAt} endedAt={message.timestamp}
          beforeReasoning={beforeReasoning}
          partView={{ attachment, lookupMessage, onNavigate, onError, canLoadReferences }} /> : null}
      {body.kind === "control" ? body.action === "failure" ? <p className="timeline-failure">
        {/* 失败就是这一轮的结果：一段话，句首小标题是全块唯一的错误色。 */}
        <strong>{controlLabels[body.action]}</strong>{body.reason}
      </p> : <p className="timeline-control-note">
        {/* 暂停、恢复、放弃是轮次里的一个事件，不是一段回复：一行次级小字即可。 */}
        {controlLabels[body.action]}{body.reason ? ` · ${body.reason}` : ""}
      </p> : hasFlow ? null : <>
        {body.kind === "tool_result" && !attachmentResult ? <div className="timeline-result-heading">
          <strong>工具结果 · {outcomeLabels[body.outcome]}</strong>
          <button type="button" onClick={() => onNavigate(body.call_ref.message_id, body.call_ref.part_index)}
            disabled={!lookupMessage(body.call_ref.message_id)}>
            {lookupMessage(body.call_ref.message_id) ? "查看调用" : "调用不在当前记录中"}
          </button>
        </div> : null}
        <UserTextClamp enabled={body.kind === "input"}>{body.parts.map((part, index) => ({ part, index })).sort((left, right) =>
          flowRank(right.part) + Number(right.part.kind === "history.transcript")
          - flowRank(left.part) - Number(left.part.kind === "history.transcript"))
          .map(({ part, index }) => isTimelinePartVisible(part) && !(input && part.kind === "artifact_ref") && !((hideBody || (attachmentResult && !renderAttachment)) && part.kind === "text") && !((hideProcess || inlineReply) && (("display" in part && part.display === "thinking") || part.kind === "tool_call" || part.kind === "history.transcript")) ? <div key={index} data-part-index={index} tabIndex={-1}>
          <TimelinePartView part={part} toolResult={body.kind === "tool_result" && !attachmentResult} userInput={input} attachment={attachment} lookupMessage={lookupMessage} canLoadReferences={canLoadReferences}
            onNavigate={onNavigate} onError={onError} processStartContent={leadingContent} />
        </div> : null)}</UserTextClamp>
      </>}
      {afterBody}
      {input ? null : attachmentList}
    </div>}
  </div>;
}

function timelineAttachmentItem(ref: TimelineAttachment): SentAttachmentItem {
  return { id: ref.artifact_id, name: ref.filename ?? "附件", mediaType: ref.media_type ?? undefined,
    url: `/api/chat/artifacts/${encodeURIComponent(ref.artifact_id)}`, sizeBytes: ref.size_bytes };
}

// 长输入（粘贴的日志、整段代码）收成约 12 行，下面一行安静的"展开全部"；短输入不受影响。
// 是否溢出取决于当前宽度，只在收起时测量，宽度变化时重测。
// 按最窄阅读宽度（约 12 个汉字一行）粗估行数；含图片、代码块、表格时高度无法从字数推断，一律测量。
function mayExceedClamp(element: HTMLElement): boolean {
  if (element.querySelector("img, pre, table")) return true;
  const lines = (element.textContent ?? "").split("\n")
    .reduce((count, line) => count + Math.max(1, Math.ceil(line.length / 12)), 0);
  return lines >= 12;
}

function UserTextClamp({ enabled, children }: { enabled: boolean; children: ReactNode }) {
  const [expanded, setExpanded] = useState(false);
  const [overflowing, setOverflowing] = useState(false);
  const box = useRef<HTMLDivElement>(null);
  const content = useRef<HTMLDivElement>(null);
  const toggle = useRef<HTMLButtonElement>(null);
  useLayoutEffect(() => {
    const element = box.current;
    const inner = content.current;
    if (!enabled || expanded || !element || !inner) return;
    // 明显不到 12 行的短输入不读几何、不挂观察器：补载历史时这些消息多在跳过渲染的区域里，逐条读 scrollHeight
    // 会逐条强制布局；历史消息带 content-visibility:auto，逐条观察会让长会话滚动多跑布局。
    if (!mayExceedClamp(element)) return;
    const measure = () => {
      setOverflowing(element.scrollHeight - element.clientHeight > 1);
    };
    // 首帧的裁剪由 CSS 无条件保证（.user-text-clamp:not([data-state="open"]) 恒裁 12 行），
    // 本处只决定“展开全部”按钮与末两行渐隐，所以不在注册时同步测量：ResizeObserver 回调在
    // 布局之后、绘制之前跑，此时读 scrollHeight 不再强制重排（design.md 3.18），零强制布局且首帧完成。
    // 曾经同步测量，切入长会话时可见区每条长提问一次，实测占切换 CPU 的 ~19%（measure，CPU×4）。
    // 可能超出的这一小部分观察折叠框与里面不限高的内容层：窗口变宽变窄、字体换用、图片加载、富渲染逐步换出
    // （源码→高亮代码块）都会改内容层高度。只看框不够：恰好 12 行的正文再晚到一张图，框仍是 12 行高，变的只有内容。
    const observer = new ResizeObserver(measure);
    observer.observe(element);
    observer.observe(inner);
    return () => observer.disconnect();
  }, [enabled, expanded]);
  if (!enabled) return <>{children}</>;
  const onToggle = () => {
    setExpanded((value) => !value);
    // 收起时上方内容变短，把按钮留在视口里，读者不会被甩到别处。
    if (expanded) requestAnimationFrame(() => toggle.current?.scrollIntoView({ block: "nearest" }));
  };
  return <>
    <div ref={box} className="user-text-clamp" data-state={expanded ? "open" : overflowing ? "clamped" : undefined}>
      <div ref={content}>{children}</div>
    </div>
    {overflowing || expanded ? <button ref={toggle} type="button" className="user-text-toggle"
      aria-expanded={expanded} onClick={onToggle}>{expanded ? "收起" : "展开全部"}</button> : null}
  </>;
}

export interface TimelineAttachmentOrigin {
  kind: TimelineMessage["body"]["kind"];
  author: string;
  tool: string | null;
  /** 附件所在消息的时间（ISO 8601）。 */
  at: string;
}

// 插件按来源决定是否认领附件；工具结果附带产生它的工具名，缺少调用记录时为 null。
function attachmentOrigin(message: TimelineMessage, lookupMessage: (id: string) => TimelineMessage | undefined): TimelineAttachmentOrigin {
  const body = message.body;
  if (body.kind !== "tool_result") return { kind: body.kind, author: message.author, tool: null, at: message.timestamp };
  const call = lookupMessage(body.call_ref.message_id);
  const part = call?.body.kind === "control" ? undefined : call?.body.parts[body.call_ref.part_index];
  return { kind: body.kind, author: message.author, tool: part?.kind === "tool_call" && !("display" in part) ? part.name : null, at: message.timestamp };
}

/** 保留一个稳定的插图容器；插件退出后用原消息文字降级，平滑收起高度。 */
function MessageAttachmentFigure({ attachment, origin, sessionId, messageId, fallback }: {
  attachment: TimelineAttachment; origin: TimelineAttachmentOrigin;
  sessionId: string; messageId: string; fallback: ReactNode;
}) {
  const box = useRef<HTMLElement>(null);
  const content = useRef<HTMLDivElement>(null);
  useLayoutEffect(() => {
    const outer = box.current!, inner = content.current!;
    const measure = () => { outer.style.height = `${inner.getBoundingClientRect().height}px`; };
    measure();
    const observer = new ResizeObserver(measure);
    observer.observe(inner);
    return () => observer.disconnect();
  }, []);
  return <section ref={box} className="message-figure" aria-label="回复附件">
    <div ref={content}>
      <PluginUiSlot name="message.attachment" quiet sessionId={sessionId} messageId={messageId}
        block={{ ...attachment, origin }} fallback={fallback} />
      <a className="message-figure-download" href={`/api/chat/artifacts/${encodeURIComponent(attachment.artifact_id)}`}
        download={attachment.filename ?? "附件"} aria-label={`下载源文件 ${attachment.filename ?? "附件"}`}><Download size={16} aria-hidden="true" /></a>
    </div>
  </section>;
}

// 用户的纯文本输入：已知技能的 /名称 显示成提及。
function UserPlainText({ text }: { text: string }) {
  return <p className="plain-message-response"><MentionText text={text} /></p>;
}

function TimelinePartView({ part, toolResult = false, userInput = false, attachment, lookupMessage, onNavigate, onError, processStartContent, canLoadReferences }: {
  canLoadReferences: boolean;
  toolResult?: boolean;
  userInput?: boolean;
  processStartContent?: ReactNode;
  part: TimelinePart;
  attachment: (id: string) => ReactNode;
  lookupMessage: (id: string) => TimelineMessage | undefined;
  onNavigate: (id: string, partIndex?: number) => void;
  onError?: (error: unknown) => void;
}) {
  if ("display" in part) return part.display === "data"
    ? <ToolResultContent value={"rendered" in part ? part.rendered : part.value} />
    : part.display === "thinking" ? part.text ? <ProcessTrace blocks={[{ kind: "thinking", content: part.text }]} streaming={false}
      interrupted={false} startContent={processStartContent} /> : null
    : <p className="timeline-state">无法展示此内容</p>;
  if ("archive" in part) return part.kind === "history.transcript" ? <TimelineTranscript archive={part.archive} startContent={processStartContent} onError={onError} /> : null;
  switch (part.kind) {
    case "context.notice": return <ContextNoticeRow text={part.value.text} />;
    case "text": return toolResult ? <ToolResultContent value={"rendered" in part ? part.rendered : part.value} />
      : userInput && !messageNeedsMarkdown(part.value) ? <UserPlainText text={part.value} />
      : <MessageBody content={part.value} streaming={false} deferRichContent onError={onError} />;
    case "artifact_ref": return attachment(part.value);
    case "reply_ref": {
      const source = lookupMessage(part.value);
      return <MessageReplyReference author={source?.author ?? "原消息"}
        preview={source ? timelineReply(source).preview : ""} unavailable={!source} canLoad={canLoadReferences}
        onNavigate={() => onNavigate(part.value)} />;
    }
    case "tool_call": return <details className="timeline-details timeline-tool-call">
      <summary>工具调用 · {part.name}</summary>
      <pre tabIndex={0}>{JSON.stringify(part.arguments, null, 2)}</pre>
    </details>;
  }
}

/** 将归档转换成既有过程组件的展示块，原日志不变。 */
function historyBlocks(archive: unknown): AgentBlock[] {
  const groups = historyTranscript(archive);
  return groups?.flatMap((group, index): AgentBlock[] => [
    ...(group.thinking ? [{ kind: "thinking" as const, content: group.thinking }] : []),
    ...(group.text ? [{ kind: "thinking" as const, content: group.text }] : []),
    ...group.calls.map((call, callIndex) => ({ kind: "tool" as const,
      callId: `history-${index}-${callIndex}`, name: call.name, input: call.arguments,
      output: call.result, status: "output-available" as const, errorText: undefined })),
  ]) ?? [];
}

function TimelineTranscript({ archive }: { archive: unknown; startContent?: ReactNode; onError?: (error: unknown) => void }) {
  if (historyTranscript(archive) === null) return <p>这段历史过程暂无法展示。</p>;
  const blocks = historyBlocks(archive);
  return blocks.length ? <ProcessTrace blocks={blocks} streaming={false} interrupted={false} /> : null;
}

const controlLabels = { pause: "已暂停", resume: "已恢复", abandon: "已放弃", failure: "执行失败" };
const outcomeLabels = { success: "成功", denied: "已拒绝", error: "失败", interrupted: "已中断" };

const MessageAttachments = memo(function MessageAttachments({ attachments, pending }: { attachments: MessageAttachment[]; pending: boolean }) {
  // 发送中的附件还是本地 Blob：骨架闪烁到上传完成换成服务端附件。
  return <SentAttachments pending={pending && attachments.some((item) => item.url.startsWith("blob:"))}
    items={attachments.map((item) => ({ id: item.id, name: item.filename ?? "附件", mediaType: item.mediaType, url: item.url, sizeBytes: item.sizeBytes }))} />;
});

type MediaItem = { url: string; label: string; kind: MediaKind; trigger: HTMLElement };
type MediaGallery = { items: MediaItem[]; index: number };

const REPLY_IMAGE_SCOPE = ".agent-content, .process-markdown";

// 可预览的媒体：带 data-media-* 的附件按钮；回复正文里的 Markdown 图片（普通 img，不在链接或按钮里）。
function mediaItem(node: Element): MediaItem | null {
  if (node instanceof HTMLElement && node.dataset.mediaUrl) {
    return { url: node.dataset.mediaUrl, label: node.dataset.mediaLabel ?? "附件",
      kind: node.dataset.mediaKind === "video" ? "video" : "image", trigger: node };
  }
  if (node instanceof HTMLImageElement && node.closest(REPLY_IMAGE_SCOPE) && !node.closest("a, button")) {
    return { url: node.currentSrc || node.src, label: node.alt || "图片", kind: "image", trigger: node };
  }
  return null;
}

// 同一条消息里的全部媒体按文档顺序成组。不跨消息：远处的消息是占位渲染，媒体不在 DOM 里，跨消息的组会时有时无。
function galleryFrom(item: MediaItem): MediaGallery {
  const scope = item.trigger.closest(".web-message-anchor");
  const items = scope ? [...scope.querySelectorAll("[data-media-url], img")].flatMap((node) => {
    const found = mediaItem(node);
    return found ? [found] : [];
  }) : [];
  const index = items.findIndex((candidate) => candidate.trigger === item.trigger);
  return index < 0 ? { items: [item], index: 0 } : { items, index };
}

// 图片与视频的查看器（参考 Memoh 的媒体灯箱）：附件与回复正文图片共用，document 级委托，不改渲染节点。
// 正文图片由这里补键盘入口（可聚焦、按钮语义，Enter/空格打开）；同一条消息里有多张时 ←/→ 切换；
// 关闭后焦点回到当前这一张的入口。
export function MediaGalleryViewer() {
  const { scrollRef } = useStickToBottomContext();
  const [gallery, setGallery] = useState<MediaGallery | null>(null);
  // 必要 effect：点击、按键委托挂在 document 上；新消息与懒加载换出的 img 由 MutationObserver 补上键盘入口。
  useEffect(() => {
    const triggerOf = (target: EventTarget | null) => target instanceof Element
      ? mediaItem(target.closest("[data-media-url]") ?? target) : null;
    const mark = (root: ParentNode) => {
      for (const img of root.querySelectorAll("img")) {
        // markstream 的图片自带 tabindex="0"：焦点属性与按钮语义分开补，不因已可聚焦就跳过角色和标签。
        if (!mediaItem(img) || img.getAttribute("role") === "button") continue;
        if (!img.hasAttribute("tabindex")) img.tabIndex = 0;
        img.setAttribute("role", "button");
        img.setAttribute("aria-label", `查看图片：${img.alt || "图片"}`);
      }
    };
    const onClick = (event: MouseEvent) => {
      const item = triggerOf(event.target);
      if (!item) return;
      event.preventDefault();
      setGallery(galleryFrom(item));
    };
    const onKey = (event: KeyboardEvent) => {
      if ((event.key !== "Enter" && event.key !== " ") || !(event.target instanceof HTMLImageElement)) return;
      const item = mediaItem(event.target);
      if (!item) return;
      event.preventDefault();
      setGallery(galleryFrom(item));
    };
    const scroller = scrollRef.current;
    const observer = new MutationObserver((records) => {
      for (const record of records) for (const node of record.addedNodes) {
        if (node instanceof HTMLImageElement) mark(node.parentElement ?? node);
        else if (node instanceof Element && node.querySelector("img")) mark(node);
      }
    });
    if (scroller) {
      mark(scroller);
      observer.observe(scroller, { childList: true, subtree: true });
    }
    document.addEventListener("click", onClick);
    document.addEventListener("keydown", onKey);
    return () => {
      observer.disconnect();
      document.removeEventListener("click", onClick);
      document.removeEventListener("keydown", onKey);
    };
  }, [scrollRef]);
  const step = (delta: number) => setGallery((current) => current
    ? { ...current, index: Math.min(Math.max(current.index + delta, 0), current.items.length - 1) } : current);
  return <Dialog open={gallery !== null} onOpenChange={(open) => { if (!open) setGallery(null); }}>
    {gallery ? <MediaViewer gallery={gallery} onStep={step} onClose={() => setGallery(null)} /> : null}
  </Dialog>;
}

type ZoomPoint = { x: number; y: number; px: number; py: number };

type ZoomState = { owner: MediaGallery; scaled: boolean; actual: boolean };

// 图片放不下画框时可切到原始像素，在同一块区域里滚动看细节。状态跟着这一次翻页（图集对象）走：
// 翻走再翻回来、同一地址在图集里出现两次，都从适应窗口开始，不需要 effect 重置。
function useImageZoom(item: MediaItem, owner: MediaGallery) {
  // 画框与图片要用 state 记：对话框内容在 Portal 挂上后才出现，图片还会随地址换节点，effect 要跟着重新接上。
  const [frame, frameRef] = useState<HTMLDivElement | null>(null);
  const [image, imageRef] = useState<HTMLImageElement | null>(null);
  const [state, setState] = useState<ZoomState | null>(null);
  const anchor = useRef<ZoomPoint | null>(null);
  const current = item.kind === "image" && state?.owner === owner ? state : null;
  const zoomable = current?.scaled ?? false;
  const actual = zoomable && (current?.actual ?? false);
  const toggle = (point?: ZoomPoint) => {
    anchor.current = point ?? null;
    setState({ owner, scaled: true, actual: !actual });
  };
  // 必要 effect：放不放得下取决于画框尺寸（转屏、分屏、拉窗口），加载完成与画框变化时都重算；放得下了就撤掉缩放。
  useEffect(() => {
    if (!frame || !image) return;
    const measure = () => {
      if (!image.naturalWidth) return;
      const scaled = image.naturalWidth > frame.clientWidth + 1 || image.naturalHeight > frame.clientHeight + 1;
      setState((previous) => {
        const mine = previous?.owner === owner ? previous : null;
        return mine?.scaled === scaled ? mine : { owner, scaled, actual: scaled && (mine?.actual ?? false) };
      });
    };
    const resize = new ResizeObserver(measure);
    resize.observe(frame);
    image.addEventListener("load", measure);
    return () => {
      resize.disconnect();
      image.removeEventListener("load", measure);
    };
  }, [owner, frame, image]);
  // 切到原始像素后让点击处留在指针下（键盘切换时停在中心）。画框不随缩放变化，按图片在框里的新位置换算。
  useLayoutEffect(() => {
    if (!actual || !frame || !image) return;
    const point = anchor.current ?? { x: 0.5, y: 0.5, px: frame.clientWidth / 2, py: frame.clientHeight / 2 };
    const box = image.getBoundingClientRect();
    const area = frame.getBoundingClientRect();
    frame.scrollLeft += box.left - area.left + point.x * box.width - point.px;
    frame.scrollTop += box.top - area.top + point.y * box.height - point.py;
  }, [actual, frame, image]);
  return { frame, frameRef, imageRef, actual, zoomable, toggle };
}

// 触屏在媒体上横滑切换上一个/下一个。原始像素下滑动是平移画面；顶栏与视频底部控制条（拖进度）上的滑动也不算。
function useSwipeStep(enabled: boolean, go: (delta: number) => void) {
  const start = useRef<{ x: number; y: number } | null>(null);
  return {
    onPointerDown: (event: ReactPointerEvent) => {
      const target = event.target instanceof Element ? event.target : null;
      const onControls = target instanceof HTMLVideoElement && event.clientY > target.getBoundingClientRect().bottom - 64;
      start.current = enabled && event.pointerType === "touch" && !onControls && !target?.closest(".media-preview__bar")
        ? { x: event.clientX, y: event.clientY } : null;
    },
    onPointerUp: (event: ReactPointerEvent) => {
      const from = start.current;
      start.current = null;
      if (!from) return;
      const dx = event.clientX - from.x;
      if (Math.abs(dx) > 48 && Math.abs(event.clientY - from.y) < 40) go(dx < 0 ? 1 : -1);
    },
  };
}

function MediaPager({ index, count, go }: { index: number; count: number; go: (delta: number) => void }) {
  return <div className="media-preview__pager">
    <button type="button" aria-label="上一个" data-tip="上一个（←）" disabled={index === 0} onClick={() => go(-1)}>
      <ChevronLeft size={18} aria-hidden="true" />
    </button>
    <span aria-live="polite">{index + 1} / {count}</span>
    <button type="button" aria-label="下一个" data-tip="下一个（→）" disabled={index === count - 1} onClick={() => go(1)}>
      <ChevronRight size={18} aria-hidden="true" />
    </button>
  </div>;
}

// 整屏压暗，视频按视口放大；图片原尺寸显示、超出视口才等比缩小，被缩小时点图（或顶栏按钮）切到原始像素。
function MediaViewer({ gallery, onStep, onClose }: { gallery: MediaGallery; onStep: (delta: number) => void; onClose: () => void }) {
  const { items, index } = gallery;
  const item = items[index];
  const closeRef = useRef<HTMLButtonElement>(null);
  const zoom = useImageZoom(item, gallery);
  // 到头停住不循环。
  const go = (delta: number) => { if (index + delta >= 0 && index + delta < items.length) onStep(delta); };
  const swipe = useSwipeStep(items.length > 1 && !zoom.actual, go);
  return <DialogContent className="media-preview" overlayClassName="media-preview-overlay" showCloseButton={false}
    aria-describedby={undefined} {...swipe}
    // 打开后焦点落在关闭钮（Esc 与回车都能离开），而不是顶栏第一个"新标签打开"；关闭后回到当前这一张的入口。
    onOpenAutoFocus={(event) => { event.preventDefault(); closeRef.current?.focus(); }}
    onCloseAutoFocus={(event) => {
      if (!item.trigger.isConnected) return;
      event.preventDefault();
      item.trigger.focus({ preventScroll: true });
    }}
    onKeyDown={(event) => {
      if (event.target instanceof HTMLVideoElement || event.altKey || event.ctrlKey || event.metaKey || event.shiftKey) return;
      if (event.key !== "ArrowLeft" && event.key !== "ArrowRight") return;
      event.preventDefault();
      go(event.key === "ArrowLeft" ? -1 : 1);
    }}
    onClick={(event) => { if (event.target === event.currentTarget || event.target === zoom.frame) onClose(); }}>
    <div className="media-preview__bar">
      <DialogTitle className="media-preview__title"><MarqueeText still>{item.label}</MarqueeText></DialogTitle>
      {items.length > 1 ? <MediaPager index={index} count={items.length} go={go} /> : null}
      {zoom.zoomable ? <button type="button" aria-pressed={zoom.actual} aria-label={zoom.actual ? "适应窗口" : "原始尺寸"}
        title={zoom.actual ? "适应窗口" : "原始尺寸"} onClick={() => zoom.toggle()}>
        {zoom.actual ? <ZoomOut size={18} aria-hidden="true" /> : <ZoomIn size={18} aria-hidden="true" />}
      </button> : null}
      <a href={item.url} target="_blank" rel="noreferrer" aria-label="在新标签页打开" data-tip="在新标签页打开"><ExternalLink size={18} aria-hidden="true" /></a>
      <a href={item.url} download={item.label} aria-label={`下载 ${item.label}`} data-tip="下载"><Download size={18} aria-hidden="true" /></a>
      <DialogClose ref={closeRef} aria-label="关闭" data-tip="关闭（Esc）"><X size={20} aria-hidden="true" /></DialogClose>
    </div>
    {item.kind === "image"
      ? <div ref={zoom.frameRef} className={`media-preview__frame${zoom.actual ? " is-actual" : ""}`}>
        <img key={item.url} ref={zoom.imageRef} alt={item.label} src={item.url}
          className={`media-preview__media${zoom.zoomable ? zoom.actual ? " is-actual" : " is-scaled" : ""}`}
          onClick={(event) => {
            if (!zoom.zoomable) return;
            const box = event.currentTarget.getBoundingClientRect();
            const frame = event.currentTarget.parentElement!.getBoundingClientRect();
            zoom.toggle({ x: (event.clientX - box.left) / box.width, y: (event.clientY - box.top) / box.height,
              px: event.clientX - frame.left, py: event.clientY - frame.top });
          }} />
      </div>
      : <video key={item.url} className="media-preview__media is-video" src={item.url} controls autoPlay playsInline />}
  </DialogContent>;
}

/** 每个过程项独立成行：思考、工具各自折叠，不共用外框。 */
const ProcessTrace = memo(function ProcessTrace({
  blocks,
  streaming,
  interrupted,
  durationMs,
  startContent,
  beforeBlock,
  onCopyToolDetail,
}: {
  blocks: AgentBlock[];
  streaming: boolean;
  interrupted: boolean;
  durationMs?: number;
  startContent?: ReactNode;
  beforeBlock?: (block: AgentBlock, index: number) => ReactNode;
  onCopyToolDetail?: (text: string) => void;
}) {
  const lastThinkingIndex = blocks.reduce((last, block, index) => block.kind === "thinking" ? index : last, -1);
  const grouped = groupToolRuns(blocks.map((block, index) => ({ block, index })),
    (entry) => entry.block.kind === "tool" && entry.block.status !== "input-available" ? entry.block : null);
  const renderBlock = (block: AgentBlock, index: number) => block.kind === "thinking" ? (
    <ThinkingRow
      content={block.content}
      streaming={streaming && index === lastThinkingIndex}
      duration={index === lastThinkingIndex && durationMs
        ? Math.max(1, Math.round(durationMs / 1000)) : undefined}
    />
  ) : (
    <ToolStep
      block={block}
      active={block.status === "input-available"}
      onCopyDetail={onCopyToolDetail}
    />
  );
  return (
    <div className="process-trace">
      {startContent}
      {grouped.map((entry) => "members" in entry ? (
        <CollapsedToolGroup key={`toolgroup-${entry.members[0].index}`} count={entry.members.length} durationMs={entry.durationMs}>
          {entry.members.map((member) => (
            <Fragment key={member.index}>
              {beforeBlock?.(member.block, member.index)}
              {renderBlock(member.block, member.index)}
            </Fragment>
          ))}
        </CollapsedToolGroup>
      ) : (
        <Fragment key={entry.block.kind === "thinking" ? `thinking-${entry.index}` : entry.block.callId}>
          {beforeBlock?.(entry.block, entry.index)}
          {renderBlock(entry.block, entry.index)}
        </Fragment>
      ))}
      {interrupted ? <div className="process-status-line">已中止</div> : null}
    </div>
  );
});

const ThinkingRow = memo(function ThinkingRow({
  content,
  streaming,
  duration,
}: {
  content: string;
  streaming: boolean;
  duration?: number;
}) {
  const features = streaming ? null : detectMessageRenderingFeatures(content);
  const needsRichContent = streaming || features?.math || features?.code;
  return (
    // 流式时只在标题行滚动最新一句，不自动展开：展开会与标题行重复同一句，结束时行重挂载又会瞬间收起。
    // Render ordinary Markdown right away; math and code still need the rich renderer.
    <Reasoning className="thinking-row" isStreaming={streaming} defaultOpen={false} duration={duration}>
      <div className="process-trigger-sticky"><ThinkingRowTrigger content={streaming ? content : ""} /></div>
      <CollapsibleContent className="process-content">
        <div className="process-text process-markdown">
          {needsRichContent ? <Suspense fallback={<span className="process-markdown-fallback">{content}</span>}>
            <LazyMessageResponse isAnimating={streaming}>{content}</LazyMessageResponse>
          </Suspense> : <StaticMessageResponse>{content}</StaticMessageResponse>}
        </div>
      </CollapsibleContent>
    </Reasoning>
  );
});

function ThinkingRowTrigger({ content }: { content: string }) {
  const { isOpen, isStreaming, duration } = useReasoning();
  // 收起长内容后视口可能跳离原位；结束后把行滚回最近可见位置。
  const handleClick = (event: ReactMouseEvent<HTMLButtonElement>) => {
    if (!isOpen) return;
    const row = event.currentTarget.closest<HTMLElement>(".thinking-row");
    requestAnimationFrame(() => requestAnimationFrame(() => {
      row?.scrollIntoView({ block: "nearest" });
    }));
  };
  return (
    <ReasoningTrigger className={`process-trigger ${isStreaming ? "is-streaming" : ""}`} onClick={handleClick}>
      <Brain className="process-trigger-icon" size={13} aria-hidden="true" />
      <span className="process-trigger-label">
        {isStreaming ? "正在思考" : duration ? `已思考 ${duration}s` : "已思考"}
      </span>
      {content ? <ThinkingTail content={content} /> : null}
      <ChevronDown className={`process-chevron ${isOpen ? "open" : ""}`} size={13} aria-hidden="true" />
    </ReasoningTrigger>
  );
}

/** 流式思考只显示最新一行的末尾：长行写满一屏后仍能看到它在往前走，换行时接着从新行开始。 */
function ThinkingTail({ content }: { content: string }) {
  const paced = usePacedText(content, true);
  const line = latestLine(paced);
  const frame = useRef<HTMLSpanElement>(null);
  // 必要 effect：只有排版后才知道这一行是否超出槽位，超出时左缘淡出、右端贴齐最新文字。
  useLayoutEffect(() => {
    const element = frame.current;
    const text = element?.firstElementChild;
    if (element && text) element.classList.toggle("is-clipped", text.scrollWidth > element.clientWidth);
  }, [line]);
  return <span ref={frame} className="process-trigger-summary is-tail"><span>{line}</span></span>;
}

// 从末尾往前找最后一个非空行，避免每帧把整段思考切成行数组。
function latestLine(text: string): string {
  let end = text.length;
  while (end > 0) {
    const start = text.lastIndexOf("\n", end - 1) + 1;
    const line = text.slice(start, end).trim();
    if (line) return line;
    end = start - 1;
  }
  return "";
}

const ToolStep = memo(function ToolStep({
  block,
  active,
  onCopyDetail,
}: {
  block: ToolBlock;
  active: boolean;
  onCopyDetail?: (text: string) => void;
}) {
  const description = toolDescription(block.input);
  const resultValue = block.output === undefined ? block.errorText : block.output;
  const hasResult = toolHasValue(resultValue) || Boolean(block.attachments?.length);
  const hasDetails = toolHasParameters(block.input) || hasResult;
  const { open, instant, rowRef, toggle } = useStickyDisclosure(block.status === "output-error");
  const parameters = useMemo(
    () => open ? toolParameters(block.input) : [],
    [block.input, open],
  );
  const result = useMemo(
    () => open ? toolDataText(resultValue) : "",
    [open, resultValue],
  );
  const parameterCopyText = useMemo(
    () => open ? toolParameterCopyText(block.input) : "",
    [block.input, open],
  );
  const [copiedDetail, setCopiedDetail] = useState<{
    section: "parameters" | "result";
    text: string;
  } | null>(null);
  useEffect(() => {
    if (copiedDetail === null) return;
    const timer = window.setTimeout(() => setCopiedDetail(null), 1600);
    return () => window.clearTimeout(timer);
  }, [copiedDetail]);
  const copyDetail = (section: "parameters" | "result", text: string) => {
    onCopyDetail?.(text);
    setCopiedDetail({ section, text });
  };
  // 成功是默认结果，不再逐行写"完成"；只有运行中、失败和耗时值得占位。
  const stateLabel = block.status === "input-available"
    ? "运行中"
    : block.status === "output-error"
      ? "失败"
      : block.durationMs === undefined
        ? ""
        : formatToolDuration(block.durationMs);

  return (
    <div
      className={`tool-step ${active ? "active" : ""} ${block.status === "output-error" ? "error" : ""}`}
    >
      <div className="tool-step-body">
        {hasDetails ? (
          <div ref={rowRef} className={`tool-step-sticky${open ? " is-open" : ""}`}>
            <button
              className="tool-step-summary"
              type="button"
              aria-expanded={open}
              onClick={toggle}
            >
              <ToolStepSummary
                block={block}
                description={description}
                stateLabel={stateLabel}
                expandable
                open={open}
              />
            </button>
          </div>
        ) : (
          <div className="tool-step-summary tool-step-summary-static">
            <ToolStepSummary
              block={block}
              description={description}
              stateLabel={stateLabel}
              expandable={false}
              open={false}
            />
          </div>
        )}
        {hasDetails ? (
          <Disclosure open={open} instant={instant}>
            <div className="tool-detail-surface">
                {parameters.length > 0 ? (
                  <section className="tool-detail-section" aria-label="工具参数">
                    <ToolDetailHeading
                      label="参数"
                      copied={copiedDetail?.section === "parameters" && copiedDetail.text === parameterCopyText}
                      onCopy={onCopyDetail ? () => copyDetail("parameters", parameterCopyText) : undefined}
                    />
                    <dl className="tool-parameter-list">
                      {parameters.map(([name, value]) => (
                        <div className="tool-parameter" key={name}>
                          <dt>{name}</dt>
                          <dd>{value}</dd>
                        </div>
                      ))}
                    </dl>
                  </section>
                ) : null}
                {open && hasResult ? (
                  <section className={`tool-detail-section${block.status === "output-error" ? " is-error" : ""}`} aria-label={block.status === "output-error" ? "工具错误" : "工具结果"}>
                    <ToolDetailHeading
                      label={block.status === "output-error" ? "错误" : "结果"}
                      copied={copiedDetail?.section === "result" && copiedDetail.text === result}
                      onCopy={onCopyDetail ? () => copyDetail("result", result) : undefined}
                    />
                    {toolHasValue(resultValue) ? <ToolResultContent value={resultValue} error={block.status === "output-error"}
                      tool={{ owner: block.owner, name: block.name, arguments: block.input }} /> : null}
                    {block.attachments?.length ? <SentAttachments items={block.attachments} /> : null}
                  </section>
                ) : null}
            </div>
          </Disclosure>
        ) : null}
      </div>
    </div>
  );
});

function ToolDetailHeading({
  label,
  copied,
  onCopy,
}: {
  label: string;
  copied: boolean;
  onCopy?: () => void;
}) {
  return (
    <div className="tool-detail-heading">
      <h4>{label}</h4>
      {onCopy ? (
        <button
          className={copied ? "tool-detail-copy copied" : "tool-detail-copy"}
          type="button"
          aria-label={copied ? `${label}已复制` : `复制工具${label}`}
          onClick={onCopy}
        >
          {copied ? <Check size={14} aria-hidden="true" /> : <Copy size={14} aria-hidden="true" />}
          <span aria-live="polite">{copied ? "已复制" : "复制"}</span>
        </button>
      ) : null}
    </div>
  );
}

// 展开后行尾提示可收起；吸顶时这一行就是随时可点的收起入口。
function CollapseHint({ open, lead = true }: { open: boolean; lead?: boolean }) {
  return open ? <span className="tool-step-collapse-hint">{lead ? " · 收起" : "收起"}</span> : null;
}

function ToolStepSummary({
  block,
  description,
  stateLabel,
  expandable,
  open,
}: {
  block: ToolBlock;
  description: ToolDescription;
  stateLabel: string;
  expandable: boolean;
  open: boolean;
}) {
  return (
    <span className="tool-step-heading">
      <span className="tool-step-title">
        <Wrench className="tool-step-icon" size={13} />
        <span>{block.name}</span>
      </span>
      <ToolDescriptionText description={description} />
      <span className="tool-step-state">{stateLabel}<CollapseHint open={open} lead={Boolean(stateLabel)} /></span>
      {expandable ? <ChevronDown className={`tool-step-chevron ${open ? "open" : ""}`} size={13} /> : null}
    </span>
  );
}

// 路径摘要放不下时先收目录、保住文件名："plugins/runtime_di…/profiling.py"，而不是把文件名截成"profilin…"。
// 路径保持一个文本节点（拆成多个 flex 子项时，选中复制会在目录与文件名之间插入换行）；
// 放不下时从左侧省略，离文件最近的目录与文件名留到最后。title 是工具参数里的原始路径。
function ToolDescriptionText({ description }: { description: ToolDescription }) {
  if (description.kind !== "path") return <span className="tool-step-description">{description.text}</span>;
  return (
    <span className="tool-step-description tool-step-description--path" title={description.full}>
      <bdi>{description.text}</bdi>
    </span>
  );
}

type ToolDescription = { kind: "text"; text: string } | { kind: "path"; text: string; full: string };

// 工具行摘要：优先模型写的 description；没有时取调用目标（文件、网址、检索词、命令），
// 不展开也知道读了哪个文件、搜了什么——只有工具名的一行等于没说。
function toolDescription(input: unknown): ToolDescription {
  if (typeof input !== "object" || input === null || Array.isArray(input)) {
    return { kind: "text", text: "" };
  }
  const args = input as Record<string, unknown>;
  const text = (name: string) => typeof args[name] === "string" ? (args[name] as string).trim() : "";
  if (text("description")) return { kind: "text", text: text("description") };
  if (text("path")) return { kind: "path", text: text("path").split(/[\\/]/u).filter(Boolean).slice(-3).join("/"), full: text("path") };
  if (text("url")) return { kind: "text", text: text("url").replace(/^https?:\/\//u, "") };
  if (text("query")) return { kind: "text", text: text("query") };
  return { kind: "text", text: text("command").split("\n")[0] ?? "" };
}

function toolParameters(input: unknown): [string, string][] {
  if (typeof input !== "object" || input === null || Array.isArray(input)) return [];
  return Object.entries(input as Record<string, unknown>)
    .filter(([name]) => name !== "description")
    .map(([name, value]) => [name, toolValue(value)]);
}

function toolParameterCopyText(input: unknown): string {
  if (typeof input !== "object" || input === null || Array.isArray(input)) return "";
  const parameters = Object.fromEntries(
    Object.entries(input as Record<string, unknown>).filter(([name]) => name !== "description"),
  );
  return JSON.stringify(parameters, null, 2);
}

function toolHasParameters(input: unknown): boolean {
  return typeof input === "object"
    && input !== null
    && !Array.isArray(input)
    && Object.keys(input).some((name) => name !== "description");
}

function toolHasValue(value: unknown): boolean {
  return value !== undefined;
}

function toolValue(value: unknown): string {
  if (value === undefined) return "";
  if (value === null) return "null";
  if (typeof value === "string") return value;
  if (typeof value === "number" || typeof value === "boolean") return String(value);
  return JSON.stringify(value, null, 2) ?? String(value);
}

