import { PluginUiSlot } from "./plugin-ui-runtime";
import { ThinkingPlaceholder } from "./thinking-placeholder";
import {
  Attachment,
  AttachmentHoverCard,
  AttachmentHoverCardContent,
  AttachmentHoverCardTrigger,
  AttachmentPreview,
  Attachments,
  getAttachmentLabel,
  getMediaCategory,
} from "@/components/ai-elements/attachments";
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
  DialogContent,
  DialogTitle,
  DialogTrigger,
} from "@/components/ui/dialog";
import { Brain, Check, ChevronDown, Copy, Download, Gauge, ImageIcon, Wrench } from "lucide-react";
import {
  Fragment,
  lazy,
  memo,
  Suspense,
  type MouseEvent as ReactMouseEvent,
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
import type { ReplyActivity, TimelineAttachment, TimelineMessage, TimelinePart } from "./message-timeline";
import { timelineReply, timelineToolOutput, historyTranscript, isTimelinePartVisible, needsBeforeReasoningFallback } from "./message-timeline";
import { MessageReplyReference } from "./message-actions";
import { StaticMessageResponse } from "./static-message-response";
import { Disclosure, formatToolDuration, ToolResultContent, useStickyDisclosure } from "./tool-result-view";

const LazyMessageResponse = lazy(() =>
  import("@/components/ai-elements/message-response").then(({ MessageResponse }) => ({ default: MessageResponse })),
);

const MessageBody = memo(function MessageBody({
  content,
  streaming,
  deferRichContent,
  onError,
}: {
  content: string;
  streaming: boolean;
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
      <LazyMessageResponse isAnimating={streaming}>{content}</LazyMessageResponse>
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
  const beforeReasoning = (message: TimelineMessage) => needsBeforeReasoningFallback(message, inputStarts)
    ? <PluginUiSlot name="turn.before_reasoning" sessionId={message.session_id} messageId={message.id}
      block={{ source: message.source }} refreshToken={refreshToken} /> : null;
  return <div className="message-row agent-row reply-activity" data-reply-handle={activity.handle}
    data-preview-message-id={draft?.message_id} aria-busy={activity.active}>
    <div className="agent-content">
      <TimelineProcess flow={flow} draftThinking={draft?.thinking}
        beforeReasoning={beforeReasoning}
        beforePart={(part, index, message) => part.kind === "tool_call" && !("display" in part) ? <PluginUiSlot
          name="turn.before_tool" sessionId={message.session_id} messageId={message.id}
          block={{ ...part, message_id: message.id, part_index: index }} /> : null} />
      {activity.active && draft?.retry_status ? <RetryStatus text={draft.retry_status} onStop={onStop} /> : null}
      {activity.active && !draft?.retry_status && !text && !draft?.thinking && !flow.length ? <ThinkingPlaceholder /> : null}
      {text ? <MessageBody content={text} streaming={Boolean(draft?.text) && activity.active} deferRichContent onError={onError} /> : null}
    </div>
  </div>;
}

/** 重试与压缩进度共用一张状态卡：首行是状态，其余是原始原因，停止就在旁边。 */
function RetryStatus({ text, onStop }: { text: string; onStop?: () => void }) {
  const [headline, ...detail] = text.split("\n");
  return <div className="retry-status" role="status">
    <div className="retry-status__text">
      <strong>{headline}</strong>
      {detail.length ? <code>{detail.join("\n")}</code> : null}
    </div>
    {onStop ? <button type="button" onClick={onStop}>停止</button> : null}
  </div>;
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
      ? <MessageAttachments attachments={message.attachments} />
      : null;
  if (message.role === "user") {
    return (
      <Message from="user" className="message-row user-row">
        <MessageContent className="user-bubble">
          {leadingContent}
          {attachments}
          {message.content ? (
            <MessageBody
              content={message.content}
              streaming={message.streaming === true}
              deferRichContent={deferRichContent}
              onError={onError}
            />
          ) : null}
        </MessageContent>
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
      if ("display" in part) return [];
      const key = `${origin.id}:${index}`;
      if (part.kind === "model.facts") {
        return part.value.thinking
          ? [{ kind: "thinking", key, origin, index, part, content: part.value.thinking }] : [];
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
          kind: "tool", callId: key, name: part.name, input: part.arguments,
          output: result ? timelineToolOutput(result) : undefined,
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

function flowRank(part: TimelinePart): number {
  if ("display" in part) return 0;
  return part.kind === "context.notice" ? 2 : part.kind === "model.facts" ? 1 : 0;
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

function FlowItemView({ item, beforePart, onCopyToolDetail, partView }: {
  item: FlowItem;
  beforePart?: (part: TimelinePart, index: number, message: TimelineMessage) => ReactNode;
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
      {beforePart?.(item.part, item.index, item.origin)}
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
function TimelineProcess({ flow, draftThinking = "", beforeReasoning, beforePart, onCopyToolDetail, partView }: {
  flow: FlowItem[];
  draftThinking?: string;
  beforeReasoning?: (message: TimelineMessage) => ReactNode;
  beforePart?: (part: TimelinePart, index: number, message: TimelineMessage) => ReactNode;
  onCopyToolDetail?: (text: string) => void;
  partView?: FlowPartViewProps;
}) {
  // 槽位锚在首个过程项之前；纯文本输出没有过程项时退到流开头，仍由提交消息提供上下文。
  const slotIndex = flow.findIndex((item) => item.kind === "thinking" || item.kind === "tool");
  const slotKey = (slotIndex >= 0 ? flow[slotIndex] : flow[0])?.key ?? null;
  const grouped = groupToolRuns(flow,
    (item) => item.kind === "tool" && item.block.status !== "input-available" ? item.block : null);
  if (!flow.length && !draftThinking) return null;
  return <>
    <div className="process-trace">
      {grouped.map((item) => "members" in item ? (
        <Fragment key={`toolgroup-${item.members[0].key}`}>
          {item.members.some((member) => member.key === slotKey) ? beforeReasoning?.(item.members[0].origin) : null}
          <CollapsedToolGroup count={item.members.length} durationMs={item.durationMs}>
            {item.members.map((member) => <FlowItemView key={member.key} item={member}
              beforePart={beforePart} onCopyToolDetail={onCopyToolDetail} partView={partView} />)}
          </CollapsedToolGroup>
        </Fragment>
      ) : (
        <Fragment key={item.key}>
          {item.key === slotKey ? beforeReasoning?.(item.origin) : null}
          <FlowItemView item={item} beforePart={beforePart} onCopyToolDetail={onCopyToolDetail} partView={partView} />
        </Fragment>
      ))}
      {draftThinking ? <ThinkingRow content={draftThinking} streaming /> : null}
    </div>
  </>;
}

/** 一轮结束后，思考、工具与中途旁白收进一行摘要；正文、附件和仍在运行的工具留在外面。 */
function SettledTurnFlow({ flow, startedAt, endedAt, beforeReasoning, beforePart, partView }: {
  flow: FlowItem[];
  startedAt?: string;
  endedAt: string;
  beforeReasoning?: (message: TimelineMessage) => ReactNode;
  beforePart?: (part: TimelinePart, index: number, message: TimelineMessage) => ReactNode;
  partView: FlowPartViewProps;
}) {
  const process = flow.filter((item) => item.kind !== "part");
  const running = process.some((item) => item.kind === "tool" && item.block.status === "input-available");
  const worthSummary = process.some((item) => item.kind === "tool") || process.length >= 2;
  if (running || !worthSummary) {
    return <TimelineProcess flow={flow} beforeReasoning={beforeReasoning} beforePart={beforePart} partView={partView} />;
  }
  // 1. 召回等插件槽位属于本轮开头，留在摘要外保持可见。
  // 2. 摘要内复用逐行过程视图，展开后与运行时看到的一致。
  return <>
    {beforeReasoning?.(process[0].origin)}
    <TurnProcessSummary items={process} durationMs={turnDurationMs(startedAt, endedAt)}>
      <TimelineProcess flow={process} beforePart={beforePart} />
    </TurnProcessSummary>
    <TimelineProcess flow={flow.filter((item) => item.kind === "part")} beforePart={beforePart} partView={partView} />
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
function InlineReplyProcess({ messages, toolResults, beforeReasoning, beforePart }: {
  messages: TimelineMessage[]; toolResults: ReadonlyMap<string, TimelineMessage>;
  beforeReasoning?: (message: TimelineMessage) => ReactNode;
  beforePart?: (part: TimelinePart, index: number, message: TimelineMessage) => ReactNode;
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
      <TimelineProcess flow={settled} beforeReasoning={beforeReasoning} beforePart={beforePart} />
    </details> : null}
    <TimelineProcess flow={running} beforeReasoning={beforeReasoning} beforePart={beforePart} />
  </div>;
}

/** 保留消息引用与 part 位置，复用原聊天的过程和正文组件。 */
export function TimelineMessageView({ message, lookupMessage, toolResults, onNavigate, onError, beforeReasoning, beforePart, afterBody, renderAttachment, hideBody = false, processMessages = [message], hideProcess = false, inlineReply = false, leadingProcess, canLoadReferences = false, turnStartedAt }: {
  message: TimelineMessage;
  /** 本轮用户输入的时间；摘要行据此给出整轮用时。 */
  turnStartedAt?: string;
  hideBody?: boolean;
  canLoadReferences?: boolean;
  toolResults: ReadonlyMap<string, TimelineMessage>;
  renderAttachment?: (attachment: TimelineAttachment) => ReactNode;
  beforeReasoning?: (message: TimelineMessage) => ReactNode;
  processMessages?: TimelineMessage[];
  hideProcess?: boolean;
  inlineReply?: boolean;
  leadingProcess?: TimelineMessage[];
  beforePart?: (part: TimelinePart, index: number, message: TimelineMessage) => ReactNode;
  afterBody?: ReactNode;
  lookupMessage: (id: string) => TimelineMessage | undefined;
  onNavigate: (id: string, partIndex?: number) => void;
  onError?: (error: unknown) => void;
}) {
  const body = message.body;
  const flow = hideProcess || inlineReply ? [] : timelineFlow(processMessages, message.id, toolResults);
  const hasFlow = body.kind === "output" && flow.length > 0;
  const leadingContent = hideProcess || hasFlow || inlineReply ? null : beforeReasoning?.(message);
  const referencedArtifacts = new Set(body.kind === "control" ? [] : body.parts.flatMap((part) =>
    !("display" in part) && part.kind === "artifact_ref" ? [part.value] : []));
  const origin = attachmentOrigin(message, lookupMessage);
  // 单文件成功结果作为回复插图；原始工具数据仍可从过程行查看。
  const attachmentResult = body.kind === "tool_result" && body.outcome === "success"
    && message.attachments.length === 1 && message.attachments[0].kind === "file"
    && body.parts.every(part => !("display" in part) && (part.kind === "text" || part.kind === "artifact_ref"));
  const fallback = attachmentResult ? <div className="attachment-description">{body.parts.map((part, index) =>
    part.kind === "text" && !("display" in part) ? <MessageBody key={index} content={part.value}
      streaming={false} deferRichContent onError={onError} /> : null)}
    {!body.parts.some(part => part.kind === "text" && !("display" in part) && part.value.trim())
      ? <p>{message.attachments[0].filename ?? "附件"}</p> : null}</div> : null;
  const attachment = (id: string) => {
    const ref = message.attachments.find(item => item.artifact_id === id)!;
    if (renderAttachment) return renderAttachment(ref);
    if (attachmentResult) return <MessageAttachmentFigure attachment={ref} origin={origin}
      sessionId={message.session_id} messageId={message.id} fallback={fallback} />;
    return <MessageAttachmentItem attachment={{ id: ref.artifact_id, type: "file",
      filename: ref.filename ?? "附件", mediaType: ref.media_type ?? "application/octet-stream",
      url: `/api/chat/artifacts/${encodeURIComponent(ref.artifact_id)}` }} />;
  };
  return <div className={`message-row timeline-message timeline-${body.kind}`}>
    <div className={body.kind === "input" ? "user-bubble" : "agent-content"}>
      {leadingContent}
      {leadingProcess ? <InlineReplyProcess messages={leadingProcess} toolResults={toolResults}
        beforeReasoning={beforeReasoning} beforePart={beforePart} /> : null}
      {hasFlow ? <SettledTurnFlow flow={flow} startedAt={turnStartedAt} endedAt={message.timestamp}
        beforeReasoning={beforeReasoning} beforePart={beforePart}
        partView={{ attachment, lookupMessage, onNavigate, onError, canLoadReferences }} /> : null}
      {body.kind === "control" ? <div className="timeline-control-summary">
        <strong>{controlLabels[body.action]}</strong>
        {body.reason !== null ? <p className="plain-message-response">{body.reason}</p> : null}
      </div> : hasFlow ? null : <>
        {body.kind === "tool_result" && !attachmentResult ? <div className="timeline-result-heading">
          <strong>工具结果 · {outcomeLabels[body.outcome]}</strong>
          <button type="button" onClick={() => onNavigate(body.call_ref.message_id, body.call_ref.part_index)}
            disabled={!lookupMessage(body.call_ref.message_id)}>
            {lookupMessage(body.call_ref.message_id) ? "查看调用" : "调用不在当前记录中"}
          </button>
        </div> : null}
        {body.parts.map((part, index) => ({ part, index })).sort((left, right) =>
          flowRank(right.part) + Number(right.part.kind === "history.transcript")
          - flowRank(left.part) - Number(left.part.kind === "history.transcript"))
          .map(({ part, index }) => isTimelinePartVisible(part) && !((hideBody || (attachmentResult && !renderAttachment)) && part.kind === "text") && !((hideProcess || inlineReply) && (part.kind === "model.facts" || part.kind === "tool_call" || part.kind === "history.transcript")) ? <div key={index} data-part-index={index} tabIndex={-1}>
          {beforePart?.(part, index, message)}
          <TimelinePartView part={part} toolResult={body.kind === "tool_result" && !attachmentResult} attachment={attachment} lookupMessage={lookupMessage} canLoadReferences={canLoadReferences}
            onNavigate={onNavigate} onError={onError} processStartContent={leadingContent} />
        </div> : null)}
      </>}
      {afterBody}
      {message.attachments.filter((ref) => !referencedArtifacts.has(ref.artifact_id)).map((ref, index) =>
        <Fragment key={`${ref.artifact_id}:${index}`}>{attachment(ref.artifact_id)}</Fragment>)}
    </div>
  </div>;
}

export interface TimelineAttachmentOrigin {
  kind: TimelineMessage["body"]["kind"];
  author: string;
  tool: string | null;
}

// 插件按来源决定是否认领附件；工具结果附带产生它的工具名，缺少调用记录时为 null。
function attachmentOrigin(message: TimelineMessage, lookupMessage: (id: string) => TimelineMessage | undefined): TimelineAttachmentOrigin {
  const body = message.body;
  if (body.kind !== "tool_result") return { kind: body.kind, author: message.author, tool: null };
  const call = lookupMessage(body.call_ref.message_id);
  const part = call?.body.kind === "control" ? undefined : call?.body.parts[body.call_ref.part_index];
  return { kind: body.kind, author: message.author, tool: part?.kind === "tool_call" && !("display" in part) ? part.name : null };
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

function TimelinePartView({ part, toolResult = false, attachment, lookupMessage, onNavigate, onError, processStartContent, canLoadReferences }: {
  canLoadReferences: boolean;
  toolResult?: boolean;
  processStartContent?: ReactNode;
  part: TimelinePart;
  attachment: (id: string) => ReactNode;
  lookupMessage: (id: string) => TimelineMessage | undefined;
  onNavigate: (id: string, partIndex?: number) => void;
  onError?: (error: unknown) => void;
}) {
  if ("display" in part) return part.display === "data"
    ? <ToolResultContent value={"rendered" in part ? part.rendered : part.value} />
    : <p className="timeline-state">无法展示此内容</p>;
  if ("archive" in part) return part.kind === "history.transcript" ? <TimelineTranscript archive={part.archive} startContent={processStartContent} onError={onError} /> : null;
  switch (part.kind) {
    case "context.notice": return <ContextNoticeRow text={part.value.text} />;
    case "text": return toolResult ? <ToolResultContent value={"rendered" in part ? part.rendered : part.value} /> : <MessageBody content={part.value} streaming={false} deferRichContent onError={onError} />;
    case "artifact_ref": return attachment(part.value);
    case "reply_ref": {
      const source = lookupMessage(part.value);
      return <MessageReplyReference author={source?.author ?? "原消息"}
        preview={source ? timelineReply(source).preview : ""} unavailable={!source} canLoad={canLoadReferences}
        onNavigate={() => onNavigate(part.value)} />;
    }
    case "model.facts": return part.value.thinking ? <ProcessTrace
      blocks={[{ kind: "thinking", content: part.value.thinking }]} streaming={false}
      interrupted={false} startContent={processStartContent} /> : null;
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

const MessageAttachments = memo(function MessageAttachments({ attachments }: { attachments: MessageAttachment[] }) {
  return (
    <Attachments className="message-attachments" variant="grid">
      {attachments.map((attachment) => (
        <MessageAttachmentItem attachment={attachment} key={attachment.id} />
      ))}
    </Attachments>
  );
});

function MessageAttachmentItem({ attachment }: { attachment: MessageAttachment }) {
  const isImage = getMediaCategory(attachment) === "image" && attachment.url;
  const label = getAttachmentLabel(attachment);

  if (isImage) {
    return <MessageContentImage attachment={attachment} label={label} />;
  }

  return (
    <AttachmentHoverCard>
      <AttachmentHoverCardTrigger asChild>
        <a className="message-attachment-download" href={attachment.url} download={label} aria-label={`下载 ${label}`}>
          <Attachment className="message-attachment-chip is-file" data={attachment}>
            <AttachmentPreview />
            <span>{label}</span>
          </Attachment>
        </a>
      </AttachmentHoverCardTrigger>
      <AttachmentHoverCardContent>
        <AttachmentHover attachment={attachment} />
      </AttachmentHoverCardContent>
    </AttachmentHoverCard>
  );
}

function MessageContentImage({
  attachment,
  label,
}: {
  attachment: MessageAttachment;
  label: string;
}) {
  const [failed, setFailed] = useState(false);

  if (failed || !attachment.url) {
    return (
      <div className="message-attachment-chip is-broken" title={label}>
        <ImageIcon size={18} aria-hidden="true" />
        <span>{label}</span>
        <small>无法预览</small>
      </div>
    );
  }

  return (
    <Dialog>
      <DialogTrigger asChild>
        <button type="button" className="message-content-image" title="点击预览图片">
          <img
            alt={label}
            className="message-content-image__media"
            src={attachment.url}
            loading="eager"
            decoding="async"
            onError={() => setFailed(true)}
          />
        </button>
      </DialogTrigger>
      <DialogContent className="image-preview-dialog">
        <DialogTitle className="sr-only">{label}</DialogTitle>
        <img alt={label} className="image-preview-full" src={attachment.url} />
      </DialogContent>
    </Dialog>
  );
}

function AttachmentHover({ attachment }: { attachment: MessageAttachment }) {
  const category = getMediaCategory(attachment);
  const label = getAttachmentLabel(attachment);

  return (
    <div className="attachment-hover">
      {category === "image" && attachment.url ? (
        <img alt={label} className="attachment-hover-image" src={attachment.url} />
      ) : (
        <div className="attachment-hover-file">
          <Attachment data={attachment}>
            <AttachmentPreview />
          </Attachment>
        </div>
      )}
      <div className="attachment-hover-title">{label}</div>
      {attachment.mediaType ? (
        <div className="attachment-hover-type">{attachment.mediaType}</div>
      ) : null}
    </div>
  );
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
  const latestLine = streaming
    ? content.split("\n").map((line) => line.trim()).filter(Boolean).at(-1) ?? ""
    : "";
  return (
    // 流式时只在标题行滚动最新一句，不自动展开：展开会与标题行重复同一句，结束时行重挂载又会瞬间收起。
    <Reasoning className="thinking-row" isStreaming={streaming} defaultOpen={false} duration={duration}>
      <div className="process-trigger-sticky"><ThinkingRowTrigger summary={latestLine} /></div>
      <CollapsibleContent className="process-content">
        <div className="process-text process-markdown">
          <Suspense fallback={<span className="process-markdown-fallback">{content}</span>}>
            <LazyMessageResponse isAnimating={streaming}>{content}</LazyMessageResponse>
          </Suspense>
        </div>
      </CollapsibleContent>
    </Reasoning>
  );
});

function ThinkingRowTrigger({ summary }: { summary: string }) {
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
      {summary ? <span className="process-trigger-summary">{summary}</span> : null}
      <ChevronDown className={`process-chevron ${isOpen ? "open" : ""}`} size={13} aria-hidden="true" />
    </ReasoningTrigger>
  );
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
  const hasDetails = toolHasParameters(block.input) || toolHasValue(resultValue);
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
                {open && toolHasValue(resultValue) ? (
                  <section className="tool-detail-section" aria-label={block.status === "output-error" ? "工具错误" : "工具结果"}>
                    <ToolDetailHeading
                      label={block.status === "output-error" ? "错误" : "结果"}
                      copied={copiedDetail?.section === "result" && copiedDetail.text === result}
                      onCopy={onCopyDetail ? () => copyDetail("result", result) : undefined}
                    />
                    <ToolResultContent value={resultValue} error={block.status === "output-error"} />
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
  description: string;
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
      <span className="tool-step-description">{description}</span>
      <span className="tool-step-state">{stateLabel}<CollapseHint open={open} lead={Boolean(stateLabel)} /></span>
      {expandable ? <ChevronDown className={`tool-step-chevron ${open ? "open" : ""}`} size={13} /> : null}
    </span>
  );
}

function toolDescription(input: unknown) {
  if (typeof input !== "object" || input === null || Array.isArray(input)) {
    return "";
  }

  const description = (input as Record<string, unknown>).description;
  return typeof description === "string" ? description.trim() : "";
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

