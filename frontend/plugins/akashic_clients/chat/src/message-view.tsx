import { chatPortalRoot } from "./chat-layout";
import { readToolIcon } from "./message-timeline";
import { PluginUiSlot, pluginSlotAccepts, pluginToolSummary, usePluginUiCatalogVersion } from "./plugin-ui-runtime";
import { ToolStepIcon } from "./tool-icon";
import type { ToolSummaryFacts, ToolTaskSummary } from "@akashic/chat-ui-v1";
import { ThinkingPlaceholder } from "./thinking-placeholder";
import { MentionText } from "./composer-suggest";
import { usePacedText } from "./stream-pace";
import { SentAttachments, type MediaKind, type SentAttachmentItem } from "./attachment-card";
import {
  Message,
  MessageContent,
} from "@/components/ai-elements/message";
import { detectMessageRenderingFeatures, messageNeedsMarkdown, readToolData, toolDataText } from "@/message-rendering-policy";
import {
  Reasoning,
  ReasoningTrigger,
  useReasoning,
} from "@/components/ai-elements/reasoning";
import {
  Dialog,
  DialogClose,
  DialogContent,
  DialogTitle,
} from "@/components/ui/dialog";
import { AlertCircle, Bot, Brain, Check, ChevronDown, ChevronLeft, ChevronRight, Copy, Download, ExternalLink, Gauge, MessagesSquare, Minimize2, Wrench, X, ZoomIn, ZoomOut } from "lucide-react";
import { useStickToBottomContext } from "use-stick-to-bottom";
import { MarqueeText } from "./marquee-text";
import { createPortal } from "react-dom";
import {
  Fragment,
  lazy,
  memo,
  Suspense,
  type ComponentType,
  type PointerEvent as ReactPointerEvent,
  type ReactNode,
  type TransitionEvent,
  useEffect,
  useId,
  useLayoutEffect,
  useRef,
  useCallback,
  useMemo,
  useState,
} from "react";
import type {
  AgentBlock,
  ChatMessage,
  MessageAttachment,
  ToolBlock,
} from "./chat-message";
import type { CompactionBoundaryValue, ModelAttempt, ReplyActivity, TimelineAttachment, TimelineMessage, TimelinePart } from "./message-timeline";
import { userInput, timelineReply, timelineTruncated, timelineToolOutput, historyTranscript, isCompactionBoundaryMessage, isTimelinePartVisible, needsBeforeReasoningFallback } from "./message-timeline";
import { MessageReplyReference } from "./message-actions";
import { StaticMessageResponse, richRendererFallback } from "./static-message-response";
import type { MessageResponseProps } from "@/components/ai-elements/message-response";
import { Disclosure, formatToolDuration, keepAt, prefersReducedMotion, ToolResultContent, useStickyDisclosure } from "./tool-result-view";
import { FindRegionCount, useFindRegion } from "./thread-find";
import { IconSwap } from "./icon-swap";

const LazyMessageResponse = lazy<ComponentType<MessageResponseProps>>(() =>
  import("@/components/ai-elements/message-response").then(({ MessageResponse }) => ({ default: MessageResponse }), richRendererFallback),
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
  if (!messageNeedsMarkdown(content)) {
    return <p className="plain-message-response">{content}</p>;
  }
  const features = detectMessageRenderingFeatures(content);
  // 打开历史会话时富文本分块可能还在下载：先用主包里的静态渲染排好标题、列表与代码块，不把 ##、``` 原文露给读者。
  // 含公式的仍放原文（静态渲染不保护公式，$a*b*c$ 会被当成强调改写）；流式也放原文（出字前已预热分块，
  // 静态的流式分段按空行切，会把围栏里的空行后半段切出代码块）。
  const fallback = streaming || features.math
    ? <p className="plain-message-response">{content}</p>
    : <StaticMessageResponse onError={onError}>{content}</StaticMessageResponse>;
  if (deferRichContent) {
    if (features.math || features.code) {
      return (
        <Suspense fallback={fallback}>
          <LazyMessageResponse isAnimating={streaming} paced={paced}>{content}</LazyMessageResponse>
        </Suspense>
      );
    }
    return <StaticMessageResponse streaming={streaming} onError={onError}>{content}</StaticMessageResponse>;
  }
  return (
    <Suspense fallback={fallback}>
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
  const flow = timelineFlow(processMessages, undefined, toolResults, activity.active);
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
  const [open, setOpen] = useState(false);
  const now = useClock(15000);
  const summary = `上游不稳定 · 第 ${attemptLabel(attempt)} 次尝试 · ${elapsedLabel(now / 1000 - attempt.failing_since)} · 上次：${attempt.last_failure}`;
  return <div className="attempt-status" role="status">
    <div className="attempt-status__details">
      {attempt.detail ? <>
        <button type="button" className="attempt-status__trigger" aria-expanded={open} onClick={() => setOpen((value) => !value)}>{summary}</button>
        <Disclosure open={open}><code>{attempt.detail}</code></Disclosure>
      </> : <span>{summary}</span>}
    </div>
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
    // 会话引用同理：只发引用时也要有一行说明，否则等待服务端确认期间是空气泡。
    const references = message.references?.length
      ? <div className="message-session-reference" aria-label="引用了会话">
          <MessagesSquare size={15} aria-hidden="true" />
          <span>引用 {message.references.length} 个会话：{message.references.map((item) => item.title || item.sessionId).join("、")}</span>
        </div>
      : null;
    return (
      <Message from="user" className="message-row user-row">
        {attachments}
        {references}
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
  | { kind: "compaction"; key: string; origin: TimelineMessage; index: number; part: TimelinePart; value: CompactionBoundaryValue }
  | { kind: "part"; key: string; origin: TimelineMessage; index: number; part: TimelinePart };

/** 组内全部成员按 part 顺序展开成独立流项：思考、工具、中途正文、末条正文各归原位。 */
function timelineFlow(messages: TimelineMessage[], finalId: string | undefined, toolResults: ReadonlyMap<string, TimelineMessage>, active = false): FlowItem[] {
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
            : block.kind === "text"
            ? [{ kind: "narration", key: `${key}:h${blockIndex}`, origin, index, part, text: block.content }]
            : [{ kind: "tool", key: `${key}:h${blockIndex}`, origin, index, part, block }])
          : [];
      }
      if (part.kind === "tool_call") {
        const result = toolResults.get(key);
        const outcome = result?.body.kind === "tool_result" ? result.body.outcome : null;
        return [{ kind: "tool", key, origin, index, part, block: {
          kind: "tool", callId: key, name: part.name, owner: part.owner, icon: readToolIcon(part.icon),
          label: typeof part.label === "string" && part.label.trim() ? part.label.trim() : undefined, input: part.arguments,
          output: result ? timelineToolOutput(result) : undefined,
          attachments: toolImages(result),
          status: outcome === null ? active ? "input-available" : "output-unknown" : outcome === "success" ? "output-available" : "output-error",
          errorText: outcome === null && !active ? "未收到工具回执；外部效果可能已经发生，请先检查状态。"
            : outcome && outcome !== "success" ? outcomeLabels[outcome] : undefined } }];
      }
      if (part.kind === "context.notice") return [{ kind: "notice", key, origin, index, part, text: part.value.text }];
      if (part.kind === "compaction.boundary") return [{ kind: "compaction", key, origin, index, part, value: part.value }];
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
  return part.kind === "context.notice" || part.kind === "compaction.boundary" ? 2 : 0;
}

function formatCompactTokenCount(tokens: number): string {
  if (tokens >= 10_000) return `${Math.round(tokens / 1000)}k`;
  if (tokens >= 1_000) return `${(tokens / 1000).toFixed(1).replace(/\.0$/u, "")}k`;
  return String(tokens);
}

const COMPACTION_TRIGGER_LABELS: Record<CompactionBoundaryValue["trigger"], string> = {
  idle: "空闲自动压缩",
  soft_limit: "容量软水位",
  context_overflow: "容量触顶",
};

interface CompactionAnchor {
  id: string;
  label: string;
  lineIndex: number;
}

function extractCompactionAnchors(lines: readonly string[]): CompactionAnchor[] {
  const anchors: CompactionAnchor[] = [];
  const seen = new Set<string>();
  const add = (raw: string, lineIndex: number) => {
    if (anchors.length >= 6) return;
    const label = raw.trim().replace(/^`|`$/g, "");
    if (!label || label.length < 3 || label.length > 36 || seen.has(label)) return;
    seen.add(label);
    anchors.push({ id: `${lineIndex}-${label}`, label, lineIndex });
  };
  for (let lineIndex = 0; lineIndex < lines.length && anchors.length < 6; lineIndex += 1) {
    const line = lines[lineIndex];
    const withoutUrls = line.replace(/https?:\/\/\S+/gu, " ");
    for (const match of line.matchAll(/`([^`\n]{3,36})`/gu)) add(match[1], lineIndex);
    for (const match of withoutUrls.matchAll(/(?:^|[\s(（【"'`])((?:\/[\w.-]+){2,})/gu)) add(match[1], lineIndex);
    for (const match of withoutUrls.matchAll(/(?:^|[^\w.])(\d+(?:\.\d+)?\s*(?:GiB|MiB|GB|MB|ms|s|px|tokens?|k|K|%))(?![\w.])/gu)) add(match[1], lineIndex);
    for (const match of withoutUrls.matchAll(/(\d+(?:\.\d+)?\s*(?:次|个|行))/gu)) add(match[1], lineIndex);
  }
  return anchors;
}

/** 压缩边界：对齐 Claude Desktop 的低调发丝线分隔条，默认收起一行，点开可看节省体积、摘要正文与边缘锚点。 */
function CompactionBoundaryRow({ boundary, standalone = false }: {
  boundary: CompactionBoundaryValue;
  standalone?: boolean;
}) {
  const [open, setOpen] = useState(false);
  const [activeAnchorId, setActiveAnchorId] = useState<string | null>(null);
  const previewRef = useRef<HTMLDivElement | null>(null);
  const lineRefs = useRef<(HTMLParagraphElement | null)[]>([]);
  const saved = Math.max(0, boundary.tokens_before - boundary.tokens_after);
  const ratio = boundary.tokens_before > 0 ? Math.round((saved / boundary.tokens_before) * 100) : 0;
  const verb = boundary.mode === "background" ? "已后台自动压缩" : "已压缩会话";
  const label = saved > 0
    ? `${verb} · 省下 ${formatCompactTokenCount(saved)} tokens`
    : boundary.tokens_before > 0
    ? `${verb} · 原 ${formatCompactTokenCount(boundary.tokens_before)} tokens`
    : verb;
  const triggerLabel = boundary.mode === "background"
    ? "后台空闲自动压缩"
    : (COMPACTION_TRIGGER_LABELS[boundary.trigger] ?? "自动压缩");
  const lines = useMemo(
    () => (boundary.summary_preview ?? "").split(/\r?\n/u).map((line) => line.trimEnd()).filter(Boolean),
    [boundary.summary_preview],
  );
  const anchors = useMemo(() => extractCompactionAnchors(lines), [lines]);
  const activeAnchor = anchors.find((anchor) => anchor.id === activeAnchorId) ?? null;
  const activeLine = activeAnchor?.lineIndex ?? null;
  const jumpToAnchor = (anchor: CompactionAnchor) => {
    setActiveAnchorId((current) => (current === anchor.id ? null : anchor.id));
    if (activeAnchorId !== anchor.id) {
      const preview = previewRef.current;
      const lineEl = lineRefs.current[anchor.lineIndex];
      if (preview && lineEl) {
        const targetTop = Math.max(0, lineEl.offsetTop - preview.offsetTop - Math.max(0, (preview.clientHeight - lineEl.offsetHeight) / 2));
        preview.scrollTo({ top: targetTop, behavior: "smooth" });
      }
    }
  };
  return <div className={`compaction-boundary${standalone ? " is-standalone" : ""}`} data-disclosure-root="" data-open={open || undefined}>
    <button type="button" className="compaction-boundary__trigger" data-disclosure-trigger="" aria-expanded={open}
      onClick={() => setOpen((current) => !current)}>
      <span className="compaction-boundary__pill">
        <Minimize2 className="compaction-boundary__icon" size={12} aria-hidden="true" />
        <span className="compaction-boundary__label">{label}</span>
        <ChevronDown className="compaction-boundary__chevron" size={12} aria-hidden="true" />
      </span>
    </button>
    <Disclosure open={open}>
      <div className="compaction-boundary__detail" role="status">
        <div className="compaction-boundary__meta-row">
          <p className="compaction-boundary__meta">
            {triggerLabel} · {boundary.tokens_before.toLocaleString()} → {boundary.tokens_after.toLocaleString()} tokens · 第 {boundary.generation} 代摘要
          </p>
          {ratio > 0 ? <span className="compaction-boundary__ratio">−{ratio}%</span> : null}
        </div>
        {lines.length ? (
          <div className={`compaction-boundary__grid${anchors.length ? " has-anchors" : ""}`}>
            <div ref={previewRef} className="compaction-boundary__preview">
              {lines.map((line, index) => (
                <p
                  key={index}
                  ref={(node) => { lineRefs.current[index] = node; }}
                  className={`compaction-boundary__line${activeLine === index ? " is-highlighted" : ""}`}
                >
                  {line}
                </p>
              ))}
            </div>
            {anchors.length ? (
              <aside className="compaction-boundary__anchors" aria-label="摘要保留的关键锚点">
                <span className="compaction-boundary__anchors-title">保留锚点 · {anchors.length}</span>
                <div className="compaction-boundary__anchors-list">
                  {anchors.map((anchor) => (
                    <button
                      key={anchor.id}
                      type="button"
                      className={`compaction-boundary__anchor${activeAnchorId === anchor.id ? " is-active" : ""}`}
                      aria-pressed={activeAnchorId === anchor.id}
                      onClick={() => jumpToAnchor(anchor)}
                    >
                      <span className="compaction-boundary__anchor-dot" aria-hidden="true" />
                      <span>{anchor.label}</span>
                    </button>
                  ))}
                </div>
              </aside>
            ) : null}
          </div>
        ) : null}
      </div>
    </Disclosure>
  </div>;
}

/** 上下文提示是本轮请求的诊断，与思考行同级：收起时一行摘要，展开看全文；压缩完成提示复用分隔条视觉。 */
function ContextNoticeRow({ text }: { text: string }) {
  const [open, setOpen] = useState(false);
  if (text.startsWith("已压缩会话")) {
    const paren = text.indexOf("（");
    const headline = paren > 0 ? text.slice(0, paren) : text;
    const detail = paren > 0 && text.endsWith("）") ? text.slice(paren + 1, -1) : text;
    return <div className="compaction-boundary" data-disclosure-root="" data-open={open || undefined}>
      <button type="button" className="compaction-boundary__trigger" data-disclosure-trigger="" aria-expanded={open}
        onClick={() => setOpen((current) => !current)}>
        <span className="compaction-boundary__pill">
          <Minimize2 className="compaction-boundary__icon" size={12} aria-hidden="true" />
          <span className="compaction-boundary__label">{headline}</span>
          <ChevronDown className="compaction-boundary__chevron" size={12} aria-hidden="true" />
        </span>
      </button>
      <Disclosure open={open}>
        <div className="compaction-boundary__detail" role="status">
          <p className="compaction-boundary__meta">{detail}</p>
        </div>
      </Disclosure>
    </div>;
  }
  return <div className="context-notice" data-disclosure-root="" data-open={open || undefined}>
    <button type="button" className="process-trigger" data-disclosure-trigger="" aria-expanded={open}
      onClick={() => setOpen((current) => !current)}>
      <Gauge className="process-trigger-icon" size={14} aria-hidden="true" />
      <span className="process-trigger-summary">{text}</span>
      <ChevronDown className={`process-chevron${open ? " open" : ""}`} size={14} aria-hidden="true" />
    </button>
    <Disclosure open={open}>
      <p className="process-content process-text" role="status">{text}</p>
    </Disclosure>
  </div>;
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
  const { version } = usePluginUiCatalogVersion();
  const task = item.kind === "tool" ? toolTaskSummary(item.block, version) : null;
  const content = item.kind === "thinking" ? <ThinkingRow content={item.content} streaming={false} />
    : item.kind === "tool" ? (task
      ? <TaskCard block={item.block} task={task} onCopyDetail={onCopyToolDetail} />
      : <ToolStep block={item.block} active={item.block.status === "input-available"} onCopyDetail={onCopyToolDetail} />)
    : item.kind === "notice" ? <ContextNoticeRow text={item.text} />
    : item.kind === "compaction" ? <CompactionBoundaryRow boundary={item.value} />
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

/** 过程按「步骤」折叠，思考也算一步：思考/shell/思考/read/思考/write 只留最后三步，
    前三个（含思考）收进一行可展开的计数。运行中的工具不参与折叠，保持逐行可见——
    注意力总在最近发生的事上。 */
const VISIBLE_STEP_TAIL = 3;

/** 运行中的工具不能收：它自己要保持可见，也切断折叠区间，避免把「正在跑」藏进计数里。
    耗时只汇总被折叠的那些，且要求它们全都有耗时——思考块没有耗时字段，只加部分成员的
    ms 会让「4 个更早步骤 120ms」读成整组只花了 120ms。 */
function groupProcessSteps<T>(
  items: T[],
  step: (item: T) => { foldable: boolean; ms?: number },
): (T | { members: T[]; durationMs?: number })[] {
  const result: (T | { members: T[]; durationMs?: number })[] = [];
  let run: { item: T; ms?: number }[] = [];
  const flush = () => {
    if (run.length > VISIBLE_STEP_TAIL) {
      const head = run.slice(0, run.length - VISIBLE_STEP_TAIL);
      const allTimed = head.every((entry) => entry.ms !== undefined);
      result.push({
        members: head.map((entry) => entry.item),
        durationMs: allTimed ? head.reduce((total, entry) => total + (entry.ms ?? 0), 0) : undefined,
      });
      result.push(...run.slice(-VISIBLE_STEP_TAIL).map((entry) => entry.item));
    } else {
      result.push(...run.map((entry) => entry.item));
    }
    run = [];
  };
  for (const item of items) {
    const info = step(item);
    if (!info.foldable) { flush(); result.push(item); continue; }
    run.push({ item, ms: info.ms });
  }
  flush();
  return result;
}

/** 收起的工具组：一行 caret + 计数，语序跟 Claude 的 "Show N earlier steps" 一致；
    展开后逐行恢复原行和详情，再收起时同一行回到原处。 */
function CollapsedToolGroup({ count, durationMs, findText, children }: {
  count: number;
  durationMs?: number;
  findText: () => string;
  children: ReactNode;
}) {
  const { open, sticky, rowRef, toggle, setOpen } = useStickyDisclosure();
  // 成员先进 DOM 或由 @starting-style 提供起始帧；悬停/聚焦提前挂载可省去点击当帧创建子树的开销。
  const [mounted, setMounted] = useState(false);
  const prefetch = () => setMounted(true);
  const matches = useFindRegion(rowRef, open, findText, (next) => { setMounted(true); setOpen(next); });
  return (
    <div className={`tool-group${sticky ? " is-open" : ""}`}>
      <div ref={rowRef} className={`tool-step-sticky${sticky ? " is-open" : ""}`}>
        <button
          type="button"
          className="tool-step-summary tool-group-summary"
          aria-expanded={open}
          onPointerEnter={prefetch}
          onFocus={prefetch}
          onClick={() => { setMounted(true); toggle(); }}
        >
          <span className="tool-step-heading">
            <span className="tool-step-title">
              {/* 箭头在前、计数在后：一眼看出这一行是折叠入口，而不是一条已完成记录。 */}
              <ChevronDown className={`tool-group-caret ${open ? "open" : ""}`} size={14} aria-hidden="true" />
              <span>{count} 个更早步骤</span>
            </span>
            <span className="tool-step-description"><FindRegionCount count={matches} /></span>
            <span className="tool-step-state">
              {durationMs !== undefined ? formatToolDuration(durationMs) : ""}
            </span>
          </span>
        </button>
      </div>
      <Disclosure open={open}>
        {mounted ? <div className="tool-group-items">{children}</div> : null}
      </Disclosure>
    </div>
  );
}

type ToolFlowItem = Extract<FlowItem, { kind: "tool" }>;
type TaskFlowMember = { item: ToolFlowItem; task: ToolTaskSummary };
type TaskBatchItem = { kind: "task-batch"; key: string; origin: TimelineMessage; members: TaskFlowMember[] };
type RenderFlowItem = FlowItem | TaskBatchItem;

/** 连续 2 个及以上的委派任务收成一组；单个任务仍是普通流项，照常参与"更早步骤"折叠。 */
function groupTaskBatches(flow: FlowItem[], version: number): RenderFlowItem[] {
  const result: RenderFlowItem[] = [];
  let batch: TaskFlowMember[] = [];
  const flush = () => {
    if (batch.length >= 2) {
      result.push({ kind: "task-batch", key: `task-batch-${batch[0].item.key}`, origin: batch[0].item.origin, members: batch });
    } else {
      result.push(...batch.map(({ item }) => item));
    }
    batch = [];
  };
  for (const item of flow) {
    const task = item.kind === "tool" ? toolTaskSummary(item.block, version) : null;
    if (item.kind === "tool" && task) {
      batch.push({ item, task });
    } else {
      flush();
      result.push(item);
    }
  }
  flush();
  return result;
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
  const { version } = usePluginUiCatalogVersion();
  const batched = groupTaskBatches(flow, version);
  const grouped = groupTools ? groupProcessSteps(batched, (item) => {
    // 思考也能收：它和工具一样是过程里的一步，展开后照样逐行恢复。
    if (item.kind === "thinking") return { foldable: true };
    if (item.kind === "tool") {
      return { foldable: item.block.status !== "input-available", ms: item.block.durationMs };
    }
    // 旁白、正文与委派任务组不是可折叠的步骤：它们切断折叠区间，自己原样留在流里。
    return { foldable: false };
  }) : batched;
  if (!flow.length && !draftThinking && !draftLead) return null;
  return <>
    <div className="process-trace">
      {grouped.map((item) => !("kind" in item) ? (
        <Fragment key={`toolgroup-${item.members[0].key}`}>
          {item.members.some((member) => member.key === slotKey) ? beforeReasoning?.(item.members[0].origin) : null}
          <CollapsedToolGroup count={item.members.length} durationMs={item.durationMs}
            findText={() => item.members.map((member) => member.kind === "task-batch" ? "" : flowItemFindText(member)).join("\n")}>
            {item.members.map((member) => member.kind === "task-batch" ? null : (
              <FlowItemView key={member.key} item={member} onCopyToolDetail={onCopyToolDetail} partView={partView} />
            ))}
          </CollapsedToolGroup>
        </Fragment>
      ) : item.kind === "task-batch" ? (
        <Fragment key={item.key}>
          {item.members.some(({ item: member }) => member.key === slotKey) ? beforeReasoning?.(item.origin) : null}
          <TaskBatchCard members={item.members} onCopyToolDetail={onCopyToolDetail} />
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

/** 结束后只收起连续的过程项，正文切断摘要区间，保持原来的阅读顺序。 */
function SettledTurnFlow({ flow, startedAt, endedAt, beforeReasoning, partView }: {
  flow: FlowItem[];
  startedAt?: string;
  endedAt: string;
  beforeReasoning?: (message: TimelineMessage) => ReactNode;
  partView: FlowPartViewProps;
}) {
  const isBody = (item: FlowItem) => item.kind === "part" || item.kind === "narration";
  const process = flow.filter((item) => !isBody(item));
  const running = process.some((item) => item.kind === "tool" && item.block.status === "input-available");
  const worthSummary = process.some((item) => item.kind === "tool") || process.length >= 2;
  if (running || !worthSummary) {
    return <TimelineProcess flow={flow} beforeReasoning={beforeReasoning} partView={partView} />;
  }
  // 1. 有末条回复时，它之前的过程连同中途旁白收成一行：读者要的是结论，旁白是工作记录。
  const answerStart = trailingAnswerStart(flow);
  if (answerStart > 0 && flow.slice(0, answerStart).some((item) => item.kind === "narration")) {
    const head = flow.slice(0, answerStart);
    return <>
      <TurnProcessSummary items={head} durationMs={turnDurationMs(startedAt, endedAt)} lead={beforeReasoning?.(process[0].origin)}>
        <TimelineProcess flow={head} groupTools={false} partView={partView} />
      </TurnProcessSummary>
      <TimelineProcess flow={flow.slice(answerStart)} partView={partView} />
    </>;
  }
  // 2. 没有末条回复（只剩旁白）时，正文不因后面还有工具调用而并入过程摘要。
  const groups: { body: boolean; items: FlowItem[] }[] = [];
  for (const item of flow) {
    const body = isBody(item);
    const previous = groups.at(-1);
    if (previous?.body === body) previous.items.push(item);
    else groups.push({ body, items: [item] });
  }
  const processGroups = groups.filter((group) => !group.body);
  const summarized = (group: { body: boolean; items: FlowItem[] }) => !group.body && (group.items.length > 1 || group.items[0].kind === "tool");
  // 3. 召回槽位只出现一次：流以摘要行开头时并进那一行，否则单独一行；整轮用时不能重复标在每段过程上。
  const recall = beforeReasoning?.(process[0].origin);
  const recallInline = summarized(groups[0]);
  return <>
    {recallInline ? null : recall}
    {groups.map((group, index) => !summarized(group)
      ? <TimelineProcess key={group.items[0].key} flow={group.items} partView={partView} />
      : <TurnProcessSummary key={group.items[0].key} items={group.items} lead={index === 0 && recallInline ? recall : undefined}
          durationMs={processGroups.length === 1 ? turnDurationMs(startedAt, endedAt) : undefined}>
          <TimelineProcess flow={group.items} groupTools={false} />
        </TurnProcessSummary>)}
  </>;
}

// 末条回复（finalId 的正文 part）在流尾连续出现；返回它的起点，没有则返回 -1。
function trailingAnswerStart(flow: FlowItem[]): number {
  let start = flow.length;
  while (start > 0 && flow[start - 1].kind === "part") start -= 1;
  return start < flow.length ? start : -1;
}

function turnDurationMs(startedAt: string | undefined, endedAt: string): number | undefined {
  const elapsed = startedAt ? Date.parse(endedAt) - Date.parse(startedAt) : Number.NaN;
  return Number.isFinite(elapsed) && elapsed >= 1_000 ? elapsed : undefined;
}

function formatTurnDuration(durationMs: number): string {
  const seconds = Math.round(durationMs / 1_000);
  return seconds < 60 ? `${seconds}s` : `${Math.floor(seconds / 60)}m ${seconds % 60}s`;
}

// 摘要写做了哪几类事（最多两类）：「读取文件、搜索网页等 4 步」比「调用 4 个工具」多说了内容。
function toolKindsLabel(tools: ToolBlock[]): string {
  const kinds = [...new Set(tools.map((block) => block.label ?? block.name))];
  const head = kinds.slice(0, 2).join("、") + (kinds.length > 2 ? "等" : "");
  return tools.length === 1 ? head : `${head} ${tools.length} 步`;
}

function TurnProcessSummary({ items, durationMs, lead, children }: {
  items: FlowItem[];
  durationMs?: number;
  /** 同一行开头的插件内容（本轮的回想）；它有自己的点按入口，不属于展开按钮。 */
  lead?: ReactNode;
  children: ReactNode;
}) {
  const { open, sticky, rowRef, toggle, setOpen } = useStickyDisclosure();
  // 成员先进 DOM 或由 @starting-style 提供起始帧；悬停/聚焦提前挂载可省去点击当帧创建子树的开销。
  const [mounted, setMounted] = useState(false);
  const prefetch = () => setMounted(true);
  const findText = useCallback(() => items.map(flowItemFindText).join("\n"), [items]);
  const matches = useFindRegion(rowRef, open, findText, (next) => { setMounted(true); setOpen(next); });
  const thinking = items.filter((item) => item.kind === "thinking").length;
  const tools = items.filter((item) => item.kind === "tool");
  const failed = tools.filter((item) => item.kind === "tool" && item.block.status === "output-error").length;
  const { version } = usePluginUiCatalogVersion();
  const allTasks = tools.length > 0 && tools.every((item) => item.kind === "tool" && toolTaskSummary(item.block, version) !== null);
  const label = tools.length
    ? (allTasks ? `委派 ${tools.length} 个任务` : toolKindsLabel(tools.flatMap((item) => item.kind === "tool" ? [item.block] : [])))
    : thinking ? `思考 ${thinking} 次` : "过程记录";
  const Icon = tools.length ? Wrench : Brain;
  return (
    <div className={`turn-process${sticky ? " is-open" : ""}`}>
      <div ref={rowRef} className={`tool-step-sticky${sticky ? " is-open" : ""}`}>
        {lead ? <div className="turn-process-lead">{lead}</div> : null}
        <button type="button" className="tool-step-summary turn-process-summary" aria-expanded={open}
          onPointerEnter={prefetch} onFocus={prefetch}
          onClick={() => { setMounted(true); toggle(); }}>
          <span className="tool-step-heading">
            <span className="tool-step-title">
              <Icon className="tool-step-icon" size={14} aria-hidden="true" />
              <span>{label}</span>
            </span>
            <span className="tool-step-description">{failed ? <em className="turn-process-failed">{failed} 个失败</em> : null}<FindRegionCount count={matches} /></span>
            <span className="tool-step-state">
              {durationMs !== undefined ? formatTurnDuration(durationMs) : ""}
            </span>
            <ChevronDown className={`tool-step-chevron ${open ? "open" : ""}`} size={14} aria-hidden="true" />
          </span>
        </button>
      </div>
      <Disclosure open={open}>
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
  const [open, setOpen] = useState(false);
  const [mounted, setMounted] = useState(false);
  const flow = timelineFlow(messages, undefined, toolResults).filter(item => item.kind === "tool" || item.kind === "thinking");
  const running = flow.filter(item => item.kind === "tool" && item.block.status === "input-available");
  const settled = flow.filter(item => !running.includes(item));
  const count = settled.filter(item => item.kind === "tool").length;
  return <div className="inline-reply-process">
    {settled.length ? <div className="context-notice inline-reply-notice" data-disclosure-root="" data-open={open || undefined}>
      <button type="button" className="process-trigger" data-disclosure-trigger="" aria-expanded={open}
        onPointerEnter={() => setMounted(true)} onFocus={() => setMounted(true)}
        onClick={() => { setMounted(true); setOpen((current) => !current); }}>
        {count ? <Wrench className="process-trigger-icon" size={14} aria-hidden="true" />
          : <Brain className="process-trigger-icon" size={14} aria-hidden="true" />}
        <span className="process-trigger-summary">{count ? `思考并调用 ${count} 个工具` : "思考过程"}</span>
        <ChevronDown className={`process-chevron${open ? " open" : ""}`} size={14} aria-hidden="true" />
      </button>
      <Disclosure open={open}>
        {mounted ? <TimelineProcess flow={settled} beforeReasoning={beforeReasoning} /> : null}
      </Disclosure>
    </div> : null}
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
  usePluginUiCatalogVersion();
  const body = message.body;
  if (isCompactionBoundaryMessage(message)) {
    return <div className="message-row compaction-row">
      {body.kind === "output" ? body.parts.map((part, index) => (
        !("display" in part) && part.kind === "compaction.boundary"
          ? <div key={index} data-part-index={index} tabIndex={-1}><CompactionBoundaryRow boundary={part.value} standalone /></div>
          : null
      )) : null}
    </div>;
  }
  // 未结束轮次仍可有正文；是否最终完成不改变文字的展示类别。
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
  const automatic = input && !userInput(message);
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
          .map(({ part, index }) => isTimelinePartVisible(part) && !(input && part.kind === "artifact_ref") && !((hideBody || (attachmentResult && !renderAttachment)) && part.kind === "text") && !((hideProcess || inlineReply) && (("display" in part && part.display === "thinking") || part.kind === "tool_call")) && !(hideProcess && part.kind === "history.transcript") ? <div key={index} data-part-index={index} tabIndex={-1}>
          <TimelinePartView part={part} toolResult={body.kind === "tool_result" && !attachmentResult} userInput={input} attachment={attachment} lookupMessage={lookupMessage} canLoadReferences={canLoadReferences}
            onNavigate={onNavigate} onError={onError} processStartContent={leadingContent} hideTranscriptProcess={inlineReply} />
        </div> : null)}</UserTextClamp>
      </>}
      {timelineTruncated(message) ? <p className="timeline-control-note">回答达到长度上限，已在此中断</p> : null}
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
  const collapsedHeightRef = useRef<number | null>(null);
  const pendingRef = useRef<{ height: number; expanding: boolean } | null>(null);
  const collapseFrameRef = useRef(0);
  const collapseTimerRef = useRef(0);
  const settleTimerRef = useRef(0);

  // 过渡中（内联 max-height）或展开态测不出“收起后是否溢出”，只在静止的收起态更新，否则收起途中按钮会闪退。
  const measureClamp = useCallback(() => {
    const element = box.current;
    if (!element || element.style.maxHeight || element.dataset.state === "open") return;
    setOverflowing(element.scrollHeight - element.clientHeight > 1);
    collapsedHeightRef.current = element.getBoundingClientRect().height;
  }, []);

  // 过渡收尾：清掉内联高度，回到 CSS 控制后重测一次溢出。
  const settleClamp = useCallback(() => {
    window.clearTimeout(settleTimerRef.current);
    settleTimerRef.current = 0;
    const element = box.current;
    if (!element) return;
    element.style.maxHeight = "";
    element.style.transition = "";
    measureClamp();
  }, [measureClamp]);

  const stopCollapseAnchor = useCallback(() => {
    window.cancelAnimationFrame(collapseFrameRef.current);
    window.clearTimeout(collapseTimerRef.current);
    collapseFrameRef.current = 0;
    collapseTimerRef.current = 0;
  }, []);

  useEffect(() => () => {
    stopCollapseAnchor();
    window.clearTimeout(settleTimerRef.current);
  }, [stopCollapseAnchor]);

  useLayoutEffect(() => {
    const element = box.current;
    const inner = content.current;
    if (!enabled || expanded || !element || !inner) return;
    // 明显不到 12 行的短输入不读几何、不挂观察器：补载历史时这些消息多在跳过渲染的区域里，逐条读 scrollHeight
    // 会逐条强制布局；历史消息带 content-visibility:auto，逐条观察会让长会话滚动多跑布局。
    if (!mayExceedClamp(element)) return;
    const observer = new ResizeObserver(measureClamp);
    observer.observe(element);
    observer.observe(inner);
    return () => observer.disconnect();
  }, [enabled, expanded, measureClamp]);

  useLayoutEffect(() => {
    const element = box.current;
    const inner = content.current;
    const before = pendingRef.current;
    if (!element || !inner || !before) return;
    pendingRef.current = null;
    if (prefersReducedMotion()) {
      element.style.maxHeight = "";
      element.style.transition = "";
      if (!before.expanding) toggle.current?.scrollIntoView({ block: "nearest" });
      return;
    }
    const targetHeight = before.expanding
      ? inner.getBoundingClientRect().height
      : (collapsedHeightRef.current ?? Math.min(before.height, Math.ceil((parseFloat(getComputedStyle(element).lineHeight) || 20) * 12)));
    element.style.transition = "none";
    element.style.maxHeight = `${before.height}px`;
    void element.offsetHeight;
    if (Math.abs(targetHeight - before.height) < 1) {
      settleClamp();
      return;
    }
    // motion-exempt: 长文本折叠的高度过渡：合成写法需要盖板与不透明底色，ADR 0166 已知限制
    element.style.transition = `max-height ${before.expanding ? "var(--ak-sys-duration-expand)" : "var(--ak-sys-duration-snap)"} var(--ak-sys-motion-standard)`;
    element.style.maxHeight = `${targetHeight}px`;
    // transitionend 可能不来（节点被隐藏、过渡被样式打断），到点兜底归位。
    window.clearTimeout(settleTimerRef.current);
    settleTimerRef.current = window.setTimeout(settleClamp, 400);
  }, [expanded, settleClamp]);

  if (!enabled) return <>{children}</>;
  const onToggle = () => {
    const element = box.current;
    const currentHeight = element?.getBoundingClientRect().height ?? 0;
    if (!expanded && element && !element.style.maxHeight) collapsedHeightRef.current = currentHeight;
    const nextExpanded = !expanded;
    pendingRef.current = { height: currentHeight, expanding: nextExpanded };
    stopCollapseAnchor();
    if (!nextExpanded && !prefersReducedMotion()) {
      const toggleBefore = toggle.current?.getBoundingClientRect().top ?? null;
      const tick = () => {
        if (toggleBefore !== null) keepAt(toggle.current, toggleBefore);
        collapseFrameRef.current = window.requestAnimationFrame(tick);
      };
      collapseFrameRef.current = window.requestAnimationFrame(tick);
      collapseTimerRef.current = window.setTimeout(stopCollapseAnchor, 220);
    }
    setExpanded(nextExpanded);
  };
  const onTransitionEnd = (event: TransitionEvent<HTMLDivElement>) => {
    if (event.target !== event.currentTarget || event.propertyName !== "max-height") return;
    stopCollapseAnchor();
    settleClamp();
  };
  return <>
    <div ref={box} className="user-text-clamp" data-state={expanded ? "open" : overflowing ? "clamped" : undefined} onTransitionEnd={onTransitionEnd}>
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
    // 内容到达撑开时直接到位，只有插件退出降级变矮才过渡：切回会话时附件异步长高，
    // 若也动画，整页会在 300ms 里每帧跟着追滚动。
    const measure = () => {
      const next = inner.getBoundingClientRect().height;
      outer.style.transition = next >= outer.getBoundingClientRect().height ? "none" : "";
      outer.style.height = `${next}px`;
    };
    measure();
    const observer = new ResizeObserver(measure);
    observer.observe(inner);
    return () => observer.disconnect();
  }, []);
  return <section ref={box} className="message-figure" aria-label="回复附件">
    <div ref={content}>
      <PluginUiSlot name="message.attachment" quiet sessionId={sessionId} messageId={messageId}
        block={{ ...attachment, origin }} fallback={fallback}
        errorFallback={<SentAttachments items={[timelineAttachmentItem(attachment)]} />} />
      <a className="message-figure-download" href={`/api/chat/artifacts/${encodeURIComponent(attachment.artifact_id)}`}
        download={attachment.filename ?? "附件"} aria-label={`下载源文件 ${attachment.filename ?? "附件"}`}><Download size={16} aria-hidden="true" /></a>
    </div>
  </section>;
}

// 用户的纯文本输入：已知技能的 /名称 显示成提及。
function UserPlainText({ text }: { text: string }) {
  return <p className="plain-message-response"><MentionText text={text} /></p>;
}

function TimelinePartView({ part, toolResult = false, userInput = false, attachment, lookupMessage, onNavigate, onError, processStartContent, canLoadReferences, hideTranscriptProcess = false }: {
  canLoadReferences: boolean;
  toolResult?: boolean;
  userInput?: boolean;
  processStartContent?: ReactNode;
  hideTranscriptProcess?: boolean;
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
  if ("archive" in part) return part.kind === "history.transcript" ? <TimelineTranscript archive={part.archive} hideProcess={hideTranscriptProcess} /> : null;
  switch (part.kind) {
    case "context.notice": return <ContextNoticeRow text={part.value.text} />;
    case "compaction.boundary": return <CompactionBoundaryRow boundary={part.value} />;
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
    case "session.reference": {
      const sessions = part.value.sessions;
      return <div className="message-session-reference" aria-label="引用了会话">
        <MessagesSquare size={15} aria-hidden="true" />
        <span>引用 {sessions.length} 个会话：{sessions.map((item) => item.title || item.session_id).join("、")}</span>
      </div>;
    }
    case "tool_call": return <TimelineToolCallRow name={part.name} arguments={part.arguments} />;
  }
}

function TimelineToolCallRow({ name, arguments: args }: { name: string; arguments: unknown }) {
  const [open, setOpen] = useState(false);
  return <div className="timeline-details timeline-tool-call" data-disclosure-root="" data-open={open || undefined}>
    <button type="button" className="timeline-details__trigger" data-disclosure-trigger="" aria-expanded={open}
      onClick={() => setOpen((current) => !current)}>
      <ChevronDown className={`timeline-details__chevron${open ? " open" : ""}`} size={14} aria-hidden="true" />
      <span>工具调用 · {name}</span>
    </button>
    <Disclosure open={open}>
      <pre tabIndex={0}>{JSON.stringify(args, null, 2)}</pre>
    </Disclosure>
  </div>;
}

/** 将归档转换成既有过程组件的展示块，原日志不变。 */
function historyBlocks(archive: unknown): AgentBlock[] {
  const groups = historyTranscript(archive);
  return groups?.flatMap((group, index): AgentBlock[] => [
    ...(group.thinking ? [{ kind: "thinking" as const, content: group.thinking }] : []),
    ...(group.text ? [{ kind: "text" as const, content: group.text }] : []),
    ...group.calls.map((call, callIndex) => ({ kind: "tool" as const,
      callId: `history-${index}-${callIndex}`, name: call.name, input: call.arguments,
      output: call.result, status: "output-available" as const, errorText: undefined })),
  ]) ?? [];
}

function TimelineTranscript({ archive, hideProcess = false }: { archive: unknown; hideProcess?: boolean }) {
  if (historyTranscript(archive) === null) return <p>这段历史过程暂无法展示。</p>;
  const blocks = historyBlocks(archive).filter((block) => !hideProcess || block.kind === "text");
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
        <IconSwap state={zoom.actual} a={<ZoomIn size={18} aria-hidden="true" />} b={<ZoomOut size={18} aria-hidden="true" />} />
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
  const grouped = groupProcessSteps(blocks.map((block, index) => ({ block, index })), (entry) =>
    entry.block.kind === "thinking"
      ? { foldable: true }
      : entry.block.kind === "tool"
      ? { foldable: entry.block.status !== "input-available", ms: entry.block.durationMs }
      : { foldable: false },
  );
  const renderBlock = (block: AgentBlock, index: number) => block.kind === "thinking" ? (
    <ThinkingRow
      content={block.content}
      streaming={streaming && index === lastThinkingIndex}
      duration={index === lastThinkingIndex && durationMs
        ? Math.max(1, Math.round(durationMs / 1000)) : undefined}
    />
  ) : block.kind === "text" ? (
    <div className="timeline-narration"><MessageBody content={block.content} streaming={streaming} deferRichContent /></div>
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
        <CollapsedToolGroup key={`toolgroup-${entry.members[0].index}`} count={entry.members.length} durationMs={entry.durationMs}
          findText={() => entry.members.map((member) => agentBlockFindText(member.block)).join("\n")}>
          {entry.members.map((member) => (
            <Fragment key={member.index}>
              {beforeBlock?.(member.block, member.index)}
              {renderBlock(member.block, member.index)}
            </Fragment>
          ))}
        </CollapsedToolGroup>
      ) : (
        <Fragment key={entry.block.kind === "tool" ? entry.block.callId : `${entry.block.kind}-${entry.index}`}>
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
  const { open, sticky, rowRef, setOpen } = useStickyDisclosure(false);
  const [mounted, setMounted] = useState(false);
  const prefetch = () => setMounted(true);
  const features = streaming ? null : detectMessageRenderingFeatures(content);
  const needsRichContent = streaming || features?.math || features?.code;
  const setFindOpen = useCallback((next: boolean) => {
    setMounted(true);
    setOpen(next);
  }, [setOpen]);
  return (
    // 流式时只在标题行滚动最新一句，不自动展开：展开会与标题行重复同一句，结束时行重挂载又会瞬间收起。
    // Render ordinary Markdown right away; math and code still need the rich renderer.
    // defaultOpen={false} 不能省：Reasoning 只凭它判断“显式关闭”，省了流式时会在渲染期自动展开。
    <Reasoning className="thinking-row" isStreaming={streaming} defaultOpen={false} open={open}
      onOpenChange={(next) => { setMounted(true); setOpen(next); }}
      data-sticky={sticky ? "true" : undefined} duration={duration}>
      <ThinkingRowTrigger content={streaming ? content : ""} findText={content} rowRef={rowRef}
        onPrefetch={prefetch} onFindOpen={setFindOpen} />
      <Disclosure open={open}>
        {mounted ? <div className="process-content" data-state={open ? "open" : "closed"}>
          <div className="process-text process-markdown">
            {needsRichContent ? <Suspense fallback={<span className="process-markdown-fallback">{content}</span>}>
              <LazyMessageResponse isAnimating={streaming}>{content}</LazyMessageResponse>
            </Suspense> : <StaticMessageResponse>{content}</StaticMessageResponse>}
          </div>
        </div> : null}
      </Disclosure>
    </Reasoning>
  );
});

function ThinkingRowTrigger({ content, findText, rowRef, onPrefetch, onFindOpen }: {
  content: string;
  findText: string;
  rowRef: React.RefObject<HTMLDivElement | null>;
  onPrefetch: () => void;
  onFindOpen: (open: boolean) => void;
}) {
  const { isOpen, isStreaming, duration } = useReasoning();
  const text = useCallback(() => findText, [findText]);
  const matches = useFindRegion(rowRef, isOpen, text, onFindOpen);
  return (
    <div ref={rowRef} className="process-trigger-sticky">
      <ReasoningTrigger className={`process-trigger ${isStreaming ? "is-streaming" : ""}`}
        onPointerEnter={onPrefetch} onFocus={onPrefetch}>
        <Brain className="process-trigger-icon" size={14} aria-hidden="true" />
        <span className="process-trigger-label">
          {isStreaming ? "正在思考" : duration ? `已思考 ${duration}s` : "已思考"}
        </span>
        {content ? <ThinkingTail content={content} /> : null}
        <FindRegionCount count={matches} />
        <ChevronDown className={`process-chevron ${isOpen ? "open" : ""}`} size={14} aria-hidden="true" />
      </ReasoningTrigger>
    </div>
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
  const hasDetails = toolHasDetails(block);
  // 失败也默认收起：行尾的红字已说明原因，整段参数与报错由读者自己点开。
  const { open, sticky, rowRef, toggle, setOpen } = useStickyDisclosure();
  const [mounted, setMounted] = useState(false);
  const prefetch = () => setMounted(true);
  const showDetails = mounted || open || sticky;
  const findText = useCallback(() => toolDetailFindText(block), [block]);
  const matches = useFindRegion(rowRef, open, findText, (next) => { setMounted(true); setOpen(next); });
  // 成功是默认结果，不再逐行写"完成"；运行中与失败常驻，耗时只在悬停时出现。
  const failure = block.status === "output-error" ? toolErrorHeadline(block) : "";
  const stateLabel = block.status === "input-available"
    ? "运行中"
    : block.status === "output-unknown"
      ? "状态未知"
    : block.status === "output-error"
      ? <span className="tool-step-error">{failure ? `失败 · ${failure}` : "失败"}</span>
      : block.durationMs === undefined
        ? ""
        : formatToolDuration(block.durationMs);
  const quietState = block.status === "output-available";

  return (
    <div
      className={`tool-step ${active ? "active" : ""} ${block.status === "output-error" ? "error" : ""}`}
    >
      <div className="tool-step-body">
        {hasDetails ? (
          <div ref={rowRef} className={`tool-step-sticky${sticky ? " is-open" : ""}`}>
            <button
              className="tool-step-summary"
              type="button"
              aria-expanded={open}
              onPointerEnter={prefetch}
              onFocus={prefetch}
              onClick={() => { setMounted(true); toggle(); }}
            >
              <ToolStepSummary
                block={block}
                description={description}
                stateLabel={stateLabel}
                quietState={quietState}
                expandable
                open={open}
                matches={matches}
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
          <Disclosure open={open}>
            {showDetails ? <ToolDetailSurface block={block} open={showDetails} onCopyDetail={onCopyDetail}
              omit={description.repeats} named /> : null}
          </Disclosure>
        ) : null}
      </div>
    </div>
  );
});

function toolHasDetails(block: ToolBlock): boolean {
  const resultValue = block.output === undefined ? block.errorText : block.output;
  return toolHasParameters(block.input) || toolHasValue(resultValue) || Boolean(block.attachments?.length);
}

// 展开后的参数与结果：工具行与任务卡共用；收起时不计算参数与结果文字。
// 标题行写过的参数只在它够短、且省掉后参数区还在时省略：长值要有处看全文，唯一参数也要留住复制入口。
function omitRepeated(parameters: [string, string][], omit: string | undefined): [string, string][] {
  const repeated = parameters.find(([name]) => name === omit);
  if (!repeated || parameters.length < 2 || repeated[1].length > 80) return parameters;
  return parameters.filter(([name]) => name !== omit);
}

function ToolDetailSurface({ block, open, onCopyDetail, className = "", omit, named = false }: {
  block: ToolBlock;
  open: boolean;
  onCopyDetail?: (text: string) => void;
  className?: string;
  /** 标题行已经完整写出的参数，展开后不再重复；复制仍给全部参数。 */
  omit?: string;
  /** 工具行显示的是动作短语时，在第一节小标题后补上工具原名。 */
  named?: boolean;
}) {
  const resultValue = block.output === undefined ? block.errorText : block.output;
  const hasResult = toolHasValue(resultValue) || Boolean(block.attachments?.length);
  const parameters = useMemo(() => open ? omitRepeated(toolParameters(block.input), omit) : [], [block.input, open, omit]);
  const errorMessage = useMemo(() => open && block.status === "output-error" ? toolErrorMessage(resultValue) : null, [block.status, open, resultValue]);
  const toolName = named ? block.name : undefined;
  const result = useMemo(() => open ? toolDataText(resultValue) : "", [open, resultValue]);
  const parameterCopyText = useMemo(() => open ? toolParameterCopyText(block.input) : "", [block.input, open]);
  const [copiedDetail, setCopiedDetail] = useState<{ section: "parameters" | "result"; text: string } | null>(null);
  useEffect(() => {
    if (copiedDetail === null) return;
    const timer = window.setTimeout(() => setCopiedDetail(null), 1600);
    return () => window.clearTimeout(timer);
  }, [copiedDetail]);
  const copy = (section: "parameters" | "result", text: string) => onCopyDetail ? () => {
    onCopyDetail(text);
    setCopiedDetail({ section, text });
  } : undefined;
  const error = block.status === "output-error";
  return (
    <div className={`tool-detail-surface${className ? ` ${className}` : ""}`}>
      {parameters.length > 0 ? (
        <section
          className="tool-detail-section"
          data-copied-highlight={copiedDetail?.section === "parameters" && copiedDetail.text === parameterCopyText ? "true" : undefined}
          aria-label="工具参数"
        >
          <ToolDetailHeading label="参数" meta={toolName} onCopy={copy("parameters", parameterCopyText)}
            copied={copiedDetail?.section === "parameters" && copiedDetail.text === parameterCopyText} />
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
        <section
          className={`tool-detail-section${error ? " is-error" : ""}`}
          data-copied-highlight={copiedDetail?.section === "result" && copiedDetail.text === result ? "true" : undefined}
          aria-label={error ? "工具错误" : "工具结果"}
        >
          <ToolDetailHeading label={error ? "错误" : "结果"} meta={parameters.length ? undefined : toolName}
            onCopy={copy("result", result)}
            copied={copiedDetail?.section === "result" && copiedDetail.text === result} />
          {errorMessage !== null ? <ToolResultContent value={errorMessage} error />
            : toolHasValue(resultValue) ? <ToolResultContent value={resultValue} error={error}
              tool={{ owner: block.owner, name: block.name, arguments: block.input }} /> : null}
          {block.attachments?.length ? <SentAttachments items={block.attachments} /> : null}
        </section>
      ) : null}
    </div>
  );
}

function ToolDetailHeading({
  label,
  meta,
  copied,
  onCopy,
}: {
  label: string;
  meta?: string;
  copied: boolean;
  onCopy?: () => void;
}) {
  return (
    <div className="tool-detail-heading">
      <h4>{label}{meta ? <span className="tool-detail-meta"> · {meta}</span> : null}</h4>
      {onCopy ? (
        <button
          className={copied ? "tool-detail-copy copied" : "tool-detail-copy"}
          type="button"
          aria-label={copied ? `${label}已复制` : `复制工具${label}`}
          onClick={onCopy}
        >
          <IconSwap state={copied} a={<Copy size={14} aria-hidden="true" />} b={<Check size={14} aria-hidden="true" />} />
          <span aria-live="polite">{copied ? "已复制" : "复制"}</span>
        </button>
      ) : null}
    </div>
  );
}

function ToolStepSummary({
  block,
  description,
  stateLabel,
  quietState = false,
  expandable,
  open,
  matches = 0,
}: {
  block: ToolBlock;
  description: ToolDescription;
  stateLabel: ReactNode;
  /** 只有耗时：悬停或展开时才出现，静止的过程行只留动作与对象。 */
  quietState?: boolean;
  expandable: boolean;
  open: boolean;
  matches?: number;
}) {
  return (
    <span className="tool-step-heading">
      <span className="tool-step-title">
        <ToolStepIcon block={block} className="tool-step-icon" size={14} />
        <span title={block.label ? block.name : undefined}>{block.label ?? block.name}</span>
      </span>
      <ToolDescriptionText description={description} />
      <span className={`tool-step-state${quietState ? " is-quiet" : ""}`}><FindRegionCount count={matches} />{stateLabel}</span>
      {expandable ? <ChevronDown className={`tool-step-chevron ${open ? "open" : ""}`} size={14} /> : null}
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

/** repeats：行里已经完整写出的参数名，展开后的参数表不再重复它。 */
type ToolDescription = ({ kind: "text"; text: string } | { kind: "path"; text: string; full: string }) & { repeats?: string };

// 工具行摘要：优先模型写的 description；没有时取调用目标（文件、网址、检索词、命令），
// 不展开也知道读了哪个文件、搜了什么——只有工具名的一行等于没说。
function toolDescription(input: unknown): ToolDescription {
  if (typeof input !== "object" || input === null || Array.isArray(input)) {
    return { kind: "text", text: "" };
  }
  const args = input as Record<string, unknown>;
  const text = (name: string) => typeof args[name] === "string" ? (args[name] as string).trim() : "";
  if (text("description")) return { kind: "text", text: text("description") };
  // 路径行只写末三段，完整路径在悬停提示里；网址只去掉协议。两者都算已写出。
  if (text("path")) return { kind: "path", text: text("path").split(/[\\/]/u).filter(Boolean).slice(-3).join("/"), full: text("path"), repeats: "path" };
  if (text("url")) return { kind: "text", text: text("url").replace(/^https?:\/\//u, ""), repeats: "url" };
  for (const name of ["query", "label", "title"]) if (text(name)) return { kind: "text", text: text(name), repeats: name };
  // 任务与命令只取首行：多行时展开后仍要看全文。
  const multiline = (name: string) => text(name).includes("\n") ? undefined : name;
  if (text("task")) return { kind: "text", text: firstLine(text("task")), repeats: multiline("task") };
  return { kind: "text", text: text("command").split("\n")[0] ?? "", repeats: text("command") ? multiline("command") : undefined };
}

const SUMMARY_STATUS: Record<ToolBlock["status"], ToolSummaryFacts["status"]> = {
  "input-available": "running",
  "output-available": "done",
  "output-error": "error",
  "output-unknown": "unknown",
};

// 同一次调用只在状态、结果或插件目录变化时重新询问 owner；参数对象来自消息，渲染之间保持同一引用。
const taskSummaryCache = new WeakMap<object, { status: ToolBlock["status"]; raw: unknown; version: number; task: ToolTaskSummary | null }>();

/** 工具 owner 把这次调用描述成委派任务时返回摘要；其他调用（含没有 owner 的归档调用）返回 null。 */
function toolTaskSummary(block: ToolBlock, version: number): ToolTaskSummary | null {
  if (!block.owner || typeof block.input !== "object" || block.input === null) return null;
  const raw = block.output === undefined ? block.errorText : block.output;
  const hit = taskSummaryCache.get(block.input);
  if (hit && hit.status === block.status && hit.raw === raw && hit.version === version) return hit.task;
  const task = pluginToolSummary({
    name: block.name, arguments: block.input, status: SUMMARY_STATUS[block.status], value: readToolData(raw),
  }, block.owner);
  taskSummaryCache.set(block.input, { status: block.status, raw, version, task });
  return task;
}

type TaskTone = "live" | "done" | "error" | "idle";

interface TaskView {
  title: string;
  tags: readonly string[];
  detail: string;
  cwd: string;
  preview: string;
  tone: TaskTone;
  stateLabel: string;
}

// 插件只给文字；状态缺省按调用回执，预览缺省取结果首行。
function taskView(block: ToolBlock, task: ToolTaskSummary): TaskView {
  const tone: TaskTone = task.state?.tone ?? (block.status === "input-available" ? "live"
    : block.status === "output-error" ? "error"
    : block.status === "output-available" ? "done"
    : "idle");
  const stateLabel = task.state?.label ?? (block.status === "input-available" ? "运行中"
    : block.status === "output-error" ? "失败"
    : block.status === "output-available" ? (block.durationMs !== undefined ? formatToolDuration(block.durationMs) : "完成")
    : "状态未知");
  // 插件给了预览就不再读结果：长结果在流式期间每帧都会重新渲染。
  const preview = task.preview?.trim() || (block.status === "input-available" ? "正在执行…"
    : firstLine(toolDataText(block.output === undefined ? block.errorText : block.output))
      || (block.status === "output-error" ? "执行失败" : ""));
  return {
    title: task.title.trim(),
    tags: task.tags ?? [],
    detail: task.detail?.trim() ?? "",
    cwd: task.cwd?.trim() ?? "",
    preview: preview.slice(0, 140),
    tone,
    stateLabel,
  };
}

function firstLine(text: string): string {
  for (let start = 0; start < text.length;) {
    const end = text.indexOf("\n", start);
    const line = text.slice(start, end < 0 ? text.length : end).trim();
    if (line) return line;
    if (end < 0) break;
    start = end + 1;
  }
  return "";
}

function TaskGlyph({ tone }: { tone: TaskTone }) {
  return (
    <span className={`delegated-glyph is-${tone}`} aria-hidden="true">
      {tone === "error" ? <AlertCircle size={12} /> : tone === "done" ? <Check size={12} /> : <Bot size={12} />}
    </span>
  );
}

// 同一时刻只留一张浮卡：鼠标停在一行、键盘焦点在另一行时，后出现的替换先出现的。
let activeHoverCard: ((anchor: DOMRect | null) => void) | null = null;
const HOVER_OPEN_MS = 280;
const HOVER_CLOSE_MS = 120;

// 悬停或键盘聚焦一段时间后浮出详情；指针可以移进浮卡，Esc、滚动、缩放窗口与离开都会收起。
function useTaskHoverCard() {
  const [anchor, setAnchor] = useState<DOMRect | null>(null);
  const timer = useRef<number | null>(null);
  const cancel = useCallback(() => {
    if (timer.current !== null) window.clearTimeout(timer.current);
    timer.current = null;
  }, []);
  const hide = useCallback(() => {
    cancel();
    if (activeHoverCard === setAnchor) activeHoverCard = null;
    setAnchor(null);
  }, [cancel]);
  const show = useCallback((el: HTMLElement) => {
    cancel();
    timer.current = window.setTimeout(() => {
      timer.current = null;
      if (!el.isConnected) return;
      if (activeHoverCard !== setAnchor) activeHoverCard?.(null);
      activeHoverCard = setAnchor;
      // 1. 在出现的这一刻量锚点：等待期间的滚动或布局变化不留下过期坐标
      setAnchor(el.getBoundingClientRect());
    }, HOVER_OPEN_MS);
  }, [cancel]);
  const leave = useCallback(() => {
    cancel();
    timer.current = window.setTimeout(hide, HOVER_CLOSE_MS);
  }, [cancel, hide]);
  useEffect(() => hide, [hide]);
  useEffect(() => {
    if (!anchor) return;
    const onKeyDown = (event: KeyboardEvent) => { if (event.key === "Escape") hide(); };
    window.addEventListener("scroll", hide, true);
    window.addEventListener("resize", hide);
    document.addEventListener("keydown", onKeyDown, true);
    return () => {
      window.removeEventListener("scroll", hide, true);
      window.removeEventListener("resize", hide);
      document.removeEventListener("keydown", onKeyDown, true);
    };
  }, [anchor, hide]);
  return { anchor, show, hide, leave, cancel };
}

function TaskHoverCard({ id, view, anchor, onPointerEnter, onPointerLeave }: {
  id: string;
  view: TaskView;
  anchor: DOMRect;
  onPointerEnter: () => void;
  onPointerLeave: () => void;
}) {
  const cardRef = useRef<HTMLDivElement | null>(null);
  const [pos, setPos] = useState<{ top: number; left: number } | null>(null);
  useLayoutEffect(() => {
    const el = cardRef.current;
    if (!el) return;
    // 1. 优先放在行上方；放不下时放到下方，并整体压回视口内
    const box = el.getBoundingClientRect();
    const edge = 8;
    const above = anchor.top - box.height - edge;
    const below = Math.min(window.innerHeight - box.height - edge, anchor.bottom + edge);
    const left = Math.min(Math.max(12, anchor.left + 16), Math.max(12, window.innerWidth - box.width - 12));
    setPos({ top: above >= edge ? above : Math.max(edge, below), left });
  }, [anchor]);
  return createPortal(
    <div ref={cardRef} id={id} className="delegated-hover-card" role="tooltip"
      onPointerEnter={onPointerEnter} onPointerLeave={onPointerLeave}
      style={pos ? { top: pos.top, left: pos.left } : { top: 0, left: 0, visibility: "hidden" }}>
      <div className="delegated-hover-card__head">
        <TaskGlyph tone={view.tone} />
        <span className="delegated-hover-card__title">{view.title}</span>
        <span className={`delegated-hover-card__status is-${view.tone}`}>{view.stateLabel}</span>
      </div>
      {view.detail || view.cwd || view.tags.length ? (
        <div className="delegated-hover-card__meta">
          {view.tags.map((tag) => <span key={tag} className="delegated-tag">{tag}</span>)}
          {view.detail ? <span>{view.detail}</span> : null}
          {view.cwd ? <code className="delegated-hover-card__cwd">{view.cwd}</code> : null}
        </div>
      ) : null}
      {view.preview ? <p className="delegated-hover-card__preview">{view.preview}</p> : null}
    </div>,
    chatPortalRoot(),
  );
}

function TaskRowContent({ view, open, expandable, matches }: { view: TaskView; open: boolean; expandable: boolean; matches: number }) {
  return (
    <span className="delegated-row-inner">
      <TaskGlyph tone={view.tone} />
      <span className="delegated-row-main">
        <span className="delegated-row-top">
          <span className="delegated-row-title">{view.title}</span>
          {view.tags.map((tag) => <span key={tag} className="delegated-tag">{tag}</span>)}
          {view.detail ? <span className="delegated-model">{view.detail}</span> : null}
        </span>
        {view.preview ? <span className={`delegated-row-preview is-${view.tone}`}>{view.preview}</span> : null}
      </span>
      <span className="delegated-row-meta">
        <FindRegionCount count={matches} />
        <span className={`delegated-state-label is-${view.tone}`}>{view.stateLabel}</span>
        {expandable ? <ChevronDown className={`tool-step-chevron ${open ? "open" : ""}`} size={14} aria-hidden="true" /> : null}
      </span>
    </span>
  );
}

const TaskCard = memo(function TaskCard({ block, task, member = false, onCopyDetail }: {
  block: ToolBlock;
  task: ToolTaskSummary;
  member?: boolean;
  onCopyDetail?: (text: string) => void;
}) {
  const view = taskView(block, task);
  const hasDetails = toolHasDetails(block);
  const { open, sticky, rowRef, toggle, setOpen } = useStickyDisclosure(block.status === "output-error");
  const findText = useCallback(() => toolDetailFindText(block), [block]);
  const matches = useFindRegion(rowRef, open, findText, setOpen);
  const tooltipId = useId();
  const hover = useTaskHoverCard();
  // 展开后详情已经在眼前，不再浮出摘要。
  const hoverProps = {
    onPointerEnter: (event: ReactPointerEvent<HTMLElement>) => { if (event.pointerType === "mouse" && !open) hover.show(event.currentTarget); },
    onPointerLeave: hover.leave,
  };
  const settled = view.tone === "done" && !open;
  const content = <TaskRowContent view={view} open={open} expandable={hasDetails} matches={matches} />;
  return (
    <div className={`delegated-card${member ? " is-member" : " is-single"}${view.tone === "error" ? " is-error" : ""}${settled ? " is-settled" : ""}`}>
      {hasDetails ? (
        <div ref={rowRef} className={`tool-step-sticky${sticky ? " is-open" : ""}`}>
          <button type="button" className="delegated-row-trigger" aria-expanded={open}
            aria-describedby={hover.anchor ? tooltipId : undefined} {...hoverProps}
            onFocus={(event) => { if (!open && event.currentTarget.matches(":focus-visible")) hover.show(event.currentTarget); }}
            onBlur={hover.hide}
            onClick={() => { hover.hide(); toggle(); }}>
            {content}
          </button>
        </div>
      ) : <div className="delegated-row-trigger is-static" {...hoverProps}>{content}</div>}
      {hover.anchor ? <TaskHoverCard id={tooltipId} view={view} anchor={hover.anchor}
        onPointerEnter={hover.cancel} onPointerLeave={hover.leave} /> : null}
      {hasDetails ? (
        <Disclosure open={open}>
          <ToolDetailSurface block={block} open={open || sticky} onCopyDetail={onCopyDetail} className="delegated-detail-surface" />
        </Disclosure>
      ) : null}
    </div>
  );
});

// 组头的计数按运行中、失败、完成、其他排列；运行中与完成用固定说法，失败与其他按卡片上的状态文字（如"已取消"）分别计数。
function taskBatchCounts(views: TaskView[]): { summary: string; live: number; failed: number } {
  const counts = new Map<string, number>();
  for (const tone of ["live", "error", "done", "idle"] as const) {
    for (const view of views.filter((item) => item.tone === tone)) {
      const label = tone === "live" ? "运行中" : tone === "done" ? "完成" : view.stateLabel;
      counts.set(label, (counts.get(label) ?? 0) + 1);
    }
  }
  return {
    summary: [...counts].map(([label, count]) => `${count} ${label}`).join(" · "),
    live: views.filter((view) => view.tone === "live").length,
    failed: views.filter((view) => view.tone === "error").length,
  };
}

function TaskBatchCard({ members, onCopyToolDetail }: {
  members: TaskFlowMember[];
  onCopyToolDetail?: (text: string) => void;
}) {
  const views = members.map(({ item, task }) => taskView(item.block, task));
  const counts = taskBatchCounts(views);
  // 有任务在跑或失败时默认展开；之后的开合只由读者决定，状态变化不把组弹开或收起。
  const { open, sticky, rowRef, toggle, setOpen } = useStickyDisclosure(counts.live > 0 || counts.failed > 0);
  const [mounted, setMounted] = useState(open);
  const prefetch = () => setMounted(true);
  const findText = useCallback(() => members.map(({ item }) => flowItemFindText(item)).join("\n"), [members]);
  const matches = useFindRegion(rowRef, open, findText, (next) => { setMounted(true); setOpen(next); });
  const settled = counts.live === 0 && counts.failed === 0 && !open;
  return (
    <div className={`delegated-group${sticky ? " is-open" : ""}${counts.failed ? " has-error" : ""}${settled ? " is-settled" : ""}`}>
      <div ref={rowRef} className={`tool-step-sticky${sticky ? " is-open" : ""}`}>
        <button type="button" className="delegated-group-summary" aria-expanded={open}
          onPointerEnter={prefetch} onFocus={prefetch} onClick={() => { setMounted(true); toggle(); }}>
          <span className="delegated-dot-stack" aria-hidden="true">
            {views.slice(0, 3).map((view, index) => <span key={members[index].item.key} className={`delegated-dot is-${view.tone}`} />)}
            {members.length > 3 ? <span className="delegated-dot-more">+{members.length - 3}</span> : null}
          </span>
          <span className="delegated-group-title">{members.length} 个委派任务</span>
          <span className={`delegated-group-counts${counts.live ? " is-live" : counts.failed ? " is-error" : ""}`}>{counts.summary}</span>
          <span className="delegated-group-right">
            <FindRegionCount count={matches} />
            <ChevronDown className={`tool-group-caret ${open ? "open" : ""}`} size={14} aria-hidden="true" />
          </span>
        </button>
      </div>
      <Disclosure open={open}>
        {mounted ? (
          <div className="delegated-group-members">
            {members.map(({ item, task }) => (
              <TaskCard key={item.key} block={item.block} task={task} member onCopyDetail={onCopyToolDetail} />
            ))}
          </div>
        ) : null}
      </Disclosure>
    </div>
  );
}

// 收起时查找要计数的文字：展开后会看到的参数与结果；整组收起时再加上每一步的摘要行。
function toolDetailFindText(block: ToolBlock): string {
  const parameters = toolParameters(block.input).map(([name, value]) => `${name} ${value}`);
  return [...parameters, toolDataText(block.output === undefined ? block.errorText : block.output)].join("\n");
}

function agentBlockFindText(block: AgentBlock): string {
  return block.kind === "tool"
    ? `${block.name} ${toolDescription(block.input).text}\n${toolDetailFindText(block)}` : block.content;
}

function flowItemFindText(item: FlowItem): string {
  return item.kind === "thinking" ? item.content
    : item.kind === "tool" ? agentBlockFindText(item.block)
    : item.kind === "narration" || item.kind === "notice" ? item.text : "";
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

// 报错常包在 JSON 里（{"error": "..."}）：取出给人读的那句；取不到时返回 null，照原样显示。
function toolErrorMessage(value: unknown): string | null {
  const data = readToolData(value);
  if (typeof data === "string") return data;
  if (data === null || typeof data !== "object" || Array.isArray(data)) return null;
  const record = data as Record<string, unknown>;
  for (const key of ["error", "message", "detail"]) {
    const item = record[key];
    if (typeof item === "string" && item.trim()) return item;
    if (item !== null && typeof item === "object" && typeof (item as Record<string, unknown>).message === "string") {
      return (item as Record<string, string>).message;
    }
  }
  return null;
}

// 失败行尾的一句原因：报错的首个非空行，长度交给样式省略。
function toolErrorHeadline(block: ToolBlock): string {
  const value = block.output === undefined ? block.errorText : block.output;
  const text = toolErrorMessage(value) ?? toolDataText(value);
  return text.split("\n").map((line) => line.trim()).find(Boolean)?.slice(0, 160) ?? "";
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

