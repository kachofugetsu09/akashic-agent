import { MobilePluginSlot } from "./mobile-plugin-runtime";
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
import { detectMessageRenderingFeatures, messageNeedsMarkdown } from "@/message-rendering-policy";
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
import { Check, ChevronDown, ChevronUp, Copy, ImageIcon, Wrench } from "lucide-react";
import {
  Fragment,
  lazy,
  memo,
  Suspense,
  type ReactNode,
  useEffect,
  useLayoutEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import type {
  AgentBlock,
  ChatMessage,
  MessageAttachment,
  ThinkingBlock,
  ToolBlock,
} from "./chat-message";
import type { ReplyActivity, TimelineAttachment, TimelineMessage, TimelinePart } from "./message-timeline";
import { timelineReply, timelineText, historyTranscript, isTimelinePartVisible } from "./message-timeline";
import { MessageReplyReference } from "./message-actions";
import { StaticMessageResponse } from "./static-message-response";

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
    if (features.math || features.mermaid || features.code) {
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
export function ReplyActivityView({ activity, committed, onError, processMessages = [], toolResults = new Map() }: {
  activity: ReplyActivity;
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
  const process = timelineProcessBlocks(processMessages, toolResults);
  const latest = processMessages.at(-1);
  const text = draft?.text || (latest ? timelineText(latest) : "");
  const beforeReasoning = (message: TimelineMessage, prefetch = false) => <MobilePluginSlot name="turn.before_reasoning"
    sessionId={message.session_id} messageId={message.id} prefetch={prefetch} />;
  return <div className="message-row agent-row reply-activity" data-reply-handle={activity.handle}
    data-preview-message-id={draft?.message_id} aria-busy={activity.active}>
    <div className="agent-content">
      <TimelineProcess process={process} draftThinking={draft?.thinking} streaming={activity.active}
        beforeReasoning={beforeReasoning}
        prefetchReasoning={(message) => beforeReasoning(message, true)}
        draftSlot={draft ? <MobilePluginSlot name="turn.before_reasoning" sessionId={activity.session_id}
          messageId={draft.message_id} block={{ source: activity.source }} /> : undefined}
        beforePart={(part, index, message) => part.kind === "tool_call" && !("display" in part) ? <MobilePluginSlot
          name="turn.before_tool" sessionId={message.session_id} messageId={message.id}
          block={{ ...part, message_id: message.id, part_index: index }} /> : null} />
      {!draft && !text && !process.length ? <ThinkingPlaceholder /> : null}
      {text ? <MessageBody content={text} streaming={Boolean(draft?.text) && activity.active} deferRichContent onError={onError} /> : null}
    </div>
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

interface TimelineProcessBlock {
  origin: TimelineMessage;
  part: TimelinePart;
  index: number;
  block: AgentBlock;
}

/** 每条消息内先展示思考，再按原消息顺序连接全部过程。 */
function timelineProcessBlocks(messages: TimelineMessage[], toolResults: ReadonlyMap<string, TimelineMessage>): TimelineProcessBlock[] {
  return messages.flatMap((origin) => origin.body.kind !== "output" ? [] : origin.body.parts.map((part, index) => ({ part, index }))
    .sort((left, right) => Number(right.part.kind === "model.facts") - Number(left.part.kind === "model.facts"))
    .flatMap(({ part, index }): TimelineProcessBlock[] => {
    if ("display" in part) return [];
    if (part.kind === "model.facts" && part.value.thinking) return [{ index, origin, part, block: { kind: "thinking", content: part.value.thinking } }];
    if ("archive" in part && part.kind === "history.transcript") {
      return historyBlocks(part.archive).map((block) => ({ index, origin, part, block }));
    }
    if (part.kind !== "tool_call") return [];
    const result = toolResults.get(`${origin.id}:${index}`);
    const outcome = result?.body.kind === "tool_result" ? result.body.outcome : null;
    return [{ index, origin, part, block: { kind: "tool", callId: `${origin.id}:${index}`, name: part.name,
      input: part.arguments, output: result ? timelineText(result) : undefined,
      status: outcome === null ? "input-available" : outcome === "success" ? "output-available" : "output-error",
      errorText: outcome && outcome !== "success" ? outcomeLabels[outcome] : undefined } }];
  }));
}

/** 历史与实时回复共用一条轨迹，节点继续引用原消息和 part。 */
function TimelineProcess({ process, streaming = false, draftThinking = "", draftSlot, beforeReasoning, prefetchReasoning, beforePart }: {
  process: TimelineProcessBlock[];
  streaming?: boolean;
  draftThinking?: string;
  draftSlot?: ReactNode;
  beforeReasoning?: (message: TimelineMessage) => ReactNode;
  prefetchReasoning?: (message: TimelineMessage) => ReactNode;
  beforePart?: (part: TimelinePart, index: number, message: TimelineMessage) => ReactNode;
}) {
  const blocks: AgentBlock[] = process.map((item) => item.block);
  if (draftThinking) blocks.push({ kind: "thinking", content: draftThinking });
  if (!blocks.length && !draftSlot) return null;
  return <>
    {!streaming && process.length ? prefetchReasoning?.(process[0].origin) : null}
    <ProcessTrace blocks={blocks} streaming={streaming} interrupted={false}
    startContent={process.length ? beforeReasoning?.(process[0].origin) : draftSlot}
    beforeBlock={(_block, index) => {
      const item = process[index];
      if (!item) return null;
      return <div data-process-message-id={item.origin.id} data-part-index={item.index} tabIndex={-1}>
        {beforePart?.(item.part, item.index, item.origin)}
      </div>;
    }} />
  </>;
}

/** 保留消息引用与 part 位置，复用原聊天的过程和正文组件。 */
export function TimelineMessageView({ message, lookupMessage, toolResults, onNavigate, onError, beforeReasoning, prefetchReasoning, beforePart, afterBody, renderAttachment, hideBody = false, processMessages = [message], hideProcess = false, canLoadReferences = false }: {
  message: TimelineMessage;
  hideBody?: boolean;
  canLoadReferences?: boolean;
  toolResults: ReadonlyMap<string, TimelineMessage>;
  renderAttachment?: (attachment: TimelineAttachment) => ReactNode;
  beforeReasoning?: (message: TimelineMessage) => ReactNode;
  prefetchReasoning?: (message: TimelineMessage) => ReactNode;
  processMessages?: TimelineMessage[];
  hideProcess?: boolean;
  beforePart?: (part: TimelinePart, index: number, message: TimelineMessage) => ReactNode;
  afterBody?: ReactNode;
  lookupMessage: (id: string) => TimelineMessage | undefined;
  onNavigate: (id: string, partIndex?: number) => void;
  onError?: (error: unknown) => void;
}) {
  const body = message.body;
  const process = hideProcess ? [] : timelineProcessBlocks(processMessages, toolResults);
  const leadingContent = hideProcess ? null : beforeReasoning?.(process[0]?.origin ?? message);
  const referencedArtifacts = new Set(body.kind === "control" ? [] : body.parts.flatMap((part) =>
    !("display" in part) && part.kind === "artifact_ref" ? [part.value] : []));
  const attachment = (id: string) => {
    const ref = message.attachments.find((item) => item.artifact_id === id)!;
    if (renderAttachment) return renderAttachment(ref);
    return <MessageAttachmentItem attachment={{ id: ref.artifact_id, type: "file",
      filename: ref.filename ?? "附件", mediaType: ref.media_type ?? "application/octet-stream",
      url: `/api/chat/artifacts/${encodeURIComponent(ref.artifact_id)}` }} />;
  };
  return <div className={`message-row timeline-message timeline-${body.kind}`}>
    <div className={body.kind === "input" ? "user-bubble" : "agent-content"}>
      {process.length === 0 ? leadingContent : null}
      {body.kind === "output" && process.length ? <TimelineProcess process={process}
        beforeReasoning={beforeReasoning} prefetchReasoning={prefetchReasoning} beforePart={beforePart} /> : null}
      {body.kind === "control" ? <div className="timeline-control-summary">
        <strong>{controlLabels[body.action]}</strong>
        {body.reason !== null ? <p className="plain-message-response">{body.reason}</p> : null}
      </div> : <>
        {body.kind === "tool_result" ? <div className="timeline-result-heading">
          <strong>工具结果 · {outcomeLabels[body.outcome]}</strong>
          <button type="button" onClick={() => onNavigate(body.call_ref.message_id, body.call_ref.part_index)}
            disabled={!lookupMessage(body.call_ref.message_id)}>
            {lookupMessage(body.call_ref.message_id) ? "查看调用" : "调用不在当前记录中"}
          </button>
        </div> : null}
        {body.parts.map((part, index) => ({ part, index })).sort((left, right) =>
          Number(right.part.kind === "model.facts" || right.part.kind === "history.transcript")
          - Number(left.part.kind === "model.facts" || left.part.kind === "history.transcript"))
          .map(({ part, index }) => isTimelinePartVisible(part) && !(hideBody && part.kind === "text") && !((hideProcess && (part.kind === "model.facts" || part.kind === "tool_call" || part.kind === "history.transcript")) || process.some((item) => item.origin.id === message.id && item.index === index)) ? <div key={index} data-part-index={index} tabIndex={-1}>
          {beforePart?.(part, index, message)}
          <TimelinePartView part={part} attachment={attachment} lookupMessage={lookupMessage} canLoadReferences={canLoadReferences}
            onNavigate={onNavigate} onError={onError} processStartContent={leadingContent} />
        </div> : null)}
      </>}
      {afterBody}
      {message.attachments.filter((ref) => !referencedArtifacts.has(ref.artifact_id)).map((ref, index) =>
        <Fragment key={`${ref.artifact_id}:${index}`}>{attachment(ref.artifact_id)}</Fragment>)}
    </div>
  </div>;
}

function TimelinePartView({ part, attachment, lookupMessage, onNavigate, onError, processStartContent, canLoadReferences }: {
  canLoadReferences: boolean;
  processStartContent?: ReactNode;
  part: TimelinePart;
  attachment: (id: string) => ReactNode;
  lookupMessage: (id: string) => TimelineMessage | undefined;
  onNavigate: (id: string, partIndex?: number) => void;
  onError?: (error: unknown) => void;
}) {
  if ("display" in part) return <p className="timeline-state">无法展示此内容</p>;
  if ("archive" in part) return part.kind === "history.transcript" ? <TimelineTranscript archive={part.archive} startContent={processStartContent} onError={onError} /> : null;
  switch (part.kind) {
    case "text": return <MessageBody content={part.value} streaming={false} deferRichContent onError={onError} />;
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
        <Attachment className="message-attachment-chip is-file" data={attachment}>
          <AttachmentPreview />
          <span>{label}</span>
        </Attachment>
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
  const processItemsRef = useRef<HTMLDivElement>(null);
  const processLineRef = useRef<HTMLDivElement>(null);
  const processFlowRef = useRef<HTMLDivElement>(null);
  let activeBlockIndex = streaming ? blocks.length - 1 : -1;
  blocks.forEach((block, index) => {
    if (block.kind === "tool" && block.status === "input-available") activeBlockIndex = index;
  });

  useLayoutEffect(() => {
    const items = processItemsRef.current;
    const line = processLineRef.current;
    const flow = processFlowRef.current;
    if (!items || !line || !flow) return;

    // 1. 只在节点结构变化时定位起点；正文增长由 CSS bottom 自动延伸。
    const firstNode = items.querySelector<HTMLElement>(".process-item .process-node");
    const firstItem = firstNode?.closest<HTMLElement>(".process-item");
    if (!firstNode || !firstItem) return;
    const lineTop = firstItem.offsetTop + firstNode.offsetTop + firstNode.offsetHeight / 2;
    line.style.top = `${lineTop}px`;

    // 2. 活动段从上一个节点延伸到当前活动内容末端（相对 items，以适配框内滚动）。
    const processItems = Array.from(items.querySelectorAll<HTMLElement>(".process-item"));
    const activeItemIndex = processItems.findIndex((item) => item.classList.contains("active"));
    if (activeItemIndex < 0) {
      flow.dataset.active = "false";
      return;
    }
    const activeItem = processItems[activeItemIndex];
    const frontierItem = processItems[Math.max(0, activeItemIndex - 1)];
    const frontierNode = frontierItem.querySelector<HTMLElement>(".process-node");
    if (!frontierNode) return;
    const flowTop = frontierItem.offsetTop + frontierNode.offsetTop + frontierNode.offsetHeight / 2;
    const flowBottom = activeItem.offsetTop + activeItem.offsetHeight;
    flow.style.top = `${flowTop}px`;
    flow.style.bottom = `${Math.max(0, items.offsetHeight - flowBottom)}px`;
    flow.dataset.active = "true";
  }, [activeBlockIndex, blocks.length]);

  return (
    <Reasoning
      className="process-trace"
      isStreaming={streaming}
      defaultOpen={streaming}
      duration={durationMs ? Math.max(1, Math.round(durationMs / 1000)) : undefined}
    >
      <ProcessTraceTrigger interrupted={interrupted} />
      <CollapsibleContent className="process-content">
        <div className="process-panel">
          <div className="process-panel-body">
            <div className="process-items" ref={processItemsRef}>
              <div className="process-line" aria-hidden="true" ref={processLineRef} />
              <div className="process-flow" aria-hidden="true" data-active="false" ref={processFlowRef} />
              {startContent}
              {blocks.map((block, index) => (
                <Fragment key={block.kind === "thinking" ? `thinking-${index}` : block.callId}>
                  {beforeBlock?.(block, index)}
                  {block.kind === "thinking" ? (
                    <ThinkingStep
                      block={block}
                      active={streaming && index === blocks.length - 1}
                      origin={index === 0}
                    />
                  ) : (
                    <ToolStep
                      block={block}
                      active={block.status === "input-available"}
                      origin={index === 0}
                      onCopyDetail={onCopyToolDetail}
                    />
                  )}
                </Fragment>
              ))}
            </div>
          </div>
          <ProcessTraceCollapse />
        </div>
      </CollapsibleContent>
    </Reasoning>
  );
});

function ProcessTraceTrigger({ interrupted }: { interrupted: boolean }) {
  const { isOpen, isStreaming, duration } = useReasoning();
  const label = interrupted
    ? `已中止${duration ? ` · ${duration}s` : ""}`
    : isStreaming
      ? "正在思考"
      : `已思考${duration ? ` ${duration}s` : ""}`;

  return (
    <ReasoningTrigger className="process-trigger">
      <span>{label}</span>
      <ChevronDown className={`process-chevron ${isOpen ? "open" : ""}`} size={15} aria-hidden="true" />
    </ReasoningTrigger>
  );
}

function ProcessTraceCollapse() {
  const { setIsOpen } = useReasoning();

  return (
    <button
      type="button"
      className="process-collapse"
      aria-label="收起思考过程"
      onClick={(event) => {
        setIsOpen(false);
        const trigger = event.currentTarget
          .closest(".process-trace")
          ?.querySelector<HTMLElement>(".process-trigger");
        trigger?.focus({ preventScroll: true });
      }}
    >
      <ChevronUp size={14} aria-hidden="true" />
      <span>收起</span>
    </button>
  );
}

const ThinkingStep = memo(function ThinkingStep({
  block,
  active,
  origin,
}: {
  block: ThinkingBlock;
  active: boolean;
  origin: boolean;
}) {
  return (
    <div className={`process-item thinking-step ${active ? "active" : ""} ${origin ? "trace-origin" : ""}`}>
      <span className="process-node circle" />
      <div className="process-text process-markdown">
        <Suspense fallback={<span className="process-markdown-fallback">{block.content}</span>}>
          <LazyMessageResponse isAnimating={active}>{block.content}</LazyMessageResponse>
        </Suspense>
      </div>
    </div>
  );
});

const ToolStep = memo(function ToolStep({
  block,
  active,
  origin,
  onCopyDetail,
}: {
  block: ToolBlock;
  active: boolean;
  origin: boolean;
  onCopyDetail?: (text: string) => void;
}) {
  const description = toolDescription(block.input);
  const resultValue = block.status === "output-error" ? block.errorText : block.output;
  const hasDetails = toolHasParameters(block.input) || toolHasValue(resultValue);
  const [open, setOpen] = useState(false);
  const parameters = useMemo(
    () => open ? toolParameters(block.input) : [],
    [block.input, open],
  );
  const result = useMemo(
    () => open ? toolValue(resultValue) : "",
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
  const stateLabel = block.status === "input-available"
    ? "运行中"
    : block.status === "output-error"
      ? "失败"
      : block.durationMs === undefined
        ? "完成"
        : `完成 · ${formatToolDuration(block.durationMs)}`;

  return (
    <div
      className={`process-item tool-step ${active ? "active" : ""} ${origin ? "trace-origin" : ""} ${block.status === "output-error" ? "error" : ""}`}
    >
      <span className="process-node diamond" />
      <div className="tool-step-body">
        {hasDetails ? (
          <button
            className="tool-step-summary"
            type="button"
            aria-expanded={open}
            onClick={() => setOpen((current) => !current)}
          >
            <ToolStepSummary
              block={block}
              description={description}
              stateLabel={stateLabel}
              expandable
              open={open}
            />
          </button>
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
          <div
            className={`tool-step-disclosure ${open ? "open" : ""}`}
            aria-hidden={!open}
            inert={!open ? true : undefined}
          >
            <div className="tool-step-disclosure-inner">
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
                {result ? (
                  <section className="tool-detail-section" aria-label={block.status === "output-error" ? "工具错误" : "工具结果"}>
                    <ToolDetailHeading
                      label={block.status === "output-error" ? "错误" : "结果"}
                      copied={copiedDetail?.section === "result" && copiedDetail.text === result}
                      onCopy={onCopyDetail ? () => copyDetail("result", result) : undefined}
                    />
                    <pre className={block.status === "output-error" ? "tool-result error" : "tool-result"}>{result}</pre>
                  </section>
                ) : null}
              </div>
            </div>
          </div>
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
    <>
          <span className="tool-step-heading">
            <span className="tool-step-title">
              <Wrench className="tool-step-icon" size={14} />
              <span>{block.name}</span>
            </span>
            <span className="tool-step-state">{stateLabel}</span>
            {expandable ? <ChevronDown className={`tool-step-chevron ${open ? "open" : ""}`} size={15} /> : null}
          </span>
          {description ? <span className="tool-step-description">{description}</span> : null}
    </>
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
  return value !== undefined && value !== null && (typeof value !== "string" || value.length > 0);
}

function toolValue(value: unknown): string {
  if (value === undefined || value === null) return "";
  if (typeof value === "string") return value;
  if (typeof value === "number" || typeof value === "boolean") return String(value);
  return JSON.stringify(value, null, 2) ?? String(value);
}

function formatToolDuration(durationMs: number): string {
  if (durationMs < 1_000) return `${Math.max(1, Math.round(durationMs))}ms`;
  return `${(durationMs / 1_000).toFixed(durationMs < 10_000 ? 1 : 0).replace(/\.0$/, "")}s`;
}
