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
  MessageResponse,
} from "@/components/ai-elements/message";
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
import { Check, ChevronDown, Copy, Wrench } from "lucide-react";
import { Fragment, type ReactNode, useEffect, useMemo, useState } from "react";
import type {
  AgentBlock,
  ChatMessage,
  MessageAttachment,
  ThinkingBlock,
  ToolBlock,
} from "./main";

export function ChatMessageView({
  message,
  leadingContent,
  attachmentContent,
  processStartContent,
  beforeProcessBlock,
  answerEndContent,
  onCopyToolDetail,
}: {
  message: ChatMessage;
  leadingContent?: ReactNode;
  attachmentContent?: ReactNode;
  processStartContent?: ReactNode;
  beforeProcessBlock?: (block: AgentBlock, index: number) => ReactNode;
  answerEndContent?: ReactNode;
  onCopyToolDetail?: (text: string) => void;
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
          {message.content ? <MessageResponse>{message.content}</MessageResponse> : null}
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
            message={message}
            startContent={processStartContent}
            beforeBlock={beforeProcessBlock}
            onCopyToolDetail={onCopyToolDetail}
          />
        ) : null}
        {attachments}
        {message.content ? <MessageResponse>{message.content}</MessageResponse> : null}
        {answerEndContent}
      </MessageContent>
    </Message>
  );
}

function MessageAttachments({ attachments }: { attachments: MessageAttachment[] }) {
  return (
    <Attachments className="message-attachments" variant="grid">
      {attachments.map((attachment) => (
        <MessageAttachmentItem attachment={attachment} key={attachment.id} />
      ))}
    </Attachments>
  );
}

function MessageAttachmentItem({ attachment }: { attachment: MessageAttachment }) {
  const isImage = getMediaCategory(attachment) === "image" && attachment.url;
  const label = getAttachmentLabel(attachment);

  if (isImage) {
    return (
      <Dialog>
        <DialogTrigger asChild>
          <Attachment
            className="previewable-attachment"
            data={attachment}
            role="button"
            tabIndex={0}
            title="点击预览图片"
          >
            <AttachmentPreview />
          </Attachment>
        </DialogTrigger>
        <DialogContent className="image-preview-dialog">
          <DialogTitle className="sr-only">{label}</DialogTitle>
          <img alt={label} className="image-preview-full" src={attachment.url} />
        </DialogContent>
      </Dialog>
    );
  }

  return (
    <AttachmentHoverCard>
      <AttachmentHoverCardTrigger asChild>
        <Attachment data={attachment}>
          <AttachmentPreview />
        </Attachment>
      </AttachmentHoverCardTrigger>
      <AttachmentHoverCardContent>
        <AttachmentHover attachment={attachment} />
      </AttachmentHoverCardContent>
    </AttachmentHoverCard>
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

function ProcessTrace({
  message,
  startContent,
  beforeBlock,
  onCopyToolDetail,
}: {
  message: ChatMessage;
  startContent?: ReactNode;
  beforeBlock?: (block: AgentBlock, index: number) => ReactNode;
  onCopyToolDetail?: (text: string) => void;
}) {
  return (
    <Reasoning
      className="process-trace"
      isStreaming={!!message.streaming}
      defaultOpen={!!message.streaming}
      duration={message.durationMs ? Math.max(1, Math.round(message.durationMs / 1000)) : undefined}
    >
      <ProcessTraceTrigger interrupted={!!message.interrupted} />
      <CollapsibleContent className="process-content">
        <div className="process-line" aria-hidden="true" />
        <div className="process-items">
          {startContent}
          {message.blocks.map((block, index) => (
            <Fragment key={block.kind === "thinking" ? `thinking-${index}` : block.callId}>
              {beforeBlock?.(block, index)}
              {block.kind === "thinking" ? (
                <ThinkingStep
                  block={block}
                  active={!!message.streaming && index === message.blocks.length - 1}
                />
              ) : (
                <ToolStep
                  block={block}
                  active={block.status === "input-available"}
                  onCopyDetail={onCopyToolDetail}
                />
              )}
            </Fragment>
          ))}
        </div>
      </CollapsibleContent>
    </Reasoning>
  );
}

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
      <ChevronDown className={`process-chevron ${isOpen ? "open" : ""}`} size={15} />
    </ReasoningTrigger>
  );
}

function ThinkingStep({ block, active }: { block: ThinkingBlock; active: boolean }) {
  return (
    <div className={`process-item thinking-step ${active ? "active" : ""}`}>
      <span className="process-node circle" />
      <div className="process-text">{block.content}</div>
    </div>
  );
}

function ToolStep({
  block,
  active,
  onCopyDetail,
}: {
  block: ToolBlock;
  active: boolean;
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
      className={`process-item tool-step ${active ? "active" : ""} ${block.status === "output-error" ? "error" : ""}`}
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
}

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
