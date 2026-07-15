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
import { ChevronDown, Wrench } from "lucide-react";
import { Fragment, type ReactNode } from "react";
import type {
  AgentBlock,
  ChatMessage,
  MessageAttachment,
  ThinkingBlock,
  ToolBlock,
} from "./main";

export function ChatMessageView({
  message,
  attachmentContent,
  processStartContent,
  beforeProcessBlock,
  answerEndContent,
}: {
  message: ChatMessage;
  attachmentContent?: ReactNode;
  processStartContent?: ReactNode;
  beforeProcessBlock?: (block: AgentBlock, index: number) => ReactNode;
  answerEndContent?: ReactNode;
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
          {attachments}
          {message.content ? <MessageResponse>{message.content}</MessageResponse> : null}
        </MessageContent>
      </Message>
    );
  }

  return (
    <Message from="assistant" className="message-row agent-row">
      <MessageContent className="agent-content">
        {message.blocks.length ? (
          <ProcessTrace
            message={message}
            startContent={processStartContent}
            beforeBlock={beforeProcessBlock}
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
}: {
  message: ChatMessage;
  startContent?: ReactNode;
  beforeBlock?: (block: AgentBlock, index: number) => ReactNode;
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

function ToolStep({ block, active }: { block: ToolBlock; active: boolean }) {
  const description = toolDescription(block.input);

  return (
    <div
      className={`process-item tool-step ${active ? "active" : ""} ${block.status === "output-error" ? "error" : ""}`}
    >
      <span className="process-node diamond" />
      <div className="tool-step-body">
        <div className="tool-step-title">
          <Wrench className="tool-step-icon" size={14} />
          <span>{block.name}</span>
        </div>
        {description ? <div className="tool-step-description">{description}</div> : null}
      </div>
    </div>
  );
}

function toolDescription(input: unknown) {
  if (typeof input !== "object" || input === null || Array.isArray(input)) {
    return "";
  }

  const description = (input as Record<string, unknown>).description;
  return typeof description === "string" ? description.trim() : "";
}
