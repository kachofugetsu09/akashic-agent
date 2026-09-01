import { Plus } from "lucide-react";
import { memo, useCallback, useEffect, useState, type ChangeEvent } from "react";
import {
  Attachment, AttachmentHoverCard, AttachmentHoverCardContent, AttachmentHoverCardTrigger,
  AttachmentPreview, AttachmentRemove, Attachments, getAttachmentLabel, getMediaCategory,
} from "@/components/ai-elements/attachments";
import {
  PromptInput, PromptInputActionAddAttachments, PromptInputActionMenu, PromptInputActionMenuContent,
  PromptInputActionMenuTrigger, PromptInputBody, PromptInputFooter, PromptInputTextarea, PromptInputTools,
  usePromptInputAttachments,
} from "@/components/ai-elements/prompt-input";
import type { ChatMessage } from "./chat-message";
import { nextComposerExpanded } from "./composer-layout";
import { ComposerActionButton } from "./composer-action";
import { ComposerReply } from "./message-actions";
import { ModelCapsulePicker } from "./model-capsule-picker";
import type { ChatModelRuntime } from "./model-capsule-data";
import { isGeneratingChatStatus, type ChatStatus } from "./web-chat-status";

export type ComposerFile = { filename?: string; mediaType?: string; url?: string };

/** Own transient editor state while the app controller owns transport and durable chat state. */
export const DesktopComposer = memo(function DesktopComposer({
  chatReady, status, stopPending, modelState, selectedRuntimeId, selectedEffort, replyTarget,
  onModelChange, onCancelReply, onSend, onStop,
}: {
  chatReady: boolean;
  status: ChatStatus;
  stopPending: boolean;
  modelState: { defaultRuntime: string; runtimes: ChatModelRuntime[] } | null;
  selectedRuntimeId: string;
  selectedEffort: string;
  replyTarget: ChatMessage | null;
  onModelChange: (runtimeId: string, effort: string) => void;
  onCancelReply: () => void;
  onSend: (text: string, files: ComposerFile[]) => Promise<void>;
  onStop: () => void;
}) {
  const [input, setInput] = useState("");
  const [expanded, setExpanded] = useState(false);
  const [hasAttachments, setHasAttachments] = useState(false);
  const syncExpanded = useCallback((textarea: HTMLTextAreaElement | null, text: string) => {
    setExpanded((wasExpanded) => nextComposerExpanded(
      wasExpanded,
      text,
      // 只在紧凑态读取一次溢出；展开后的宽度变化不能反向改写布局状态。
      () => textarea ? textarea.scrollHeight > textarea.clientHeight : false,
    ));
  }, []);
  const onInputChange = useCallback((event: ChangeEvent<HTMLTextAreaElement>) => {
    const next = event.target.value;
    setInput(next);
    syncExpanded(event.target, next);
  }, [syncExpanded]);
  const submit = useCallback(async (text: string, files: ComposerFile[]) => {
    const wasExpanded = expanded;
    setInput("");
    setExpanded(false);
    try {
      await onSend(text, files);
    } catch (error) {
      setInput((current) => current || text);
      setExpanded(wasExpanded);
      throw error;
    }
  }, [expanded, onSend]);
  const shellExpanded = expanded || hasAttachments || Boolean(replyTarget);
  return (
    <PromptInput
      className={`composer ${shellExpanded ? "is-expanded" : "is-compact"} ${input.trim() || replyTarget ? "has-text" : "empty"}`}
      multiple
      onSubmit={(message) => submit(message.text, message.files)}
    >
      {replyTarget ? <ComposerReply role={replyTarget.role} preview={desktopComposerReplyPreview(replyTarget)} onCancel={onCancelReply} /> : null}
      <PromptInputBody>
        <ComposerAttachments onPresenceChange={setHasAttachments} />
        <PromptInputTextarea
          className="composer__textarea !min-h-0"
          value={input}
          onChange={onInputChange}
          disabled={!chatReady}
          placeholder={chatReady ? "继续布置任务…" : "连接模型后即可开始对话"}
        />
      </PromptInputBody>
      <PromptInputFooter className="composer__bar">
        <PromptInputTools className="composer__lead">
          {modelState ? <ModelCapsulePicker
            compact
            defaultRuntime={modelState.defaultRuntime}
            runtimes={modelState.runtimes}
            selectedRuntimeId={selectedRuntimeId}
            selectedEffort={selectedEffort}
            disabled={status !== "idle"}
            onChange={onModelChange}
          /> : null}
        </PromptInputTools>
        <PromptInputTools className="composer__trail">
          <PromptInputActionMenu>
            <PromptInputActionMenuTrigger aria-label="添加文件" className="composer-tool" tooltip="添加文件"><Plus size={18} /></PromptInputActionMenuTrigger>
            <PromptInputActionMenuContent><PromptInputActionAddAttachments label="上传文件" /></PromptInputActionMenuContent>
          </PromptInputActionMenu>
          <ComposerSubmit input={input} status={status} stopPending={stopPending} onStop={onStop} disabled={!chatReady} />
        </PromptInputTools>
      </PromptInputFooter>
    </PromptInput>
  );
});

function ComposerAttachments({ onPresenceChange }: { onPresenceChange: (hasAttachments: boolean) => void }) {
  const attachments = usePromptInputAttachments();
  const hasAttachments = attachments.files.length > 0;
  useEffect(() => {
    onPresenceChange(hasAttachments);
  }, [hasAttachments, onPresenceChange]);
  if (attachments.files.length === 0) return null;
  return (
    <Attachments className="composer-attachments" variant="grid">
      {attachments.files.map((attachment) => {
        const category = getMediaCategory(attachment);
        const isMedia = category === "image" || category === "video";
        return (
          <AttachmentHoverCard key={attachment.id}>
            <AttachmentHoverCardTrigger asChild>
              <Attachment
                className={`attachment-chip ${isMedia ? "is-media" : "is-file"}`}
                data={attachment}
                onRemove={() => attachments.remove(attachment.id)}
              >
                <div className="attachment-preview-slot">
                  <div className="attachment-preview-icon">
                    <AttachmentPreview />
                  </div>
                  <AttachmentRemove className="attachment-remove-inline" />
                </div>
                {isMedia ? null : <span>{getAttachmentLabel(attachment)}</span>}
              </Attachment>
            </AttachmentHoverCardTrigger>
            <AttachmentHoverCardContent>
              <AttachmentHover attachment={attachment} />
            </AttachmentHoverCardContent>
          </AttachmentHoverCard>
        );
      })}
    </Attachments>
  );
}

function AttachmentHover({ attachment }: { attachment: ReturnType<typeof usePromptInputAttachments>["files"][number] }) {
  const category = getMediaCategory(attachment);
  const label = getAttachmentLabel(attachment);
  return <div className="attachment-hover">
    {category === "image" && attachment.url ? <img alt={label} className="attachment-hover-image" src={attachment.url} /> : <div className="attachment-hover-file"><Attachment data={attachment}><AttachmentPreview /></Attachment></div>}
    <div className="attachment-hover-title">{label}</div>
    {attachment.mediaType ? <div className="attachment-hover-type">{attachment.mediaType}</div> : null}
  </div>;
}

function ComposerSubmit({ input, status, stopPending, onStop, disabled }: { input: string; status: ChatStatus; stopPending: boolean; onStop: () => void; disabled: boolean }) {
  const attachments = usePromptInputAttachments();
  const generating = isGeneratingChatStatus(status);
  return <ComposerActionButton
    mode={generating ? "stop" : "send"}
    label={stopPending ? "正在停止" : generating ? "中止回答" : "发送消息"}
    type={generating ? "button" : "submit"}
    onClick={generating ? onStop : undefined}
    disabled={disabled || stopPending || (!generating && !input.trim() && attachments.files.length === 0)}
  />;
}

export function desktopComposerReplyPreview(message: ChatMessage) {
  return message.content.split(/\s+/u).filter(Boolean).join(" ").slice(0, 512)
    || (message.attachments?.length ? "[附件]" : "[无文字消息]");
}
