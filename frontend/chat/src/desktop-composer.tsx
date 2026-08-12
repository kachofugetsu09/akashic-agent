import { Plus } from "lucide-react";
import { memo, useCallback, useState } from "react";
import {
  Attachment, AttachmentHoverCard, AttachmentHoverCardContent, AttachmentHoverCardTrigger,
  AttachmentInfo, AttachmentPreview, AttachmentRemove, Attachments, getAttachmentLabel, getMediaCategory,
} from "@/components/ai-elements/attachments";
import {
  PromptInput, PromptInputActionAddAttachments, PromptInputActionMenu, PromptInputActionMenuContent,
  PromptInputActionMenuTrigger, PromptInputBody, PromptInputFooter, PromptInputTextarea, PromptInputTools,
  usePromptInputAttachments,
} from "@/components/ai-elements/prompt-input";
import type { ChatMessage } from "./chat-message";
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
  const submit = useCallback(async (text: string, files: ComposerFile[]) => {
    setInput("");
    try {
      await onSend(text, files);
    } catch (error) {
      setInput((current) => current || text);
      throw error;
    }
  }, [onSend]);
  return (
    <>
      {modelState ? <ModelCapsulePicker
        defaultRuntime={modelState.defaultRuntime}
        runtimes={modelState.runtimes}
        selectedRuntimeId={selectedRuntimeId}
        selectedEffort={selectedEffort}
        disabled={status !== "idle"}
        onChange={onModelChange}
      /> : null}
      <PromptInput className="composer" multiple onSubmit={(message) => submit(message.text, message.files)}>
        {replyTarget ? <ComposerReply role={replyTarget.role} preview={desktopComposerReplyPreview(replyTarget)} onCancel={onCancelReply} /> : null}
        <PromptInputBody>
          <ComposerAttachments />
          <PromptInputTextarea value={input} onChange={(event) => setInput(event.target.value)} disabled={!chatReady} placeholder={chatReady ? "有问题，尽管问" : "连接模型后即可开始对话"} />
        </PromptInputBody>
        <PromptInputFooter>
          <PromptInputTools>
            <PromptInputActionMenu>
              <PromptInputActionMenuTrigger aria-label="添加文件" className="composer-tool" tooltip="添加文件"><Plus size={20} /></PromptInputActionMenuTrigger>
              <PromptInputActionMenuContent><PromptInputActionAddAttachments label="上传文件" /></PromptInputActionMenuContent>
            </PromptInputActionMenu>
          </PromptInputTools>
          <PromptInputTools><ComposerSubmit input={input} status={status} stopPending={stopPending} onStop={onStop} disabled={!chatReady} /></PromptInputTools>
        </PromptInputFooter>
      </PromptInput>
    </>
  );
});

function ComposerAttachments() {
  const attachments = usePromptInputAttachments();
  if (attachments.files.length === 0) return null;
  return <Attachments className="composer-attachments" variant="inline">{attachments.files.map((attachment) => (
    <AttachmentHoverCard key={attachment.id}>
      <AttachmentHoverCardTrigger asChild>
        <Attachment data={attachment} onRemove={() => attachments.remove(attachment.id)}>
          <div className="attachment-preview-slot"><div className="attachment-preview-icon"><AttachmentPreview /></div><AttachmentRemove className="attachment-remove-inline" /></div>
          <AttachmentInfo />
        </Attachment>
      </AttachmentHoverCardTrigger>
      <AttachmentHoverCardContent><AttachmentHover attachment={attachment} /></AttachmentHoverCardContent>
    </AttachmentHoverCard>
  ))}</Attachments>;
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
