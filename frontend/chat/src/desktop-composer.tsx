import { Plus } from "lucide-react";
import { memo, useCallback, useEffect, useRef, useState, type ChangeEvent } from "react";
import {
  Attachment, AttachmentHoverCard, AttachmentHoverCardContent, AttachmentHoverCardTrigger,
  AttachmentPreview, AttachmentRemove, Attachments, getAttachmentLabel, getMediaCategory,
} from "@/components/ai-elements/attachments";
import {
  PromptInput, PromptInputActionAddAttachments, PromptInputActionMenu, PromptInputActionMenuContent,
  PromptInputActionMenuTrigger, PromptInputBody, PromptInputFooter, PromptInputTextarea, PromptInputTools,
  usePromptInputAttachments,
} from "@/components/ai-elements/prompt-input";
import type { TimelineReply } from "./message-timeline";
import { nextComposerExpanded } from "./composer-layout";
import { ComposerActionButton } from "./composer-action";
import { ComposerReply } from "./message-actions";
import { ModelCapsulePicker } from "./model-capsule-picker";
import type { ChatModelRuntime } from "./model-capsule-data";
import { isGeneratingChatStatus, type ChatStatus } from "./web-chat-status";

export type ComposerFile = { filename?: string; mediaType?: string; url?: string };

/** Own transient editor state while the app controller owns transport and durable chat state. */
export const DesktopComposer = memo(function DesktopComposer({
  chatReady, canSend, modelProblem, draftKey, status, stopPending, modelState, selectedRuntimeId, selectedEffort, replyTarget,
  onModelChange, onCancelReply, onSend, onStop,
}: {
  chatReady: boolean;
  canSend: boolean;
  modelProblem: string;
  draftKey: string;
  status: ChatStatus;
  stopPending: boolean;
  modelState: { defaultRuntime: string; runtimes: ChatModelRuntime[] } | null;
  selectedRuntimeId: string;
  selectedEffort: string;
  replyTarget: TimelineReply | null;
  onModelChange: (runtimeId: string, effort: string) => void;
  onCancelReply: () => void;
  onSend: (text: string, files: ComposerFile[]) => Promise<void>;
  onStop: () => void;
}) {
  // 标签页文本草稿是展示状态，既不上传，也不创建 Message。
  const drafts = useRef(new Map<string, string>());
  const [draftVersion, setDraftVersion] = useState(0);
  const [draftStorageError, setDraftStorageError] = useState(false);
  let input = drafts.current.get(draftKey);
  if (input === undefined) {
    try { input = sessionStorage.getItem(`akashic.chat.draft:${draftKey}`) ?? ""; }
    catch (error) { if (!(error instanceof DOMException)) throw error; input = ""; }
    drafts.current.set(draftKey, input);
  }
  const setDraft = useCallback((key: string, text: string) => {
    drafts.current.set(key, text);
    try {
      if (text) sessionStorage.setItem(`akashic.chat.draft:${key}`, text);
      else sessionStorage.removeItem(`akashic.chat.draft:${key}`);
    } catch (error) {
      if (!(error instanceof DOMException)) throw error;
      setDraftStorageError(true);
    }
    setDraftVersion((version) => version + 1);
  }, []);
  const setInput = useCallback((text: string) => setDraft(draftKey, text), [draftKey, setDraft]);
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
    if (!canSend) throw new Error(modelProblem || "聊天服务暂不可用，请稍后重试。");
    const wasExpanded = expanded;
    setInput("");
    setExpanded(false);
    try {
      await onSend(text, files);
    } catch (error) {
      setDraft(draftKey, drafts.current.get(draftKey) || text);
      setExpanded(wasExpanded);
      throw error;
    }
  }, [canSend, modelProblem, expanded, onSend, setInput, setDraft, draftKey]);
  const shellExpanded = expanded || hasAttachments || Boolean(replyTarget);
  return (
    <>
    {draftStorageError ? <p role="status">浏览器无法保存本页草稿。刷新前请复制已输入的文字。</p> : null}
    <PromptInput
      className={`composer ${shellExpanded ? "is-expanded" : "is-compact"} ${input.trim() || replyTarget ? "has-text" : "empty"}`}
      multiple
      data-draft-version={draftVersion}
      onSubmit={(message) => submit(message.text, message.files)}
    >
      {replyTarget ? <ComposerReply author={replyTarget.author} preview={replyTarget.preview} onCancel={onCancelReply} /> : null}
      <PromptInputBody>
        <ComposerAttachments onPresenceChange={setHasAttachments} />
        <PromptInputTextarea
          className="composer__textarea !min-h-0"
          value={input}
          onChange={onInputChange}
          aria-label="消息"
          aria-describedby={modelProblem ? "chat-model-reason" : undefined}
          disabled={!chatReady}
          placeholder={canSend ? "继续布置任务…" : "先写下想说的话…"}
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
            disabled={isGeneratingChatStatus(status)}
            onChange={onModelChange}
          /> : null}
        </PromptInputTools>
        <PromptInputTools className="composer__trail">
          <PromptInputActionMenu>
            <PromptInputActionMenuTrigger aria-label="添加文件" className="composer-tool" tooltip="添加文件"><Plus size={18} /></PromptInputActionMenuTrigger>
            <PromptInputActionMenuContent><PromptInputActionAddAttachments label="上传文件" /></PromptInputActionMenuContent>
          </PromptInputActionMenu>
          <ComposerSubmit input={input} status={status} stopPending={stopPending} onStop={onStop} disabled={!canSend} />
        </PromptInputTools>
      </PromptInputFooter>
    </PromptInput>
    </>
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
    disabled={stopPending || (!generating && (disabled || (!input.trim() && attachments.files.length === 0)))}
  />;
}
