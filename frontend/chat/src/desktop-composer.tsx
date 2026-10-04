import { Paperclip } from "lucide-react";
import { memo, useCallback, useEffect, useImperativeHandle, useRef, useState, type ChangeEvent, type DragEvent, type Ref } from "react";
import {
  Attachment, AttachmentHoverCard, AttachmentHoverCardContent, AttachmentHoverCardTrigger,
  AttachmentPreview, AttachmentRemove, Attachments, getAttachmentLabel, getMediaCategory,
} from "@/components/ai-elements/attachments";
import {
  PromptInput, PromptInputBody, PromptInputButton, PromptInputFooter, PromptInputTextarea, PromptInputTools,
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

/** 空态推荐等外部入口写入草稿的窄接口。 */
export interface ComposerApi {
  insertDraft: (text: string) => void;
}

/** Own transient editor state while the app controller owns transport and durable chat state. */
export const DesktopComposer = memo(function DesktopComposer({
  chatReady, canSend, modelProblem, draftKey, autoFocus = false, status, stopPending, modelState, selectedRuntimeId, selectedEffort, replyTarget,
  onModelChange, onCancelReply, onSend, onStop, ref,
}: {
  chatReady: boolean;
  canSend: boolean;
  modelProblem: string;
  draftKey: string;
  /** 空态（新会话/无消息）时输入框接收焦点。 */
  autoFocus?: boolean;
  status: ChatStatus;
  stopPending: boolean;
  modelState: { defaultRuntime: string; runtimes: ChatModelRuntime[] } | null;
  selectedRuntimeId: string;
  selectedEffort: string;
  replyTarget: TimelineReply | null;
  onModelChange: (runtimeId: string, effort: string) => void;
  onCancelReply: () => void;
  onSend: (text: string, files: ComposerFile[]) => Promise<string | undefined>;
  onStop: () => void;
  ref?: Ref<ComposerApi>;
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
  const textareaRef = useRef<HTMLTextAreaElement | null>(null);
  useImperativeHandle(ref, () => ({
    // 写入后把焦点与光标放到草稿末尾，用户接着补完即可。
    insertDraft: (text) => {
      setDraft(draftKey, text);
      window.requestAnimationFrame(() => {
        const textarea = textareaRef.current;
        if (!textarea) return;
        textarea.focus();
        textarea.setSelectionRange(textarea.value.length, textarea.value.length);
      });
    },
  }), [draftKey, setDraft]);
  useEffect(() => {
    if (autoFocus && chatReady) textareaRef.current?.focus();
  }, [autoFocus, chatReady, draftKey]);
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
  }, [setInput, syncExpanded]);
  const submit = useCallback(async (text: string, files: ComposerFile[]) => {
    if (!canSend) throw new Error(modelProblem || "聊天服务暂不可用，请稍后重试。");
    const wasExpanded = expanded;
    setInput("");
    setExpanded(false);
    try {
      const sessionId = await onSend(text, files);
      // 首次接纳只是给同一编辑位置确定会话身份，后写的草稿随它保留。
      if (sessionId && draftKey.startsWith("new:")) {
        const next = drafts.current.get(draftKey) || "";
        setDraft(sessionId, next);
        setDraft(draftKey, "");
      }
    } catch (error) {
      const later = drafts.current.get(draftKey) || "";
      setDraft(draftKey, [text, later].filter(Boolean).join("\n\n"));
      setExpanded(wasExpanded);
      throw error;
    }
  }, [canSend, modelProblem, expanded, onSend, setInput, setDraft, draftKey]);
  const shellExpanded = expanded || hasAttachments || Boolean(replyTarget);
  // 文件拖拽强调态：dragenter/dragleave 用深度计数去抖，子元素间移动不成对出入时也不会闪烁。
  const dragDepth = useRef(0);
  const [fileDragActive, setFileDragActive] = useState(false);
  const isFileDrag = useCallback((event: DragEvent<HTMLFormElement>) =>
    Array.from(event.dataTransfer?.types ?? []).includes("Files"), []);
  const onDragEnter = useCallback((event: DragEvent<HTMLFormElement>) => {
    if (!isFileDrag(event)) return;
    event.preventDefault();
    dragDepth.current += 1;
    setFileDragActive(true);
  }, [isFileDrag]);
  const onDragOver = useCallback((event: DragEvent<HTMLFormElement>) => {
    if (!isFileDrag(event)) return;
    event.preventDefault();
    event.dataTransfer.dropEffect = "copy";
  }, [isFileDrag]);
  const onDragLeave = useCallback((event: DragEvent<HTMLFormElement>) => {
    if (!isFileDrag(event)) return;
    dragDepth.current = Math.max(0, dragDepth.current - 1);
    if (dragDepth.current === 0) setFileDragActive(false);
  }, [isFileDrag]);
  const onDrop = useCallback((event: DragEvent<HTMLFormElement>) => {
    dragDepth.current = 0;
    setFileDragActive(false);
    // 附件入列由 PromptInput 内部共享的 drop 处理（与粘贴/选择器同一条 add 路径）；这里只防浏览器直接打开文件。
    if (isFileDrag(event)) event.preventDefault();
  }, [isFileDrag]);
  return (
    <>
    {draftStorageError ? <p role="status">浏览器无法保存本页草稿。刷新前请复制已输入的文字。</p> : null}
    <PromptInput
      className={`composer ${shellExpanded ? "is-expanded" : "is-compact"} ${input.trim() || replyTarget ? "has-text" : "empty"}${fileDragActive ? " is-drop-target" : ""}`}
      multiple
      data-draft-version={draftVersion}
      onSubmit={(message) => submit(message.text, message.files)}
      onDragEnter={onDragEnter}
      onDragOver={onDragOver}
      onDragLeave={onDragLeave}
      onDrop={onDrop}
    >
      {replyTarget ? <ComposerReply author={replyTarget.author} preview={replyTarget.preview} onCancel={onCancelReply} /> : null}
      {fileDragActive ? <div className="composer-drop-hint" aria-hidden="true">松开以添加文件</div> : null}
      <PromptInputBody>
        <ComposerAttachments onPresenceChange={setHasAttachments} />
        <PromptInputTextarea
          ref={textareaRef}
          autoFocus={autoFocus}
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
          <ComposerAttachmentButton />
          <ComposerSubmit input={input} status={status} stopPending={stopPending} onStop={onStop} disabled={!canSend} />
        </PromptInputTools>
      </PromptInputFooter>
    </PromptInput>
    </>
  );
});

function ComposerAttachmentButton() {
  const attachments = usePromptInputAttachments();
  return (
    <PromptInputButton
      aria-label="添加文件"
      className="composer-tool"
      tooltip="添加文件"
      onClick={() => attachments.openFileDialog()}
    >
      <Paperclip size={18} />
    </PromptInputButton>
  );
}

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
    label={stopPending ? "正在停止" : status === "uploading" ? "取消上传" : generating ? "中止回答" : "发送消息"}
    type={generating ? "button" : "submit"}
    onClick={generating ? (event) => {
      // 取消会把同一按钮变回 submit；先阻断这次点击的默认提交。
      event.preventDefault();
      onStop();
    } : undefined}
    disabled={stopPending || (!generating && (disabled || (!input.trim() && attachments.files.length === 0)))}
  />;
}
