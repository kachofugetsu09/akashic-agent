import { Paperclip } from "lucide-react";
import { memo, useCallback, useEffect, useImperativeHandle, useRef, useState, type ChangeEvent, type FocusEvent, type Ref } from "react";
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
import { ModelCapsulePicker, ModelEffortAction, resolveVisibleRuntime } from "./model-capsule-picker";
import { compatibleEffort, type ChatModelRuntime } from "./model-capsule-data";
import { isGeneratingChatStatus, type ChatStatus } from "./web-chat-status";

export type ComposerFile = { filename?: string; mediaType?: string; url?: string };

/** 空态推荐等外部入口写入草稿的窄接口。 */
export interface ComposerApi {
  insertDraft: (text: string) => void;
}

/** Own transient editor state while the app controller owns transport and durable chat state. */
export const DesktopComposer = memo(function DesktopComposer({
  chatReady, canSend, modelProblem, draftKey, autoFocus = false, fresh = false, status, stopPending, modelState, selectedRuntimeId, selectedEffort, replyTarget,
  onModelChange, onCancelReply, onSend, onStop, ref,
}: {
  chatReady: boolean;
  canSend: boolean;
  modelProblem: string;
  draftKey: string;
  /** 空态（新会话/无消息）时输入框接收焦点。 */
  autoFocus?: boolean;
  /** 会话还没有消息；占位文案邀请开场，而不是"继续"。 */
  fresh?: boolean;
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
  useTypeToCompose(textareaRef, chatReady);
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
  // 窄屏阅读时输入卡只保留一行。用户点进卡片才展开（程序 autofocus 不算）；
  // 焦点离开整张卡（含它弹出的模型菜单）才收回。
  const [engaged, setEngaged] = useState(false);
  const onComposerBlur = (event: FocusEvent<HTMLFormElement>) => {
    const next = event.relatedTarget;
    if (next instanceof Element && (event.currentTarget.contains(next) || next.closest("[data-radix-popper-content-wrapper]"))) return;
    setEngaged(false);
  };
  const resting = !engaged && !shellExpanded && !input.trim();
  const [fileDragActive, setFileDragActive] = useState(false);
  return (
    <>
    {draftStorageError ? <p role="status">浏览器无法保存本页草稿。刷新前请复制已输入的文字。</p> : null}
    <PromptInput
      className={`composer ${shellExpanded ? "is-expanded" : "is-compact"} ${input.trim() || replyTarget ? "has-text" : "empty"}${resting ? " is-resting" : ""}${fileDragActive ? " is-drop-target" : ""}`}
      multiple
      data-draft-version={draftVersion}
      onSubmit={(message) => submit(message.text, message.files)}
      onClick={(event) => {
        // 触摸的 click 在 pointerdown 之后按当时布局命中目标；展开必须等这次点击落定，
        // 否则工具行会顶到手指下，把点击送进模型选择。卡片空白处的点击等同点进输入框。
        const target = event.target instanceof Element ? event.target : null;
        if (target?.closest("button, a, [role='button'], [data-radix-popper-content-wrapper]")) return;
        setEngaged(true);
        if (!target?.closest("textarea")) textareaRef.current?.focus();
      }}
      onKeyDown={() => setEngaged(true)}
      onBlur={onComposerBlur}
    >
      <WindowFileDrop enabled={chatReady} onActiveChange={setFileDragActive} />
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
          placeholder={!canSend ? "先写下想说的话…" : fresh ? "说说想做什么…" : "继续…"}
        />
      </PromptInputBody>
      <PromptInputFooter className="composer__bar">
        <PromptInputTools className="composer__lead">
          {modelState ? (
            <>
              <ModelCapsulePicker
                defaultRuntime={modelState.defaultRuntime}
                runtimes={modelState.runtimes}
                selectedRuntimeId={selectedRuntimeId}
                selectedEffort={selectedEffort}
                disabled={isGeneratingChatStatus(status)}
                onChange={onModelChange}
              />
              <ComposerEffortAction
                modelState={modelState}
                selectedRuntimeId={selectedRuntimeId}
                selectedEffort={selectedEffort}
                disabled={isGeneratingChatStatus(status)}
                onChange={onModelChange}
              />
            </>
          ) : null}
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

// 读回复时直接打字就进入输入框：焦点不在任何可编辑处、也没有浮层时，把这次按键的焦点交给输入框，
// 字符随按键的默认动作落进去。触屏设备没有这种输入方式，不挂监听。
function useTypeToCompose(textareaRef: { current: HTMLTextAreaElement | null }, enabled: boolean) {
  useEffect(() => {
    if (!enabled || !window.matchMedia("(hover: hover) and (pointer: fine)").matches) return;
    const onKeyDown = (event: KeyboardEvent) => {
      // 1. 只接普通可见字符；组合键、输入法组合中与已被处理的按键都不碰。
      if (event.defaultPrevented || event.isComposing || event.ctrlKey || event.metaKey || event.altKey) return;
      if (event.key.length !== 1 || event.key === " ") return;
      // 2. 焦点已在可编辑处，或有对话框/菜单/浮层打开时，按键属于它们。
      const active = document.activeElement;
      if (active instanceof HTMLElement && active.closest("input, textarea, select, [contenteditable]:not([contenteditable='false'])")) return;
      if (document.querySelector("[role='dialog'], [role='menu'], [role='listbox'], [data-radix-popper-content-wrapper]")) return;
      const textarea = textareaRef.current;
      if (!textarea || textarea.disabled) return;
      textarea.focus();
      textarea.setSelectionRange(textarea.value.length, textarea.value.length);
    };
    document.addEventListener("keydown", onKeyDown);
    return () => document.removeEventListener("keydown", onKeyDown);
  }, [enabled, textareaRef]);
}

function ComposerEffortAction({ modelState, selectedRuntimeId, selectedEffort, disabled, onChange }: {
  modelState: { defaultRuntime: string; runtimes: ChatModelRuntime[] };
  selectedRuntimeId: string;
  selectedEffort: string;
  disabled: boolean;
  onChange: (runtimeId: string, effort: string) => void;
}) {
  const { visibleModel, explicitModel } = resolveVisibleRuntime(modelState.runtimes, selectedRuntimeId, modelState.defaultRuntime);
  if (!visibleModel) return null;
  return (
    <ModelEffortAction
      runtime={visibleModel}
      effort={compatibleEffort(visibleModel, selectedEffort)}
      explicit={Boolean(explicitModel)}
      disabled={disabled}
      onChange={onChange}
    />
  );
}

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

// 文件拖进窗口任意位置都由输入区接住：卡片显示"松开以添加文件"，松开即入列（与粘贴、选择器同一条 add 路径）。
// 落在输入卡里的 drop 由 PromptInput 自己处理；落在别处时这里入列并阻止浏览器直接打开文件、离开聊天页。
// 已被其它放置区（如对话框里的上传框）处理的 drop 不重复入列；对话框打开时不显示强调态。
function WindowFileDrop({ enabled, onActiveChange }: { enabled: boolean; onActiveChange: (active: boolean) => void }) {
  const attachments = usePromptInputAttachments();
  const add = attachments.add;
  useEffect(() => {
    // 1. dragenter/dragleave 在每个元素上成对出入，用深度计数判断是否还在窗口内。
    let depth = 0;
    const isFileDrag = (event: DragEvent) => Array.from(event.dataTransfer?.types ?? []).includes("Files");
    const reset = () => { depth = 0; onActiveChange(false); };
    const onEnter = (event: DragEvent) => {
      if (!isFileDrag(event)) return;
      depth += 1;
      onActiveChange(enabled && !document.querySelector("[role='dialog']"));
    };
    const onLeave = (event: DragEvent) => {
      if (!isFileDrag(event)) return;
      depth = Math.max(0, depth - 1);
      if (depth === 0 || event.relatedTarget === null) reset();
    };
    // 2. dragover 必须 preventDefault，drop 才会派发而不是被浏览器当作打开文件。
    const onOver = (event: DragEvent) => {
      if (!isFileDrag(event)) return;
      event.preventDefault();
      if (event.dataTransfer) event.dataTransfer.dropEffect = enabled ? "copy" : "none";
    };
    // 3. 卡片内的 drop 已由 PromptInput 入列；其余位置在这里入列。
    const onDrop = (event: DragEvent) => {
      reset();
      if (!isFileDrag(event) || event.defaultPrevented) return;
      event.preventDefault();
      const insideComposer = event.target instanceof Element && event.target.closest("form.composer");
      if (enabled && !insideComposer && !document.querySelector("[role='dialog']") && event.dataTransfer?.files.length) {
        add(event.dataTransfer.files);
      }
    };
    window.addEventListener("dragenter", onEnter);
    window.addEventListener("dragleave", onLeave);
    window.addEventListener("dragover", onOver);
    window.addEventListener("drop", onDrop);
    window.addEventListener("dragend", reset);
    return () => {
      window.removeEventListener("dragenter", onEnter);
      window.removeEventListener("dragleave", onLeave);
      window.removeEventListener("dragover", onOver);
      window.removeEventListener("drop", onDrop);
      window.removeEventListener("dragend", reset);
    };
  }, [add, enabled, onActiveChange]);
  return null;
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
  // 正在停止时用 aria-disabled 而非 disabled：禁用按钮会被浏览器夺走键盘焦点，状态只剩读屏文本。
  const stopping = generating && stopPending;
  return <ComposerActionButton
    mode={generating ? "stop" : "send"}
    label={stopPending ? "正在停止" : status === "uploading" ? "取消上传" : generating ? "中止回答" : "发送消息"}
    type={generating ? "button" : "submit"}
    aria-disabled={stopping || undefined}
    onClick={generating ? (event) => {
      // 取消会把同一按钮变回 submit；先阻断这次点击的默认提交。
      event.preventDefault();
      if (!stopping) onStop();
    } : undefined}
    disabled={!generating && (stopPending || disabled || (!input.trim() && attachments.files.length === 0))}
  />;
}
