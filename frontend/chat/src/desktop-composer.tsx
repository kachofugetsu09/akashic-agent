import { Clock3, Paperclip, Undo2 } from "lucide-react";
import { memo, useCallback, useEffect, useImperativeHandle, useRef, useState, type ChangeEvent, type FocusEvent, type KeyboardEvent as ReactKeyboardEvent, type Ref } from "react";
import {
  Attachment, AttachmentFileBody, AttachmentHoverCard, AttachmentHoverCardContent, AttachmentHoverCardTrigger,
  AttachmentPreview, AttachmentRemove, Attachments, getAttachmentLabel, getMediaCategory,
} from "@/components/ai-elements/attachments";
import {
  PromptInput, PromptInputBody, PromptInputButton, PromptInputFooter, PromptInputTextarea, PromptInputTools,
  touchOnly, usePromptInputAttachments,
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
  /** append：接在已有草稿之后（空行分隔），否则替换。 */
  insertDraft: (text: string, options?: { append?: boolean }) => void;
  /** 把焦点交给输入框并把光标放到末尾（如选择回复引用之后）。 */
  focus: () => void;
}

/** Own transient editor state while the app controller owns transport and durable chat state. */
// 浮层打开时按键与拖放都属于浮层：读回复打字、窗口拖放都让路。
const FLOATING_LAYER = "[role='dialog'], [role='menu'], [role='listbox'], [data-radix-popper-content-wrapper]";

// 自己在 document 上处理 Esc 的浮层：对话框、菜单、列表，以及标了 data-escape-layer 的非模态面板。
const ESCAPE_LAYERS = "[role='dialog'], [role='menu'], [role='listbox'], [data-radix-popper-content-wrapper], [data-escape-layer]";
// 与服务端单个上传上限（web_chat.MAX_UPLOAD_BYTES）一致。
const MAX_ATTACHMENT_MB = 50;

export const DesktopComposer = memo(function DesktopComposer({
  chatReady, canSend, modelProblem, draftKey, autoFocus = false, fresh = false, status, stopPending, stopCount, inputId, turnCompleted, modelState, selectedRuntimeId, selectedEffort, replyTarget,
  onModelChange, onCancelReply, onSend, onStop, lastInput = "", ref,
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
  stopCount: number;
  inputId: string;
  turnCompleted: boolean;
  modelState: { defaultRuntime: string; runtimes: ChatModelRuntime[] } | null;
  selectedRuntimeId: string;
  selectedEffort: string;
  replyTarget: TimelineReply | null;
  onModelChange: (runtimeId: string, effort: string) => void;
  onCancelReply: () => void;
  onSend: (text: string, files: ComposerFile[]) => Promise<string | undefined>;
  onStop: () => void;
  /** 当前会话最后一条用户提问；空输入框按 ↑ 取回编辑。 */
  lastInput?: string;
  ref?: Ref<ComposerApi>;
}) {
  // 标签页文本草稿是展示状态，既不上传，也不创建 Message。
  const drafts = useRef(new Map<string, string>());
  // 同一草稿存储保留排队文字；刷新后只恢复编辑内容，不恢复自动发送意图。
  const queuedDrafts = useRef(new Map<string, string>());
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
      const saved = [queuedDrafts.current.get(key), text].filter(Boolean).join("\n\n");
      if (saved) sessionStorage.setItem(`akashic.chat.draft:${key}`, saved);
      else sessionStorage.removeItem(`akashic.chat.draft:${key}`);
    } catch (error) {
      if (!(error instanceof DOMException)) throw error;
      setDraftStorageError(true);
    }
    setDraftVersion((version) => version + 1);
  }, []);
  const setInput = useCallback((text: string) => setDraft(draftKey, text), [draftKey, setDraft]);
  const textareaRef = useRef<HTMLTextAreaElement | null>(null);
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
  useImperativeHandle(ref, () => ({
    focus: () => window.requestAnimationFrame(() => {
      const textarea = textareaRef.current;
      if (!textarea) return;
      textarea.focus({ preventScroll: true });
      textarea.setSelectionRange(textarea.value.length, textarea.value.length);
    }),
    // 写入后把焦点与光标放到草稿末尾，用户接着补完即可。
    insertDraft: (text, options) => {
      const current = options?.append ? drafts.current.get(draftKey)?.trim() : "";
      const next = current ? `${current}\n\n${text}` : text;
      // 走浏览器的编辑命令整段替换：进原生撤销栈，误点"引用""放回输入框编辑"后 Ctrl+Z 即可撤回；
      // 它触发的 input 事件照常经 onInputChange 写回草稿与展开状态。不支持时退回直接写值。
      const textarea = textareaRef.current;
      if (textarea && !textarea.disabled) {
        textarea.focus();
        textarea.setSelectionRange(0, textarea.value.length);
        if (document.execCommand("insertText", false, next)) return;
      }
      setDraft(draftKey, next);
      window.requestAnimationFrame(() => {
        const textarea = textareaRef.current;
        if (!textarea) return;
        textarea.focus();
        textarea.setSelectionRange(textarea.value.length, textarea.value.length);
        // 放回的原文可能有多行：按实际溢出展开，与手动输入同一判定。
        syncExpanded(textarea, textarea.value);
      });
    },
  }), [draftKey, setDraft, syncExpanded]);
  const onInputChange = useCallback((event: ChangeEvent<HTMLTextAreaElement>) => {
    const next = event.target.value;
    setInput(next);
    syncExpanded(event.target, next);
  }, [setInput, syncExpanded]);
  const readDraft = useCallback((key: string) => drafts.current.get(key) ?? "", []);
  const followUp = useFollowUpQueue({ draftKey, status, stopPending, stopCount, inputId, turnCompleted, canSend: canSend && !replyTarget, readDraft, setDraft, queuedDrafts, onSend });
  const moveQueue = followUp.move;
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
        moveQueue(draftKey, sessionId);
        setDraft(sessionId, next);
        setDraft(draftKey, "");
        return sessionId;
      }
      return undefined;
    } catch (error) {
      const later = drafts.current.get(draftKey) || "";
      setDraft(draftKey, [text, later].filter(Boolean).join("\n\n"));
      setExpanded(wasExpanded);
      throw error;
    }
  }, [canSend, modelProblem, expanded, onSend, setInput, setDraft, draftKey, moveQueue]);
  // 输入卡内一行轻提示（排队被拒、附件超限），自动消失；同一时间只有一条。
  const [notice, setNotice] = useState<{ text: string; ms: number; title?: string } | null>(null);
  useEffect(() => {
    if (!notice) return;
    const timer = window.setTimeout(() => setNotice(null), notice.ms);
    return () => window.clearTimeout(timer);
  }, [notice]);
  const rejectFiles = useCallback((files: File[]) => {
    const names = files.map((file) => file.name || "未命名");
    // 过长的文件名中间省略：开头与带扩展名的结尾最能区分文件，完整名称放在 title 里。
    const first = [...names[0]].length > 40 ? `${[...names[0]].slice(0, 20).join("")}…${[...names[0]].slice(-16).join("")}` : names[0];
    const label = names.length > 1 ? `${first} 等 ${names.length} 个文件` : first;
    setNotice({ text: `${label} 超过 ${MAX_ATTACHMENT_MB}MB，未添加`, ms: 4000, title: names.join("\n") });
  }, []);
  const onComposerKeyDown = (event: ReactKeyboardEvent<HTMLTextAreaElement>) => {
    const native = event.nativeEvent;
    // 生成中在输入框按 Esc 停止本轮（coding 工具的中断习惯）；输入法组合中的 Esc 属于输入法。
    // 有浮层开着时 Esc 先关浮层（浮层在 document 上自己监听），这一下不能顺带停掉生成。
    if (event.key === "Escape" && !native.isComposing && native.keyCode !== 229
      && isGeneratingChatStatus(status) && !stopPending && !document.querySelector(ESCAPE_LAYERS)) {
      event.preventDefault();
      onStop();
      return;
    }
    // 空输入框按 ↑ 取回上一条提问（终端与聊天工具的习惯）；有内容时 ↑ 仍是移动光标。
    if (event.key === "ArrowUp" && !event.shiftKey && !event.altKey && !event.metaKey && !event.ctrlKey
      && !native.isComposing && !input && lastInput && !hasAttachments) {
      event.preventDefault();
      setInput(lastInput);
      const textarea = event.currentTarget;
      window.requestAnimationFrame(() => {
        textarea.setSelectionRange(textarea.value.length, textarea.value.length);
        syncExpanded(textarea, textarea.value);
      });
      return;
    }
    queueOnEnter(event);
  };
  const queueOnEnter = (event: ReactKeyboardEvent<HTMLTextAreaElement>) => {
    // 本轮未落定时，桌面键盘按回车把纯文字排到本轮之后。
    const native = event.nativeEvent;
    if (event.key !== "Enter" || event.shiftKey || native.isComposing || native.keyCode === 229 || touchOnly()) return;
    const text = input.trim();
    if (!turnBusy(status) || !text) return;
    event.preventDefault();
    if (hasAttachments || replyTarget) return setNotice({ text: "带附件或引用的消息，本轮结束后再发送", ms: 2400 });
    followUp.enqueue(text);
    setExpanded(false);
  };
  const shellExpanded = expanded || hasAttachments || Boolean(replyTarget) || Boolean(followUp.queued);
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
      attachmentKey={draftKey}
      maxFileSize={MAX_ATTACHMENT_MB * 1024 * 1024}
      onRejectFiles={rejectFiles}
      data-draft-version={draftVersion}
      onSubmit={(message) => {
        // 提示是关于这次输入的；发出后输入卡收成紧凑条，不该把上一条提示带进去。
        setNotice(null);
        // 触屏发送后收起键盘、输入卡回到静止态：接下来是读回复，半屏键盘和展开的工具行只会挡住它。
        if (!window.matchMedia("(hover: hover) and (pointer: fine)").matches) {
          setEngaged(false);
          if (document.activeElement instanceof HTMLElement) document.activeElement.blur();
        }
        return submit(message.text, message.files);
      }}
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
      {followUp.queued ? <div className="composer-reply composer-queued" aria-label="待发送文字">
        <Clock3 size={17} aria-hidden="true" />
        <div>
          <strong role="status">{followUp.stopped ? "已停止，结束后退回输入框" : "本轮结束后发送"}</strong>
          <span>{followUp.queued.text}</span>
        </div>
        <button type="button" onClick={() => followUp.queued && followUp.restore(followUp.queued)} aria-label="撤回到输入框" title="撤回到输入框">
          <Undo2 size={18} aria-hidden="true" />
        </button>
      </div> : null}
      {notice ? <p className="composer-queue-refused" role="status" title={notice.title}>{notice.text}</p> : null}
      {fileDragActive ? <div className="composer-drop-hint" aria-hidden="true">松开以添加文件</div> : null}
      <PromptInputBody>
        <ComposerAttachments onPresenceChange={setHasAttachments} />
        <PromptInputTextarea
          ref={textareaRef}
          autoFocus={autoFocus}
          className="composer__textarea !min-h-0"
          value={input}
          onChange={onInputChange}
          onKeyDown={onComposerKeyDown}
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

interface QueuedFollowUp {
  key: string;
  inputId: string;
  text: string;
  /** 排队时或之后用户按过停止：本轮结束后退回草稿，不自动发送。 */
  stopped: boolean;
  stopCount: number;
}

// 排队入口与落定判定共用本轮是否结束的规则。
function turnBusy(status: ChatStatus): boolean {
  return isGeneratingChatStatus(status) || status === "finalizing";
}

// 本轮未落定时按回车把文字排进本轮之后；本轮正常落定再发出，停止、出错或切走会话则退回草稿。
function useFollowUpQueue({ draftKey, status, stopPending, stopCount, inputId, turnCompleted, canSend, readDraft, setDraft, queuedDrafts, onSend }: {
  draftKey: string;
  status: ChatStatus;
  stopPending: boolean;
  stopCount: number;
  inputId: string;
  turnCompleted: boolean;
  queuedDrafts: { current: Map<string, string> };
  canSend: boolean;
  readDraft: (key: string) => string;
  setDraft: (key: string, text: string) => void;
  onSend: (text: string, files: ComposerFile[]) => Promise<string | undefined>;
}) {
  const [queued, setQueued] = useState<QueuedFollowUp | null>(null);
  const restore = useCallback((item: QueuedFollowUp) => {
    setQueued((current) => current === item ? null : current);
    queuedDrafts.current.delete(item.key);
    setDraft(item.key, [item.text, readDraft(item.key)].filter(Boolean).join("\n\n"));
  }, [readDraft, setDraft, queuedDrafts]);
  const stopped = Boolean(queued && (queued.stopped || queued.stopCount !== stopCount));
  const enqueue = useCallback((text: string) => {
    const next = {
      key: draftKey,
      inputId: queued?.inputId ?? inputId,
      text: queued ? `${queued.text}\n\n${text}` : text,
      stopped: stopped || stopPending,
      stopCount,
    };
    setQueued(next);
    queuedDrafts.current.set(draftKey, next.text);
    setDraft(draftKey, "");
  }, [draftKey, inputId, queued, stopped, stopPending, stopCount, setDraft, queuedDrafts]);

  // 首次消息接纳只确定原草稿的会话身份，不是切换会话。
  const move = useCallback((from: string, to: string) => {
    const text = queuedDrafts.current.get(from);
    if (text === undefined) return;
    queuedDrafts.current.delete(from);
    queuedDrafts.current.set(to, text);
    setQueued((current) => current?.key === from ? { ...current, key: to } : current);
  }, [queuedDrafts]);

  // 传输状态与最终消息共同决定发送时机；idle 也可能来自暂停、失败或能力撤回。
  useEffect(() => {
    if (!queued) return;
    // 1. 切走会话：排队内容回到原会话草稿。
    if (queued.key !== draftKey || queued.inputId !== inputId) return restore(queued);
    if (stopped && !queued.stopped) {
      setQueued({ ...queued, stopped: true });
      return;
    }
    if (turnBusy(status)) return;
    // 2. 用户停止、本轮出错或暂不可发：不替用户做决定。
    if (stopped || status === "error" || !canSend || !turnCompleted) return restore(queued);
    // 3. 正常落定：发出；失败同样退回草稿。
    setQueued(null);
    queuedDrafts.current.delete(queued.key);
    setDraft(queued.key, readDraft(queued.key));
    onSend(queued.text, []).catch(() => {
      // 发送失败可能晚于下一条入队，不能清掉新的排队意图。
      setDraft(queued.key, [queued.text, readDraft(queued.key)].filter(Boolean).join("\n\n"));
    });
  }, [queued, draftKey, inputId, status, stopped, turnCompleted, canSend, onSend, restore, queuedDrafts, setDraft, readDraft]);
  return { queued, stopped, enqueue, restore, move };
}

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
      if (document.querySelector(FLOATING_LAYER)) return;
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
                {isMedia ? null : <AttachmentFileBody name={getAttachmentLabel(attachment)} />}
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
// 落在输入卡里的 drop 由 PromptInput 自己处理；落在别处时这里入列。
// 对话框、菜单、选择器打开或聊天未就绪时不入列、不显示强调态；已被其它放置区处理的拖放不干预。
// 无论是否入列，都阻止浏览器把文件当页面打开——那会直接离开聊天页、丢掉草稿。
function WindowFileDrop({ enabled, onActiveChange }: { enabled: boolean; onActiveChange: (active: boolean) => void }) {
  const attachments = usePromptInputAttachments();
  const add = attachments.add;
  useEffect(() => {
    // 1. dragenter/dragleave 在每个元素上成对出入，用深度计数判断是否还在窗口内。
    let depth = 0;
    const isFileDrag = (event: DragEvent) => Array.from(event.dataTransfer?.types ?? []).includes("Files");
    const accepting = () => enabled && !document.querySelector(FLOATING_LAYER);
    const reset = () => { depth = 0; onActiveChange(false); };
    const onEnter = (event: DragEvent) => {
      if (!isFileDrag(event)) return;
      depth += 1;
      onActiveChange(accepting());
    };
    const onLeave = (event: DragEvent) => {
      if (!isFileDrag(event)) return;
      depth = Math.max(0, depth - 1);
      if (depth === 0 || event.relatedTarget === null) reset();
    };
    // 2. dragover 必须 preventDefault，drop 才会派发而不是被浏览器当作打开文件。
    const onOver = (event: DragEvent) => {
      if (!isFileDrag(event) || event.defaultPrevented) return;
      event.preventDefault();
      if (event.dataTransfer) event.dataTransfer.dropEffect = accepting() ? "copy" : "none";
    };
    // 3. 卡片内的 drop 已由 PromptInput 入列；其余位置在这里入列。
    const onDrop = (event: DragEvent) => {
      reset();
      if (!isFileDrag(event) || event.defaultPrevented) return;
      event.preventDefault();
      const insideComposer = event.target instanceof Element && event.target.closest("form.composer");
      if (accepting() && !insideComposer && event.dataTransfer?.files.length) {
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

// 输入卡里的附件块会截断文件名：悬停卡给出全名与类型；图片附带预览，文件不再垫一块只有小图标的空白。
function AttachmentHover({ attachment }: { attachment: ReturnType<typeof usePromptInputAttachments>["files"][number] }) {
  const category = getMediaCategory(attachment);
  const label = getAttachmentLabel(attachment);
  return <div className="attachment-hover">
    {category === "image" && attachment.url ? <img alt={label} className="attachment-hover-image" src={attachment.url} /> : null}
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
    label={stopPending ? "正在停止" : status === "finalizing" ? "正在收尾" : status === "uploading" ? "取消上传" : generating ? "中止回答" : "发送消息"}
    type={generating ? "button" : "submit"}
    title={generating && !stopping ? "中止回答（Esc）" : undefined}
    aria-keyshortcuts={generating ? "Escape" : undefined}
    aria-disabled={stopping || undefined}
    onClick={generating ? (event) => {
      // 取消会把同一按钮变回 submit；先阻断这次点击的默认提交。
      event.preventDefault();
      if (!stopping) onStop();
    } : undefined}
    disabled={!generating && (status === "finalizing" || stopPending || disabled || (!input.trim() && attachments.files.length === 0))}
  />;
}
