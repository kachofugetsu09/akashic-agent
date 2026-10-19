import { Clock3, FileText, FolderOpen, ImageIcon, Paperclip, Undo2 } from "lucide-react";
import { memo, useCallback, useEffect, useImperativeHandle, useLayoutEffect, useRef, useState, type ChangeEvent, type CSSProperties, type FocusEvent, type KeyboardEvent as ReactKeyboardEvent, type Ref, type RefObject } from "react";
import { createPortal, flushSync } from "react-dom";
import {
  PromptInput, PromptInputBody, PromptInputButton, PromptInputFooter, PromptInputTextarea, PromptInputTools,
  touchOnly, usePromptInputAttachments, type PromptInputFile, type RejectReason,
} from "@/components/ai-elements/prompt-input";
import {
  AttachmentCard, attachmentCardKind, isTextLike, mediaAttributes, PASTED_TEXT_NAME, readAttachmentText, useAttachmentPreview, useTextFacts,
} from "./attachment-card";
import type { TimelineReply } from "./message-timeline";
import { nextComposerExpanded } from "./composer-layout";
import { ComposerActionButton } from "./composer-action";
import { ComposerReply } from "./message-actions";
import { ModelCapsulePicker, ModelEffortAction, resolveVisibleRuntime } from "./model-capsule-picker";
import { compatibleEffort, type ChatModelRuntime } from "./model-capsule-data";
import { isGeneratingChatStatus, type ChatStatus } from "./web-chat-status";
import { useComposerSuggest } from "./composer-suggest";
import { setComposerBusy } from "./composer-contributions";
import { PluginUiSlot } from "./plugin-ui-runtime";

export type ComposerFile = { filename?: string; mediaType?: string; url?: string; sizeBytes?: number };

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
// 一条消息最多带的附件数（与 Desktop 相同）。
const MAX_ATTACHMENTS = 20;
// 粘贴的纯文本达到 4KB（UTF-8）就转成附件（与 Desktop 相同）。
const PASTE_AS_FILE_BYTES = 4096;

export const DesktopComposer = memo(function DesktopComposer({
  chatReady, canSend, sendProblem, draftKey, autoFocus = false, fresh = false, status, stopPending, stopCount, inputId, turnCompleted, modelState, selectedRuntimeId, selectedEffort, replyTarget,
  onModelChange, onCancelReply, onSend, onStop, lastInput = "", ref,
}: {
  chatReady: boolean;
  canSend: boolean;
  /** 不能发送的原因；输入框以 #chat-send-reason 关联它。 */
  sendProblem: string;
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
  const writeMention = useCallback((next: string) => {
    setInput(next);
    syncExpanded(textareaRef.current, next);
  }, [setInput, syncExpanded]);
  const mention = useComposerSuggest({ value: input, scope: draftKey, textareaRef, setValue: writeMention });
  const syncMention = mention.sync;
  const onInputChange = useCallback((event: ChangeEvent<HTMLTextAreaElement>) => {
    const next = event.target.value;
    setInput(next);
    syncExpanded(event.target, next);
    syncMention();
  }, [setInput, syncExpanded, syncMention]);
  const generating = isGeneratingChatStatus(status);
  // 布局阶段先写入，工具位插件在随后的失效回调里读到的就是新状态。
  useLayoutEffect(() => { setComposerBusy(generating); }, [generating]);
  const readDraft = useCallback((key: string) => drafts.current.get(key) ?? "", []);
  const followUp = useFollowUpQueue({ draftKey, status, stopPending, stopCount, inputId, turnCompleted, canSend: canSend && !replyTarget, readDraft, setDraft, queuedDrafts, onSend });
  const moveQueue = followUp.move;
  const submit = useCallback(async (text: string, files: ComposerFile[]) => {
    if (!canSend) throw new Error(sendProblem || "聊天服务暂不可用，请稍后重试。");
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
  }, [canSend, sendProblem, expanded, onSend, setInput, setDraft, draftKey, moveQueue]);
  // 输入卡内一行轻提示（排队被拒、附件超限），自动消失；同一时间只有一条。
  const [notice, setNotice] = useState<{ text: string; ms: number; title?: string } | null>(null);
  useEffect(() => {
    if (!notice) return;
    const timer = window.setTimeout(() => setNotice(null), notice.ms);
    return () => window.clearTimeout(timer);
  }, [notice]);
  const rejectFiles = useCallback((files: File[], reason: RejectReason) => {
    const names = files.map((file) => file.name || "未命名");
    if (reason === "count") {
      setNotice({ text: `每条消息最多 ${MAX_ATTACHMENTS} 个附件，多出的 ${files.length} 个未添加`, ms: 4000, title: names.join("\n") });
      return;
    }
    // 过长的文件名中间省略：开头与带扩展名的结尾最能区分文件，完整名称放在 title 里。
    const first = [...names[0]].length > 40 ? `${[...names[0]].slice(0, 20).join("")}…${[...names[0]].slice(-16).join("")}` : names[0];
    const label = names.length > 1 ? `${first} 等 ${names.length} 个文件` : first;
    setNotice({ text: `${label} 超过 ${MAX_ATTACHMENT_MB}MB，未添加`, ms: 4000, title: names.join("\n") });
  }, []);
  // 粘贴卡还原成文字：接在已有草稿之后，光标放到末尾。
  const restoreText = useCallback((text: string) => {
    const current = drafts.current.get(draftKey) ?? "";
    const next = current ? `${current}\n${text}` : text;
    setInput(next);
    window.requestAnimationFrame(() => {
      const textarea = textareaRef.current;
      if (!textarea) return;
      textarea.focus();
      textarea.setSelectionRange(next.length, next.length);
      syncExpanded(textarea, next);
    });
  }, [draftKey, setInput, syncExpanded]);
  const pasteAsFile = { bytes: PASTE_AS_FILE_BYTES, filename: PASTED_TEXT_NAME,
    onConverted: (lines: number) => setNotice({ text: `${lines} 行文本已作为附件（按住 Shift 粘贴可直接插入）`, ms: 3200 }) };
  const onComposerKeyDown = (event: ReactKeyboardEvent<HTMLTextAreaElement>) => {
    if (mention.onKeyDown(event)) return;
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
    {mention.menu ? <div className="composer-suggest-anchor">{mention.menu}</div> : null}
    <PromptInput
      className={`composer ${shellExpanded ? "is-expanded" : "is-compact"} ${input.trim() || replyTarget ? "has-text" : "empty"}${resting ? " is-resting" : ""}${fileDragActive ? " is-drop-target" : ""}`}
      multiple
      attachmentKey={draftKey}
      maxFileSize={MAX_ATTACHMENT_MB * 1024 * 1024}
      maxFiles={MAX_ATTACHMENTS}
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
        <button type="button" onClick={() => followUp.queued && followUp.restore(followUp.queued)} aria-label="撤回到输入框" data-tip="撤回到输入框">
          <Undo2 size={18} aria-hidden="true" />
        </button>
      </div> : null}
      {notice ? <p className="composer-queue-refused" role="status" title={notice.title}>{notice.text}</p> : null}
      {fileDragActive ? <ComposerDropScrim /> : null}
      <PromptInputBody>
        <ComposerAttachments onPresenceChange={setHasAttachments} onRestoreText={restoreText} />
        <PromptInputTextarea
          ref={textareaRef}
          autoFocus={autoFocus}
          className="composer__textarea !min-h-0"
          value={input}
          onChange={onInputChange}
          onKeyDown={onComposerKeyDown}
          onSelect={syncMention}
          onBlur={mention.close}
          {...mention.inputProps}
          pasteAsFile={pasteAsFile}
          aria-label="消息"
          aria-describedby={sendProblem ? "chat-send-reason" : undefined}
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
          {/* 插件的输入框工具（随草稿重新挂载，不同新对话的选择互不串）。 */}
          <PluginUiSlot key={draftKey} name="composer.toolbar" sessionId={draftKey.startsWith("new:") ? undefined : draftKey}
            refreshToken={generating ? 1 : 0} quiet />
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

// 输入卡里的附件托盘：单行横滚，溢出一侧渐隐；新卡入列后滚到末尾。
// 删除用 View Transition：被删的卡缩小淡出，其余卡平滑补位；聚焦卡上按 Delete/Backspace 删，焦点移到相邻卡或回到输入框。
function ComposerAttachments({ onPresenceChange, onRestoreText }: {
  onPresenceChange: (hasAttachments: boolean) => void;
  onRestoreText: (text: string) => void;
}) {
  const attachments = usePromptInputAttachments();
  const { files } = attachments;
  const trayRef = useRef<HTMLDivElement>(null);
  const transitionRef = useRef<ViewTransition | null>(null);
  const preview = useAttachmentPreview();
  const hasAttachments = files.length > 0;
  useEffect(() => {
    onPresenceChange(hasAttachments);
  }, [hasAttachments, onPresenceChange]);
  useTrayEdges(trayRef, files.length);
  const remove = (id: string, refocus: boolean) => {
    const index = files.findIndex((file) => file.id === id);
    const next = files[index + 1] ?? files[index - 1];
    // 删掉最后一张后托盘卸载、ref 变空：正文框要在删除前取好。
    const textarea = trayRef.current?.closest("form")?.querySelector("textarea");
    const apply = () => {
      flushSync(() => attachments.remove(id));
      if (!refocus) return;
      const target = next && trayRef.current?.querySelector<HTMLElement>(`[data-attachment-id="${next.id}"] .ak-attachment__surface`);
      (target ?? textarea)?.focus();
    };
    const tray = trayRef.current;
    if (!tray || !document.startViewTransition || window.matchMedia("(prefers-reduced-motion: reduce)").matches) return apply();
    tray.dataset.reflow = "";
    // 动画没放完又删一张：新过渡会跳过旧的，旧过渡的 ready 按预期拒绝；只由最新的过渡收尾。
    const transition = document.startViewTransition(apply);
    transitionRef.current = transition;
    transition.ready.catch(() => undefined);
    void transition.finished.finally(() => {
      if (transitionRef.current === transition) delete tray.dataset.reflow;
    });
  };
  if (!hasAttachments) return null;
  return <>
    <div ref={trayRef} className="composer-tray" role="list" aria-label={`附件，${files.length} 个`}>
      {files.map((file) => <div key={file.id} role="listitem" data-attachment-id={file.id} style={{ "--reflow-name": `attachment-${file.id}` } as CSSProperties}>
        <ComposerAttachment file={file} onRemove={(refocus) => remove(file.id, refocus)} onRestoreText={onRestoreText} onPreview={preview.open} />
      </div>)}
    </div>
    {preview.dialog}
  </>;
}

function ComposerAttachment({ file, onRemove, onRestoreText, onPreview }: {
  file: PromptInputFile;
  onRemove: (refocus: boolean) => void;
  onRestoreText: (text: string) => void;
  onPreview: ReturnType<typeof useAttachmentPreview>["open"];
}) {
  const name = file.filename || "未命名";
  const kind = attachmentCardKind(file.mediaType, name);
  const textual = kind === "paste" || (kind === "file" && isTextLike(file.mediaType, name));
  const facts = useTextFacts(file.url, textual);
  const media = kind === "image" || kind === "video";
  const restore = async () => {
    if (!file.url) return;
    onRestoreText(await readAttachmentText(file.url));
    onRemove(false);
  };
  return <AttachmentCard
    data={{ id: file.id, name, kind, mediaType: file.mediaType, url: file.url, sizeBytes: file.sizeBytes,
      lines: facts?.lines, previewText: kind === "paste" ? facts?.preview : undefined }}
    mediaTrigger={media && file.url ? mediaAttributes(file.url, name, kind) : undefined}
    onOpen={!media && textual && file.url ? () => onPreview({ name, url: file.url!, sizeBytes: file.sizeBytes }) : undefined}
    onRemove={() => onRemove(false)}
    onEdit={kind === "paste" ? () => void restore() : undefined}
    onSurfaceKeyDown={(event) => {
      if (event.key !== "Delete" && event.key !== "Backspace") return;
      event.preventDefault();
      onRemove(true);
    }}
  />;
}

// 托盘两侧有被遮住的卡时渐隐提示还能滚；附件数变多时滚到末尾露出新卡。
function useTrayEdges(trayRef: RefObject<HTMLDivElement | null>, count: number) {
  const previous = useRef(count);
  // 必要 layout effect：渐隐取决于实际滚动宽度，绘制前算好，避免先露硬边再淡出。
  useLayoutEffect(() => {
    const tray = trayRef.current;
    if (!tray) return;
    const measure = () => {
      tray.toggleAttribute("data-fade-start", tray.scrollLeft > 1);
      tray.toggleAttribute("data-fade-end", tray.scrollLeft + tray.clientWidth < tray.scrollWidth - 1);
    };
    if (count > previous.current) tray.scrollTo({ left: tray.scrollWidth, behavior: "smooth" });
    previous.current = count;
    measure();
    const resize = new ResizeObserver(measure);
    resize.observe(tray);
    tray.addEventListener("scroll", measure, { passive: true });
    return () => {
      resize.disconnect();
      tray.removeEventListener("scroll", measure);
    };
  }, [trayRef, count]);
}

// 拖文件进窗口：整窗模糊遮罩，内缩的高亮框里三个图标依次浮起（Desktop ChatComposerDropScrim）。
// 挂到 body 上：输入卡祖先的 backdrop-filter 会成为 fixed 的包含块，遮罩会被困在卡片里。
function ComposerDropScrim() {
  return createPortal(<div className="composer-drop-scrim" aria-hidden="true">
    <div className="composer-drop-scrim__ring">
      <span className="composer-drop-scrim__icons"><ImageIcon /><FileText /><FolderOpen /></span>
      松开以添加文件
    </div>
  </div>, document.body);
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
