import { Copy, PencilLine, Reply, X } from "lucide-react";

import { CheckGlyph, IconSwap } from "./icon-swap";

export type ReplyRole = "user" | "assistant";

export function SharedMessageActions({
  canReply,
  canCopy,
  copied,
  onReply,
  onCopy,
  onEdit,
}: {
  canReply: boolean;
  canCopy: boolean;
  copied: boolean;
  onReply: () => void;
  onCopy: () => void;
  /** 只给用户自己的提问：把原文放回输入框，改一改再问。 */
  onEdit?: () => void;
}) {
  return (
    <div className="shared-message-actions">
      {onEdit ? (
        <button type="button" onClick={onEdit} aria-label="放回输入框编辑" data-tip="放回输入框编辑">
          <PencilLine size={16} aria-hidden="true" />
        </button>
      ) : null}
      {canReply ? (
        <button type="button" onClick={onReply} aria-label="引用此消息">
          <Reply size={16} aria-hidden="true" />
        </button>
      ) : null}
      {canCopy ? (
        <button className={copied ? "copied" : ""} type="button" onClick={onCopy} aria-label={copied ? "已复制" : "复制消息"}>
          <IconSwap state={copied} a={<Copy size={16} aria-hidden="true" />} b={<CheckGlyph />} />
        </button>
      ) : null}
    </div>
  );
}

export function MessageReplyReference({
  role,
  author,
  preview,
  unavailable,
  canLoad = false,
  onNavigate,
}: {
  role?: ReplyRole;
  author?: string;
  preview: string;
  unavailable: boolean;
  canLoad?: boolean;
  onNavigate: () => void;
}) {
  return (
    <button
      className={`message-reply-reference ${unavailable ? "unavailable" : ""}`}
      type="button"
      onClick={onNavigate}
      disabled={unavailable && !canLoad}
      aria-label={`查看引用的 ${author ?? (role === "assistant" ? "Akashic" : "你")} 消息`}
    >
      <span>{author ?? (role === "assistant" ? "Akashic" : "你")}</span>
      <p aria-live="polite">{unavailable ? "原消息不在当前记录中" : preview}</p>
    </button>
  );
}

// 引用条显示给人看的名字：内部角色名不外露，其它来源（渠道用户名等）原样显示。
function replyAuthorLabel(author: string | undefined, role: ReplyRole | undefined): string {
  const who = author ?? role;
  if (who === "assistant") return "Akashic";
  if (who === "user" || who === undefined) return "你";
  return who;
}

export function ComposerReply({
  role,
  author,
  preview,
  onCancel,
}: {
  role?: ReplyRole;
  author?: string;
  preview: string;
  onCancel: () => void;
}) {
  return (
    <div className="composer-reply" aria-label={`正在回复 ${replyAuthorLabel(author, role)}`}>
      <Reply size={18} aria-hidden="true" />
      <div>
        <strong>回复 {replyAuthorLabel(author, role)}</strong>
        <span>{preview}</span>
      </div>
      <button type="button" onClick={onCancel} aria-label="取消引用"><X size={19} /></button>
    </div>
  );
}

/** 展开合并轨迹后按原消息和 part 定位，不把相同 part index 混为一个节点。 */
export function focusMessagePart(element: HTMLElement, messageId: string, partIndex?: number): void {
  const find = () => partIndex === undefined ? element :
    element.querySelector<HTMLElement>(`[data-process-message-id="${CSS.escape(messageId)}"][data-part-index="${partIndex}"]`)
      ?? (element.dataset.messageId === messageId ? element.querySelector<HTMLElement>(`[data-part-index="${partIndex}"]`) : null);
  const focus = () => {
    const target = find() ?? element;
    const disclosure = target.closest("details");
    if (disclosure) disclosure.open = true;
    target.focus({ preventScroll: true });
    target.scrollIntoView({ block: "center", behavior: "instant" });
  };
  if (partIndex !== undefined && !find()) {
    element.querySelector<HTMLButtonElement>('.process-trigger[aria-expanded="false"]')?.click();
    requestAnimationFrame(focus);
  } else focus();
}
