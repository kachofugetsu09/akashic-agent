import { Check, Copy, Reply, X } from "lucide-react";

export type ReplyRole = "user" | "assistant";

export function SharedMessageActions({
  canReply,
  canCopy,
  copied,
  onReply,
  onCopy,
}: {
  canReply: boolean;
  canCopy: boolean;
  copied: boolean;
  onReply: () => void;
  onCopy: () => void;
}) {
  return (
    <div className="shared-message-actions">
      {canReply ? (
        <button type="button" onClick={onReply} aria-label="引用此消息">
          <Reply size={16} aria-hidden="true" />
        </button>
      ) : null}
      {canCopy ? (
        <button className={copied ? "copied" : ""} type="button" onClick={onCopy} aria-label={copied ? "已复制" : "复制消息"}>
          {copied ? <Check size={16} aria-hidden="true" /> : <Copy size={16} aria-hidden="true" />}
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
  onNavigate,
}: {
  role?: ReplyRole;
  author?: string;
  preview: string;
  unavailable: boolean;
  onNavigate: () => void;
}) {
  return (
    <button
      className={`message-reply-reference ${unavailable ? "unavailable" : ""}`}
      type="button"
      onClick={onNavigate}
      disabled={unavailable}
      aria-label={`查看引用的 ${author ?? (role === "assistant" ? "Akashic" : "你")} 消息`}
    >
      <span>{author ?? (role === "assistant" ? "Akashic" : "你")}</span>
      <p aria-live="polite">{unavailable ? "原消息不在当前记录中" : preview}</p>
    </button>
  );
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
    <div className="composer-reply" aria-label={`正在回复 ${author ?? (role === "assistant" ? "Akashic" : "你")}`}>
      <Reply size={18} aria-hidden="true" />
      <div>
        <strong>回复 {author ?? (role === "assistant" ? "Akashic" : "你")}</strong>
        <span>{preview}</span>
      </div>
      <button type="button" onClick={onCancel} aria-label="取消引用"><X size={19} /></button>
    </div>
  );
}
