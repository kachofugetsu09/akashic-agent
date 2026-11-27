import { Reply, X } from "lucide-react";

export type ReplyRole = "user" | "assistant";

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
    <div className="composer-reply" data-akashic-private="container" aria-label={`正在回复 ${replyAuthorLabel(author, role)}`}>
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
  // 整条引用：行本身就是这条消息时定位行；它被并进别的轮次（如落定后收起的中途旁白）时定位过程里的原消息。
  const find = () => partIndex === undefined
    ? element.dataset.messageId === messageId ? element
      : element.querySelector<HTMLElement>(`.flow-narration[data-process-message-id="${CSS.escape(messageId)}"]`)
        ?? element.querySelector<HTMLElement>(`[data-process-message-id="${CSS.escape(messageId)}"]`)
    : element.querySelector<HTMLElement>(`[data-process-message-id="${CSS.escape(messageId)}"][data-part-index="${partIndex}"]`)
      ?? (element.dataset.messageId === messageId ? element.querySelector<HTMLElement>(`[data-part-index="${partIndex}"]`) : null);
  const focus = () => {
    const target = find() ?? element;
    const disclosure = target.closest("details");
    if (disclosure) disclosure.open = true;
    // 自定义折叠可能包着目标，也可能就在目标（整条过程项）里面。
    const customDisclosure = target.closest("[data-disclosure-root]") ?? target.querySelector("[data-disclosure-root]");
    customDisclosure?.querySelector<HTMLButtonElement>('[data-disclosure-trigger][aria-expanded="false"]')?.click();
    target.focus({ preventScroll: true });
    target.scrollIntoView({ block: "center", behavior: "instant" });
  };
  const found = find();
  // 目标还收在过程里（未挂载，或挂载着但轮次摘要收起）：先展开轮次摘要（点击同步提交挂载），再展开其中的折叠组。
  const folded = found?.closest(".turn-process:not(.is-open)");
  if (!found || folded) {
    (folded ?? element).querySelector<HTMLButtonElement>('.turn-process-summary[aria-expanded="false"]')?.click();
    if (!found) element.querySelector<HTMLButtonElement>('.process-trigger[aria-expanded="false"]')?.click();
    requestAnimationFrame(focus);
  } else focus();
}
