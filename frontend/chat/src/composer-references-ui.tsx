import { MessagesSquare, X } from "lucide-react";
import { useEffect, useState, type RefObject } from "react";
import { addReference, MAX_REFERENCES, readSessionDrag, removeReference, useComposerReferences } from "./composer-references";

/**
 * 输入卡里的会话引用条。
 *
 * 引用不是「把正文贴进来」，所以这里只显示会话标题，正文由模型在需要时用
 * `read_session` 去读。这样聊天里能看清引用了谁，上下文预算也仍然由当前会话掌握。
 */
export function ComposerReferences({ draftKey }: { draftKey: string }) {
  const references = useComposerReferences(draftKey);
  const [overflow, setOverflow] = useState(false);
  useEffect(() => {
    if (!overflow) return;
    const timer = window.setTimeout(() => setOverflow(false), 2600);
    return () => window.clearTimeout(timer);
  }, [overflow]);
  // 超出上限的拖入由 addReference 返回 false 表示；这里用一个事件把提示引出来。
  useEffect(() => {
    const onRejected = () => setOverflow(true);
    window.addEventListener("akashic:reference-rejected", onRejected);
    return () => window.removeEventListener("akashic:reference-rejected", onRejected);
  }, []);
  if (references.length === 0 && !overflow) return null;
  return (
    <>
      {references.length > 0 ? (
        <div className="composer-references" aria-label="已引用的会话">
          {references.map((reference) => (
            <span className="composer-reference" key={reference.sessionId}>
              <MessagesSquare size={15} aria-hidden="true" />
              <span className="composer-reference__title" title={reference.title}>{reference.title}</span>
              <button type="button" aria-label={`取消引用 ${reference.title}`}
                onClick={() => removeReference(draftKey, reference.sessionId)}>
                <X size={15} aria-hidden="true" />
              </button>
            </span>
          ))}
        </div>
      ) : null}
      {overflow ? <p className="composer-reference__notice" role="status">
        一次最多引用 {MAX_REFERENCES} 个会话
      </p> : null}
    </>
  );
}

/**
 * 会话行拖进输入卡的接收区。
 *
 * 只在 dataTransfer 带我们的自定义类型时介入：文件拖放、侧栏内部排序都不受影响，
 * 已被其它放置区处理过的（defaultPrevented）也不再插手。
 */
export function useReferenceDrop(target: RefObject<HTMLElement | null>, draftKey: string, enabled: boolean) {
  const [active, setActive] = useState(false);
  useEffect(() => {
    if (!enabled) { setActive(false); return; }
    const element = target.current;
    if (!element) return;
    const isSessionDrag = (event: DragEvent) => Boolean(event.dataTransfer?.types.includes("application/x-akashic-session"));
    const onOver = (event: DragEvent) => {
      if (!isSessionDrag(event) || event.defaultPrevented) return;
      event.preventDefault();
      if (event.dataTransfer) event.dataTransfer.dropEffect = "copy";
      setActive(true);
    };
    const onLeave = (event: DragEvent) => {
      if (!isSessionDrag(event)) return;
      const next = event.relatedTarget;
      if (next instanceof Node && element.contains(next)) return;
      setActive(false);
    };
    const onDrop = (event: DragEvent) => {
      setActive(false);
      if (event.defaultPrevented) return;
      const reference = readSessionDrag(event.dataTransfer);
      if (!reference) return;
      event.preventDefault();
      if (!addReference(draftKey, reference)) {
        window.dispatchEvent(new Event("akashic:reference-rejected"));
      }
    };
    const onEnd = () => setActive(false);
    element.addEventListener("dragover", onOver);
    element.addEventListener("dragleave", onLeave);
    element.addEventListener("drop", onDrop);
    element.addEventListener("dragend", onEnd);
    return () => {
      element.removeEventListener("dragover", onOver);
      element.removeEventListener("dragleave", onLeave);
      element.removeEventListener("drop", onDrop);
      element.removeEventListener("dragend", onEnd);
    };
  }, [target, draftKey, enabled]);
  return active;
}
