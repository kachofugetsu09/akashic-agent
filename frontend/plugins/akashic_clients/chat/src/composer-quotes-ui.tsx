import { Quote, X } from "lucide-react";
import { removeComposerQuote, useComposerQuotes, type ComposerQuoteItem } from "./composer-quotes";
import { DOCUMENT_INTENT, openInWorkbench } from "./shell-workbench";

/**
 * 输入卡里的工作台索引引用条（PDF 划词、Office 选区、代码行、Git Diff 等）。
 *
 * 引用以紧凑索引 Chip 挂在输入框上方，不再把大段文本硬塞进 textarea；
 * 悬停可预览完整摘录，带路径锚点时点击可直接跳回侧边栏对应位置。
 */
export function ComposerQuotes({ draftKey, sessionId }: { draftKey: string; sessionId: string }) {
  const quotes = useComposerQuotes(draftKey);
  if (quotes.length === 0) return null;

  const jumpToSource = (item: ComposerQuoteItem) => {
    if (!sessionId || !item.anchor?.path) return;
    void openInWorkbench(DOCUMENT_INTENT, sessionId, item.anchor);
  };

  return (
    <div className="composer-references" aria-label="已引用的文档片段">
      {quotes.map((item) => {
        const tooltip = item.preview ? `${item.source}\n\n${item.text.slice(0, 400)}` : item.source;
        const canJump = Boolean(sessionId && item.anchor?.path);
        return (
          <span className="composer-reference" key={item.id}>
            <Quote size={14} aria-hidden="true" />
            {canJump ? (
              <button
                type="button"
                className="composer-reference__title composer-reference__jump"
                data-akashic-private="title"
                title={tooltip}
                onClick={() => jumpToSource(item)}
              >
                {item.label}
                {item.preview ? ` · ${item.preview}` : ""}
              </button>
            ) : (
              <span className="composer-reference__title" data-akashic-private="title" title={tooltip}>
                {item.label}
                {item.preview ? ` · ${item.preview}` : ""}
              </span>
            )}
            <button
              type="button"
              aria-label={`移除引用 ${item.label}`}
              onClick={() => removeComposerQuote(draftKey, item.id)}
            >
              <X size={14} aria-hidden="true" />
            </button>
          </span>
        );
      })}
    </div>
  );
}
