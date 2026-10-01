import { memo, useCallback, useMemo } from "react";
import { copyReadingCode } from "./markdown-reading";
import { renderStaticMarkdown } from "./static-markdown";

export const StaticMessageResponse = memo(function StaticMessageResponse({
  children,
  onError,
}: {
  children: string;
  onError?: (error: unknown) => void;
}) {
  const html = useMemo(() => renderStaticMarkdown(children), [children]);
  const copyCode = useCallback((event: React.MouseEvent<HTMLDivElement>) => {
    const target = event.target;
    if (!(target instanceof Element)) return;
    const button = target.closest<HTMLButtonElement>("[data-static-code-copy]");
    if (!button) return;
    const code = button
      .closest(".static-code-block")
      ?.querySelector("code")
      ?.textContent;
    if (code === null || code === undefined) return;
    void copyReadingCode(code).then(() => {
      button.dataset.copied = "true";
      button.setAttribute("aria-label", "已复制");
      button.title = "已复制";
      window.setTimeout(() => {
        if (button.isConnected) {
          delete button.dataset.copied;
          button.setAttribute("aria-label", "复制代码");
          button.title = "复制代码";
        }
      }, 1500);
    }).catch((error: unknown) => {
      button.setAttribute("aria-label", "复制失败，点击重试");
      button.title = "复制失败，点击重试";
      if (onError) onError(error);
      else console.error("复制代码失败", error);
    });
  }, [onError]);

  return (
    <div
      className="static-message-response markdown-reading"
      onClick={copyCode}
      dangerouslySetInnerHTML={{ __html: html }}
    />
  );
});
