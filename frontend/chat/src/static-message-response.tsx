import { memo, useCallback, useMemo } from "react";
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
    void navigator.clipboard.writeText(code).then(() => {
      button.textContent = "已复制";
      window.setTimeout(() => {
        if (button.isConnected) button.textContent = "复制";
      }, 1500);
    }).catch((error: unknown) => {
      if (onError) onError(error);
      else console.error("复制代码失败", error);
    });
  }, [onError]);

  return (
    <div
      className="static-message-response"
      onClick={copyCode}
      dangerouslySetInnerHTML={{ __html: html }}
    />
  );
});
