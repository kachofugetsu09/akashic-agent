import { memo, useCallback, useEffect, useMemo, useRef } from "react";
import { copyReadingCode, jumpToFootnote, watchOverflowEdges } from "./markdown-reading";
import { FILE_INTENT, openFileInWorkbench, useWorkbenchAccepts, workspaceFileRef } from "./shell-workbench";
import { renderStaticMarkdown } from "./static-markdown";

export const StaticMessageResponse = memo(function StaticMessageResponse({
  children,
  onError,
}: {
  children: string;
  onError?: (error: unknown) => void;
}) {
  const html = useMemo(() => renderStaticMarkdown(children), [children]);
  const ref = useRef<HTMLDivElement>(null);
  // 必要 effect：静态 HTML 里的宽表同样要标出哪边还有列；内容换了就重新接上。
  useEffect(() => {
    const tables = ref.current ? [...ref.current.querySelectorAll<HTMLElement>(".reading-table")] : [];
    const stops = tables.map(watchOverflowEdges);
    return () => stops.forEach((stop) => stop());
  }, [html]);
  const acceptsFiles = useWorkbenchAccepts(FILE_INTENT);
  // 必要 effect：行内代码里的文件路径包成按钮（与流式路径的 ReadingInlineCode 一致）；工作台不接受文件意图时拆回去。
  useEffect(() => {
    const root = ref.current;
    if (!root) return;
    for (const link of root.querySelectorAll<HTMLButtonElement>("button.reading-file-link")) link.replaceWith(...link.childNodes);
    if (!acceptsFiles) return;
    for (const code of root.querySelectorAll<HTMLElement>(":not(pre) > code")) {
      // 已经在链接里的代码不再包按钮：<a> 里不能嵌交互控件，点击也会同时打开链接。
      const file = code.closest("a") ? null : workspaceFileRef(code.textContent ?? "");
      if (!file) continue;
      const button = document.createElement("button");
      Object.assign(button, { type: "button", className: "reading-file-link", title: `在工作台里打开 ${file.path}` });
      button.dataset.filePath = file.path;
      if (file.line !== null) button.dataset.fileLine = String(file.line);
      code.replaceWith(button);
      button.append(code);
    }
  }, [html, acceptsFiles]);
  const openFile = useCallback((event: React.MouseEvent<HTMLDivElement>) => {
    const link = event.target instanceof Element ? event.target.closest<HTMLElement>("[data-file-path]") : null;
    if (link?.dataset.filePath) void openFileInWorkbench({ path: link.dataset.filePath, line: link.dataset.fileLine ? Number(link.dataset.fileLine) : null });
  }, []);
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
      ref={ref}
      className="static-message-response markdown-reading"
      onClick={(event) => { copyCode(event); openFile(event); }}
      onClickCapture={jumpToFootnote}
      dangerouslySetInnerHTML={{ __html: html }}
    />
  );
});
