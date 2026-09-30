import { gfm, gfmHtml } from "micromark-extension-gfm";
import { micromark } from "micromark";
import { isConceptLabel, isNumericColumn, isPlainTextLanguage } from "./markdown-reading";

/** Render settled GFM while keeping raw HTML and unsafe protocols inert. */
export function renderStaticMarkdown(markdown: string) {
  const html = micromark(markdown, {
    extensions: [gfm()],
    htmlExtensions: [gfmHtml()],
  });
  // 1. 只处理 micromark 已转义、已过滤协议的输出；模板不挂到页面。
  const template = document.createElement("template");
  template.innerHTML = html;
  const root = template.content;

  for (const paragraph of root.querySelectorAll("p")) {
    const content = Array.from(paragraph.childNodes).filter((child) => child.nodeType !== Node.TEXT_NODE || child.textContent?.trim());
    if (content.length !== 1 || !(content[0] instanceof HTMLElement) || content[0].tagName !== "STRONG") continue;
    paragraph.classList.add((content[0].textContent?.length ?? 0) < 65 ? "reading-topic" : "reading-emphasis");
  }

  // 2. 保留列表语义和每个文字节点，只给标签与说明添加排版容器。
  for (const list of root.querySelectorAll("ul")) {
    const items = Array.from(list.children);
    const labels = items.map((item) => {
      const paragraph = item.firstElementChild?.tagName === "P" ? item.firstElementChild : item;
      // 只跳过空白文字；图片、勾选框等没有 textContent 的元素仍占据原文位置。
      const first = Array.from(paragraph.childNodes).find((child) => child.nodeType !== Node.TEXT_NODE || child.textContent?.trim());
      if (!(first instanceof HTMLElement) || first.tagName !== "STRONG") return null;
      const description = item.cloneNode(true) as Element;
      description.querySelector("strong")!.remove();
      return isConceptLabel(first.textContent ?? "", description.textContent ?? "") ? first : null;
    });
    if (items.length < 2 || labels.some((label) => label === null)) continue;
    list.classList.add("reading-concepts");
    list.setAttribute("role", "list");
    items.forEach((item, index) => {
      const label = labels[index]!;
      const labelColumn = document.createElement("span");
      labelColumn.className = "reading-concept-label";
      labelColumn.append(label);
      const description = document.createElement("div");
      description.className = "reading-concept-description";
      description.append(...item.childNodes);
      item.append(labelColumn, description);
    });
  }

  // 3. 宽表只在自己的容器内滚动，显式 Markdown 对齐优先。
  for (const table of root.querySelectorAll("table")) {
    const headers = Array.from(table.querySelectorAll("thead th"));
    const rows = Array.from(table.querySelectorAll("tbody tr"));
    headers.forEach((header, index) => {
      const cells = rows.map((row) => row.children[index]);
      if (cells.some((cell) => !cell)) return;
      if (!isNumericColumn(cells.map((cell) => cell.textContent ?? ""))) return;
      header.classList.add("reading-number");
      cells.forEach((cell) => cell.classList.add("reading-number"));
    });
    const frame = document.createElement("div");
    frame.className = "reading-table";
    table.replaceWith(frame);
    frame.append(table);
  }

  // 4. 代码文本原样保留，纯文本围栏不显示语言标题。
  for (const pre of root.querySelectorAll("pre")) {
    const code = pre.firstElementChild;
    if (!code || code.tagName !== "CODE") continue;
    const language = Array.from(code.classList).find((name) => name.startsWith("language-"))?.slice(9) ?? "";
    const frame = document.createElement("div");
    frame.className = `static-code-block${isPlainTextLanguage(language) ? " reading-plain-text" : ""}`;
    const button = document.createElement("button");
    button.type = "button";
    button.className = "reading-code-copy";
    button.dataset.staticCodeCopy = "";
    button.setAttribute("aria-label", "复制代码");
    button.title = "复制代码";
    button.innerHTML = '<svg aria-hidden="true" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><rect x="8" y="8" width="12" height="12" rx="2"/><path d="M16 8V4a2 2 0 0 0-2-2H4a2 2 0 0 0-2 2v10a2 2 0 0 0 2 2h4"/></svg>';
    pre.replaceWith(frame);
    frame.append(pre, button);
  }
  return template.innerHTML;
}
