import { micromark } from "micromark";
import { gfm, gfmHtml } from "micromark-extension-gfm";
import "./markdown-prose.css";

// 图片交给调用方：返回 true 表示调用方接管（例如按相对路径异步加载），否则替换成省略说明。
export type MarkdownImageHandler = (image: HTMLImageElement, src: string) => boolean;

export interface MarkdownProseOptions {
  className?: string;
  // 同页多段渲染（如消息流）时关掉标题锚点，避免重复 id。
  headingIds?: boolean;
  onImage?: MarkdownImageHandler;
}

// 文件预览与插件面板共用的 Markdown 阅读渲染：GFM、原始 HTML 一律转义，链接外开、表格可横滚、任务框只读。
export function renderMarkdownProse(text: string, options: MarkdownProseOptions = {}): HTMLDivElement {
  const template = document.createElement("template");
  template.innerHTML = micromark(text, { extensions: [gfm()], htmlExtensions: [gfmHtml()] });
  template.content.querySelectorAll("img").forEach((image) => {
    const src = (image.getAttribute("src") || "").trim();
    image.removeAttribute("src");
    if (options.onImage?.(image, src)) return;
    const note = document.createElement("span");
    note.className = "md-image-note";
    note.textContent = `[图片：${image.alt || "外部或不支持的资源"}；已省略]`;
    image.replaceWith(note);
  });
  const prose = document.createElement("div");
  prose.className = options.className ? `md-prose ${options.className}` : "md-prose";
  prose.append(template.content);
  decorateProse(prose, options.headingIds ?? true);
  return prose;
}

// 1. 标题生成稳定锚点 2. 外链新窗口打开 3. 表格包一层横向滚动 4. 任务框只读。
function decorateProse(container: HTMLElement, headingIds: boolean): void {
  const slugs = new Map<string, number>();
  container.querySelectorAll("h1,h2,h3,h4,h5,h6").forEach((heading) => {
    if (!headingIds || heading.id) return;
    const slug = (heading.textContent || "section").toLowerCase()
      .replace(/[^\p{L}\p{N}\s_-]/gu, "").replace(/\s+/g, "-");
    const count = slugs.get(slug) || 0;
    slugs.set(slug, count + 1);
    heading.id = `md-${count ? `${slug}-${count}` : slug}`;
  });
  container.querySelectorAll("a").forEach((link) => {
    if (/^https?:/i.test(link.getAttribute("href") || "")) {
      link.target = "_blank";
      link.rel = "noopener noreferrer";
    }
  });
  container.querySelectorAll("table").forEach((table) => {
    const wrapper = document.createElement("div");
    wrapper.className = "md-table";
    wrapper.tabIndex = 0;
    table.replaceWith(wrapper);
    wrapper.append(table);
  });
  container.querySelectorAll("input").forEach((input) => {
    input.disabled = true;
  });
}
