import { gfm, gfmHtml } from "micromark-extension-gfm";
import { micromark } from "micromark";

/** Render settled GFM while keeping raw HTML and unsafe protocols inert. */
export function renderStaticMarkdown(markdown: string) {
  const html = micromark(markdown, {
    extensions: [gfm()],
    htmlExtensions: [gfmHtml()],
  });
  return html.replaceAll(
    "<pre><code",
    '<pre class="static-code-block"><button type="button" class="static-code-copy" data-static-code-copy aria-label="复制代码">复制</button><code',
  );
}
