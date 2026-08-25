import { gfm, gfmHtml } from "micromark-extension-gfm";
import { micromark } from "micromark";

/** Render settled GFM while keeping raw HTML and unsafe protocols inert. */
export function renderStaticMarkdown(markdown: string) {
  const html = micromark(markdown, {
    extensions: [gfm()],
    htmlExtensions: [gfmHtml()],
  });
  // Memoh-style shell: code scrolls in its column; copy is a trailing sibling.
  return html.replace(
    /<pre><code([^>]*)>([\s\S]*?)<\/code><\/pre>/g,
    (_match, attrs: string, code: string) =>
      `<div class="static-code-block"><pre><code${attrs}>${code}</code></pre><button type="button" class="static-code-copy" data-static-code-copy aria-label="复制代码">复制</button></div>`,
  );
}
