import SyntaxWorker from "./syntax.worker?worker&inline";
import type { SyntaxRow } from "./syntax.worker";

const extensions: Record<string, string> = {
  ts: "typescript", mts: "typescript", cts: "typescript", js: "javascript", mjs: "javascript", cjs: "javascript",
  tsx: "typescript", jsx: "javascript", py: "python", json: "json", sh: "bash", bash: "bash", css: "css", html: "xml", htm: "xml", go: "go", rs: "rust",
};
interface Token { text: string; classes: string }
interface Highlighted { indices: number[]; html: string }

// Parse escaped highlighter output in an inert template. Only text and allowlisted span
// class names are copied into the live DOM. Source HTML is never inserted into the page.
function tokenLines(html: string): Token[][] {
  const template = document.createElement("template");
  template.innerHTML = html;
  const lines: Token[][] = [[]];
  const visit = (node: Node, classes: string) => {
    if (node.nodeType === Node.TEXT_NODE) {
      const parts = (node.textContent ?? "").split("\n");
      parts.forEach((text, i) => {
        if (i) lines.push([]);
        if (text) lines[lines.length - 1].push({ text, classes });
      });
    } else if (node instanceof HTMLSpanElement) {
      const names = [...node.classList].filter(name => /^(hljs-[a-z_-]+|[a-z]+_+)$/.test(name));
      for (const child of node.childNodes) visit(child, `${classes} ${names.join(" ")}`.trim());
    }
  };
  for (const child of template.content.childNodes) visit(child, "");
  return lines;
}

/** At most one worker/job per mounted tab. Hard deadline bounds pathological regexes.
 * Failed/oversized jobs remain plain text; no main-thread tokenization fallback.
 */
export class DiffSyntax {
  private worker: Worker | null = null;
  private timer = 0;
  private frame = 0;

  cancel(): void {
    clearTimeout(this.timer);
    cancelAnimationFrame(this.frame);
    this.worker?.terminate();
    this.worker = null;
  }

  highlight(path: string, rows: SyntaxRow[], root: HTMLElement): void {
    this.cancel();
    const language = extensions[path.split(".").pop()?.toLowerCase() ?? ""];
    if (!language || rows.length > 1500 || rows.reduce((n, r) => n + r.text.length, 0) > 96_000 || rows.some(r => r.text.length > 2000)) return;
    this.frame = requestAnimationFrame(() => {
      if (!root.isConnected) return;
      try {
        const worker = this.worker = new SyntaxWorker();
        this.timer = window.setTimeout(() => this.cancel(), 4000);
        worker.onerror = () => this.cancel();
        worker.onmessage = (event: MessageEvent<Highlighted[] | null>) => {
          clearTimeout(this.timer);
          worker.terminate();
          this.worker = null;
          if (!root.isConnected || !event.data) return;
          const groups = event.data;
          const nodes = root.querySelectorAll<HTMLElement>(".changes-code");
          let group = 0;
          let line = 0;
          let tokens: Token[][] | null = null;
          const paint = () => {
            if (!root.isConnected) return;
            const deadline = performance.now() + 4;
            while (group < groups.length) {
              const current = groups[group];
              tokens ??= tokenLines(current.html);
              while (line < current.indices.length) {
                const index = current.indices[line];
                const parts = tokens[line];
                if (parts && parts.length <= 256 && parts.map(t => t.text).join("") === rows[index].text.slice(1)) {
                  const content = document.createDocumentFragment();
                  content.append(rows[index].text[0]);
                  for (const token of parts) {
                    const span = document.createElement("span");
                    span.textContent = token.text;
                    span.className = `changes-token ${token.classes}`;
                    content.append(span);
                  }
                  nodes[index].replaceChildren(content);
                }
                line++;
                if (performance.now() >= deadline) { this.frame = requestAnimationFrame(paint); return; }
              }
              group++; line = 0; tokens = null;
            }
          };
          this.frame = requestAnimationFrame(paint);
        };
        worker.postMessage({ language, rows });
      } catch { this.cancel(); }
    });
  }
}
