import SyntaxWorker from "./syntax.worker?worker&inline";
import type { GrammarName } from "./syntax.worker";

const EXTENSIONS: Record<string, GrammarName> = {
  ts: "typescript", mts: "typescript", cts: "typescript", tsx: "typescript",
  js: "javascript", mjs: "javascript", cjs: "javascript", jsx: "javascript",
  py: "python", pyi: "python",
  json: "json", jsonl: "json",
  sh: "bash", bash: "bash", zsh: "bash",
  css: "css", scss: "css",
  html: "xml", htm: "xml", xml: "xml", svg: "xml", vue: "xml",
  go: "go", rs: "rust",
  toml: "ini", ini: "ini", cfg: "ini", conf: "ini",
  yaml: "yaml", yml: "yaml",
  sql: "sql",
  diff: "diff", patch: "diff",
  md: "markdown", markdown: "markdown", mdown: "markdown", mdx: "markdown",
};

interface Token { text: string; classes: string }

function languageOf(path: string): GrammarName | undefined {
  const base = path.slice(path.lastIndexOf("/") + 1);
  const dot = base.lastIndexOf(".");
  if (dot <= 0) return undefined;
  return EXTENSIONS[base.slice(dot + 1).toLowerCase()];
}

function tokenLines(html: string): Token[][] {
  const template = document.createElement("template");
  template.innerHTML = html;
  const lines: Token[][] = [[]];
  const visit = (node: Node, classes: string) => {
    if (node.nodeType === Node.TEXT_NODE) {
      const parts = (node.textContent ?? "").split("\n");
      parts.forEach((text, index) => {
        if (index) lines.push([]);
        if (text) lines[lines.length - 1].push({ text, classes });
      });
    } else if (node instanceof HTMLSpanElement) {
      const names = [...node.classList].filter((name) => /^(hljs-[a-z_-]+|[a-z]+_+)$/.test(name));
      for (const child of node.childNodes) visit(child, `${classes} ${names.join(" ")}`.trim());
    }
  };
  for (const child of template.content.childNodes) visit(child, "");
  return lines;
}

export class FileSyntax {
  private worker: Worker | null = null;
  private timer = 0;
  private frame = 0;

  cancel(): void {
    window.clearTimeout(this.timer);
    cancelAnimationFrame(this.frame);
    this.worker?.terminate();
    this.worker = null;
  }

  highlight(path: string, lines: readonly string[], root: HTMLElement): void {
    this.cancel();
    const language = languageOf(path);
    if (!language || !lines.length || lines.length > 1500) return;
    const total = lines.reduce((sum, line) => sum + line.length, 0);
    if (total > 96_000 || lines.some((line) => line.length > 2000)) return;
    this.frame = requestAnimationFrame(() => {
      if (!root.isConnected) return;
      try {
        const worker = this.worker = new SyntaxWorker();
        this.timer = window.setTimeout(() => this.cancel(), 4000);
        worker.onerror = () => this.cancel();
        worker.onmessage = (event: MessageEvent<string | null>) => {
          window.clearTimeout(this.timer);
          worker.terminate();
          this.worker = null;
          if (!root.isConnected || typeof event.data !== "string") return;
          const parsed = tokenLines(event.data);
          const nodes = root.querySelectorAll<HTMLElement>(".md-code-text");
          let index = 0;
          const paint = () => {
            if (!root.isConnected) return;
            const deadline = performance.now() + 4;
            while (index < nodes.length && index < parsed.length) {
              const parts = parsed[index];
              if (parts && parts.length <= 256 && parts.map((item) => item.text).join("") === lines[index]) {
                const fragment = document.createDocumentFragment();
                for (const token of parts) {
                  const span = document.createElement("span");
                  span.textContent = token.text;
                  span.className = token.classes ? `md-token ${token.classes}` : "md-token";
                  fragment.append(span);
                }
                nodes[index].replaceChildren(fragment);
              }
              index += 1;
              if (performance.now() >= deadline) {
                this.frame = requestAnimationFrame(paint);
                return;
              }
            }
          };
          this.frame = requestAnimationFrame(paint);
        };
        worker.postMessage({ language, code: lines.join("\n") });
      } catch {
        this.cancel();
      }
    });
  }
}
