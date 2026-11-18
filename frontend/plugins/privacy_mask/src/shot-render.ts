import { domToBlob } from "modern-screenshot";
import type { MessageSelection } from "@akashic/chat-ui-v1";

// 对话长图：在宿主排好的只读副本上打码、加页脚，按画布上限分段拍成 PNG。

export type PiiKind = "phone" | "email" | "user" | "secret";
export type MaskLevel = "off" | "pii" | "all";

export interface ShotOptions {
  mask: MaskLevel;
  /** 「敏感信息」档下用户取消遮挡的类型。 */
  unmasked: Partial<Record<PiiKind, boolean>>;
  wide: boolean;
  process: boolean;
  time: boolean;
  footer: boolean;
}

export interface ShotImage {
  blob: Blob;
  width: number;
  height: number;
}

export interface ShotResult {
  images: ShotImage[];
  /** 选中范围里识别到的敏感信息，按类型计数；与是否遮挡无关。 */
  found: Partial<Record<PiiKind, number>>;
}

export const PII_LABEL: Record<PiiKind, string> = { phone: "手机号", email: "邮箱", user: "用户名", secret: "密钥" };

const PII_PATTERNS: Record<PiiKind, RegExp> = {
  phone: /(?<!\d)(?:\+?86[\s-]?)?1[3-9]\d(?:[\s-]?\d{4}){2}(?!\d)/g,
  email: /[\w.+-]+@[\w-]+(?:\.[\w-]+)+/g,
  user: /(?<=(?:\/home\/|\/Users\/|[A-Za-z]:\\Users\\))[^/\\\s"'`]+/g,
  secret: /\b(?:sk|pk|rk|ghp|gho|ghs|github_pat|xox[abpr])[-_][A-Za-z0-9_-]{16,}/g,
};

// iOS Safari 单张画布的像素上限；超过就按消息边界切成几张。
const MAX_CANVAS_AREA = 16_777_216;
const CONTINUATION_HEIGHT = 40;

// 1. 找出一段文字里互不重叠的敏感片段，按出现顺序
function findPii(text: string): { start: number; end: number; kind: PiiKind }[] {
  const hits: { start: number; end: number; kind: PiiKind }[] = [];
  for (const kind of Object.keys(PII_PATTERNS) as PiiKind[]) {
    for (const match of text.matchAll(PII_PATTERNS[kind])) {
      const start = match.index ?? 0;
      hits.push({ start, end: start + match[0].length, kind });
    }
  }
  hits.sort((a, b) => a.start - b.start || b.end - a.end);
  return hits.filter((hit, index) => hits.findIndex((other) => other.start <= hit.start && other.end > hit.start) === index);
}

// 副本里的文字节点，连同代码块等 open shadow root 里的。
function textNodes(root: Node, into: Text[] = []): Text[] {
  const walker = document.createTreeWalker(root, NodeFilter.SHOW_TEXT | NodeFilter.SHOW_ELEMENT);
  for (let node = walker.nextNode(); node; node = walker.nextNode()) {
    if (node instanceof Text) into.push(node);
    else if (node instanceof Element && node.shadowRoot) textNodes(node.shadowRoot, into);
  }
  return into;
}

// 计数全部识别结果；需要遮挡的类型包成色块。色块写内联样式：影子树里外部样式表够不着。
function markPii(root: HTMLElement, cover: (kind: PiiKind) => boolean): Partial<Record<PiiKind, number>> {
  const found: Partial<Record<PiiKind, number>> = {};
  const nodes = textNodes(root);
  for (const node of nodes) {
    const text = node.data;
    const hits = findPii(text);
    if (hits.length === 0) continue;
    const parts: (string | HTMLElement)[] = [];
    let at = 0;
    for (const hit of hits) {
      found[hit.kind] = (found[hit.kind] ?? 0) + 1;
      if (!cover(hit.kind)) continue;
      parts.push(text.slice(at, hit.start));
      const block = document.createElement("span");
      block.className = "privacy-shot-pii";
      block.style.cssText = "background: var(--ak-ink-primary); color: transparent; border-radius: 3px; -webkit-box-decoration-break: clone; box-decoration-break: clone;";
      block.textContent = text.slice(hit.start, hit.end);
      parts.push(block);
      at = hit.end;
    }
    if (at > 0) node.replaceWith(...parts, text.slice(at));
  }
  return found;
}

function footerElement(): HTMLElement {
  const footer = document.createElement("div");
  footer.className = "privacy-shot-footer";
  const day = new Date();
  const date = `${day.getFullYear()}-${String(day.getMonth() + 1).padStart(2, "0")}-${String(day.getDate()).padStart(2, "0")}`;
  footer.innerHTML = `<span>${date}</span><b>Akashic</b>`;
  return footer;
}

async function imagesSettled(root: HTMLElement): Promise<void> {
  await Promise.all([...root.querySelectorAll("img")].map((image) => image.complete ? undefined
    : new Promise<void>((resolve) => { image.addEventListener("load", () => resolve(), { once: true }); image.addEventListener("error", () => resolve(), { once: true }); })));
}

// 2. 按消息边界贪心分段：每段连同上下留白不超过画布上限对应的高度
function planSegments(root: HTMLElement, rows: HTMLElement[], maxHeight: number): HTMLElement[][] {
  if (root.offsetHeight <= maxHeight || rows.length < 2) return [rows];
  const top = parseFloat(getComputedStyle(root).paddingTop);
  const end = rows[rows.length - 1].offsetTop + rows[rows.length - 1].offsetHeight;
  const tail = root.offsetHeight - end;
  const segments: HTMLElement[][] = [];
  let current: HTMLElement[] = [];
  let height = top;
  rows.forEach((row, index) => {
    const size = (index < rows.length - 1 ? rows[index + 1].offsetTop : end) - row.offsetTop;
    if (current.length && height + size + tail > maxHeight) {
      segments.push(current);
      current = [];
      height = top + CONTINUATION_HEIGHT;
    }
    current.push(row);
    height += size;
  });
  segments.push(current);
  return segments;
}

// 3. 逐段只显示本段消息拍照；单条仍超上限时降倍率
async function captureSegments(root: HTMLElement, rows: HTMLElement[], segments: HTMLElement[][], scale: number, alive: () => boolean): Promise<ShotImage[] | null> {
  const footer = root.querySelector<HTMLElement>(".privacy-shot-footer");
  const label = document.createElement("div");
  label.className = "privacy-shot-continued";
  const images: ShotImage[] = [];
  for (const [index, segment] of segments.entries()) {
    for (const row of rows) row.style.display = segment.includes(row) ? "" : "none";
    if (footer) footer.style.display = index === segments.length - 1 ? "" : "none";
    if (index > 0) {
      label.textContent = `接上一张 · ${index + 1}/${segments.length}`;
      root.prepend(label);
    }
    const width = root.offsetWidth;
    const height = root.offsetHeight;
    // 库把每层盒子的实测高度写成内联，外边距穿不出父级、段落挤在一起；高度交还给排版。
    const properties = [...getComputedStyle(root)].filter((name) => name !== "height" && name !== "block-size");
    const fit = Math.min(scale, Math.floor(Math.sqrt(MAX_CANVAS_AREA / (width * height)) * 100) / 100);
    const blob = await domToBlob(root, {
      scale: fit,
      type: "image/png",
      backgroundColor: getComputedStyle(root).backgroundColor,
      includeStyleProperties: properties,
      // 代码块等横向滚动区在图里只剩一条没用的滚动条。
      onCloneEachNode: (node) => { if (node instanceof HTMLElement) node.style.scrollbarWidth = "none"; },
      style: { position: "static", left: "auto", top: "auto", insetInlineStart: "auto", insetBlockStart: "auto" },
    });
    if (!alive()) return null;
    images.push({ blob, width: Math.round(width * fit), height: Math.round(height * fit) });
  }
  return images;
}

export async function renderShot(selection: MessageSelection, options: ShotOptions, alive: () => boolean): Promise<ShotResult | null> {
  const width = options.wide ? 720 : 390;
  const scale = options.wide ? 2 : 3;
  const stage = selection.stage({ width, process: options.process, time: options.time });
  try {
    const root = stage.element;
    const rows = [...root.children] as HTMLElement[];
    root.classList.add("privacy-shot");
    root.dataset.mask = options.mask;
    const found = markPii(root, (kind) => options.mask === "pii" && !options.unmasked[kind]);
    if (options.footer) root.append(footerElement());
    await imagesSettled(root);
    const segments = planSegments(root, rows, Math.floor(MAX_CANVAS_AREA / (width * scale * scale)));
    const images = await captureSegments(root, rows, segments, scale, alive);
    return images ? { images, found } : null;
  } finally {
    stage.release();
  }
}
