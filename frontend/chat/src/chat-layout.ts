// 聊天的响应式断点按承载聊天的元素自身宽度判断，不按窗口宽度：同文档挂载时聊天只占页面一部分（工作台展开时更窄）。
// 样式表读 frame 上的属性（data-chat-width 列出已满足的 max-width 断点），脚本读 chatLayout()；两边同一个来源。
// rem 断点保留原媒体查询的语义：随用户的默认字号缩放（外壳不改 html 的字号，根节点字号就是默认字号）。
const MAX_WIDTHS: readonly (readonly [token: string, px: (rem: number) => number])[] = [
  ["820", () => 820], ["600", () => 600], ["36rem", (rem) => 36 * rem], ["32rem", (rem) => 32 * rem], ["480", () => 480],
];
const WIDE = 1200;
const LOW = "(max-height: 30rem)";
const SHORT_COARSE = "(max-height: 30rem) and (pointer: coarse)";
const TOUCH = "(hover: none)";

export interface ChatLayout {
  /** 紧凑外壳：宽度不超过 820px，或横屏手机这类矮的触屏。 */
  compact: boolean;
  /** 窄阅读列：宽度不超过 36rem。 */
  narrow: boolean;
  /** 右侧留白放得下提问目录：宽度至少 1200px。 */
  wide: boolean;
}

let current: ChatLayout = { compact: false, narrow: false, wide: true };
const listeners = new Set<() => void>();

export function chatLayout(): ChatLayout {
  return current;
}

export function subscribeChatLayout(listener: () => void): () => void {
  listeners.add(listener);
  return () => { listeners.delete(listener); };
}

function toggle(frame: HTMLElement, name: string, on: boolean): void {
  if (on) frame.dataset[name] = "";
  else delete frame.dataset[name];
}

// frame 是 html 时（独立页面与 iframe）按视口宽度算，与原来的媒体查询一致；否则按聊天根节点的宽度。
export function observeChatLayout(frame: HTMLElement): () => void {
  const low = window.matchMedia(LOW);
  const shortCoarse = window.matchMedia(SHORT_COARSE);
  const touch = window.matchMedia(TOUCH);
  const apply = () => {
    const width = frame === document.documentElement ? window.innerWidth : frame.getBoundingClientRect().width;
    const rem = Number.parseFloat(getComputedStyle(document.documentElement).fontSize) || 16;
    frame.dataset.chatWidth = MAX_WIDTHS.filter(([, px]) => width <= px(rem)).map(([token]) => token).join(" ");
    toggle(frame, "chatWide", width >= WIDE);
    toggle(frame, "chatLow", low.matches);
    toggle(frame, "chatShort", shortCoarse.matches);
    toggle(frame, "chatTouch", touch.matches);
    const next = { compact: width <= 820 || shortCoarse.matches, narrow: width <= 36 * rem, wide: width >= WIDE };
    if (next.compact === current.compact && next.narrow === current.narrow && next.wide === current.wide) return;
    current = next;
    for (const listener of [...listeners]) listener();
  };
  const observer = new ResizeObserver(apply);
  observer.observe(frame);
  const queries = [low, shortCoarse, touch];
  for (const query of queries) query.addEventListener("change", apply);
  window.addEventListener("resize", apply);
  apply();
  return () => {
    observer.disconnect();
    for (const query of queries) query.removeEventListener("change", apply);
    window.removeEventListener("resize", apply);
  };
}
