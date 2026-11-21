// 展开/收起的合成线程写法（design.md §2.3）：布局一步到位，下方元素用 transform 从旧位置滑到新位置，内容只动 opacity。
// 只产生 transform / opacity 动画，主线程忙时照样逐帧推进。

export type MotionPhase = "open" | "close";

// 跟随者在滑动期间盖住还没露出（或正在收起）的内容：不透明底色、层级高于展开区，由 CSS 按此类名给出。
const COVER_CLASS = "is-motion-cover";

// CSS 时长转毫秒：构建会把 240ms 压成 .24s，单位必须按 s/ms 换算。
export function cssDurationMs(value: string): number {
  const match = value.trim().match(/^(-?[\d.]+)(ms|s)$/);
  if (!match) return 0;
  const amount = Number.parseFloat(match[1]);
  if (!Number.isFinite(amount)) return 0;
  return match[2] === "s" ? amount * 1000 : amount;
}

// 读时长/曲线 token；减少动态效果时时长 token 为 0。
export function motionTiming(element: Element, phase: MotionPhase): { duration: number; easing: string } {
  const styles = getComputedStyle(element);
  const duration = cssDurationMs(styles.getPropertyValue(`--ak-sys-duration-${phase}`));
  const easing = styles.getPropertyValue(`--ak-sys-motion-${phase}`).trim();
  return { duration, easing: easing || "ease" };
}

// host 之后、boundary 之内的元素：沿祖先链逐层收集后继兄弟，只留滑动途中会进入 boundary 可见区的（其余直接落位，看不见）。
// travel 是滑动距离：展开后被推到可见区下方、收起时将从下方滑进来的元素都还在这段范围里。
export function followingInView(host: Element, boundary: Element, travel = 0): HTMLElement[] {
  const view = boundary.getBoundingClientRect();
  const out: HTMLElement[] = [];
  for (let node: Element | null = host; node && node !== boundary; node = node.parentElement) {
    for (let next = node.nextElementSibling; next; next = next.nextElementSibling) {
      if (!(next instanceof HTMLElement)) continue;
      const rect = next.getBoundingClientRect();
      if (rect.height === 0 && rect.width === 0) continue;
      if (rect.bottom < view.top || rect.top - travel > view.bottom) continue;
      out.push(next);
    }
  }
  return out;
}

// 展开之后调用（内容已在布局里，高度 height）：跟随者从上移 height 处滑回原位，内容渐显。
export function slideOpen(host: Element, followers: HTMLElement[], content: HTMLElement, height: number, timing: { duration: number; easing: string }): Animation[] {
  if (timing.duration <= 0 || height <= 0) return [];
  const options: KeyframeAnimationOptions = { duration: timing.duration, easing: timing.easing };
  const gaps = coverGaps(host, followers);
  const animations = followers.map((element, index) => coverWhile(element, gaps[index], element.animate(
    [{ transform: `translateY(${-height}px)` }, { transform: "none" }], options)));
  animations.push(content.animate([{ opacity: 0, transform: "translateY(-4px)" }, { opacity: 1, transform: "none" }], options));
  return animations;
}

// 收起：布局先不动，跟随者上移 height 盖住内容、内容渐隐；停在终点（fill: forwards），由调用方在卸载内容的同一提交里 cancel。
export function slideClose(host: Element, followers: HTMLElement[], content: HTMLElement, height: number, timing: { duration: number; easing: string }): Animation[] {
  if (timing.duration <= 0 || height <= 0) return [];
  const options: KeyframeAnimationOptions = { duration: timing.duration, easing: timing.easing, fill: "forwards" };
  const gaps = coverGaps(host, followers);
  const animations = followers.map((element, index) => coverWhile(element, gaps[index], element.animate(
    [{ transform: "none" }, { transform: `translateY(${-height}px)` }], options)));
  animations.push(content.animate([{ opacity: 1 }, { opacity: 0 }], options));
  return animations;
}

// 每个跟随者当前负责遮盖的动画：上一轮动画的 cancel 事件是异步派发的，不能撤掉新一轮加上的遮盖。
const coverOwner = new WeakMap<HTMLElement, Animation>();

// 跟随者与上一个元素（第一个是 host）之间的缝：网格间距、外边距。遮盖层向上补齐这段，内容不从缝里透出来。
function coverGaps(host: Element, followers: HTMLElement[]): number[] {
  return gapsBelow(host.getBoundingClientRect().bottom, followers.map((element) => element.getBoundingClientRect()));
}

function gapsBelow(start: number, rects: Array<{ top: number; bottom: number }>): number[] {
  let previous = start;
  return rects.map((rect) => {
    const gap = Math.max(0, rect.top - previous);
    previous = Math.max(previous, rect.bottom);
    return gap;
  });
}

function coverWhile(element: HTMLElement, gap: number, animation: Animation): Animation {
  coverOwner.set(element, animation);
  element.classList.add(COVER_CLASS);
  // 多盖 1px：滑动中位置落在亚像素上，缝边会有抗锯齿残影。
  element.style.setProperty("--motion-cover-gap", `${gap > 0 ? Math.ceil(gap) + 1 : 0}px`);
  const release = () => {
    if (coverOwner.get(element) !== animation) return;
    coverOwner.delete(element);
    element.classList.remove(COVER_CLASS);
    element.style.removeProperty("--motion-cover-gap");
  };
  animation.addEventListener("finish", () => { if (animation.effect?.getComputedTiming().fill !== "forwards") release(); });
  animation.addEventListener("cancel", release);
  return animation;
}

// 展开区开合的 FLIP（消息区）：apply(open) 切布局并做滚动修正。滚动位置可能被钳制、被修正，
// 所以不按高度推算，而是量出 boundary 内每个元素前后的实际位置，按位置差滑动；内容只动 opacity，跟着标题行走。
// 展开：布局一步到位后起步。收起：先量终点再退回起点布局，动画停在终点（fill: forwards），
// 调用方在 finished 后再 apply(false) 并 settle，同一任务里落位。上一轮没播完时从此刻看到的位置接着走。
export interface DisclosureFlip {
  host: HTMLElement;
  content: HTMLElement;
  boundary: HTMLElement;
  // 吸顶元素的选择器：吸住的元素不随祖先位移，含有吸住元素的祖先要拆开逐个滑动。
  pinned: string;
  open: boolean;
  apply: (open: boolean) => void;
}

export function flipDisclosure(flip: DisclosureFlip, previous: Animation[]): Animation[] {
  const { host, content, boundary, open, apply } = flip;
  const timing = motionTiming(host, open ? "open" : "close");
  // 1. 此刻看到的位置（含上一轮动画的位移）；再撤掉上一轮。
  const movers = timing.duration > 0 ? collectMovers(flip) : [];
  const seen = movers.map(top);
  const fade = previous.length ? Number(getComputedStyle(content).opacity) : open ? 0 : 1;
  settle(previous);
  if (timing.duration <= 0) {
    apply(open);
    return [];
  }
  // 2. 起点、终点布局；收起要退回起点布局播放，滚动位置一并还原。
  const before = movers.map((element) => element.getBoundingClientRect());
  const hostBefore = host.getBoundingClientRect().bottom;
  const scroll = boundary.scrollTop;
  apply(open);
  const after = movers.map((element) => element.getBoundingClientRect());
  const hostAfter = host.getBoundingClientRect().bottom;
  if (!open) {
    apply(true);
    boundary.scrollTop = scroll;
  }
  const base = open ? after : before;
  const view = boundary.getBoundingClientRect();
  const shows = (rect: DOMRect) => rect.bottom > view.top && rect.top < view.bottom;
  const options: KeyframeAnimationOptions = { duration: timing.duration, easing: timing.easing, fill: open ? "auto" : "forwards" };
  // 3. 跟随者带底色盖住内容；标题行之前的元素只在滚动被修正时才会移动，不需要遮盖。
  //    内容跟着标题行移动时（滚到底收起、scrollTop 被钳制），停着不动的跟随者也要成层、带底色，内容不能从它们身上穿过去。
  const lead = movers.indexOf(host.previousElementSibling as HTMLElement);
  const leadFrom = lead < 0 ? 0 : seen[lead] - base[lead].top;
  const leadTo = lead < 0 ? 0 : after[lead].top - base[lead].top;
  const follows = (element: HTMLElement) => Boolean(host.compareDocumentPosition(element) & Node.DOCUMENT_POSITION_FOLLOWING);
  const moving = movers.map((element, index) => ({ element, index, from: seen[index] - base[index].top, to: after[index].top - base[index].top }))
    .filter(({ element, index, from, to }) => (shows(before[index]) || shows(after[index]))
      && (Math.abs(from) > 0.5 || Math.abs(to) > 0.5 || (follows(element) && Math.abs(leadFrom) + Math.abs(leadTo) > 0.5)));
  const followers = moving.filter(({ element }) => follows(element));
  const gaps = gapsBelow(open ? hostAfter : hostBefore, followers.map(({ index }) => base[index]));
  const animations = moving.map(({ element, from, to }) => {
    const animation = element.animate([{ transform: `translateY(${from}px)` }, { transform: `translateY(${to}px)` }], options);
    const order = followers.findIndex((item) => item.element === element);
    return order < 0 ? animation : coverWhile(element, gaps[order], animation);
  });
  // 4. 内容跟着标题行走（标题行吸住时不动），透明度从此刻的值接着变。
  const nudge = open && !previous.length ? -4 : 0;
  const reveal = content.animate(leadFrom || leadTo || nudge
    ? [{ opacity: fade, transform: `translateY(${leadFrom + nudge}px)` }, { opacity: open ? 1 : 0, transform: `translateY(${leadTo}px)` }]
    : [{ opacity: fade }, { opacity: open ? 1 : 0 }], options);
  if (open) followGrowth(host, followers.map(({ from }, order) => ({ animation: animations[moving.indexOf(followers[order])], from })), reveal);
  return [...animations, reveal];
}

// 展开后内容还可能再长（子组件下一次提交才排出结果区）：跟随者的布局位置随之下移，起点要同步上移，
// 保持此刻看到的位置不变、仍在原时刻落到终点。
function followGrowth(host: HTMLElement, followers: Array<{ animation: Animation; from: number }>, reveal: Animation): void {
  watchHeight(host, reveal, (delta) => {
    for (const item of followers) {
      const left = 1 - (item.animation.effect?.getComputedTiming().progress ?? 1);
      if (left < 0.05) continue;
      item.from -= delta / left;
      (item.animation.effect as KeyframeEffect).setKeyframes([{ transform: `translateY(${item.from}px)` }, { transform: "none" }]);
    }
  });
}

// animation 播放期间 host 高度一变就回调（变化量）。ResizeObserver 在布局后、绘制前回调，不会先画出跳位的一帧。
export function watchHeight(host: HTMLElement, animation: Animation, onChange: (delta: number) => void): void {
  let height = host.getBoundingClientRect().height;
  const observer = new ResizeObserver(() => {
    const next = host.getBoundingClientRect().height;
    const delta = next - height;
    height = next;
    if (Math.abs(delta) >= 0.5) onChange(delta);
  });
  observer.observe(host);
  const stop = () => observer.disconnect();
  animation.addEventListener("finish", stop);
  animation.addEventListener("cancel", stop);
}

const top = (element: HTMLElement): number => element.getBoundingClientRect().top;

// host 前后、boundary 之内的元素（沿祖先链逐层的兄弟，文档序）。含有正吸在顶上的元素时拆到子元素：
// transform 加在吸顶元素的祖先上，吸住的位置会被一起带走。
function collectMovers({ host, boundary, pinned }: DisclosureFlip): HTMLElement[] {
  const view = boundary.getBoundingClientRect();
  const visit = (element: Element): HTMLElement[] => {
    if (!(element instanceof HTMLElement)) return [];
    const rect = element.getBoundingClientRect();
    if (rect.height === 0) return [];
    const inView = rect.bottom > view.top && rect.top < view.bottom;
    if (inView && !element.matches(pinned) && holdsStuck(element, pinned, view.top)) return [...element.children].flatMap(visit);
    return [element];
  };
  const before: HTMLElement[] = [];
  const after: HTMLElement[] = [];
  for (let node: Element | null = host; node && node !== boundary; node = node.parentElement) {
    const level: HTMLElement[] = [];
    for (let prev = node.previousElementSibling; prev; prev = prev.previousElementSibling) level.unshift(...visit(prev));
    before.unshift(...level);
    for (let next = node.nextElementSibling; next; next = next.nextElementSibling) after.push(...visit(next));
  }
  return [...before, ...after];
}

// 吸顶元素正停在吸附线上（scrollport 顶 + 它的 top）。
function holdsStuck(element: HTMLElement, pinned: string, viewTop: number): boolean {
  return [...element.querySelectorAll<HTMLElement>(pinned)].some((item) => {
    const styles = getComputedStyle(item);
    if (styles.position !== "sticky") return false;
    const offset = Number.parseFloat(styles.top);
    return Number.isFinite(offset) && Math.abs(item.getBoundingClientRect().top - viewTop - offset) < 1;
  });
}

// 收尾：撤掉停在终点的动画与遮盖层。卸载内容的那次提交里（useLayoutEffect）调用，下一帧排版位置与动画终点重合。
export function settle(animations: Animation[]): void {
  for (const animation of animations) animation.cancel();
}
