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
  let previous = host.getBoundingClientRect().bottom;
  return followers.map((element) => {
    const rect = element.getBoundingClientRect();
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

// 收尾：撤掉停在终点的动画与遮盖层。卸载内容的那次提交里（useLayoutEffect）调用，下一帧排版位置与动画终点重合。
export function settle(animations: Animation[]): void {
  for (const animation of animations) animation.cancel();
}
