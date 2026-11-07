import { useLayoutEffect, useState, type RefObject } from "react";

export function nextComposerExpanded(
  wasExpanded: boolean,
  text: string,
  isOverflowing: () => boolean,
): boolean {
  if (!text) return false;
  if (wasExpanded || text.includes("\n")) return true;
  return isOverflowing();
}

// 单行输入卡挤不下工具位时改用两行：输入框守住保底宽度后，工具位里的控件（插件按各自最小宽度）
// 仍超出分到的宽度就算挤。两行时卡片或工具位尺寸再变（转屏、换模型、多出思考强度），回到单行重新量一次。
export function useComposerCrowded(card: RefObject<HTMLElement | null>, singleLine: boolean): boolean {
  const [crowded, setCrowded] = useState(false);
  useLayoutEffect(() => {
    const root = card.current;
    const lead = root?.querySelector<HTMLElement>(".composer__lead");
    if (!root || !lead || !singleLine) return;
    // 1. 单行时：工具位里有控件被压到自身最小宽度以下（内容溢出到旁边的控件上）就转两行；
    //    插件改了控件（换模型名、多一个钮）也要重量。
    if (!crowded) {
      const measure = () => { if (toolbarOverflows(lead)) setCrowded(true); };
      measure();
      const resize = new ResizeObserver(measure);
      resize.observe(root);
      resize.observe(lead);
      const mutation = new MutationObserver(measure);
      mutation.observe(lead, { childList: true, subtree: true, characterData: true });
      return () => { resize.disconnect(); mutation.disconnect(); };
    }
    // 2. 两行时：记下当前尺寸，变了才回单行（观察开始时的首次回调不算变化）。
    const size = () => `${root.clientWidth}:${lead.offsetWidth}`;
    const start = size();
    const resize = new ResizeObserver(() => { if (size() !== start) setCrowded(false); });
    resize.observe(root);
    resize.observe(lead);
    return () => resize.disconnect();
  }, [card, crowded, singleLine]);
  return crowded && singleLine;
}

// 溢出可能落在相邻控件上、不越过工具位外缘，所以逐个看不裁切的元素；本来就省略号裁切的文字不算。
// 控件常用伪元素把热区左右外扩几像素，scrollWidth 会算进去，所以超出 8px 以上才算挤。
function toolbarOverflows(lead: HTMLElement): boolean {
  return [lead, ...lead.querySelectorAll<HTMLElement>("*")].some((element) => element.clientWidth > 0
    && element.scrollWidth > element.clientWidth + 8 && getComputedStyle(element).overflowX === "visible");
}
