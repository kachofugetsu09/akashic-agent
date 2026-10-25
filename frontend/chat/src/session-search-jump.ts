import type { SearchHit } from "./session-search";

/**
 * 把一次搜索命中变成「真的看见了」：滚到那个 part、展开挡住它的折叠、闪一下落点。
 *
 * 折叠是这里唯一不能省的一步：收起的步骤在展开前根本不在 DOM 里，直接按
 * `data-part-index` 找不到节点。所以先点开承载它的折叠行，等一帧让内容挂上，
 * 再按 part 下标定位。
 */

/** 命中所在消息的定位元素：正常消息是锚点，收进折叠的是 .flow-item。 */
function findAnchor(messageId: string): HTMLElement | null {
  const escaped = CSS.escape(messageId);
  return document.querySelector<HTMLElement>(`.web-message-anchor[data-message-id="${escaped}"]`)
    ?? document.querySelector<HTMLElement>(`.flow-item[data-process-message-id="${escaped}"]`);
}

/**
 * 展开页面上所有收起的折叠行。
 *
 * 工具调用与工具结果各自是独立消息，收起时它们连消息锚点都不在 DOM 里，
 * 按 id 找不到。一次全展开再按 id 找，比按位置猜要稳。
 */
async function expandEveryCollapsedFold(): Promise<void> {
  const buttons = [...document.querySelectorAll<HTMLButtonElement>(".tool-step-summary")]
    .filter((button) => button.getAttribute("aria-expanded") !== "true");
  if (!buttons.length) return;
  for (const button of buttons) button.click();
  await nextFrame();
  await nextFrame();
}

/**
 * 命中所在 part 的 DOM 节点。
 *
 * 同一个 part 有两种落点，取决于它是否被收进过程折叠：
 * - 正常渲染：`.web-message-anchor[data-message-id]` 里的 `[data-part-index]`；
 * - 收进折叠：折叠内的 `.flow-item` 不带消息锚点，用 `data-process-message-id`
 *   记住它来自哪条消息（工具调用与结果各自是独立消息，收起时连锚点都不存在）。
 */
function partNode(messageId: string, partIndex: number): HTMLElement | null {
  const escaped = CSS.escape(messageId);
  const anchor = document.querySelector<HTMLElement>(
    `.web-message-anchor[data-message-id="${escaped}"]`,
  );
  const inAnchor = anchor?.querySelector<HTMLElement>(`[data-part-index="${partIndex}"]`);
  if (inAnchor) return inAnchor;
  return document.querySelector<HTMLElement>(
    `.flow-item[data-process-message-id="${escaped}"][data-part-index="${partIndex}"]`,
  );
}

/**
 * 展开挡住目标 part 的折叠。
 *
 * 会话里有两条折叠路径，都是 `tool-step-summary` + `aria-expanded`：
 * - 未落定的一轮走 `CollapsedToolGroup`（`.tool-group`，收的是同一条消息里的过程项）；
 * - 落定后整轮过程收进 `TurnProcess` 摘要行（`.turn-process`，标题是「调用 N 个工具」）。
 * 只看 `.tool-group` 会漏掉后一种——那恰好是读旧会话时最常见的样子。
 *
 * 折叠行的内容是懒挂载的（展开那一帧才进 DOM），所以要等挂载后的下一帧。
 */
async function revealFold(node: HTMLElement): Promise<void> {
  // 由内向外找最近的折叠容器；目标可能嵌在组里，也可能直接在一轮摘要里。
  // 传进来的也可能是消息锚点本身（命中还没渲染、只能按消息定位的情况），
  // 那时折叠容器在它内部而不是外部，所以要两个方向都看。
  const containers = new Set<HTMLElement>();
  for (let current: HTMLElement | null = node; current; current = current.parentElement) {
    if (current.classList.contains("tool-group") || current.classList.contains("turn-process")) {
      containers.add(current);
      break;
    }
  }
  for (const nested of node.querySelectorAll<HTMLElement>(".tool-group, .turn-process")) {
    containers.add(nested);
  }
  if (!containers.size) return;

  const collapsed: HTMLButtonElement[] = [];
  for (const container of containers) {
    const summary = container.querySelector<HTMLButtonElement>(":scope > .tool-step-sticky > .tool-step-summary")
      ?? container.querySelector<HTMLButtonElement>(".tool-step-summary");
    if (summary && summary.getAttribute("aria-expanded") !== "true") collapsed.push(summary);
  }
  if (!collapsed.length) return;
  for (const summary of collapsed) summary.click();
  // 展开是 0fr → 1fr 的过渡，内容在同一帧挂上；等两帧让布局稳定再滚动。
  await nextFrame();
  await nextFrame();
}

function nextFrame(): Promise<void> {
  return new Promise((resolve) => requestAnimationFrame(() => resolve()));
}

/**
 * 等搜索浮层从 DOM 上撤掉：点击折叠行之前遮罩必须先让开。
 *
 * 用 MutationObserver 而不是数帧：React 的状态提交时机不在我们的控制里，
 * 固定等几帧在慢机器上会不够，在快机器上又会白等。
 */
function waitForOverlayGone(): Promise<void> {
  if (!document.querySelector(".session-search")) return Promise.resolve();
  return new Promise((resolve) => {
    const observer = new MutationObserver(() => {
      if (document.querySelector(".session-search")) return;
      observer.disconnect();
      resolve();
    });
    observer.observe(document.body, { childList: true, subtree: true });
    // 兜底：浮层迟迟不撤（状态没更新）也不能把跳转永久挂住。
    window.setTimeout(() => { observer.disconnect(); resolve(); }, 400);
  });
}

/** 落点提示：只闪一下，不留元素、不改变文档状态。 */
function flashLanded(node: HTMLElement): void {
  for (const previous of document.querySelectorAll(".session-search-landed")) {
    previous.classList.remove("session-search-landed");
  }
  void node.offsetWidth;
  node.classList.add("session-search-landed");
  const clear = () => node.classList.remove("session-search-landed");
  node.addEventListener("animationend", clear, { once: true });
  // prefers-reduced-motion 下没有 animationend，靠定时器收尾。
  window.setTimeout(clear, 2000);
}

/**
 * 跳到一条命中。返回是否真的找到了节点——找不到时面板不关闭，
 * 免得用户以为跳成功了却停在原地。
 */
export async function jumpToHit(hit: SearchHit): Promise<boolean> {
  // 调用方已经 setSearchOpen(false)，但 React 还没提交这一帧。
  // 等遮罩真的从 DOM 上撤掉再点折叠行，否则点在遮罩上什么也不会发生。
  await waitForOverlayGone();

  let anchor = findAnchor(hit.message_id);
  if (!anchor) {
    // 工具调用各自是独立消息，收起时它们的锚点根本不在 DOM 里——
    // 只有承载整轮摘要的那个锚点存在。先把页面上所有收起的折叠展开，
    // 让这些消息挂进来，再按 id 找。
    await expandEveryCollapsedFold();
    anchor = findAnchor(hit.message_id);
  }
  if (!anchor) return false;

  await revealFold(anchor);
  let node = partNode(hit.message_id, hit.part_index);
  if (!node) {
    await expandEveryCollapsedFold();
    node = partNode(hit.message_id, hit.part_index);
  }
  if (!node) return false;
  await revealFold(node);
  const target = partNode(hit.message_id, hit.part_index) ?? node;
  const reduce = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  target.scrollIntoView({ block: "center", behavior: reduce ? "auto" : "smooth" });
  flashLanded(target);
  return true;
}
