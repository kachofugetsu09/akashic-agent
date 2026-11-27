// 页面上"前景浮层"的唯一定义：模态对话框、打开的 dialog、对话框与菜单角色。
// 插件判断"现在有浮层挡在前面"时都用这一份，不各自手写选择器。
export const overlaySelector = "[aria-modal='true'], dialog[open], [role='dialog'], [role='alertdialog'], [role='menu']";

// 返回最上层真正可见的浮层；隐藏占位（hidden/inert/aria-hidden、无盒子、visibility: hidden）不算。
export function foregroundOverlay(doc: Document = document): HTMLElement | null {
  const nodes = [...doc.querySelectorAll<HTMLElement>(overlaySelector)];
  for (let index = nodes.length - 1; index >= 0; index -= 1) {
    const node = nodes[index];
    if (node.closest("[hidden], [inert], [aria-hidden='true']")) continue;
    if (node.getClientRects().length === 0) continue;
    if (doc.defaultView?.getComputedStyle(node).visibility === "hidden") continue;
    return node;
  }
  return null;
}
