/** 将原版 Chromium snapshot 转成带身份校验的 AX 文本；不加载闭源 WASM。 */
const roleNames = new Map(
  Object.entries({
    RootWebArea: "AXWebArea",
    WebArea: "AXWebArea",
    button: "button",
    link: "link",
    textbox: "text field",
    searchbox: "search text field",
    spinbutton: "stepper",
    slider: "slider",
    combobox: "combo box",
    MenuListPopup: "menu",
    menu: "menu",
    menuitem: "menu item",
    menuitemcheckbox: "menu item",
    menuitemradio: "menu item",
    StaticText: "text",
    InlineTextBox: "text",
    heading: "heading",
    list: "content list",
    checkbox: "checkbox",
    radio: "radio button",
    switch: "switch",
    table: "table",
    grid: "table",
    row: "row",
    cell: "cell",
    gridcell: "cell",
    columnheader: "column header",
    rowheader: "row header",
    image: "image",
    progressbar: "progress indicator",
    separator: "splitter",
    tab: "tab",
    tablist: "tab group",
    tree: "outline",
    treeitem: "row",
    option: "option",
  }),
);
const value = (node, name) => node.properties[name]?.value;
const text = (input) =>
  String(input).replaceAll("\n", "\\n").replaceAll("\t", "\\t");

/** 使用与原版 Qy 完全相同的 UTF-8 长度前缀，避免 iframe 身份碰撞。 */
export function elementIdentity(tabId, node) {
  return [
    String(tabId),
    node.targetID ?? "",
    node.backendDOMNodeID == null ? node.nodeID : String(node.backendDOMNodeID),
  ]
    .map((part) => `${Buffer.byteLength(part)}:${part}`)
    .join("|");
}

/** 原版只把 settable 的文本/组合框公开为可直接赋值；其他控件使用其动作 API。 */
function isSettable(node) {
  return (
    ["textbox", "searchbox", "combobox"].includes(node.role) &&
    value(node, "settable") === true
  );
}

function renderNode(node, index, tab) {
  const role = roleNames.get(node.role) ?? "container";
  const flags = [];
  for (const name of [
    "disabled",
    "focused",
    "selected",
    "required",
    "multiline",
  ]) {
    if (value(node, name) === true) flags.push(name);
  }
  if (value(node, "expanded") != null)
    flags.push(value(node, "expanded") ? "expanded" : "collapsed");
  if (isSettable(node)) flags.push("settable");
  const parts = [];
  if (node.name) parts.push(`Description: ${text(node.name)}`);
  if (role === "AXWebArea" && tab.url)
    parts.push(`URL: ${text(tab.url.replace(/^https?:\/\//, ""))}`);
  if (node.description) parts.push(`Help: ${text(node.description)}`);
  const current = value(node, "valuetext") ?? node.value;
  if (current != null && current !== "") parts.push(`Value: ${text(current)}`);
  for (const name of [
    "checked",
    "pressed",
    "level",
    "valuemin",
    "valuemax",
    "orientation",
    "placeholder",
  ]) {
    const current = value(node, name);
    if (current != null) parts.push(`${name}: ${text(current)}`);
  }
  if (value(node, "expanded") != null)
    parts.push(
      `Secondary Actions: ${value(node, "expanded") ? "Collapse" : "Expand"}`,
    );
  let detail = parts.join(", ");
  if (parts.length === 1 && parts[0].startsWith("Description: "))
    detail = text(node.name);
  return `${index} ${role}${flags.length ? ` (${flags.join(", ")})` : ""}${detail ? ` ${detail}` : ""}`;
}

export function createAccessibilityCore() {
  const runtime = {};
  class Revision {
    constructor(snapshot, previous) {
      this.runtime = runtime;
      const identities = snapshot.nodes.map((node) =>
        elementIdentity(snapshot.tab.id, node),
      );
      const present = new Set(identities);
      const oldIds = new Map(
        previous?.entries
          .filter((entry) => present.has(entry.identity))
          .map((entry) => [entry.identity, entry.index]),
      );
      const used = new Set(oldIds.values());
      let nextIndex = 0;
      this.entries = snapshot.nodes.map((node, order) => {
        let index = oldIds.get(identities[order]);
        if (index == null) {
          while (used.has(nextIndex)) nextIndex += 1;
          index = nextIndex++;
          used.add(index);
        }
        return { index, identity: identities[order], node };
      });
      const children = new Map();
      for (let order = 0; order < this.entries.length; order++) {
        const parent = this.entries[order].node.parentIndex;
        if (!Number.isInteger(parent) || parent < -1 || parent >= order)
          throw new Error("AX snapshot parent must precede its child");
        if (!children.has(parent)) children.set(parent, []);
        children.get(parent).push(order);
      }
      // CDP 按广度返回节点；文本必须按深度遍历，否则缩进会把节点挂到错误父级。
      const stack = (children.get(-1) ?? [])
        .toReversed()
        .map((order) => [order, 0]);
      const lines = [];
      while (stack.length) {
        const [order, depth] = stack.pop();
        const { node, index } = this.entries[order];
        lines.push(
          `${"\t".repeat(depth)}${renderNode(node, index, snapshot.tab)}`,
        );
        for (const child of (children.get(order) ?? []).toReversed())
          stack.push([child, depth + 1]);
      }
      this.text =
        lines
          .concat(snapshot.warnings.map((warning) => `Warning: ${warning}`))
          .join("\n") + "\n";
    }
    identityForElement(index) {
      if (!Number.isSafeInteger(index))
        throw new TypeError("Accessibility element ID must be a safe integer");
      return this.entries.find((entry) => entry.index === index)?.identity;
    }
    isValueSettableForElement(index) {
      if (!Number.isSafeInteger(index))
        throw new TypeError("Accessibility element ID must be a safe integer");
      const entry = this.entries.find((entry) => entry.index === index);
      return entry == null ? undefined : isSettable(entry.node);
    }
  }
  return {
    buildRevision(previous, snapshot, options = {}) {
      if (
        previous != null &&
        (!(previous instanceof Revision) || previous.runtime !== runtime)
      ) {
        throw new TypeError(
          "Previous accessibility revision belongs to another runtime",
        );
      }
      if (
        !snapshot ||
        !Array.isArray(snapshot.nodes) ||
        !snapshot.tab ||
        !Array.isArray(snapshot.warnings)
      ) {
        throw new TypeError("Invalid accessibility snapshot");
      }
      if (!["auto", "full"].includes(options.mode ?? "auto"))
        throw new TypeError(
          "Accessibility rendering mode must be auto or full",
        );
      // 完整文本是有意差异：保留全部信息，不猜测原 WASM 的差分压缩阈值。
      return new Revision(snapshot, previous);
    },
  };
}
