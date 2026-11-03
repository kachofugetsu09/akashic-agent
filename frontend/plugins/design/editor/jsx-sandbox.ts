import { buildComponent, createElement, resolveToTree } from "@open-pencil/design-jsx";

// 跑在 sandbox="allow-scripts" 的不透明源 iframe 里：Agent 写的 JSX 是可执行代码，只在这里求值，
// 拿不到 akashic 的 Cookie、页面与接口；交回去的只是一棵纯数据的图层树。
const VARIABLE = Symbol.for("open-pencil.variable");

function encode(value: unknown, depth = 0): unknown {
  if (depth > 200) throw new Error("JSX 嵌套过深");
  if (typeof value === "function") throw new Error("JSX 结果里不能含函数");
  if (value === null || typeof value !== "object") return value;
  if (Array.isArray(value)) return value.map((item) => encode(item, depth + 1));
  const record = value as Record<string | symbol, unknown>;
  const out: Record<string, unknown> = {};
  if (record[VARIABLE] === true) out.__designVariable = true;
  for (const [key, item] of Object.entries(record)) out[key] = encode(item, depth + 1);
  return out;
}

window.addEventListener("message", (event: MessageEvent<{ id?: number; jsx?: string }>) => {
  if (event.source !== parent || typeof event.data?.id !== "number" || typeof event.data.jsx !== "string") return;
  const { id, jsx } = event.data;
  const warnings: string[] = [];
  try {
    const tree = resolveToTree(createElement(buildComponent(jsx, warnings), null));
    parent.postMessage({ id, tree: encode(tree), warnings }, "*");
  } catch (error) {
    parent.postMessage({ id, error: error instanceof Error ? error.message : String(error) }, "*");
  }
});
parent.postMessage({ ready: true }, "*");
