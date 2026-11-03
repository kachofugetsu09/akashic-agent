// 编辑器这一侧的 JSX 求值：构建时把 open-pencil renderJSX 里的 new Function 换成调用这里，
// 交给不透明源沙箱求值（见 jsx-sandbox.ts），编辑器页面本身仍然不允许 eval。
const SANDBOX = "/design/assets/jsx.html";
const TIMEOUT_MS = 10_000;
const VARIABLE = Symbol.for("open-pencil.variable");

type Reply = { id: number; tree?: unknown; warnings?: string[]; error?: string };

let frame: HTMLIFrameElement | null = null;
let ready: Promise<HTMLIFrameElement> | null = null;
let next = 0;
const pending = new Map<number, (reply: Reply) => void>();

window.addEventListener("message", (event: MessageEvent<Reply>) => {
  if (!frame || event.source !== frame.contentWindow || typeof event.data?.id !== "number") return;
  pending.get(event.data.id)?.(event.data);
  pending.delete(event.data.id);
});

function sandbox(): Promise<HTMLIFrameElement> {
  ready ??= new Promise((resolve, reject) => {
    const element = document.createElement("iframe");
    element.sandbox.add("allow-scripts");
    element.src = SANDBOX;
    element.hidden = true;
    const timer = window.setTimeout(() => { reset(); reject(new Error("JSX 沙箱没有启动")); }, TIMEOUT_MS);
    const onReady = (event: MessageEvent<{ ready?: boolean }>) => {
      if (event.source !== element.contentWindow || event.data?.ready !== true) return;
      window.removeEventListener("message", onReady);
      window.clearTimeout(timer);
      resolve(element);
    };
    window.addEventListener("message", onReady);
    frame = element;
    document.body.append(element);
  });
  return ready;
}

// 死循环之类卡住的 JSX：整个沙箱丢掉重建，下一次调用拿到干净的。
function reset(): void {
  frame?.remove();
  frame = null;
  ready = null;
  for (const resolve of pending.values()) resolve({ id: -1, error: "JSX 求值被中止" });
  pending.clear();
}

function revive(value: unknown): unknown {
  if (value === null || typeof value !== "object") return value;
  if (Array.isArray(value)) return value.map(revive);
  const { __designVariable, ...rest } = value as Record<string, unknown>;
  const out: Record<string | symbol, unknown> = Object.fromEntries(Object.entries(rest).map(([key, item]) => [key, revive(item)]));
  if (__designVariable === true) out[VARIABLE] = true;
  return out;
}

export async function evaluateJSX(jsx: string, warnings: string[]): Promise<unknown> {
  const target = await sandbox();
  const id = ++next;
  const reply = await new Promise<Reply>((resolve) => {
    const timer = window.setTimeout(() => {
      pending.delete(id);
      reset();
      resolve({ id, error: `JSX ${TIMEOUT_MS / 1000} 秒内没有求值完成（死循环或计算太重），沙箱已重建` });
    }, TIMEOUT_MS);
    pending.set(id, (value) => { window.clearTimeout(timer); resolve(value); });
    target.contentWindow?.postMessage({ id, jsx }, "*");
  });
  if (reply.error) throw new Error(reply.error);
  warnings.push(...(reply.warnings ?? []));
  return revive(reply.tree);
}
