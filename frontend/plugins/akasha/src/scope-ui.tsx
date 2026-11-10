import { useEffect, useRef, useState } from "react";
import { ChoiceMenu, render, type PluginUiContext, type ScopeFieldControls } from "@akashic/chat-ui-v1";

// 记忆范围与 Akasha 的 (learn, recall) 一一对应；“不使用记忆”即不学习也不召回。
type Memory = "global" | "isolated" | "off" | "none";

const CHOICES: readonly { value: Memory; label: string; description: string }[] = [
  { value: "global", label: "共享全局记忆", description: "对话参与全局学习，也能回忆其他对话。" },
  { value: "isolated", label: "独立记忆", description: "只在本项目内学习和回忆，不影响全局。" },
  { value: "none", label: "不使用记忆", description: "关闭 Akasha 学习与召回；保留 Markdown 档案、会话上下文和聊天记录。" },
  { value: "off", label: "不学习", description: "对话不写入记忆，仍可回忆全局记忆。" },
];

const policyOf = (memory: Memory) => ({ learn: memory === "none" ? "off" : memory, recall: memory !== "none" });

function memoryOf(result: Record<string, unknown>): Memory {
  const learn = result.learn;
  // 旧策略记录与旧服务端默认召回。
  if (result.recall !== undefined && typeof result.recall !== "boolean") throw new Error("召回策略无效");
  if (learn === "off" && result.recall === false) return "none";
  if (result.recall === false) throw new Error("此记忆策略组合无法显示");
  if (learn === "global" || learn === "isolated" || learn === "off") return learn;
  throw new Error("记忆策略无效");
}

// 同一范围的策略本页只读一次：分组区重绘、新建对话区重复挂载都复用这一份；上次读到的记在本机，挂载时先画。
// 策略只经本模块的 scope.policy.set 改动，改后直接更新这里。
const POLICY_KEY = "akashic.akasha.scope-policy.v1";
const policies = new Map<string, Promise<Memory>>();

function storedPolicies(): Record<string, Memory> {
  try {
    const value: unknown = JSON.parse(localStorage.getItem(POLICY_KEY) ?? "{}");
    return value && typeof value === "object" && !Array.isArray(value) ? value as Record<string, Memory> : {};
  } catch {
    return {};
  }
}

function rememberPolicy(key: string, memory: Memory) {
  policies.set(key, Promise.resolve(memory));
  try { localStorage.setItem(POLICY_KEY, JSON.stringify({ ...storedPolicies(), [key]: memory })); }
  catch { /* 禁用存储时下次打开等读取结果再画。 */ }
}

// 同一时刻挂载的徽标合成一次批量读取：侧栏有几个分组就省几个请求，弱网 HTTP/1.1 下不占满 6 条连接。
type PolicyWaiter = { scope: { dimension: string; value: string }; resolve: (memory: Memory) => void; reject: (error: unknown) => void };
let queued: PolicyWaiter[] = [];
let queuedContext: PluginUiContext<"scope.badge"> | null = null;

function flushPolicies() {
  const waiters = queued;
  const context = queuedContext;
  queued = [];
  queuedContext = null;
  if (!context) return;
  context.query("scope.policy.list", { scopes: waiters.map((waiter) => waiter.scope) }).then((result) => {
    const rows = Array.isArray(result.policies) ? result.policies as unknown[] : [];
    waiters.forEach((waiter, index) => {
      const row = rows[index];
      try { waiter.resolve(memoryOf(row && typeof row === "object" ? row as Record<string, unknown> : {})); }
      catch (error) { waiter.reject(error); }
    });
  }, (error: unknown) => waiters.forEach((waiter) => waiter.reject(error)));
}

function readPolicy(context: PluginUiContext<"scope.badge">, key: string): Promise<Memory> {
  const cached = policies.get(key);
  if (cached) return cached;
  const { dimension, value } = context.scope;
  const pending = new Promise<Memory>((resolve, reject) => {
    queued.push({ scope: { dimension, value }, resolve, reject });
    queuedContext = context;
    if (queued.length === 1) window.setTimeout(flushPolicies, 0);
  });
  policies.set(key, pending);
  pending.then((memory) => rememberPolicy(key, memory), () => { if (policies.get(key) === pending) policies.delete(key); });
  return pending;
}

// 新建表单里的记忆范围：创建前由宿主调用提交，策略在第一个会话出现前固定。
function MemoryField({ context, fields }: { context: PluginUiContext<"scope.fields">; fields: ScopeFieldControls }) {
  const [memory, setMemory] = useState<Memory>("global");
  const [busy, setBusy] = useState(() => fields.busy());
  const chosen = useRef(memory);
  chosen.current = memory;
  useEffect(() => {
    const { dimension } = context.scope;
    const stopBusy = context.onInvalidate(() => setBusy(fields.busy()));
    const stopCommit = fields.onCommit(async (value) => {
      const wanted = chosen.current;
      const result = await context.query("scope.policy.set", { dimension, value, ...policyOf(wanted) });
      if (memoryOf(result) !== wanted) throw new Error("记忆策略与创建请求不一致");
      rememberPolicy(`${dimension}:${value}`, wanted);
    });
    return () => { stopBusy(); stopCommit(); };
  }, [context, fields]);
  return <ChoiceMenu ariaLabel="记忆设置" value={memory} choices={CHOICES} note="创建后，记忆选项将固定。"
    disabled={busy} onChange={(value) => setMemory(value as Memory)} />;
}

export function mountMemoryField(host: HTMLElement, context: PluginUiContext<"scope.fields">) {
  return render(host, <MemoryField context={context} fields={context.fields} />);
}

// 偏离共享全局的策略各用一种图标线型：自成一圈是双线，只读不写是虚线，不留痕是点线。
const LINES: Partial<Record<Memory, string>> = { isolated: "double", off: "dashed", none: "dotted" };

// 名字后的记忆标签：共享全局时不显示，读不出时如实说明；带线型时宿主把它画到分组图标上。
export function mountMemoryBadge(host: HTMLElement, context: PluginUiContext<"scope.badge">) {
  const { dimension, value } = context.scope;
  let active = true;
  const show = (text: string, memory?: Memory) => {
    const badge = document.createElement("small");
    badge.className = "akasha-scope-badge";
    badge.textContent = text;
    const choice = CHOICES.find((item) => item.value === memory);
    if (memory && LINES[memory]) badge.dataset.scopeLine = LINES[memory];
    if (choice) badge.title = `${choice.label}：${choice.description}`;
    host.replaceChildren(badge);
  };
  const key = `${dimension}:${value}`;
  const paint = (memory: Memory) => {
    if (!active) return;
    if (memory === "global") host.replaceChildren();
    else show(CHOICES.find((choice) => choice.value === memory)?.label ?? "", memory);
  };
  const stored = storedPolicies()[key];
  if (stored && CHOICES.some((choice) => choice.value === stored)) paint(stored);
  // 共享的读取随先挂的那处卸载而中止时，用自己的上下文再读一次。
  const load = (): void => void readPolicy(context, key).then(paint, (error: unknown) => {
    if (!active) return;
    if (error instanceof DOMException && error.name === "AbortError") load();
    else show("当前策略未能读取");
  });
  load();
  return () => { active = false; };
}
