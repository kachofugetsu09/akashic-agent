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
  context.query("scope.policy.get", { dimension, value }).then((result) => {
    const memory = memoryOf(result);
    if (active && memory !== "global") show(CHOICES.find((choice) => choice.value === memory)?.label ?? "", memory);
  }).catch(() => { if (active) show("当前策略未能读取"); });
  return () => { active = false; };
}
