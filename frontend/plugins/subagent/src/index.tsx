import { createRoot } from "react-dom/client";
import type { ConversationTabView } from "@akashic/conversation-tools-v2";
import type { WebHostContextV1, WebUiDisposer } from "@akashic/web-ui-v1";
import { registerForm, Field, ModelField } from "../../../../packages/akashic-plugin-kit/src/configuration";
import { SubagentWorkbench } from "./workbench";

// 分叉流/子任务图标：显式 fill="none" 与 stroke="currentColor"，在任何容器内都不出现实心黑块。
const SUBAGENT_ICON = '<svg viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.45" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><circle cx="4.5" cy="4" r="1.75"/><circle cx="4.5" cy="12" r="1.75"/><circle cx="11.5" cy="8" r="1.75"/><path d="M4.5 5.75v4.5M6.1 4.6c2.2.3 3.2 1.2 3.7 2.2M6.1 11.4c2.2-.3 3.2-1.2 3.7-2.2"/></svg>';

// 子任务的运行参数与右侧工作台标签；留空即用默认值。
export function activate(ctx: WebHostContextV1): WebUiDisposer {
  const disposeSettings = registerForm(ctx, {
    id: "subagent",
    title: "子任务",
    description: "模型派出的独立子任务：同时能跑几个、每个最多走多少步。",
    family: "agent",
    familyLabel: "对话与智能体",
    switchable: false,
    fields: ({ values, change, status }) => <>
      <ModelField label="子任务模型" name="model" value={values.model} change={change} status={status} hint="跟随当前对话：用派出它的那轮对话所选的模型" />
      <Field label="最大委派深度" name="max_depth" type="number" min={0} max={5} step={1} inputMode="numeric" value={values.max_depth} fallback={status.defaults?.max_depth ?? 1} change={change} hint="默认 1（只允许直接子任务）；0 表示禁用子任务" />
      <Field label="每棵会话树最大并发数" name="max_active" type="number" min={1} max={20} step={1} inputMode="numeric" value={values.max_active} fallback={status.defaults?.max_active ?? 8} change={change} hint="1–20（默认 8，按顶层对话隔离）；超出时新的子任务会被拒绝" />
      <Field label="每个子任务最多步数" name="max_steps" type="number" min={1} step={1} inputMode="numeric" value={values.max_steps} fallback={status.defaults?.max_steps ?? "不限"} change={change} hint="一步是一次模型输出" />
      <Field label="单次输出 token 上限" name="max_output_tokens" type="number" min={1} step={1} inputMode="numeric" value={values.max_output_tokens} fallback={status.defaults?.max_output_tokens ?? "不限"} change={change} hint="留空不额外限制，由模型自身上限决定" />
      <Field label="子任务结果字数上限" name="result_chars" type="number" min={1000} max={1000000} step={1} inputMode="numeric" value={values.result_chars} fallback={status.defaults?.result_chars} change={change} hint="前台子任务交回的结果超出时截断" />
    </>,
  });

  const disposeTab = ctx.ui.inject("conversation.tools.v2", (mount) => mount.register({
    id: "subagent",
    label: "子任务",
    order: 25,
    background: true,
    icon: SUBAGENT_ICON,
    hint: "实时查看子任务进展并插话/续聊",
    accepts: ["subagent.open.v1"],
    render(host, _entryView, rawView) {
      const root = createRoot(host);
      root.render(<SubagentWorkbench ctx={ctx} view={rawView as ConversationTabView} />);
      return () => root.unmount();
    },
  }));

  return () => {
    disposeSettings();
    disposeTab();
  };
}
