import { useCallback, useEffect, useRef, useState } from "react";
import { render, type ComposerControls, type PluginUiContext, type PluginUiDefinition } from "@akashic/chat-ui-v1";
import { compatibleEffort, chatModelState, modelProblem, type ChatModelState } from "./capsule-data";
import { ModelCapsulePicker, ModelEffortAction, resolveVisibleRuntime } from "./capsule-picker";
import "./message-ui.css";

const number = (value: unknown) => new Intl.NumberFormat("zh-CN").format(Number(value || 0));

// 回答尾注：显示本轮全部模型调用的真实输出 token；没有完整用量时不占位。
function answerUsage(host: HTMLElement, context: PluginUiContext) {
  let active = true;
  if (!context.messageId || !context.sessionId) throw new Error("Turn token 统计缺少消息上下文");
  host.classList.add("models-usage-tail-host");
  context.query("usage.message", { message_id: context.messageId }, { cache: "immutable" }).then((result) => {
    if (!active || typeof result.output_tokens !== "number") return;
    const row = document.createElement("div");
    row.className = "models-usage-tail";
    row.setAttribute("aria-label", "本轮输出 Token");
    row.innerHTML = `<span>输出 <strong>${number(result.output_tokens)} tokens</strong></span>`;
    host.replaceChildren(row);
  }).catch((error: unknown) => {
    if (!active) return;
    host.classList.add("models-usage-tail-host--error");
    host.textContent = error instanceof Error ? `Token 统计不可用：${error.message}` : "Token 统计不可用";
  });
  return () => { active = false; };
}

interface Selection { runtimeId: string; effort: string; dirty: boolean }

// 工具位随草稿重新挂载；新对话发出第一条后换成会话身份，沿用刚才的目录与选择，避免胶囊闪空。
const lastState = new Map<string, ChatModelState>();
let carried: { state: ChatModelState; selection: Selection } | null = null;

// 输入框里的模型胶囊：读本插件的 chat.models，选择随下一条消息作为 model.selection 片段发出，不可用时阻止发送。
function ModelCapsule({ context, composer }: { context: PluginUiContext; composer: ComposerControls }) {
  const key = context.sessionId ?? "";
  const [adopted] = useState(() => {
    const taken = context.sessionId && !lastState.has(key) ? carried : null;
    carried = null;
    return taken;
  });
  const [state, setState] = useState<ChatModelState | null>(adopted?.state ?? lastState.get(key) ?? null);
  const [error, setError] = useState("");
  const [selection, setSelection] = useState<Selection>(adopted?.selection ?? { runtimeId: "", effort: "", dirty: false });
  const [busy, setBusy] = useState(() => composer.busy());
  const latest = useRef(0);
  const current = useRef({ state, selection });
  current.current = { state, selection };

  // 1. 读取：后发先至的旧结果丢弃；用户改过选择或正在发送时不覆盖选择。
  const load = useCallback(async () => {
    const request = ++latest.current;
    try {
      const next = chatModelState(await context.query("chat.models"));
      if (request !== latest.current) return;
      lastState.set(key, next);
      setState(next);
      setError("");
      if (!current.current.selection.dirty && !composer.busy()) {
        setSelection({ runtimeId: next.sessionOverride, effort: next.sessionSelection.reasoningEffort, dirty: false });
      }
    } catch (reason) {
      if (request !== latest.current) return;
      setError(`模型列表加载失败：${reason instanceof Error ? reason.message : String(reason)}。发送时仍使用会话已保存的模型。`);
    }
  }, [composer, context, key]);

  // 2. 回到页面或模型设置变更时重读；生成状态变化时刷新禁用态。
  useEffect(() => {
    void load();
    const visible = () => { if (document.visibilityState === "visible") void load(); };
    const changed = (event: MessageEvent<unknown>) => {
      const data = event.data as { type?: unknown } | null;
      if (event.origin === window.location.origin && event.source === window.parent && data?.type === "akashic.models.changed") void load();
    };
    const stopInvalidate = context.onInvalidate(() => setBusy(composer.busy()));
    const stopSent = composer.onSent(() => {
      const { state: sent, selection: chosen } = current.current;
      if (!context.sessionId && sent) carried = { state: sent, selection: { ...chosen, dirty: false } };
      setSelection((previous) => ({ ...previous, dirty: false }));
    });
    window.addEventListener("focus", visible);
    document.addEventListener("visibilitychange", visible);
    window.addEventListener("message", changed);
    return () => {
      stopInvalidate();
      stopSent();
      window.removeEventListener("focus", visible);
      document.removeEventListener("visibilitychange", visible);
      window.removeEventListener("message", changed);
    };
  }, [composer, context, load]);

  // 3. 宿主只看片段与阻止原因：改过的选择随下一条消息发出，当前模型不可用时说明原因。
  const modelId = (selection.dirty ? selection.runtimeId : selection.runtimeId || state?.sessionOverride) || state?.defaultRuntime || "";
  const problem = modelProblem(state, modelId);
  useEffect(() => { composer.block(problem || null); }, [composer, problem]);
  useEffect(() => {
    composer.contribute(selection.dirty ? [{ kind: "model.selection",
      value: { model_id: selection.runtimeId || null, reasoning_effort: selection.effort || null } }] : []);
  }, [composer, selection]);

  const choose = (runtimeId: string, effort: string) => setSelection({ runtimeId, effort, dirty: true });
  if (!state) return error ? <button type="button" className="models-capsule-retry" title={error} onClick={() => void load()}>重试读取模型</button> : null;
  const { visibleModel, explicitModel } = resolveVisibleRuntime(state.runtimes, selection.runtimeId, state.defaultRuntime);
  return <>
    <ModelCapsulePicker defaultRuntime={state.defaultRuntime} runtimes={state.runtimes} selectedRuntimeId={selection.runtimeId}
      selectedEffort={selection.effort} disabled={busy} onChange={choose} />
    {visibleModel ? <ModelEffortAction runtime={visibleModel} effort={compatibleEffort(visibleModel, selection.effort)}
      explicit={Boolean(explicitModel)} disabled={busy} onChange={choose} /> : null}
    {problem ? <a className="models-capsule-settings" href="/#models" target="_blank" rel="noopener">模型设置</a> : null}
  </>;
}

const definition: PluginUiDefinition = {
  slots: {
    "turn.after_answer": { mount: answerUsage },
    "composer.toolbar": {
      mount: (host, context) => {
        if (!context.composer) throw new Error("输入框工具位缺少宿主能力");
        return render(host, <ModelCapsule context={context} composer={context.composer} />);
      },
    },
  },
};

export default definition;
