import { useCallback, useEffect, useId, useRef, useState } from "react";
import { render, type ComposerControls, type PluginUiContext, type PluginUiDefinition } from "@akashic/chat-ui-v1";
import { compatibleEffort, chatModelState, modelProblem, type ChatModelState } from "./capsule-data";
import { ModelCapsulePicker, ModelEffortAction, resolveVisibleRuntime } from "./capsule-picker";
import { formatModelCallStats, readModelCallStats } from "./call-stats";
import "./message-ui.css";

const number = (value: unknown) => new Intl.NumberFormat("zh-CN").format(Number(value || 0));

// 回答尾注：显示本轮全部模型调用的真实输出 token；没有完整用量时不占位。
function answerUsage(host: HTMLElement, context: PluginUiContext<"turn.after_answer">) {
  let active = true;
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

// 输入框下方状态行：宿主在看着一轮跑完时挂载，这里读该会话最近一次调用给出终态摘要；读取失败只影响遥测，不报错。
function callSummary(host: HTMLElement, context: PluginUiContext<"composer.status">) {
  let active = true;
  context.query("call.latest").then((result) => {
    if (!active || result.call === null) return;
    const stats = readModelCallStats(result.call);
    const text = formatModelCallStats(stats);
    const line = document.createElement("span");
    line.setAttribute("aria-live", "polite");
    line.title = `${text} · ${stats.model}`;
    line.textContent = text;
    host.replaceChildren(line);
  }).catch(() => undefined);
  return () => { active = false; };
}

interface Selection { runtimeId: string; effort: string; dirty: boolean }

// 工具位随草稿重新挂载；新对话发出第一条后换成会话身份，沿用刚才的目录与选择，避免胶囊闪空。
// 产品里同一页面只有一个输入框工具位，carried 由下一个挂载取走；多工具位并存时需按草稿与会话身份交接。
const lastState = new Map<string, ChatModelState>();
let carried: { state: ChatModelState; selection: Selection } | null = null;

const savedSelection = (state: ChatModelState): Selection =>
  ({ runtimeId: state.sessionOverride, effort: state.sessionSelection.reasoningEffort, dirty: false });

// 已发出的选择就是会话现在保存的选择；缓存随之更新，回到该会话时不会先显示旧值。
const withSaved = (state: ChatModelState, selection: Selection): ChatModelState => ({ ...state,
  sessionOverride: selection.runtimeId, sessionSelection: { modelRef: selection.runtimeId, reasoningEffort: selection.effort } });

interface CapsuleState {
  state: ChatModelState | null;
  error: string;
  selection: Selection;
  busy: boolean;
  load: () => Promise<void>;
  choose: (runtimeId: string, effort: string) => void;
}

// 胶囊的目录与选择：从同会话缓存或新对话交接恢复，读取结果按选择编辑版本取舍。
function useCapsuleState(context: PluginUiContext<"composer.toolbar">, composer: ComposerControls): CapsuleState {
  const key = context.sessionId ?? "";
  const [adopted] = useState(() => {
    const taken = context.sessionId && !lastState.has(key) ? carried : null;
    carried = null;
    return taken;
  });
  const cached = adopted ? null : lastState.get(key) ?? null;
  const [state, setState] = useState<ChatModelState | null>(adopted?.state ?? cached);
  const [error, setError] = useState("");
  const [selection, setSelection] = useState<Selection>(adopted?.selection
    ?? (cached ? savedSelection(cached) : { runtimeId: "", effort: "", dirty: false }));
  const latest = useRef(0);
  // 选择编辑版本：读取开始后用户改过或发出过选择，迟到的结果就不能再覆盖它。
  const edits = useRef(0);
  const current = useRef({ state, selection });
  current.current = { state, selection };

  // 后发先至的旧结果丢弃；读取期间选择变过时只取新目录，会话选择沿用本地已知的值。
  const load = useCallback(async () => {
    const request = ++latest.current;
    const edit = edits.current;
    try {
      const next = chatModelState(await context.query("chat.models"));
      if (request !== latest.current) return;
      const known = current.current.state;
      const stale = edits.current !== edit;
      const merged = stale && known ? { ...next, sessionOverride: known.sessionOverride, sessionSelection: known.sessionSelection } : next;
      lastState.set(key, merged);
      setState(merged);
      setError("");
      if (!stale && !current.current.selection.dirty && !composer.busy()) setSelection(savedSelection(next));
    } catch (reason) {
      if (request !== latest.current) return;
      setError(`模型列表加载失败：${reason instanceof Error ? reason.message : String(reason)}。发送时仍使用会话已保存的模型。`);
    }
  }, [composer, context, key]);

  // 发出后这次选择就是会话保存的选择；新对话把它交给换成会话身份的下一个挂载。
  const sent = useCallback(() => {
    const { state: known, selection: chosen } = current.current;
    edits.current += 1;
    const saved = known && chosen.dirty ? withSaved(known, chosen) : known;
    if (saved && context.sessionId) lastState.set(key, saved);
    if (saved && !context.sessionId) carried = { state: saved, selection: { ...chosen, dirty: false } };
    if (saved) setState(saved);
    setSelection((previous) => ({ ...previous, dirty: false }));
  }, [context, key]);

  const busy = useCapsuleSubscriptions(context, composer, load, sent);
  const choose = useCallback((runtimeId: string, effort: string) => {
    edits.current += 1;
    setSelection({ runtimeId, effort, dirty: true });
  }, []);
  return { state, error, selection, busy, load, choose };
}

// 回到页面或模型设置变更时重读；生成状态变化时刷新禁用态；宿主确认送达后交接选择。
function useCapsuleSubscriptions(context: PluginUiContext<"composer.toolbar">, composer: ComposerControls, load: () => Promise<void>, sent: () => void): boolean {
  const [busy, setBusy] = useState(() => composer.busy());
  useEffect(() => {
    const stopShown = context.onShown(() => void load());
    void load();
    const stopInvalidate = context.onInvalidate(() => setBusy(composer.busy()));
    const stopSent = composer.onSent(sent);
    return () => {
      stopShown();
      stopInvalidate();
      stopSent();
    };
  }, [composer, context, load, sent]);
  return busy;
}

// 输入框里的模型胶囊：改过的选择随下一条消息作为 model.selection 片段发出，当前模型不可用时阻止发送并说明原因。
function ModelCapsule({ context, composer }: { context: PluginUiContext<"composer.toolbar">; composer: ComposerControls }) {
  const { state, error, selection, busy, load, choose } = useCapsuleState(context, composer);
  const errorId = useId();
  const modelId = (selection.dirty ? selection.runtimeId : selection.runtimeId || state?.sessionOverride) || state?.defaultRuntime || "";
  const problem = modelProblem(state, modelId);
  useEffect(() => { composer.block(problem || null); }, [composer, problem]);
  useEffect(() => {
    composer.contribute(selection.dirty ? [{ kind: "model.selection",
      value: { model_id: selection.runtimeId || null, reasoning_effort: selection.effort || null } }] : []);
  }, [composer, selection]);

  // 读取失败时保留上次的目录，同时给出原因与重试入口；完整原因经 aria-describedby 关联。
  const retry = error ? <>
    <button type="button" className="models-capsule-retry" title={error} aria-describedby={errorId} onClick={() => void load()}>重试读取模型</button>
    <span id={errorId} hidden>{error}</span>
  </> : null;
  if (!state) return retry;
  const { visibleModel, explicitModel } = resolveVisibleRuntime(state.runtimes, selection.runtimeId, state.defaultRuntime);
  return <>
    <ModelCapsulePicker defaultRuntime={state.defaultRuntime} runtimes={state.runtimes} selectedRuntimeId={selection.runtimeId}
      selectedEffort={selection.effort} disabled={busy} onChange={choose} />
    {visibleModel ? <ModelEffortAction runtime={visibleModel} effort={compatibleEffort(visibleModel, selection.effort)}
      explicit={Boolean(explicitModel)} disabled={busy} onChange={choose} /> : null}
    {problem ? <a className="models-capsule-settings" href="/#models" target="_blank" rel="noopener">模型设置</a> : null}
    {retry}
  </>;
}

const definition: PluginUiDefinition = {
  slots: {
    "turn.after_answer": { mount: answerUsage },
    "composer.status": { mount: callSummary },
    "composer.toolbar": {
      mount: (host, context) => render(host, <ModelCapsule context={context} composer={context.composer} />),
    },
  },
};

export default definition;
