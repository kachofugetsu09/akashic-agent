import {
  Check,
  ArrowLeft,
  ChevronDown,
  ChevronRight,
  Eye,
  EyeOff,
  KeyRound,
  LoaderCircle,
  Palette,
  RefreshCw,
  Search,
  ShieldCheck,
  Smartphone,
  Trash2,
  RotateCcw,
  X,
} from "lucide-react";
import { FormEvent, useEffect, useMemo, useRef, useState } from "react";
import { createPortal } from "react-dom";
import codexIcon from "./assets/provider-icons/codex.svg";
import deepseekIcon from "./assets/provider-icons/deepseek.svg";
import opencodeIcon from "./assets/provider-icons/opencode.svg";
import { cycleTheme, useTheme } from "../../theme/src/theme-runtime";
import { Dialog, DialogContent, DialogDescription, DialogTitle } from "./components/ui/dialog";
import { MemorySettings, type MemorySettingsState } from "./memory-settings";
import { MobilePairingDialog } from "./mobile-pairing-dialog";
import "./settings.css";

const isEmbeddedShell = new URLSearchParams(window.location.search).get("embedded") === "1";
type ConnectionKind = "api" | "opencode-go" | "codex";
type ModelRole = "default" | "fast" | "agent" | "vision";

interface RuntimeSummary {
  id: string;
  provider: string;
  model: string;
  sourceId: string;
  sourceName: string;
  catalogProvider: string;
  baseUrl: string;
  contextWindow: number;
  maxOutputTokens: number;
  inputModalities: string[];
  reasoningEffort: string;
  supportedReasoningEfforts: string[];
  credential: { id: string; configured: boolean; source: string };
}

interface RoleBinding {
  modelId: string;
  reasoningEffort: string;
}

interface SettingsState {
  mode: "needs_setup" | "needs_repair" | "ready";
  workspace: string;
  error?: string;
  activeRuntime: string | null;
  runtimes: RuntimeSummary[];
  roleBindings: Partial<Record<ModelRole, RoleBinding>>;
  modelRevision: number;
  codexConfigured: boolean;
  localOpenCodeConfigured: boolean;
  configRevision: string;
  onboarding: {
    step: "welcome" | "model" | "memory" | "channel" | "done";
    completed: boolean;
    memoryDecision: "pending" | "configured" | "skipped";
    channelDecision: "pending" | "configured" | "skipped";
  };
  memory: MemorySettingsState;
  proactive: ProactiveSettingsState;
  channels: ChannelsSettingsState;
  mobileRealtime: { enabled: boolean; port: number; lanHostname: string; publicUrl: string };
}

interface ModelOption {
  id: string;
  contextWindow?: number;
  maxOutputTokens?: number;
  inputModalities?: string[];
  supportedReasoningEfforts?: string[];
  defaultReasoningEffort?: string;
}

interface CodexLoginState {
  loginId: string;
  status: "waiting" | "completed" | "failed";
  userCode: string;
  verificationUri: string;
  interval: number;
  error: string;
}

interface ConnectionGroup {
  sourceId: string;
  sourceName: string;
  provider: string;
  baseUrl: string;
  runtimes: RuntimeSummary[];
}

interface ConnectionDraft {
  sourceId: string;
  sourceName: string;
  kind: ConnectionKind;
  provider: string;
  baseUrl: string;
  apiKey: string;
  credentialId: string;
  model: string;
  reasoningEffort: string;
}

const PROVIDER_TEMPLATES = [
  { kind: "codex" as const, provider: "codex", name: "Codex", detail: "ChatGPT 订阅登录", baseUrl: "", icon: codexIcon },
  { kind: "opencode-go" as const, provider: "opencode-go", name: "OpenCode Go", detail: "本机登录或 API Key", baseUrl: "https://opencode.ai/zen/go/v1", icon: opencodeIcon },
  { kind: "api" as const, provider: "", name: "API 服务", detail: "Base URL、API Key 与模型", baseUrl: "", icon: "" },
];

const API_PRESETS = [
  { provider: "", name: "自定义", baseUrl: "" },
  { provider: "deepseek", name: "DeepSeek", baseUrl: "https://api.deepseek.com/v1" },
];

const ROLE_LABELS: Record<ModelRole, { title: string; detail: string }> = {
  default: { title: "默认模型", detail: "普通模型调用与系统默认" },
  agent: { title: "对话模型", detail: "对话与计划任务的 ReAct 执行" },
  fast: { title: "轻量模型", detail: "压缩、标签与后台提取" },
  vision: { title: "视觉模型", detail: "包含图片的输入" },
};

type ConfigSection = "model" | "memory" | "proactive" | "channels" | "roles";

const CONFIG_SECTIONS: { key: ConfigSection; label: string; detail: string }[] = [
  { key: "model", label: "模型连接", detail: "服务商、凭据与模型" },
  { key: "memory", label: "记忆", detail: "引擎与向量模型" },
  { key: "proactive", label: "主动推送", detail: "推送方式与节奏" },
  { key: "channels", label: "联系方式", detail: "Telegram、QQBot 与手机" },
  { key: "roles", label: "系统模型", detail: "轻量与视觉角色" },
];

interface ProactiveSettingsState {
  configured: boolean;
  enabled: boolean;
  lifecycle: "default" | "wake";
  profile: "daily" | "quiet" | "dev_verify";
  targetChannel: string;
  targetChatId: string;
  driftEnabled: boolean;
  driftMaxSteps: number;
  driftMinIntervalHours: number;
}

interface ChannelsSettingsState {
  telegramConfigured: boolean;
  telegramUsername: string;
  qqConfigured: boolean;
  qqbotConfigured: boolean;
  qqbotTargetId: string;
}

type PushTarget = "web" | "telegram" | "qqbot" | "mobile";

const PUSH_TARGET_LABEL: Record<PushTarget, string> = {
  web: "网页会话",
  telegram: "Telegram",
  qqbot: "QQBot",
  mobile: "手机",
};

async function requestJson<T>(url: string, init?: RequestInit): Promise<T> {
  let response: Response;
  try {
    response = await fetch(url, {
      ...init,
      headers: { "Content-Type": "application/json", "X-Akasic-CSRF": "1", ...init?.headers },
    });
  } catch (reason) {
    if (reason instanceof TypeError) throw new Error("无法连接 Akashic。请确认服务仍在运行，然后重试。", { cause: reason });
    throw reason;
  }
  const text = await response.text();
  let payload: { detail?: string; message?: string };
  try {
    payload = text ? JSON.parse(text) as { detail?: string; message?: string } : {};
  } catch {
    throw new Error(`设置服务返回了无效响应 (${response.status})`);
  }
  if (!response.ok) throw new Error(payload.detail || payload.message || `请求失败 (${response.status})`);
  return payload as T;
}

function connectionKind(provider: string): ConnectionKind {
  if (provider === "codex") return "codex";
  if (provider === "opencode-go") return "opencode-go";
  return "api";
}

function providerIcon(provider: string): string {
  if (provider === "deepseek") return deepseekIcon;
  return PROVIDER_TEMPLATES.find((item) => item.provider === provider)?.icon || "";
}

function createDraft(template = PROVIDER_TEMPLATES[0], existing?: ConnectionGroup): ConnectionDraft {
  return {
    sourceId: existing?.sourceId || `source-${crypto.randomUUID()}`,
    sourceName: existing?.sourceName || (template.provider ? template.name : ""),
    kind: existing ? connectionKind(existing.provider) : template.kind,
    provider: existing?.provider || template.provider,
    baseUrl: existing?.baseUrl || template.baseUrl,
    apiKey: "",
    credentialId: existing?.runtimes[0]?.credential.id || "",
    model: existing?.runtimes[0]?.model || "",
    reasoningEffort: existing?.runtimes[0]?.reasoningEffort || "",
  };
}

function ConnectionMark({ provider, name }: { provider: string; name: string }) {
  const icon = providerIcon(provider);
  return <span className="settings-connection-mark" aria-hidden="true">{icon ? <img src={icon} alt="" /> : provider ? name.slice(0, 1).toUpperCase() : <KeyRound size={20} />}</span>;
}

function OnboardingProgress({ current }: { current: 1 | 2 | 3 }) {
  return (
    <ol className="onboard-progress" aria-label={`设置进度，第 ${current} 步，共 3 步`}>
      {["模型", "记忆", "联系方式"].map((label, index) => {
        const step = index + 1;
        return <li key={label} className={step === current ? "is-current" : step < current ? "is-complete" : ""} aria-current={step === current ? "step" : undefined}><span>{step}</span>{label}</li>;
      })}
    </ol>
  );
}

export function SettingsApp() {
  const theme = useTheme();
  const [state, setState] = useState<SettingsState | null>(null);
  const [query, setQuery] = useState("");
  const [section, setSection] = useState<ConfigSection>("model");
  const [draft, setDraft] = useState<ConnectionDraft | null>(null);
  const [models, setModels] = useState<ModelOption[]>([]);
  const [discovering, setDiscovering] = useState(false);
  const [saving, setSaving] = useState(false);
  const [showKey, setShowKey] = useState(false);
  const [toast, setToast] = useState("");
  const [error, setError] = useState("");
  const [onboardingPairingOpen, setOnboardingPairingOpen] = useState(false);
  const [codexLogin, setCodexLogin] = useState<CodexLoginState | null>(null);
  const [removeConnection, setRemoveConnection] = useState<ConnectionGroup | null>(null);
  const dialogRef = useRef<HTMLDivElement>(null);
  const dialogReturnFocusRef = useRef<HTMLElement | null>(null);
  const openDraftId = draft?.sourceId;

  async function refreshState() {
    const next = await requestJson<SettingsState>("/api/settings/state");
    setState(next);
    return next;
  }

  useEffect(() => { refreshState().catch((reason: Error) => setError(reason.message)); }, []);

  useEffect(() => {
    if (!codexLogin || codexLogin.status !== "waiting") return;
    const timer = window.setInterval(async () => {
      try {
        const next = await requestJson<CodexLoginState>(`/api/settings/codex-login/${codexLogin.loginId}`);
        setCodexLogin(next);
        if (next.status === "completed") {
          await refreshState();
          setToast("Codex 登录已完成，可以发现模型了");
        }
      } catch (reason) {
        setError(reason instanceof Error ? reason.message : String(reason));
      }
    }, Math.max(3, codexLogin.interval) * 1000);
    return () => window.clearInterval(timer);
  }, [codexLogin]);

  useEffect(() => {
    if (!openDraftId) return;
    dialogReturnFocusRef.current = document.activeElement instanceof HTMLElement ? document.activeElement : null;
    const first = dialogRef.current?.querySelector<HTMLElement>(".settings-dialog-body input, .settings-dialog-body button, .settings-dialog-body select");
    first?.focus();
    function onKeyDown(event: KeyboardEvent) {
      if (event.key === "Escape") closeDialog();
      if (event.key !== "Tab" || !dialogRef.current) return;
      const focusable = [...dialogRef.current.querySelectorAll<HTMLElement>("button, input, select")].filter((item) => !item.hasAttribute("disabled"));
      const firstItem = focusable[0];
      const lastItem = focusable.at(-1);
      if (event.shiftKey && document.activeElement === firstItem) { event.preventDefault(); lastItem?.focus(); }
      if (!event.shiftKey && document.activeElement === lastItem) { event.preventDefault(); firstItem?.focus(); }
    }
    document.addEventListener("keydown", onKeyDown);
    return () => document.removeEventListener("keydown", onKeyDown);
  }, [openDraftId]);

  const connections = useMemo(() => {
    const groups = new Map<string, ConnectionGroup>();
    for (const runtime of state?.runtimes || []) {
      const sourceId = runtime.sourceId || runtime.id;
      const current = groups.get(sourceId);
      if (current) current.runtimes.push(runtime);
      else groups.set(sourceId, { sourceId, sourceName: runtime.sourceName || runtime.provider, provider: runtime.provider, baseUrl: runtime.baseUrl, runtimes: [runtime] });
    }
    const normalized = query.trim().toLowerCase();
    return [...groups.values()].filter((group) => `${group.sourceName} ${group.provider} ${group.runtimes.map((item) => item.model).join(" ")}`.toLowerCase().includes(normalized));
  }, [query, state]);
  function closeDialog() {
    setDraft(null);
    setModels([]);
    setError("");
    window.setTimeout(() => dialogReturnFocusRef.current?.focus(), 0);
  }

  async function beginOnboarding() {
    setSaving(true);
    setError("");
    try {
      await requestJson("/api/settings/onboarding/start", { method: "POST", body: "{}" });
      await refreshState();
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : String(reason));
    } finally {
      setSaving(false);
    }
  }

  async function advanceOnboarding(step: "model" | "memory" | "channel", decision?: "configured" | "skipped") {
    await requestJson("/api/settings/onboarding/advance", {
      method: "POST",
      body: JSON.stringify({ step, decision }),
    });
    return refreshState();
  }

  async function backOnboarding() {
    setError("");
    await requestJson("/api/settings/onboarding/back", { method: "POST", body: "{}" });
    await refreshState();
  }

  async function waitForMobilePairingReady() {
    const deadline = Date.now() + 60_000;
    let lastFailure = "移动网关尚未就绪";
    while (Date.now() < deadline) {
      try {
        await requestJson<{ status: "ready" }>("/api/chat/mobile-pairing");
        return;
      } catch (reason) {
        lastFailure = reason instanceof Error ? reason.message : String(reason);
      }
      await new Promise((resolve) => window.setTimeout(resolve, 1_500));
    }
    throw new Error(`移动网关启动超时：${lastFailure}`);
  }

  async function finishOnboarding(destination: "chat" | "configuration" | "mobile") {
    setSaving(true);
    setError("");
    try {
      await requestJson("/api/settings/onboarding/complete", { method: "POST", body: "{}" });
      if (destination === "mobile") {
        await waitForMobilePairingReady();
        setOnboardingPairingOpen(true);
        return;
      }
      await refreshState();
      if (isEmbeddedShell) {
        window.parent.postMessage(
          { type: "akashic.onboarding.completed", destination },
          window.location.origin,
        );
      } else if (destination === "chat") {
        window.location.href = "/";
      }
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : String(reason));
    } finally {
      setSaving(false);
    }
  }

  async function leaveOnboardingAfterPairing() {
    setOnboardingPairingOpen(false);
    setSaving(true);
    setError("");
    try {
      await refreshState();
      if (isEmbeddedShell) {
        window.parent.postMessage(
          { type: "akashic.onboarding.completed", destination: "chat" },
          window.location.origin,
        );
      } else {
        window.location.href = "/";
      }
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : String(reason));
    } finally {
      setSaving(false);
    }
  }

  async function discoverModels() {
    if (!draft) return;
    setDiscovering(true);
    setError("");
    try {
      const result = await requestJson<{ models: ModelOption[] }>("/api/settings/models", {
        method: "POST",
        body: JSON.stringify({
          provider: draft.provider,
          model: "",
          api_key: draft.apiKey,
          base_url: draft.baseUrl,
          credential_id: draft.kind === "codex" ? "codex_default" : draft.credentialId,
          use_local_opencode: draft.kind === "opencode-go" && Boolean(state?.localOpenCodeConfigured) && !draft.apiKey,
        }),
      });
      setModels(result.models);
      if (result.models[0]) {
        setDraft((current) => current ? { ...current, model: current.model || result.models[0].id, reasoningEffort: current.reasoningEffort || result.models[0].defaultReasoningEffort || "" } : current);
      }
      if (!result.models.length) setError("没有发现模型。请确认 Base URL 和认证，或手动填写模型名。");
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : String(reason));
    } finally {
      setDiscovering(false);
    }
  }

  async function saveConnection(event: FormEvent) {
    event.preventDefault();
    if (!draft) return;
    setSaving(true);
    setError("");
    try {
      const accountCatalog = draft.kind === "codex" || draft.kind === "opencode-go";
      const selected = accountCatalog ? undefined : models.find((item) => item.id === draft.model);
      const onboardingConnection = state?.onboarding.step === "model" && !state.onboarding.completed;
      await requestJson("/api/settings/apply", {
        method: "POST",
        body: JSON.stringify({
          provider: draft.provider,
          model: accountCatalog ? "" : draft.model,
          source_id: draft.sourceId,
          source_name: draft.sourceName,
          api_key: draft.apiKey,
          base_url: draft.baseUrl,
          credential_id: draft.kind === "codex" ? "codex_default" : draft.credentialId,
          use_local_opencode: draft.kind === "opencode-go" && Boolean(state?.localOpenCodeConfigured) && !draft.apiKey,
          reasoning_effort: draft.reasoningEffort,
          context_window: selected?.contextWindow || 0,
          max_output_tokens: selected?.maxOutputTokens || 0,
          input_modalities: selected?.inputModalities,
          expected_config_revision: state?.configRevision || "",
          defer_restart: onboardingConnection,
        }),
      });
      if (onboardingConnection) await advanceOnboarding("model");
      else await refreshState();
      setToast(onboardingConnection ? `${draft.sourceName} 已保存，接下来选择记忆方式` : `${draft.sourceName} 已保存，密钥不会显示在页面中`);
      closeDialog();
      if (isEmbeddedShell && !onboardingConnection) {
        window.parent.postMessage(
          { type: "akashic.settings.applied" },
          window.location.origin,
        );
      }
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : String(reason));
    } finally {
      setSaving(false);
    }
  }

  async function beginCodexLogin() {
    setError("");
    try {
      setCodexLogin(await requestJson<CodexLoginState>("/api/settings/codex-login", { method: "POST", body: "{}" }));
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : String(reason));
    }
  }

  async function updateRole(role: ModelRole, modelId: string) {
    setError("");
    try {
      const binding = state?.roleBindings[role];
      await requestJson("/api/settings/roles", {
        method: "POST",
        body: JSON.stringify({ role, model_id: modelId, reasoning_effort: binding?.reasoningEffort || "", expected_revision: state?.modelRevision }),
      });
      await refreshState();
      setToast(`${ROLE_LABELS[role].title}已更新；正在运行的任务继续使用旧快照`);
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : String(reason));
    }
  }

  async function confirmRemoveConnection() {
    if (!removeConnection) return;
    setSaving(true);
    setError("");
    try {
      const removedName = removeConnection.sourceName;
      await requestJson(`/api/settings/model-connections/${encodeURIComponent(removeConnection.sourceId)}/remove`, {
        method: "POST",
        body: JSON.stringify({ expected_revision: state?.modelRevision }),
      });
      setRemoveConnection(null);
      await refreshState();
      setToast(`${removedName} 已移除；相关系统角色已切换到可用模型`);
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : String(reason));
    } finally {
      setSaving(false);
    }
  }

  if (!state && !error) return <div className="settings-loading" role="status"><LoaderCircle className="is-spinning" aria-hidden="true" />正在读取模型连接</div>;
  if (!state) return <main className="settings-page"><section className="settings-repair"><ShieldCheck aria-hidden="true" /><h1>暂时无法读取配置</h1><p>{error}</p><button type="button" className="settings-primary-button" onClick={() => { setError(""); void refreshState().catch((reason: Error) => setError(reason.message)); }}>重新读取</button></section></main>;
  if (state?.mode === "needs_repair") return <main className="settings-page"><section className="settings-repair"><ShieldCheck /><h1>配置需要手动处理</h1><p>{state.error}</p></section></main>;

  // 1. 进度只来自 workspace 的 onboarding owner，不再由旧配置或浏览器会话猜测。
  const activeStep = state.onboarding.completed ? null : state.onboarding.step;

  if (activeStep === "welcome" || activeStep === "model") {
    return (
      <main className="settings-page">
        <div className="settings-shell settings-shell--onboarding">
          {activeStep === "welcome" ? (
            <section className="onboard-welcome" aria-labelledby="onboard-welcome-title">
              <h1 id="onboard-welcome-title">欢迎使用 Akashic</h1>
              <p className="onboard-welcome-lead">先连接一个可用模型，再决定是否启用记忆与主动推送。每一步都会说明需要什么，也可以稍后回来修改。</p>
              <ol className="onboard-welcome-steps">
                <li><span>1</span><div><strong>连接模型</strong><small>登录 Codex、使用 OpenCode，或填写 API 服务</small></div></li>
                <li><span>2</span><div><strong>选择记忆方式</strong><small>可启用 Akasha、经典记忆，或暂不启用</small></div></li>
                <li><span>3</span><div><strong>设置联系方式</strong><small>Telegram、QQBot、手机与主动推送都可以稍后设置</small></div></li>
              </ol>
              <div className="onboard-welcome-note" role="note"><ShieldCheck size={16} aria-hidden="true" /><p>请准备 ChatGPT 订阅、OpenCode 登录或一个兼容服务的 API Key。凭据保存在当前 workspace。</p></div>
              <div className="onboard-footer"><button type="button" className="settings-primary-button" onClick={() => void beginOnboarding()} disabled={saving}>{saving && <LoaderCircle className="is-spinning" size={17} />}{saving ? "正在开始" : "开始配置"}</button></div>
            </section>
          ) : (
            <section className="onboard-model" aria-labelledby="onboard-model-title">
              <OnboardingProgress current={1} />
              <header className="onboard-step-head"><h1 id="onboard-model-title">连接模型</h1><p>选择一种连接方式。账号目录支持的模型会自动同步，API 服务也可以手动填写模型名。</p></header>
              {state.runtimes.length ? <section className="settings-section settings-section--group onboard-existing">
                <header><div><h2>当前模型可以继续使用</h2><p>{state.runtimes.length} 个模型已经可用；也可以在下面添加或更新连接。</p></div></header>
                <div className="settings-actions"><button type="button" className="settings-primary-button" onClick={() => void advanceOnboarding("model")} disabled={saving}>继续使用当前模型</button></div>
              </section> : null}
              <div className="settings-gallery">
                {PROVIDER_TEMPLATES.map((template) => <button type="button" className="settings-connection-card" key={template.provider} onClick={() => setDraft(createDraft(template))}>
                  <ConnectionMark provider={template.provider} name={template.name} /><span className="settings-card-copy"><strong>{template.name}</strong><small>{template.detail}</small></span><ChevronRight className="settings-template-action" size={18} aria-hidden="true" />
                </button>)}
              </div>
              {error && !draft && <p className="settings-inline-error" role="alert">{error}</p>}
            </section>
          )}
        </div>

        {draft && createPortal(<div className="settings-scrim" onMouseDown={(event) => { if (event.target === event.currentTarget) closeDialog(); }}>
          <div ref={dialogRef} className="settings-dialog" role="dialog" aria-modal="true" aria-labelledby="settings-dialog-title">
            <header><div><h2 id="settings-dialog-title">{connections.some((item) => item.sourceId === draft.sourceId) ? `编辑 ${draft.sourceName}` : draft.kind === "codex" ? "连接 Codex" : draft.kind === "opencode-go" ? "连接 OpenCode Go" : "连接 API 服务"}</h2><p>{draft.kind === "codex" ? "授权 ChatGPT 订阅账号，保存后自动同步可用模型。" : draft.kind === "opencode-go" ? "使用本机 OpenCode 登录或单独的 API Key，模型会自动同步。" : "选择服务预设，或填写兼容服务的地址、凭据与模型。"}</p></div><button type="button" className="settings-icon-button" onClick={closeDialog} aria-label="关闭"><X size={20} aria-hidden="true" /></button></header>
            <form onSubmit={saveConnection}>
              <div className="settings-dialog-body">
                <div className="settings-form-grid">
                  {draft.kind === "api" && <label className="is-wide"><span>服务预设</span><select aria-label="服务预设" value={draft.provider === "deepseek" ? "deepseek" : "custom"} onChange={(event) => { const preset = API_PRESETS.find((item) => item.provider === (event.target.value === "custom" ? "" : event.target.value))!; setDraft({ ...draft, provider: preset.provider, baseUrl: preset.baseUrl, sourceName: preset.provider ? preset.name : draft.sourceName }); }}><option value="custom">自定义兼容服务</option><option value="deepseek">DeepSeek</option></select></label>}
                  <label className="is-wide"><span>连接名称</span><input aria-label="连接名称" required value={draft.sourceName} onChange={(event) => setDraft({ ...draft, sourceName: event.target.value })} placeholder={draft.provider === "deepseek" ? "例如：DeepSeek 官方" : "例如：公司网关"} /></label>
                  {draft.kind === "codex" ? <div className="settings-login-card is-wide"><ShieldCheck size={20} /><span><strong>{state?.codexConfigured || codexLogin?.status === "completed" ? "Codex 已登录" : "使用 ChatGPT 订阅登录"}</strong><small>授权凭据保存在当前 workspace，不会显示在页面中。</small></span><button type="button" onClick={beginCodexLogin}>{state?.codexConfigured ? "重新登录" : "开始登录"}</button></div> : <>
                    {draft.kind === "api" && <label><span>Provider ID</span><input aria-label="Provider ID" required value={draft.provider} onChange={(event) => setDraft({ ...draft, provider: event.target.value })} placeholder="例如：openai" /></label>}
                    <label className={draft.kind === "opencode-go" ? "is-wide" : ""}><span>Base URL</span><input aria-label="Base URL" required type="url" value={draft.baseUrl} onChange={(event) => setDraft({ ...draft, baseUrl: event.target.value })} placeholder="https://api.example.com/v1" /></label>
                    <label className="settings-secret is-wide"><span>API Key{draft.kind === "opencode-go" && state?.localOpenCodeConfigured ? "（可留空使用本机登录）" : ""}</span><input aria-label="API Key" required={draft.kind === "api" && !connections.some((item) => item.sourceId === draft.sourceId)} type={showKey ? "text" : "password"} value={draft.apiKey} onChange={(event) => setDraft({ ...draft, apiKey: event.target.value })} autoComplete="off" placeholder="sk-…" /><button type="button" onClick={() => setShowKey((value) => !value)} aria-label={showKey ? "隐藏 API Key" : "显示 API Key"}>{showKey ? <EyeOff size={18} /> : <Eye size={18} />}</button></label>
                  </>}
                </div>

                {codexLogin?.status === "waiting" && draft.kind === "codex" ? <div className="settings-device-login"><span>验证码</span><strong>{codexLogin.userCode}</strong><a href={codexLogin.verificationUri} target="_blank" rel="noreferrer">打开登录页面</a></div> : null}

                {draft.kind === "api" ? <section className="settings-model-discovery">
                  <header><div><h3>可用模型</h3><p>先自动检测；服务不提供目录时再手动填写。</p></div><button type="button" className="settings-quiet-button" onClick={discoverModels} disabled={discovering}>{discovering ? <LoaderCircle className="is-spinning" size={16} /> : <RefreshCw size={16} />}{discovering ? "检测中" : "检测模型"}</button></header>
                  <div className="settings-form-grid">
                    <label className="is-wide"><span>模型名称</span>{models.length ? <select aria-label="模型名称" required value={draft.model} onChange={(event) => { const model = models.find((item) => item.id === event.target.value); setDraft({ ...draft, model: event.target.value, reasoningEffort: model?.defaultReasoningEffort || draft.reasoningEffort }); }}><option value="">选择模型</option>{models.map((model) => <option value={model.id} key={model.id}>{model.id}</option>)}</select> : <input aria-label="模型名称" required value={draft.model} onChange={(event) => setDraft({ ...draft, model: event.target.value })} placeholder={draft.provider === "deepseek" ? "例如：deepseek-chat" : "例如：your-model-name"} />}</label>
                    {(() => { const selected = models.find((item) => item.id === draft.model); const efforts = selected?.supportedReasoningEfforts || []; return efforts.length ? <label className="is-wide"><span>默认思考强度</span><select aria-label="默认思考强度" value={draft.reasoningEffort} onChange={(event) => setDraft({ ...draft, reasoningEffort: event.target.value })}>{efforts.map((effort) => <option value={effort} key={effort}>{effort}</option>)}</select></label> : null; })()}
                  </div>
                  <p>上下文窗口、多模态、推理能力和用量字段会自动归一化。</p>
                </section> : <section className="settings-model-discovery settings-model-discovery--automatic"><header><div><h3>模型自动同步</h3><p>保存后读取账号当前可用的全部模型，无需手动选择。</p></div></header></section>}

                {error && <p className="settings-inline-error" role="alert">{error}</p>}
              </div>
              <footer><span><ShieldCheck size={15} />凭据保存后不会显示在页面中</span><button type="submit" className="settings-primary-button" disabled={saving}>{saving ? <LoaderCircle className="is-spinning" size={17} /> : null}{saving ? "保存中" : draft.kind === "api" ? "保存连接" : "保存并同步模型"}</button></footer>
            </form>
          </div>
        </div>, document.body)}

        <div className="settings-toast-region" aria-live="polite" aria-atomic="true">{toast && <div className="settings-toast" role="status"><Check size={18} /><span><strong>{toast}</strong></span><button type="button" onClick={() => setToast("")} aria-label="关闭通知"><X size={16} /></button></div>}</div>
      </main>
    );
  }

  if (activeStep === "memory") {
    return <main className="settings-page">
      <div className="settings-shell settings-shell--onboarding">
        <OnboardingProgress current={2} />
        <header className="settings-header">
          <div><button type="button" className="settings-back-button" onClick={() => void backOnboarding()}><ArrowLeft size={17} />返回模型</button><h1>选择记忆方式</h1><p>可以现在启用或明确关闭记忆；如果还没准备好向量服务，也可以跳过，之后在配置中心设置。</p></div>
        </header>
        <MemorySettings
          memory={state!.memory}
          modelRevision={state!.modelRevision}
          onboarding
          deferRestart
          onRefresh={async () => (await refreshState()).memory}
          onError={setError}
          onNotice={setToast}
          onComplete={async (message) => {
            await advanceOnboarding("memory", "configured");
            setToast(message);
          }}
          onSkip={async () => {
            await advanceOnboarding("memory", "skipped");
            setToast("已记录为稍后设置；不会改动当前记忆配置");
          }}
        />
        {error && <p className="settings-inline-error" role="alert">{error}</p>}
      </div>
      <div className="settings-toast-region" aria-live="polite" aria-atomic="true">{toast && <div className="settings-toast" role="status"><Check size={18} /><span><strong>{toast}</strong></span><button type="button" onClick={() => setToast("")} aria-label="关闭通知"><X size={16} /></button></div>}</div>
    </main>;
  }

  if (activeStep === "channel") {
    return (
      <main className="settings-page">
        <div className="settings-shell settings-shell--onboarding">
          <OnboardingProgress current={3} />
          <header className="settings-header">
            <div><button type="button" className="settings-back-button" onClick={() => void backOnboarding()}><ArrowLeft size={17} />返回记忆</button><h1>联系方式与主动推送</h1><p>先连接要使用的频道，再选择主动推送目标；也可以跳过整步，之后在配置中心继续。</p></div>
          </header>
          <OnboardingChannelStep
            proactive={state!.proactive}
            channels={state!.channels}
            mobileRealtime={state!.mobileRealtime}
            configRevision={state!.configRevision}
            onNotice={setToast}
            onError={setError}
            onAdvance={async (decision) => {
              await advanceOnboarding("channel", decision);
            }}
          />
          {error && <p className="settings-inline-error" role="alert">{error}</p>}
        </div>
        <div className="settings-toast-region" aria-live="polite" aria-atomic="true">{toast && <div className="settings-toast" role="status"><Check size={18} /><span><strong>{toast}</strong></span><button type="button" onClick={() => setToast("")} aria-label="关闭通知"><X size={16} /></button></div>}</div>
      </main>
    );
  }

  if (activeStep === "done") {
    return (
      <main className="settings-page">
        <div className="settings-shell settings-shell--onboarding">
          <section className="onboard-complete" aria-labelledby="onboard-complete-title">
            <span className="onboard-complete-check" aria-hidden="true"><Check size={24} /></span>
            <h1 id="onboard-complete-title">设置已保存，准备启动</h1>
            <p className="onboard-welcome-lead">先核对下面的选择。返回修改不会丢失已经保存的连接。</p>
            <dl className="onboard-summary">
              <div><dt>模型</dt><dd>{state!.runtimes.length} 个可用模型</dd></div>
              <div><dt>记忆</dt><dd>{state!.onboarding.memoryDecision === "skipped" ? "已跳过 · 可在配置中心设置" : state!.memory.enabled ? (state!.memory.engine === "akasha" ? "Akasha 已启用" : "经典记忆已启用") : "已选择关闭"}</dd></div>
              <div><dt>联系方式与推送</dt><dd>{state!.onboarding.channelDecision === "skipped" ? "已跳过 · 可在配置中心设置" : state!.proactive.enabled ? `已开启 · ${PUSH_TARGET_LABEL[state!.proactive.targetChannel as PushTarget] || "网页会话"}` : state!.channels.telegramConfigured ? "Telegram 已连接 · 主动推送关闭" : state!.channels.qqbotConfigured ? "QQBot 已连接 · 主动推送关闭" : state!.mobileRealtime.enabled ? "手机连接已开启 · 主动推送关闭" : "已选择保持关闭"}</dd></div>
              <div><dt>Android 手机</dt><dd>{state!.mobileRealtime.enabled ? "移动网关将在启动后开启" : "暂未启用 · 可在配置中心连接"}</dd></div>
            </dl>
            {error && <p className="settings-inline-error" role="alert">{error}</p>}
            <div className="onboard-footer onboard-footer--actions">
              <button type="button" className="settings-quiet-button" onClick={() => void backOnboarding()} disabled={saving}><ArrowLeft size={17} />返回修改联系方式</button>
              <button type="button" className="settings-quiet-button" onClick={() => void finishOnboarding("configuration")} disabled={saving}>完成并查看配置中心</button>
              {state!.mobileRealtime.enabled ? <button type="button" className="settings-quiet-button" onClick={() => void finishOnboarding("chat")} disabled={saving}>暂不配对，开始对话</button> : null}
              <button type="button" className="settings-primary-button" onClick={() => void finishOnboarding(state!.mobileRealtime.enabled ? "mobile" : "chat")} disabled={saving}>{saving ? <LoaderCircle className="is-spinning" size={17} /> : null}{saving ? "正在启动" : state!.mobileRealtime.enabled ? "启动并连接手机" : "完成并开始对话"}</button>
            </div>
          </section>
        </div>
        <MobilePairingDialog open={onboardingPairingOpen} onOpenChange={(open) => { setOnboardingPairingOpen(open); if (!open) void leaveOnboardingAfterPairing(); }} />
        <div className="settings-toast-region" aria-live="polite" aria-atomic="true">{toast && <div className="settings-toast" role="status"><Check size={18} /><span><strong>{toast}</strong></span><button type="button" onClick={() => setToast("")} aria-label="关闭通知"><X size={16} /></button></div>}</div>
      </main>
    );
  }

  return (
    <main className="settings-page">
      <div className="config-center">
        <aside className="config-nav" aria-label="配置分类">
          {CONFIG_SECTIONS.map((item) => (
            <button type="button" key={item.key} className={`config-nav-item ${section === item.key ? "is-active" : ""}`} onClick={() => setSection(item.key)} aria-current={section === item.key ? "page" : undefined}>
              <span><strong>{item.label}</strong><small>{item.detail}</small></span>
            </button>
          ))}
        </aside>

        <div className="config-content">
          {section === "model" ? (
            <div className="settings-shell settings-shell--center">
              <header className="settings-header">
                <div><h1>模型连接</h1><p>每套账号或 API Key 都是独立连接；保存后自动识别模型能力。</p></div>
                <div className="settings-header-actions">
                  <button type="button" className="settings-quiet-button" onClick={() => void beginOnboarding()} disabled={saving}><RotateCcw size={17} />重新运行引导</button>
                  {!isEmbeddedShell && <button type="button" className="settings-quiet-button" onClick={cycleTheme}><Palette size={17} />{theme.label}</button>}
                </div>
              </header>

              <label className="settings-search"><Search size={18} aria-hidden="true" /><span className="sr-only">搜索模型连接</span><input value={query} onChange={(event) => setQuery(event.target.value)} placeholder="搜索连接或模型" /></label>

              <section className="settings-section">
                <header><div><h2>已连接</h2><p>同一供应商可以添加多个账号，模型选择时按连接名称区分。</p></div><span>{connections.length} 个</span></header>
                <div className="settings-gallery">
                  {connections.map((group) => <div className="settings-connection-row" key={group.sourceId}>
                    <button type="button" className="settings-connection-card" onClick={() => setDraft(createDraft(PROVIDER_TEMPLATES[0], group))}>
                      <ConnectionMark provider={group.provider} name={group.sourceName} />
                      <span className="settings-card-copy"><strong>{group.sourceName}</strong><small>{group.provider} · {group.runtimes.map((item) => item.model).join("、")}</small></span>
                      <span className="settings-card-meta"><span className="settings-connection-status">已连接</span><small>{group.runtimes.length} 个模型</small></span>
                      <ChevronRight size={18} aria-hidden="true" />
                    </button>
                    <button type="button" className="settings-row-action settings-row-action--danger" onClick={() => setRemoveConnection(group)} disabled={connections.length <= 1} aria-label={`移除 ${group.sourceName}`} title={connections.length <= 1 ? "至少保留一个模型连接" : "移除连接"}><Trash2 size={18} aria-hidden="true" /></button>
                  </div>)}
                </div>
              </section>

              <section className="settings-section settings-section--templates">
                <header><div><h2>添加其他连接</h2><p>可以继续添加另一个账号或服务。</p></div></header>
                <div className="settings-gallery">
                  {PROVIDER_TEMPLATES.map((template) => <button type="button" className="settings-connection-card" key={template.provider} onClick={() => setDraft(createDraft(template))}>
                    <ConnectionMark provider={template.provider} name={template.name} /><span className="settings-card-copy"><strong>{template.name}</strong><small>{template.detail}</small></span><ChevronRight className="settings-template-action" size={18} aria-hidden="true" />
                  </button>)}
                </div>
              </section>
              {error && !draft && <p className="settings-inline-error" role="alert">{error}</p>}
            </div>
          ) : section === "memory" ? (
            <div className="settings-shell settings-shell--center">
              <MemorySettings
                memory={state!.memory}
                modelRevision={state!.modelRevision}
                onRefresh={async () => (await refreshState()).memory}
                onError={setError}
                onNotice={setToast}
                onComplete={async (message) => { setToast(message); await refreshState(); }}
              />
            </div>
          ) : section === "roles" ? (
            <div className="settings-shell settings-shell--center">
              <section className="settings-section settings-roles">
                <header><div><h2>系统模型</h2><p>修改后不重启进程；正在运行的完整 turn 保持旧快照，下一个执行读取最新绑定。</p></div></header>
                <div className="settings-role-grid">
                  {(Object.keys(ROLE_LABELS) as ModelRole[]).map((role) => <label key={role}><span><strong>{ROLE_LABELS[role].title}</strong><small>{ROLE_LABELS[role].detail}</small></span><select value={state!.roleBindings[role]?.modelId || state!.activeRuntime || ""} onChange={(event) => updateRole(role, event.target.value)}>{state!.runtimes.map((runtime) => <option key={runtime.id} value={runtime.id}>{runtime.model}：{runtime.sourceName}</option>)}</select></label>)}
                </div>
              </section>
            </div>
          ) : section === "proactive" ? (
            <ProactiveSection
              proactive={state!.proactive}
              channels={state!.channels}
              mobileRealtime={state!.mobileRealtime}
              configRevision={state!.configRevision}
              error={error}
              onOpenChannels={() => { setError(""); setSection("channels"); }}
              onRefresh={refreshState}
              onNotice={setToast}
              onError={setError}
            />
          ) : (
            <ChannelsSection
              channels={state!.channels}
              mobileRealtime={state!.mobileRealtime}
              configRevision={state!.configRevision}
              error={error}
              waitForMobileReady={waitForMobilePairingReady}
              onContinueToProactive={() => { setError(""); setSection("proactive"); }}
              onRefresh={refreshState}
              onNotice={setToast}
              onError={setError}
            />
          )}
        </div>
      </div>

      {draft && createPortal(<div className="settings-scrim" onMouseDown={(event) => { if (event.target === event.currentTarget) closeDialog(); }}>
        <div ref={dialogRef} className="settings-dialog" role="dialog" aria-modal="true" aria-labelledby="settings-dialog-title">
          <header><div><h2 id="settings-dialog-title">{connections.some((item) => item.sourceId === draft.sourceId) ? `编辑 ${draft.sourceName}` : draft.kind === "codex" ? "连接 Codex" : draft.kind === "opencode-go" ? "连接 OpenCode Go" : "连接 API 服务"}</h2><p>{draft.kind === "codex" ? "授权 ChatGPT 订阅账号，保存后自动同步可用模型。" : draft.kind === "opencode-go" ? "使用本机 OpenCode 登录或单独的 API Key，模型会自动同步。" : "选择服务预设，或填写兼容服务的地址、凭据与模型。"}</p></div><button type="button" className="settings-icon-button" onClick={closeDialog} aria-label="关闭"><X size={20} aria-hidden="true" /></button></header>
          <form onSubmit={saveConnection}>
            <div className="settings-dialog-body">
              <div className="settings-form-grid">
                {draft.kind === "api" && <label className="is-wide"><span>服务预设</span><select aria-label="服务预设" value={draft.provider === "deepseek" ? "deepseek" : "custom"} onChange={(event) => { const preset = API_PRESETS.find((item) => item.provider === (event.target.value === "custom" ? "" : event.target.value))!; setDraft({ ...draft, provider: preset.provider, baseUrl: preset.baseUrl, sourceName: preset.provider ? preset.name : draft.sourceName }); }}><option value="custom">自定义兼容服务</option><option value="deepseek">DeepSeek</option></select></label>}
                <label className="is-wide"><span>连接名称</span><input aria-label="连接名称" required value={draft.sourceName} onChange={(event) => setDraft({ ...draft, sourceName: event.target.value })} placeholder={draft.provider === "deepseek" ? "例如：DeepSeek 官方" : "例如：公司网关"} /></label>
                {draft.kind === "codex" ? <div className="settings-login-card is-wide"><ShieldCheck size={20} /><span><strong>{state?.codexConfigured || codexLogin?.status === "completed" ? "Codex 已登录" : "使用 ChatGPT 订阅登录"}</strong><small>授权凭据保存在当前 workspace，不会显示在页面中。</small></span><button type="button" onClick={beginCodexLogin}>{state?.codexConfigured ? "重新登录" : "开始登录"}</button></div> : <>
                  {draft.kind === "api" && <label><span>Provider ID</span><input aria-label="Provider ID" required value={draft.provider} onChange={(event) => setDraft({ ...draft, provider: event.target.value })} placeholder="例如：openai" /></label>}
                  <label className={draft.kind === "opencode-go" ? "is-wide" : ""}><span>Base URL</span><input aria-label="Base URL" required type="url" value={draft.baseUrl} onChange={(event) => setDraft({ ...draft, baseUrl: event.target.value })} placeholder="https://api.example.com/v1" /></label>
                  <label className="settings-secret is-wide"><span>API Key{draft.kind === "opencode-go" && state?.localOpenCodeConfigured ? "（可留空使用本机登录）" : ""}</span><input aria-label="API Key" required={draft.kind === "api" && !connections.some((item) => item.sourceId === draft.sourceId)} type={showKey ? "text" : "password"} value={draft.apiKey} onChange={(event) => setDraft({ ...draft, apiKey: event.target.value })} autoComplete="off" placeholder="sk-…" /><button type="button" onClick={() => setShowKey((value) => !value)} aria-label={showKey ? "隐藏 API Key" : "显示 API Key"}>{showKey ? <EyeOff size={18} /> : <Eye size={18} />}</button></label>
                </>}
              </div>

              {codexLogin?.status === "waiting" && draft.kind === "codex" ? <div className="settings-device-login"><span>验证码</span><strong>{codexLogin.userCode}</strong><a href={codexLogin.verificationUri} target="_blank" rel="noreferrer">打开登录页面</a></div> : null}

              {draft.kind === "api" ? <section className="settings-model-discovery">
                <header><div><h3>可用模型</h3><p>先自动检测；服务不提供目录时再手动填写。</p></div><button type="button" className="settings-quiet-button" onClick={discoverModels} disabled={discovering}>{discovering ? <LoaderCircle className="is-spinning" size={16} /> : <RefreshCw size={16} />}{discovering ? "检测中" : "检测模型"}</button></header>
                <div className="settings-form-grid">
                  <label className="is-wide"><span>模型名称</span>{models.length ? <select aria-label="模型名称" required value={draft.model} onChange={(event) => { const model = models.find((item) => item.id === event.target.value); setDraft({ ...draft, model: event.target.value, reasoningEffort: model?.defaultReasoningEffort || draft.reasoningEffort }); }}><option value="">选择模型</option>{models.map((model) => <option value={model.id} key={model.id}>{model.id}</option>)}</select> : <input aria-label="模型名称" required value={draft.model} onChange={(event) => setDraft({ ...draft, model: event.target.value })} placeholder={draft.provider === "deepseek" ? "例如：deepseek-chat" : "例如：your-model-name"} />}</label>
                  {(() => { const selected = models.find((item) => item.id === draft.model); const efforts = selected?.supportedReasoningEfforts || []; return efforts.length ? <label className="is-wide"><span>默认思考强度</span><select aria-label="默认思考强度" value={draft.reasoningEffort} onChange={(event) => setDraft({ ...draft, reasoningEffort: event.target.value })}>{efforts.map((effort) => <option value={effort} key={effort}>{effort}</option>)}</select></label> : null; })()}
                </div>
                <p>上下文窗口、多模态、推理能力和用量字段会自动归一化。</p>
              </section> : <section className="settings-model-discovery settings-model-discovery--automatic"><header><div><h3>模型自动同步</h3><p>保存后读取账号当前可用的全部模型，无需手动选择。</p></div></header></section>}

              {error && <p className="settings-inline-error" role="alert">{error}</p>}
            </div>
            <footer><span><ShieldCheck size={15} />凭据保存后不会显示在页面中</span><button type="submit" className="settings-primary-button" disabled={saving}>{saving ? <LoaderCircle className="is-spinning" size={17} /> : null}{saving ? "保存中" : draft.kind === "api" ? "保存连接" : "保存并同步模型"}</button></footer>
          </form>
        </div>
        </div>, document.body)}

      <Dialog open={Boolean(removeConnection)} onOpenChange={(open) => { if (!open) setRemoveConnection(null); }}>
        <DialogContent className="settings-dialog" overlayClassName="settings-scrim" showCloseButton={false} aria-describedby="remove-connection-description">
          <header><div><DialogTitle>移除模型连接</DialogTitle><DialogDescription id="remove-connection-description">将停用 {removeConnection?.sourceName} 及其模型。正在使用这些模型的系统角色会切换到其他可用模型；恢复备份会保留在当前 workspace。</DialogDescription></div></header>
          {error && <p className="settings-inline-error settings-dialog-inline" role="alert">{error}</p>}
          <div className="settings-dialog-actions"><button type="button" className="settings-quiet-button" onClick={() => setRemoveConnection(null)} disabled={saving}>保留连接</button><button type="button" className="settings-danger-button" onClick={() => void confirmRemoveConnection()} disabled={saving}>{saving ? <LoaderCircle className="is-spinning" size={17} /> : <Trash2 size={17} />}{saving ? "正在移除" : "移除连接"}</button></div>
        </DialogContent>
      </Dialog>

      <div className="settings-toast-region" aria-live="polite" aria-atomic="true">{toast && <div className="settings-toast" role="status"><Check size={18} /><span><strong>{toast}</strong></span><button type="button" onClick={() => setToast("")} aria-label="关闭通知"><X size={16} /></button></div>}</div>
    </main>
  );
}

const PROACTIVE_MODES: { key: "daily" | "quiet" | "dev_verify"; name: string; detail: string }[] = [
  { key: "daily", name: "日常陪伴（推荐）", detail: "按固定节奏活跃推送订阅内容和提醒" },
  { key: "quiet", name: "安静模式", detail: "低频，只推送重要内容" },
  { key: "dev_verify", name: "演示模式", detail: "2～5 分钟内可见推送效果" },
];

const PUSH_ENGINES: { key: "default" | "wake"; name: string; detail: string }[] = [
  { key: "default", name: "日常推送 · 定时轮询", detail: "按节奏定期检查订阅内容；活跃时检查更频繁" },
  { key: "wake", name: "智能唤醒 · 事件驱动", detail: "收到新内容事件时评估是否需要发送" },
];

function ProactiveSection({ proactive, channels, mobileRealtime, configRevision, error, onOpenChannels, onRefresh, onNotice, onError }: {
  proactive: ProactiveSettingsState;
  channels: ChannelsSettingsState;
  mobileRealtime: { enabled: boolean };
  configRevision: string;
  error: string;
  onOpenChannels: () => void;
  onRefresh: () => Promise<unknown>;
  onNotice: (message: string) => void;
  onError: (message: string) => void;
}) {
  const configuredTarget: PushTarget = ["web", "telegram", "qqbot", "mobile"].includes(proactive.targetChannel) ? proactive.targetChannel as PushTarget : "web";
  const [enabled, setEnabled] = useState(proactive.enabled);
  const [engine, setEngine] = useState<"default" | "wake">(proactive.lifecycle);
  const [mode, setMode] = useState<"daily" | "quiet" | "dev_verify">(proactive.profile);
  const [target, setTarget] = useState<PushTarget>(configuredTarget);
  const [targetId, setTargetId] = useState(proactive.targetChatId || (configuredTarget === "web" ? "web:default" : configuredTarget === "mobile" ? "default" : ""));
  const [driftEnabled, setDriftEnabled] = useState(proactive.driftEnabled);
  const driftMaxSteps = proactive.driftMaxSteps;
  const driftMinHours = proactive.driftMinIntervalHours;
  const [saving, setSaving] = useState(false);
  const [driftOpen, setDriftOpen] = useState(false);
  const [discoveringTarget, setDiscoveringTarget] = useState(false);

  useEffect(() => {
    setEnabled(proactive.enabled);
    setEngine(proactive.lifecycle);
    setMode(proactive.profile);
    const nextTarget: PushTarget = ["web", "telegram", "qqbot", "mobile"].includes(proactive.targetChannel) ? proactive.targetChannel as PushTarget : "web";
    setTarget(nextTarget);
    setTargetId(proactive.targetChatId || (nextTarget === "web" ? "web:default" : nextTarget === "mobile" ? "default" : ""));
    setDriftEnabled(proactive.driftEnabled);
  }, [proactive.driftEnabled, proactive.enabled, proactive.lifecycle, proactive.profile, proactive.targetChannel, proactive.targetChatId]);

  async function handleSave() {
    setSaving(true);
    try {
      const chatId = target === "web" ? "web:default" : target === "mobile" ? (targetId.trim() || "default") : targetId.trim();
      if (enabled && !channelReady) {
        onError(`请先连接${target === "telegram" ? " Telegram" : target === "qqbot" ? " QQBot" : "手机"}，再保存主动推送。`);
        return;
      }
      if (enabled && !chatId) {
        onError("请输入这个频道用于接收推送的目标 ID。");
        return;
      }
      await requestJson("/api/settings/proactive", {
        method: "POST",
        body: JSON.stringify({
          enabled,
          lifecycle: engine,
          profile: mode,
          target_channel: target,
          target_chat_id: chatId,
          drift_enabled: driftEnabled,
          drift_max_steps: driftMaxSteps,
          drift_min_interval_hours: driftMinHours,
          expected_revision: configRevision,
        }),
      });
      await onRefresh();
      onNotice("主动推送已保存，正在重启生效");
      if (isEmbeddedShell) window.parent.postMessage({ type: "akashic.settings.applied" }, window.location.origin);
    } catch (reason) {
      onError(reason instanceof Error ? reason.message : String(reason));
    } finally {
      setSaving(false);
    }
  }

  async function discoverTarget() {
    if (target !== "telegram" && target !== "qqbot") return;
    setDiscoveringTarget(true);
    onError("");
    try {
      const result = await requestJson<{ targetId: string }>(`/api/settings/channels/${target}/discover-target`, {
        method: "POST",
        body: "{}",
      });
      setTargetId(result.targetId);
      onNotice(`已识别${target === "telegram" ? " Telegram" : " QQBot"} 推送目标`);
    } catch (reason) {
      onError(reason instanceof Error ? reason.message : String(reason));
    } finally {
      setDiscoveringTarget(false);
    }
  }

  const channelReady = target === "web" || (target === "telegram" && channels.telegramConfigured) || (target === "qqbot" && channels.qqbotConfigured) || (target === "mobile" && mobileRealtime.enabled);

  function changeTarget(next: PushTarget) {
    setTarget(next);
    if (next === proactive.targetChannel) setTargetId(proactive.targetChatId);
    else if (next === "web") setTargetId("web:default");
    else if (next === "mobile") setTargetId("default");
    else if (next === "qqbot") setTargetId(channels.qqbotTargetId);
    else setTargetId("");
    onError("");
  }

  return (
    <div className="settings-shell settings-shell--center">
      <header className="settings-header">
        <div><h1>主动推送</h1><p>按选定方式检查订阅内容，并发送到指定频道。</p></div>
      </header>

      <section className="settings-section settings-section--group settings-toggle-section">
        <header><div><h2>启用主动推送</h2><p>关闭后只在对话中回应，不会主动发送消息。</p></div><label className="settings-toggle"><input type="checkbox" checked={enabled} onChange={(event) => setEnabled(event.target.checked)} /><span className="settings-toggle-track" aria-hidden="true" /><span className="sr-only">启用主动推送</span></label></header>
      </section>

      {enabled ? <>
        <section className="settings-section settings-section--group">
          <header><div><h2>推送方式</h2><p>两种引擎行为不同，随时可以切换。</p></div></header>
          <fieldset className="settings-choice-group">
            <legend className="sr-only">推送方式</legend>
            {PUSH_ENGINES.map((item) => <label key={item.key} className={engine === item.key ? "is-selected" : ""}><input type="radio" name="proactive-engine" checked={engine === item.key} onChange={() => setEngine(item.key)} /><span><strong>{item.name}</strong><small>{item.detail}</small></span></label>)}
          </fieldset>
        </section>

        {engine === "default" ? <section className="settings-section settings-section--group">
          <header><div><h2>推送节奏</h2><p>日常推送下的活跃程度。</p></div></header>
          <fieldset className="settings-choice-group">
            <legend className="sr-only">推送节奏</legend>
            {PROACTIVE_MODES.map((item) => <label key={item.key} className={mode === item.key ? "is-selected" : ""}><input type="radio" name="proactive-mode" checked={mode === item.key} onChange={() => setMode(item.key)} /><span><strong>{item.name}</strong><small>{item.detail}</small></span></label>)}
          </fieldset>
        </section> : <p className="settings-note">智能唤醒由新内容事件触发，不使用日常推送节奏。</p>}

        <section className="settings-section settings-section--group">
          <header><div><h2>推送目标</h2><p>主动消息发送到哪里。</p></div></header>
          <label className="settings-field"><span>目标频道</span>
            <select value={target} onChange={(event) => changeTarget(event.target.value as PushTarget)}>
              <option value="web">网页会话（默认，无需配置）</option>
              <option value="telegram">Telegram{channels.telegramConfigured ? " · 已连接" : " · 需要先连接"}</option>
              <option value="qqbot">QQBot{channels.qqbotConfigured ? " · 已连接" : " · 需要先连接"}</option>
              <option value="mobile">手机{mobileRealtime.enabled ? " · 已开启" : " · 需要先开启"}</option>
            </select>
          </label>
          {!channelReady ? <div className="settings-next-action"><p>这个目标还不能接收消息。</p><button type="button" className="settings-quiet-button" onClick={onOpenChannels}>先连接{target === "telegram" ? " Telegram" : target === "qqbot" ? " QQBot" : "手机"}</button></div> : null}
          {channelReady && (target === "telegram" || target === "qqbot") ? <div className="settings-stack settings-stack--inline"><label className="settings-field"><span>{target === "telegram" ? "Telegram chat ID" : "QQBot C2C 目标"}</span><input required={enabled} value={targetId} onChange={(event) => setTargetId(event.target.value)} placeholder={target === "telegram" ? "例如：123456789" : "例如：c2c:USER_OPENID"} /><small>这是主动消息的接收目标，不是 Bot 的账号。</small></label><div className="settings-target-discovery"><p>点一下后，在 45 秒内向 Bot 发一条私聊消息。</p><button type="button" className="settings-quiet-button" onClick={() => void discoverTarget()} disabled={discoveringTarget}>{discoveringTarget ? <LoaderCircle className="is-spinning" size={17} /> : null}{discoveringTarget ? "等待消息…" : "从消息识别目标"}</button></div></div> : null}
          {channelReady && target === "mobile" ? <p className="settings-note">主动消息会进入手机端默认会话，并同步到已配对设备。</p> : null}
        </section>

        <section className="settings-section settings-section--group">
          <button type="button" className="settings-disclosure" onClick={() => setDriftOpen((value) => !value)} aria-expanded={driftOpen}>
            <span><strong>高级：空闲时自主执行任务（Drift）</strong><small>没有待推送内容时执行自主任务。建议先验证主动推送。</small></span>
            <ChevronDown className={driftOpen ? "is-rotated" : ""} size={18} aria-hidden="true" />
          </button>
          {driftOpen ? <>
          <label className="settings-toggle settings-toggle--row"><input type="checkbox" checked={driftEnabled} onChange={(event) => setDriftEnabled(event.target.checked)} /><span className="settings-toggle-track" aria-hidden="true" /><span className="settings-toggle-copy"><strong>启用 Drift</strong><small>允许 Akashic 在空闲时执行自主任务</small></span></label>
        </> : null}
        </section>
      </> : null}

      {error && <p className="settings-inline-error" role="alert">{error}</p>}
      <div className="settings-actions"><button type="button" className="settings-primary-button" onClick={handleSave} disabled={saving}>{saving ? <LoaderCircle className="is-spinning" size={17} /> : null}{saving ? "保存中" : "保存主动推送"}</button></div>
    </div>
  );
}

function ChannelsSection({ channels, mobileRealtime, configRevision, error, waitForMobileReady, onContinueToProactive, onRefresh, onNotice, onError }: {
  channels: ChannelsSettingsState;
  mobileRealtime: { enabled: boolean; port: number; lanHostname: string; publicUrl: string };
  configRevision: string;
  error: string;
  waitForMobileReady: () => Promise<void>;
  onContinueToProactive: () => void;
  onRefresh: () => Promise<unknown>;
  onNotice: (message: string) => void;
  onError: (message: string) => void;
}) {
  const [telegramToken, setTelegramToken] = useState("");
  const [telegramUsername, setTelegramUsername] = useState(channels.telegramUsername);
  const [qqbotAppId, setQqbotAppId] = useState("");
  const [qqbotSecret, setQqbotSecret] = useState("");
  const [qqbotTargetId, setQqbotTargetId] = useState(channels.qqbotTargetId);
  const [publicUrl, setPublicUrl] = useState(mobileRealtime.publicUrl);
  const [showSecret, setShowSecret] = useState(false);
  const [guideOpen, setGuideOpen] = useState(false);
  const [qqGuideOpen, setQqGuideOpen] = useState(false);
  const [saving, setSaving] = useState(false);
  const [pairingOpen, setPairingOpen] = useState(false);
  const [enabling, setEnabling] = useState(false);
  const [disconnecting, setDisconnecting] = useState<"telegram" | "qqbot" | null>(null);

  async function saveChannel(event: FormEvent, kind: "telegram" | "qqbot") {
    event.preventDefault();
    setSaving(true);
    onError("");
    try {
      await requestJson("/api/settings/channels", {
        method: "POST",
        body: JSON.stringify({
          telegram_token: kind === "telegram" ? telegramToken : "",
          telegram_username: kind === "telegram" ? telegramUsername : "",
          qq_app_id: "",
          qq_client_secret: "",
          qqbot_app_id: kind === "qqbot" ? qqbotAppId : "",
          qqbot_client_secret: kind === "qqbot" ? qqbotSecret : "",
          qqbot_target_id: kind === "qqbot" ? qqbotTargetId : "",
          expected_revision: configRevision,
        }),
      });
      await onRefresh();
      onNotice(`${kind === "telegram" ? "Telegram" : "QQBot"} 已连接；现在可以把它设为主动推送目标`);
      if (kind === "telegram") setTelegramToken("");
      else { setQqbotAppId(""); setQqbotSecret(""); }
      if (isEmbeddedShell) window.parent.postMessage({ type: "akashic.settings.applied" }, window.location.origin);
    } catch (reason) {
      onError(reason instanceof Error ? reason.message : String(reason));
    } finally {
      setSaving(false);
    }
  }

  async function saveMobile(pairAfter: boolean) {
    setEnabling(true);
    onError("");
    try {
      await requestJson("/api/settings/mobile-realtime", {
        method: "POST",
        body: JSON.stringify({ enabled: true, public_url: publicUrl, expected_revision: configRevision }),
      });
      await onRefresh();
      onNotice(publicUrl.trim() ? "手机公网入口已保存" : "手机局域网入口已保存");
      if (pairAfter) {
        await waitForMobileReady();
        setPairingOpen(true);
      }
    } catch (reason) {
      onError(reason instanceof Error ? reason.message : String(reason));
    } finally {
      setEnabling(false);
    }
  }

  async function disconnectChannel() {
    if (!disconnecting) return;
    setSaving(true);
    onError("");
    try {
      const label = disconnecting === "telegram" ? "Telegram" : "QQBot";
      await requestJson(`/api/settings/channels/${disconnecting}/disconnect`, { method: "POST", body: "{}" });
      setDisconnecting(null);
      await onRefresh();
      onNotice(`${label} 已断开`);
    } catch (reason) {
      onError(reason instanceof Error ? reason.message : String(reason));
    } finally {
      setSaving(false);
    }
  }

  return (
    <div className="settings-shell settings-shell--center">
      <header className="settings-header">
        <div><h1>连接频道与手机</h1><p>先建立消息入口，再到主动推送中选择接收目标。</p></div>
      </header>

        <section className="settings-section settings-section--group">
          <header><div><h2>Telegram</h2><p>{channels.telegramConfigured ? "已连接，可以接收对话与主动推送。" : "连接 Bot 后可以接收对话与主动推送。"}</p></div><span className={channels.telegramConfigured ? "settings-status is-ready" : "settings-status"}>{channels.telegramConfigured ? "已连接" : "未连接"}</span></header>
          <button type="button" className="settings-disclosure" onClick={() => setGuideOpen((value) => !value)} aria-expanded={guideOpen}>
            <span><strong>还没有 bot？按下面步骤创建</strong></span>
            <ChevronDown className={guideOpen ? "is-rotated" : ""} size={18} aria-hidden="true" />
          </button>
          {guideOpen ? <ol className="settings-guide">
            <li>打开 Telegram，搜索 <strong>@BotFather</strong></li>
            <li>发送 <code>/newbot</code>，按提示给 bot 起名</li>
            <li>BotFather 回复一串 token，格式：<code>123456789:AAFxxx…</code></li>
          </ol> : null}
          <form className="settings-stack" onSubmit={(event) => void saveChannel(event, "telegram")}>
            <label className="settings-field"><span>Bot token{channels.telegramConfigured ? "（留空保留现有 token）" : ""}</span><span className="settings-secret-control"><input required={!channels.telegramConfigured} type={showSecret ? "text" : "password"} autoComplete="off" value={telegramToken} onChange={(event) => setTelegramToken(event.target.value)} placeholder={channels.telegramConfigured ? "已保存" : "123456789:AAFxxx…"} aria-label="Bot token" /><button type="button" onClick={() => setShowSecret((value) => !value)} aria-label={showSecret ? "隐藏 Bot token" : "显示 Bot token"}>{showSecret ? <EyeOff size={18} aria-hidden="true" /> : <Eye size={18} aria-hidden="true" />}</button></span></label>
            <label className="settings-field"><span>你的 Telegram 用户名</span><input value={telegramUsername} onChange={(event) => setTelegramUsername(event.target.value)} placeholder="不带 @，例如 your_name" /></label>
            <div className="settings-actions settings-actions--split">{channels.telegramConfigured ? <button type="button" className="settings-danger-text-button" onClick={() => setDisconnecting("telegram")}>断开 Telegram</button> : <span /> }<button type="submit" className="settings-primary-button" disabled={saving}>{saving ? <LoaderCircle className="is-spinning" size={17} /> : null}{saving ? "保存中" : channels.telegramConfigured ? "更新 Telegram" : "连接 Telegram"}</button></div>
          </form>
        </section>

        <section className="settings-section settings-section--group">
          <header><div><h2>QQBot</h2><p>{channels.qqbotConfigured ? "已连接腾讯官方 QQBot。" : "使用腾讯开放平台 AppID 与 AppSecret 连接。"}</p></div><span className={channels.qqbotConfigured ? "settings-status is-ready" : "settings-status"}>{channels.qqbotConfigured ? "已连接" : "未连接"}</span></header>
          <button type="button" className="settings-disclosure" onClick={() => setQqGuideOpen((value) => !value)} aria-expanded={qqGuideOpen}><span><strong>创建 QQBot 与取得 C2C 目标</strong></span><ChevronDown className={qqGuideOpen ? "is-rotated" : ""} size={18} aria-hidden="true" /></button>
          {qqGuideOpen ? <ol className="settings-guide"><li>在腾讯开放平台创建机器人应用并开启私聊权限</li><li>记录 AppID 与 AppSecret</li><li>向 Bot 发送消息，从调试日志或已有配置取得 user_openid</li></ol> : null}
          <form className="settings-stack" onSubmit={(event) => void saveChannel(event, "qqbot")}>
            <label className="settings-field"><span>AppID{channels.qqbotConfigured ? "（更新时填写）" : ""}</span><input required={!channels.qqbotConfigured} value={qqbotAppId} onChange={(event) => setQqbotAppId(event.target.value)} autoComplete="off" placeholder={channels.qqbotConfigured ? "已保存" : "机器人 AppID"} /></label>
            <label className="settings-field"><span>AppSecret{channels.qqbotConfigured ? "（更新时填写）" : ""}</span><span className="settings-secret-control"><input required={!channels.qqbotConfigured} type={showSecret ? "text" : "password"} value={qqbotSecret} onChange={(event) => setQqbotSecret(event.target.value)} autoComplete="off" placeholder={channels.qqbotConfigured ? "已保存" : "机器人 AppSecret"} /><button type="button" onClick={() => setShowSecret((value) => !value)} aria-label={showSecret ? "隐藏 AppSecret" : "显示 AppSecret"}>{showSecret ? <EyeOff size={18} aria-hidden="true" /> : <Eye size={18} aria-hidden="true" />}</button></span></label>
            <label className="settings-field"><span>C2C 推送目标（可稍后填写）</span><input value={qqbotTargetId} onChange={(event) => setQqbotTargetId(event.target.value)} placeholder="c2c:USER_OPENID" /><small>主动推送到 QQBot 时需要；仅聊天可以暂时留空。</small></label>
            <div className="settings-actions settings-actions--split">{channels.qqbotConfigured ? <button type="button" className="settings-danger-text-button" onClick={() => setDisconnecting("qqbot")}>断开 QQBot</button> : <span /> }<button type="submit" className="settings-primary-button" disabled={saving}>{saving ? <LoaderCircle className="is-spinning" size={17} /> : null}{saving ? "保存中" : channels.qqbotConfigured ? "更新 QQBot" : "连接 QQBot"}</button></div>
          </form>
        </section>

        <section className="settings-section settings-section--group">
          <header><div><h2>连接手机</h2><p>可以使用局域网地址，也可以配置公网 WSS 入口。</p></div><span className={mobileRealtime.enabled ? "settings-status is-ready" : "settings-status"}>{mobileRealtime.enabled ? "已开启" : "未开启"}</span></header>
          <label className="settings-field"><span>公网 WSS 地址（可选）</span><input type="url" value={publicUrl} onChange={(event) => setPublicUrl(event.target.value)} placeholder="wss://mobile.example.com/ws" /><small>填写后手机和电脑无需处在同一网络；地址必须以 wss:// 开头并以 /ws 结尾。</small></label>
          <p className="settings-note">{mobileRealtime.enabled
            ? mobileRealtime.publicUrl
              ? `公网入口已配置（${mobileRealtime.publicUrl}），手机和电脑无需在同一网络。`
              : `当前只有局域网入口（${mobileRealtime.lanHostname}:${mobileRealtime.port}），配对时手机和电脑需在同一网络。`
            : "移动网关尚未启用，点击后会自动开启并重启，稍等片刻即可使用。"}</p>
          <div className="settings-actions settings-actions--split"><button type="button" className="settings-quiet-button" onClick={() => void saveMobile(false)} disabled={enabling}>{enabling ? "正在保存" : "保存入口"}</button><button type="button" className="settings-primary-button" onClick={() => void saveMobile(true)} disabled={enabling}><Smartphone size={17} />{enabling ? "正在准备移动网关…" : "保存并连接手机"}</button></div>
        </section>

      {error && <p className="settings-inline-error" role="alert">{error}</p>}
      {(channels.telegramConfigured || channels.qqbotConfigured || mobileRealtime.enabled) ? <div className="settings-flow-next"><div><strong>下一步：选择推送目标</strong><p>频道连接完成后，再决定主动消息发送到哪里。</p></div><button type="button" className="settings-primary-button" onClick={onContinueToProactive}>设置主动推送</button></div> : null}

      <Dialog open={Boolean(disconnecting)} onOpenChange={(open) => { if (!open) setDisconnecting(null); }}>
        <DialogContent className="settings-dialog" overlayClassName="settings-scrim" showCloseButton={false} aria-describedby="disconnect-channel-description"><header><div><DialogTitle>断开{disconnecting === "telegram" ? " Telegram" : " QQBot"}</DialogTitle><DialogDescription id="disconnect-channel-description">频道凭据将从当前 workspace 移除。若它仍是主动推送目标，需要先更换目标或关闭主动推送。</DialogDescription></div></header><div className="settings-dialog-actions"><button type="button" className="settings-quiet-button" onClick={() => setDisconnecting(null)}>保留连接</button><button type="button" className="settings-danger-button" onClick={() => void disconnectChannel()} disabled={saving}>断开频道</button></div></DialogContent>
      </Dialog>

      <MobilePairingDialog open={pairingOpen} onOpenChange={setPairingOpen} />
    </div>
  );
}

function OnboardingChannelStep({ proactive, channels, mobileRealtime, configRevision, onNotice, onError, onAdvance }: {
  proactive: ProactiveSettingsState;
  channels: ChannelsSettingsState;
  mobileRealtime: { enabled: boolean; port: number; lanHostname: string; publicUrl: string };
  configRevision: string;
  onNotice: (message: string) => void;
  onError: (message: string) => void;
  onAdvance: (decision: "configured" | "skipped") => Promise<void>;
}) {
  const initialPushTarget: PushTarget = ["web", "telegram", "qqbot", "mobile"].includes(proactive.targetChannel) ? proactive.targetChannel as PushTarget : "web";
  const [pushTarget, setPushTarget] = useState<PushTarget>(initialPushTarget);
  const [targetId, setTargetId] = useState(proactive.targetChatId || (initialPushTarget === "web" ? "web:default" : initialPushTarget === "mobile" ? "default" : ""));
  const [enabled, setEnabled] = useState(proactive.enabled);
  const [engine, setEngine] = useState<"default" | "wake">(proactive.lifecycle);
  const [mode, setMode] = useState<"daily" | "quiet" | "dev_verify">(proactive.profile);
  const driftEnabled = proactive.driftEnabled;
  const driftMaxSteps = proactive.driftMaxSteps;
  const driftMinHours = proactive.driftMinIntervalHours;
  const [telegramToken, setTelegramToken] = useState("");
  const [telegramUsername, setTelegramUsername] = useState(channels.telegramUsername);
  const [showToken, setShowToken] = useState(false);
  const [telegramOpen, setTelegramOpen] = useState(false);
  const [qqbotOpen, setQqbotOpen] = useState(false);
  const [qqbotAppId, setQqbotAppId] = useState("");
  const [qqbotSecret, setQqbotSecret] = useState("");
  const [showQqbotSecret, setShowQqbotSecret] = useState(false);
  const [guideOpen, setGuideOpen] = useState(false);
  const [mobileEnabled, setMobileEnabled] = useState(mobileRealtime.enabled);
  const [mobilePublicUrl, setMobilePublicUrl] = useState(mobileRealtime.publicUrl);
  const [saving, setSaving] = useState(false);
  const [discoveringTarget, setDiscoveringTarget] = useState(false);

  async function handleSave() {
    setSaving(true);
    try {
      const telegramReady = Boolean(channels.telegramConfigured || telegramToken.trim());
      const qqbotReady = Boolean(channels.qqbotConfigured || (qqbotAppId.trim() && qqbotSecret.trim()));
      const targetReady = pushTarget === "web" || (pushTarget === "telegram" && telegramReady) || (pushTarget === "qqbot" && qqbotReady) || (pushTarget === "mobile" && mobileEnabled);
      const targetChatId = pushTarget === "web" ? "web:default" : pushTarget === "mobile" ? (targetId.trim() || "default") : targetId.trim();
      if (enabled && !targetReady) {
        onError("先完成所选频道的连接，再开启主动推送。");
        return;
      }
      if (enabled && !targetChatId) {
        onError("请输入所选频道用于接收推送的目标 ID。");
        return;
      }
      await requestJson("/api/settings/onboarding-channel", {
        method: "POST",
        body: JSON.stringify({
          telegram_token: telegramToken.trim(),
          telegram_username: telegramUsername,
          proactive_enabled: enabled,
          lifecycle: engine,
          profile: mode,
          target_channel: pushTarget,
          target_chat_id: targetChatId,
          drift_enabled: driftEnabled,
          drift_max_steps: driftMaxSteps,
          drift_min_interval_hours: driftMinHours,
          mobile_realtime_enabled: mobileEnabled,
          mobile_public_url: mobilePublicUrl,
          qqbot_app_id: qqbotAppId,
          qqbot_client_secret: qqbotSecret,
          qqbot_target_id: pushTarget === "qqbot" ? targetChatId : channels.qqbotTargetId,
          expected_revision: configRevision,
        }),
      });
      await onAdvance("configured");
      onNotice("设置已保存");
    } catch (reason) {
      onError(reason instanceof Error ? reason.message : String(reason));
    } finally {
      setSaving(false);
    }
  }

  async function discoverOnboardingTarget() {
    if (pushTarget !== "telegram" && pushTarget !== "qqbot") return;
    setDiscoveringTarget(true);
    onError("");
    try {
      const result = await requestJson<{ targetId: string }>(`/api/settings/channels/${pushTarget}/discover-target`, {
        method: "POST",
        body: JSON.stringify(pushTarget === "telegram"
          ? { token: telegramToken, username: telegramUsername }
          : { app_id: qqbotAppId, client_secret: qqbotSecret }),
      });
      setTargetId(result.targetId);
      onNotice(`已识别${pushTarget === "telegram" ? " Telegram" : " QQBot"} 推送目标`);
    } catch (reason) {
      onError(reason instanceof Error ? reason.message : String(reason));
    } finally {
      setDiscoveringTarget(false);
    }
  }

  const selectedTargetReady = pushTarget === "web" || (pushTarget === "telegram" && (channels.telegramConfigured || Boolean(telegramToken.trim()))) || (pushTarget === "qqbot" && (channels.qqbotConfigured || Boolean(qqbotAppId.trim() && qqbotSecret.trim()))) || (pushTarget === "mobile" && mobileEnabled);

  function changePushTarget(next: PushTarget) {
    setPushTarget(next);
    if (next === proactive.targetChannel) setTargetId(proactive.targetChatId);
    else if (next === "web") setTargetId("web:default");
    else if (next === "mobile") setTargetId("default");
    else if (next === "qqbot") setTargetId(channels.qqbotTargetId);
    else setTargetId("");
    onError("");
  }

  async function handleSkip() {
    setSaving(true);
    onError("");
    try {
      await onAdvance("skipped");
      onNotice("已记录为稍后设置；不会改动当前联系方式和主动推送配置");
    } catch (reason) {
      onError(reason instanceof Error ? reason.message : String(reason));
    } finally {
      setSaving(false);
    }
  }

  return (
    <section className="onboard-channel" aria-labelledby="onboard-channel-title">
      <section className="settings-section settings-section--group">
        <button type="button" className="settings-disclosure" onClick={() => setTelegramOpen((value) => !value)} aria-expanded={telegramOpen}>
          <span><strong>连接 Telegram（可选）</strong><small>{channels.telegramConfigured ? "已连接；可以继续使用现有凭据" : "需要 Bot token；也可以之后在配置中心连接"}</small></span>
          <ChevronDown className={telegramOpen ? "is-rotated" : ""} size={18} aria-hidden="true" />
        </button>
        {telegramOpen ? <div className="settings-stack settings-stack--inline">
            <button type="button" className="settings-disclosure" onClick={() => setGuideOpen((value) => !value)} aria-expanded={guideOpen}>
              <span><strong>如何创建 Telegram bot</strong></span>
              <ChevronDown className={guideOpen ? "is-rotated" : ""} size={18} aria-hidden="true" />
            </button>
            {guideOpen ? <ol className="settings-guide">
              <li>打开 Telegram，搜索 <strong>@BotFather</strong></li>
              <li>发送 <code>/newbot</code>，按提示给 bot 起名</li>
              <li>BotFather 回复一串 token，格式：<code>123456789:AAFxxx…</code></li>
            </ol> : null}
            <label className="settings-field"><span>Bot token{channels.telegramConfigured ? "（留空保留现有 token）" : ""}</span><span className="settings-secret-control"><input required={!channels.telegramConfigured} type={showToken ? "text" : "password"} autoComplete="off" value={telegramToken} onChange={(event) => setTelegramToken(event.target.value)} placeholder={channels.telegramConfigured ? "已保存" : "123456789:AAFxxx…"} aria-label="Bot token" /><button type="button" onClick={() => setShowToken((value) => !value)} aria-label={showToken ? "隐藏 Bot token" : "显示 Bot token"}>{showToken ? <EyeOff size={18} aria-hidden="true" /> : <Eye size={18} aria-hidden="true" />}</button></span></label>
            <label className="settings-field"><span>你的 Telegram 用户名</span><input value={telegramUsername} onChange={(event) => setTelegramUsername(event.target.value)} placeholder="不带 @，例如 your_name" /></label>
          </div> : null}
      </section>

      <section className="settings-section settings-section--group">
        <button type="button" className="settings-disclosure" onClick={() => setQqbotOpen((value) => !value)} aria-expanded={qqbotOpen}>
          <span><strong>连接 QQBot（可选）</strong><small>{channels.qqbotConfigured ? "已连接；可以继续使用现有凭据" : "需要腾讯开放平台 AppID 与 AppSecret"}</small></span>
          <ChevronDown className={qqbotOpen ? "is-rotated" : ""} size={18} aria-hidden="true" />
        </button>
        {qqbotOpen ? <div className="settings-stack settings-stack--inline">
          <label className="settings-field"><span>AppID{channels.qqbotConfigured ? "（留空保留现有连接）" : ""}</span><input required={!channels.qqbotConfigured} value={qqbotAppId} onChange={(event) => setQqbotAppId(event.target.value)} placeholder={channels.qqbotConfigured ? "已保存" : "机器人 AppID"} /></label>
          <label className="settings-field"><span>AppSecret{channels.qqbotConfigured ? "（留空保留现有连接）" : ""}</span><span className="settings-secret-control"><input required={!channels.qqbotConfigured} type={showQqbotSecret ? "text" : "password"} value={qqbotSecret} onChange={(event) => setQqbotSecret(event.target.value)} autoComplete="off" placeholder={channels.qqbotConfigured ? "已保存" : "机器人 AppSecret"} /><button type="button" onClick={() => setShowQqbotSecret((value) => !value)} aria-label={showQqbotSecret ? "隐藏 AppSecret" : "显示 AppSecret"}>{showQqbotSecret ? <EyeOff size={18} aria-hidden="true" /> : <Eye size={18} aria-hidden="true" />}</button></span></label>
          <p className="settings-note">如果要接收主动推送，请向 Bot 发送一条私聊消息，并在下面填写对应的 C2C user_openid。</p>
        </div> : null}
      </section>

      <section className="settings-section settings-section--group settings-toggle-section">
        <header><div><h2>连接 Android 手机（可选）</h2><p>完成设置并启动后，用手机扫描二维码，再在两端核对 6 位确认码。</p></div><label className="settings-toggle"><input type="checkbox" checked={mobileEnabled} onChange={(event) => setMobileEnabled(event.target.checked)} /><span className="settings-toggle-track" aria-hidden="true" /><span className="sr-only">启动移动网关并在完成后配对手机</span></label></header>
        {mobileEnabled ? <div className="settings-stack settings-stack--inline"><label className="settings-field"><span>公网 WSS 地址（可选）</span><input type="url" value={mobilePublicUrl} onChange={(event) => setMobilePublicUrl(event.target.value)} placeholder="wss://mobile.example.com/ws" /><small>填写后手机和电脑无需处于同一网络；留空则使用局域网入口。</small></label><p className="settings-note settings-note--with-icon"><ShieldCheck size={16} aria-hidden="true" />移动网关会和其他设置一起启动。下一页可以直接开始配对。</p></div> : null}
      </section>

      <section className="settings-section settings-section--group settings-toggle-section">
        <header><div><h2>主动推送（可选）</h2><p>开启后会按选定方式检查订阅内容，并发送到指定频道。</p></div><label className="settings-toggle"><input type="checkbox" checked={enabled} onChange={(event) => setEnabled(event.target.checked)} /><span className="settings-toggle-track" aria-hidden="true" /><span className="sr-only">启用主动推送</span></label></header>
        {enabled ? <>
          <fieldset className="settings-choice-group settings-stack--inline">
            <legend>推送方式</legend>
            {PUSH_ENGINES.map((item) => <label key={item.key} className={engine === item.key ? "is-selected" : ""}><input type="radio" name="onboarding-proactive-engine" checked={engine === item.key} onChange={() => setEngine(item.key)} /><span><strong>{item.name}</strong><small>{item.detail}</small></span></label>)}
          </fieldset>
          {engine === "default" ? <fieldset className="settings-choice-group settings-stack--inline">
            <legend>推送节奏</legend>
            {PROACTIVE_MODES.map((item) => <label key={item.key} className={mode === item.key ? "is-selected" : ""}><input type="radio" name="onboarding-proactive-mode" checked={mode === item.key} onChange={() => setMode(item.key)} /><span><strong>{item.name}</strong><small>{item.detail}</small></span></label>)}
          </fieldset> : null}

          <div className="settings-stack settings-stack--inline">
            <label className="settings-field"><span>推送目标</span><select value={pushTarget} onChange={(event) => changePushTarget(event.target.value as PushTarget)}><option value="web">网页会话（默认）</option><option value="telegram">Telegram{channels.telegramConfigured || telegramToken.trim() ? " · 已连接" : " · 需要先连接"}</option><option value="qqbot">QQBot{channels.qqbotConfigured || (qqbotAppId.trim() && qqbotSecret.trim()) ? " · 已连接" : " · 需要先连接"}</option><option value="mobile">手机{mobileEnabled ? " · 已开启" : " · 需要先开启"}</option></select></label>
            {!selectedTargetReady ? <p className="settings-inline-error" role="alert">先在上方完成这个频道的连接，才能把它设为推送目标。</p> : null}
            {selectedTargetReady && (pushTarget === "telegram" || pushTarget === "qqbot") ? <><label className="settings-field"><span>{pushTarget === "telegram" ? "Telegram chat ID" : "QQBot C2C 目标"}</span><input required={enabled} value={targetId} onChange={(event) => setTargetId(event.target.value)} placeholder={pushTarget === "telegram" ? "例如：123456789" : "c2c:USER_OPENID"} /></label><div className="settings-target-discovery"><p>点一下后，在 45 秒内向 Bot 发一条私聊消息。</p><button type="button" className="settings-quiet-button" onClick={() => void discoverOnboardingTarget()} disabled={discoveringTarget}>{discoveringTarget ? <LoaderCircle className="is-spinning" size={17} /> : null}{discoveringTarget ? "等待消息…" : "从消息识别目标"}</button></div></> : null}
            {selectedTargetReady && pushTarget === "mobile" ? <p className="settings-note">主动消息会进入手机端默认会话，并同步到已配对设备。</p> : null}
          </div>

        </> : null}
      </section>
      <div className="onboard-footer onboard-footer--actions">
        <button type="button" className="settings-quiet-button" onClick={() => void handleSkip()} disabled={saving}>跳过，稍后设置</button>
        <button type="button" className="settings-primary-button" onClick={() => void handleSave()} disabled={saving}>{saving ? <LoaderCircle className="is-spinning" size={17} /> : null}{saving ? "保存中" : "保存选择并继续"}</button>
      </div>
    </section>
  );
}
