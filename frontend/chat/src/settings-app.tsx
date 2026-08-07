import {
  Check,
  ChevronRight,
  Eye,
  EyeOff,
  KeyRound,
  LoaderCircle,
  Palette,
  RefreshCw,
  Search,
  ShieldCheck,
  X,
} from "lucide-react";
import { FormEvent, useEffect, useMemo, useRef, useState } from "react";
import { createPortal } from "react-dom";
import codexIcon from "./assets/provider-icons/codex.svg";
import deepseekIcon from "./assets/provider-icons/deepseek.svg";
import opencodeIcon from "./assets/provider-icons/opencode.svg";
import { cycleTheme, useTheme } from "../../theme/src/theme-runtime";
import { MemorySettings, type MemorySettingsState } from "./memory-settings";
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
  memory: MemorySettingsState;
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
  { kind: "api" as const, provider: "deepseek", name: "DeepSeek", detail: "官方 API", baseUrl: "https://api.deepseek.com/v1", icon: deepseekIcon },
  { kind: "api" as const, provider: "", name: "自定义 API", detail: "连接任意兼容服务", baseUrl: "", icon: "" },
];

const ROLE_LABELS: Record<ModelRole, { title: string; detail: string }> = {
  default: { title: "默认模型", detail: "普通模型调用与系统默认" },
  agent: { title: "Agent 模型", detail: "被动对话与计划任务 ReAct" },
  fast: { title: "轻量模型", detail: "压缩、标签与后台提取" },
  vision: { title: "视觉模型", detail: "包含图片的输入" },
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

export function SettingsApp() {
  const theme = useTheme();
  const [state, setState] = useState<SettingsState | null>(null);
  const [query, setQuery] = useState("");
  const [draft, setDraft] = useState<ConnectionDraft | null>(null);
  const [models, setModels] = useState<ModelOption[]>([]);
  const [discovering, setDiscovering] = useState(false);
  const [saving, setSaving] = useState(false);
  const [showKey, setShowKey] = useState(false);
  const [toast, setToast] = useState("");
  const [error, setError] = useState("");
  const [codexLogin, setCodexLogin] = useState<CodexLoginState | null>(null);
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
  const hasConnections = Boolean(state?.runtimes.length);

  function closeDialog() {
    setDraft(null);
    setModels([]);
    setError("");
    window.setTimeout(() => dialogReturnFocusRef.current?.focus(), 0);
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
      const firstConnection = !state?.runtimes.length;
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
          defer_restart: firstConnection,
        }),
      });
      await refreshState();
      setToast(firstConnection ? `${draft.sourceName} 已保存，接下来配置记忆` : `${draft.sourceName} 已保存，密钥不会显示在页面中`);
      closeDialog();
      if (isEmbeddedShell && !firstConnection) {
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

  if (!state && !error) return <div className="settings-loading"><LoaderCircle className="is-spinning" />正在读取模型连接</div>;
  if (state?.mode === "needs_repair") return <main className="settings-page"><section className="settings-repair"><ShieldCheck /><h1>配置需要手动处理</h1><p>{state.error}</p></section></main>;
  if (state?.runtimes.length && !state.memory.configured) return <main className="settings-page">
    <div className="settings-shell settings-shell--onboarding">
      <MemorySettings
        memory={state.memory}
        modelRevision={state.modelRevision}
        onboarding
        onRefresh={async () => (await refreshState()).memory}
        onError={setError}
        onNotice={setToast}
        onComplete={(message) => {
          setToast(message);
          if (isEmbeddedShell) window.parent.postMessage({ type: "akashic.settings.applied" }, window.location.origin);
          window.setTimeout(() => {
            if (isEmbeddedShell) window.parent.location.href = "/";
            else window.location.href = "/";
          }, 350);
        }}
      />
      {error && <p className="settings-inline-error" role="alert">{error}</p>}
    </div>
    <div className="settings-toast-region" aria-live="polite" aria-atomic="true">{toast && <div className="settings-toast" role="status"><Check size={18} /><span><strong>{toast}</strong></span><button type="button" onClick={() => setToast("")} aria-label="关闭通知"><X size={16} /></button></div>}</div>
  </main>;

  return (
    <main className="settings-page">
      <div className={`settings-shell ${hasConnections ? "" : "settings-shell--first-run"}`}>
        <header className="settings-header">
          <div><h1>{hasConnections ? "模型连接" : "连接你的第一个模型"}</h1><p>{hasConnections ? "每套账号或 API Key 都是独立连接；保存后自动识别模型能力。" : "选择登录方式或 API 服务。连接成功后，再决定是否启用记忆。"}</p></div>
          <div className="settings-header-actions">
            {!isEmbeddedShell && <button type="button" className="settings-quiet-button" onClick={cycleTheme}><Palette size={17} />{theme.label}</button>}
          </div>
        </header>

        {hasConnections && <label className="settings-search"><Search size={18} aria-hidden="true" /><span className="sr-only">搜索模型连接</span><input value={query} onChange={(event) => setQuery(event.target.value)} placeholder="搜索连接或模型" /></label>}

        {hasConnections && <section className="settings-section">
          <header><div><h2>已连接</h2><p>同一供应商可以添加多个账号，模型选择时按连接名称区分。</p></div><span>{connections.length} 个</span></header>
          <div className="settings-gallery">
            {connections.map((group) => <button type="button" className="settings-connection-card" key={group.sourceId} onClick={() => setDraft(createDraft(PROVIDER_TEMPLATES[0], group))}>
              <ConnectionMark provider={group.provider} name={group.sourceName} />
              <span className="settings-card-copy"><strong>{group.sourceName}</strong><small>{group.provider} · {group.runtimes.map((item) => item.model).join("、")}</small></span>
              <span className="settings-card-meta"><i><span />已连接</i><small>{group.runtimes.length} 个模型</small></span>
              <ChevronRight size={18} aria-hidden="true" />
            </button>)}
          </div>
        </section>}

        <section className={`settings-section settings-section--templates ${hasConnections ? "" : "is-first-run"}`}>
          <header><div><h2>{hasConnections ? "添加其他连接" : "选择连接方式"}</h2><p>{hasConnections ? "可以继续添加另一个账号或服务。" : "Codex 与 OpenCode 登录后自动同步模型；API 服务会先检测模型目录。"}</p></div></header>
          <div className="settings-gallery">
            {PROVIDER_TEMPLATES.map((template) => <button type="button" className="settings-connection-card" key={template.provider} onClick={() => setDraft(createDraft(template))}>
              <ConnectionMark provider={template.provider} name={template.name} /><span className="settings-card-copy"><strong>{template.name}</strong><small>{template.detail}</small></span><ChevronRight className="settings-template-action" size={18} aria-hidden="true" />
            </button>)}
          </div>
        </section>

        {state?.runtimes.length ? <section className="settings-section settings-roles">
          <header><div><h2>系统模型</h2><p>修改后不重启进程；正在运行的完整 turn 保持旧快照，下一个执行读取最新绑定。</p></div></header>
          <div className="settings-role-grid">
            {(Object.keys(ROLE_LABELS) as ModelRole[]).map((role) => <label key={role}><span><strong>{ROLE_LABELS[role].title}</strong><small>{ROLE_LABELS[role].detail}</small></span><select value={state.roleBindings[role]?.modelId || state.activeRuntime || ""} onChange={(event) => updateRole(role, event.target.value)}>{state.runtimes.map((runtime) => <option key={runtime.id} value={runtime.id}>{runtime.model}：{runtime.sourceName}</option>)}</select></label>)}
          </div>
        </section> : null}

        {state?.runtimes.length ? <MemorySettings
          memory={state.memory}
          modelRevision={state.modelRevision}
          onRefresh={async () => (await refreshState()).memory}
          onError={setError}
          onNotice={setToast}
          onComplete={async (message) => { setToast(message); await refreshState(); }}
        /> : null}
        {error && !draft && <p className="settings-inline-error" role="alert">{error}</p>}
      </div>

      {draft && createPortal(<div className="settings-scrim" onMouseDown={(event) => { if (event.target === event.currentTarget) closeDialog(); }}>
        <div ref={dialogRef} className="settings-dialog" role="dialog" aria-modal="true" aria-labelledby="settings-dialog-title">
          <header><div><h2 id="settings-dialog-title">{connections.some((item) => item.sourceId === draft.sourceId) ? `编辑 ${draft.sourceName}` : draft.kind === "codex" ? "连接 Codex" : draft.kind === "opencode-go" ? "连接 OpenCode Go" : draft.provider === "deepseek" ? "连接 DeepSeek" : "连接自定义 API"}</h2><p>{draft.kind === "codex" ? "授权 ChatGPT 订阅账号，保存后自动同步可用模型。" : draft.kind === "opencode-go" ? "使用本机 OpenCode 登录或单独的 API Key，模型会自动同步。" : draft.provider === "deepseek" ? "填写 API Key 并选择一个可用模型，其余能力自动识别。" : "填写服务地址与凭据；支持模型目录时会自动检测。"}</p></div><button type="button" className="settings-icon-button" onClick={closeDialog} aria-label="关闭"><X size={20} /></button></header>
          <form onSubmit={saveConnection}>
            <div className="settings-dialog-body">
              <div className="settings-form-grid">
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
