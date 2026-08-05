import { useEffect, useMemo, useState } from "react";
import type { ReactNode } from "react";
import { Check, ChevronRight, KeyRound, LoaderCircle, Palette, RefreshCw, Settings2 } from "lucide-react";
import { Input } from "@/components/ui/input";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import "./settings.css";
import { cycleTheme, useTheme } from "../../theme/src/theme-runtime";
import { MaterialButton, MaterialFilterChip } from "../../theme/src/material-react";

type ProviderKind = "api" | "opencode-go" | "codex";

interface RuntimeSummary {
  id: string;
  provider: string;
  model: string;
  baseUrl: string;
  contextWindow: number;
  maxOutputTokens: number;
  inputModalities: string[];
  reasoningEffort: string;
  credential: { configured: boolean; source: string };
}

interface SettingsState {
  mode: "needs_setup" | "needs_repair" | "ready";
  workspace: string;
  error?: string;
  activeRuntime: string | null;
  runtimes: RuntimeSummary[];
  codexConfigured: boolean;
  localOpenCodeConfigured: boolean;
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

const providers: Array<{ id: ProviderKind; name: string; note: string }> = [
  { id: "api", name: "API Key", note: "任意 OpenAI Chat Completions 端点" },
  { id: "opencode-go", name: "OpenCode Go", note: "使用订阅内可用的 Chat 模型" },
  { id: "codex", name: "Codex Auth", note: "复用本机 ChatGPT Codex 登录" },
];

async function requestJson<T>(url: string, init?: RequestInit): Promise<T> {
  const response = await fetch(url, {
    ...init,
    headers: {
      "Content-Type": "application/json",
      "X-Akasic-CSRF": "1",
      ...init?.headers,
    },
  });
  const text = await response.text();
  let payload: { detail?: string; message?: string };
  try {
    payload = text ? JSON.parse(text) as { detail?: string; message?: string } : {};
  } catch {
    if (!response.ok) throw new Error(`设置服务请求失败 (${response.status})`);
    throw new Error("设置服务返回了无效响应");
  }
  if (!response.ok) throw new Error(payload.detail || payload.message || `请求失败 (${response.status})`);
  return payload as T;
}

function runtimeKind(runtime: RuntimeSummary): ProviderKind {
  if (runtime.provider === "opencode-go") return "opencode-go";
  if (runtime.provider === "codex") return "codex";
  return "api";
}

export function SettingsApp() {
  const theme = useTheme();
  const [state, setState] = useState<SettingsState | null>(null);
  const [kind, setKind] = useState<ProviderKind>("api");
  const [provider, setProvider] = useState("openai");
  const [baseUrl, setBaseUrl] = useState("https://api.openai.com/v1");
  const [apiKey, setApiKey] = useState("");
  const [model, setModel] = useState("");
  const [contextWindow, setContextWindow] = useState("128000");
  const [maxOutputTokens, setMaxOutputTokens] = useState("0");
  const [reasoningEffort, setReasoningEffort] = useState("");
  const [models, setModels] = useState<ModelOption[]>([]);
  const [loadingModels, setLoadingModels] = useState(false);
  const [saving, setSaving] = useState(false);
  const [saved, setSaved] = useState(false);
  const [error, setError] = useState("");
  const [codexLogin, setCodexLogin] = useState<CodexLoginState | null>(null);

  useEffect(() => {
    requestJson<SettingsState>("/api/settings/state")
      .then((next) => {
        setState(next);
        const active = next.runtimes.find((item) => item.id === next.activeRuntime);
        if (active) selectRuntime(active);
      })
      .catch((reason: Error) => setError(reason.message));
  }, []);

  useEffect(() => {
    if (!codexLogin || codexLogin.status !== "waiting") return;
    const timer = window.setInterval(async () => {
      const next = await requestJson<CodexLoginState>(`/api/settings/codex-login/${codexLogin.loginId}`);
      setCodexLogin(next);
      if (next.status === "completed") {
        setState(await requestJson<SettingsState>("/api/settings/state"));
      }
    }, Math.max(3, codexLogin.interval) * 1000);
    return () => window.clearInterval(timer);
  }, [codexLogin]);

  const selectedRuntime = useMemo(
    () => state?.runtimes.find((item) => runtimeKind(item) === kind),
    [kind, state],
  );

  const selectedModel = models.find((item) => item.id === model);
  const effortOptions = selectedModel?.supportedReasoningEfforts ?? [];

  function selectRuntime(runtime: RuntimeSummary) {
    setKind(runtimeKind(runtime));
    setProvider(runtime.provider);
    setBaseUrl(runtime.baseUrl);
    setModel(runtime.model);
    setContextWindow(String(runtime.contextWindow || 128000));
    setMaxOutputTokens(String(runtime.maxOutputTokens ?? 0));
    setReasoningEffort(runtime.reasoningEffort || "");
    setApiKey("");
    setModels([]);
  }

  function chooseProvider(next: ProviderKind) {
    setKind(next);
    setApiKey("");
    setModels([]);
    setSaved(false);
    setError("");
    const existing = state?.runtimes.find((item) => runtimeKind(item) === next);
    if (existing) {
      selectRuntime(existing);
      return;
    }
    if (next === "opencode-go") {
      setProvider("opencode-go");
      setBaseUrl("https://opencode.ai/zen/go/v1");
      setModel("");
      setContextWindow("128000");
      setMaxOutputTokens("0");
      setReasoningEffort("");
    } else if (next === "codex") {
      setProvider("codex");
      setBaseUrl("");
      setModel("");
      setContextWindow("128000");
      setMaxOutputTokens("0");
      setReasoningEffort("");
    } else {
      setProvider("openai");
      setBaseUrl("https://api.openai.com/v1");
      setModel("");
      setMaxOutputTokens("0");
      setReasoningEffort("");
    }
  }

  async function loadModels() {
    setLoadingModels(true);
    setError("");
    try {
      const result = await requestJson<{ models: ModelOption[] }>("/api/settings/models", {
        method: "POST",
        body: JSON.stringify({
          provider,
          api_key: apiKey,
          base_url: baseUrl,
          use_local_opencode: kind === "opencode-go" && Boolean(state?.localOpenCodeConfigured),
        }),
      });
      setModels(result.models);
      if (!model && result.models[0]) applyModel(result.models[0]);
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : String(reason));
    } finally {
      setLoadingModels(false);
    }
  }

  function applyModel(option: ModelOption) {
    setModel(option.id);
    if (option.contextWindow) setContextWindow(String(option.contextWindow));
    if (!reasoningEffort && option.defaultReasoningEffort) {
      setReasoningEffort(option.defaultReasoningEffort);
    }
  }

  async function save() {
    setSaving(true);
    setSaved(false);
    setError("");
    try {
      await requestJson("/api/settings/apply", {
        method: "POST",
        body: JSON.stringify({
          provider,
          model,
          api_key: apiKey,
          credential_id: kind === "codex" ? "codex_default" : "",
          use_local_opencode: kind === "opencode-go" && !apiKey && Boolean(state?.localOpenCodeConfigured),
          base_url: baseUrl,
          context_window: Number(contextWindow),
          max_output_tokens: Number(maxOutputTokens),
          reasoning_effort: reasoningEffort,
          input_modalities: ["text"],
        }),
      });
      setApiKey("");
      setSaved(true);
      setState(await requestJson<SettingsState>("/api/settings/state"));
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : String(reason));
    } finally {
      setSaving(false);
    }
  }

  async function beginCodexLogin() {
    setError("");
    try {
      const login = await requestJson<CodexLoginState>("/api/settings/codex-login", {
        method: "POST",
        body: "{}",
      });
      setCodexLogin(login);
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : String(reason));
    }
  }

  if (!state && !error) {
    return <div className="settings-loading"><LoaderCircle className="animate-spin" /> 正在读取设置</div>;
  }

  if (state?.mode === "needs_repair") {
    return (
      <main className="settings-page">
        <section className="settings-repair">
          <Settings2 aria-hidden="true" />
          <h1>配置需要手动处理</h1>
          <p>{state.error || "当前 config.toml 不是受支持的新格式。"}</p>
          <p className="settings-muted">本版本不会自动迁移旧配置，也不会覆盖原文件。</p>
        </section>
      </main>
    );
  }

  const keyConfigured = selectedRuntime?.credential.configured || false;
  const authReady = kind === "codex"
    ? Boolean(state?.codexConfigured)
    : kind === "opencode-go"
      ? Boolean(apiKey || keyConfigured || state?.localOpenCodeConfigured)
      : Boolean(apiKey || keyConfigured);
  const canSave = Boolean(model && provider && contextWindow && maxOutputTokens && authReady);

  return (
    <main className="settings-page">
      <div className="settings-shell">
        <header className="settings-header">
          <div>
            <h1>{state?.mode === "needs_setup" ? "连接你的模型" : "模型与认证"}</h1>
            <p>选择一个 Provider，验证后安全切换。已保存的密钥不会显示在页面中。</p>
          </div>
          <div className="settings-header-actions">
            <button className="settings-theme-button" type="button" onClick={cycleTheme}>
              <Palette aria-hidden="true" /> {theme.label}
            </button>
            {state?.mode === "ready" && (
              <a className="settings-chat-link" href={`http://${window.location.hostname}:6322`}>
                打开聊天 <ChevronRight />
              </a>
            )}
          </div>
        </header>

        <div className="settings-layout">
          <nav className="provider-list" aria-label="Provider">
            {providers.map((item) => {
              const runtime = state?.runtimes.find((entry) => runtimeKind(entry) === item.id);
              const active = runtime?.id === state?.activeRuntime;
              return (
                <button
                  className={`provider-option ${kind === item.id ? "is-selected" : ""}`}
                  key={item.id}
                  onClick={() => chooseProvider(item.id)}
                  type="button"
                >
                  <span className="provider-title">{item.name}</span>
                  <span className="provider-note">{item.note}</span>
                  {active && <span className="provider-active"><Check /> 当前使用</span>}
                </button>
              );
            })}
          </nav>

          <section
            className="settings-panel"
            key={kind}
          >
            <div className="panel-heading">
              <div className="panel-icon"><KeyRound /></div>
              <div>
                <h2>{providers.find((item) => item.id === kind)?.name}</h2>
                <p>{keyConfigured ? "已保存认证；留空即可继续使用" : "完成认证并选择模型"}</p>
              </div>
            </div>

            {kind === "api" && (
              <div className="field-grid two-columns">
                <Field label="Provider ID"><Input value={provider} onChange={(event) => setProvider(event.target.value)} /></Field>
                <Field label="Base URL"><Input value={baseUrl} onChange={(event) => setBaseUrl(event.target.value)} /></Field>
              </div>
            )}

            {kind !== "codex" && (
              <Field label={kind === "opencode-go" ? "OpenCode Go Key" : "API Key"} hint={
                kind === "opencode-go" && state?.localOpenCodeConfigured
                  ? "已检测到本机 OpenCode Go 登录，可直接使用"
                  : keyConfigured ? "已配置；只在需要替换时输入" : undefined
              }>
                <Input
                  autoComplete="new-password"
                  type="password"
                  value={apiKey}
                  onChange={(event) => setApiKey(event.target.value)}
                  placeholder={authReady ? "••••••••（已配置）" : "输入密钥"}
                />
              </Field>
            )}

            {kind === "codex" && (
              <div className={`auth-status ${state?.codexConfigured ? "is-ready" : ""}`}>
                {state?.codexConfigured ? (
                  <span>本机 Codex 登录可用</span>
                ) : codexLogin ? (
                  <div className="codex-device-login">
                    <span>{codexLogin.status === "waiting" ? "在 OpenAI 页面输入代码" : codexLogin.error}</span>
                    <strong>{codexLogin.userCode}</strong>
                    <a href={codexLogin.verificationUri} target="_blank" rel="noreferrer">打开授权页面</a>
                  </div>
                ) : (
                  <>
                    <span>尚未找到 Codex 登录</span>
                    <MaterialButton variant="outlined" onClick={beginCodexLogin}>登录 Codex</MaterialButton>
                  </>
                )}
              </div>
            )}

            <div className="model-row">
              <Field label="模型">
                {models.length ? (
                  <Select value={model} onValueChange={(value) => applyModel(models.find((item) => item.id === value)!)}>
                    <SelectTrigger><SelectValue placeholder="选择模型" /></SelectTrigger>
                    <SelectContent>{models.map((item) => <SelectItem key={item.id} value={item.id}>{item.id}</SelectItem>)}</SelectContent>
                  </Select>
                ) : (
                  <Input value={model} onChange={(event) => setModel(event.target.value)} placeholder="模型 ID" />
                )}
              </Field>
              {kind !== "api" && (
                <MaterialButton variant="tonal" onClick={loadModels} disabled={loadingModels || !authReady} loading={loadingModels}>
                  {!loadingModels && <RefreshCw />}
                  探测模型与档位
                </MaterialButton>
              )}
            </div>

            <Field label="思考强度" hint={effortOptions.length ? "候选来自所选模型的实时目录，也可以输入自定义值" : "填写模型支持的推理强度，留空使用 Provider 默认"}>
              <div className="effort-control">
                <Input
                  aria-label="自定义思考强度"
                  value={reasoningEffort}
                  onChange={(event) => setReasoningEffort(event.target.value)}
                  placeholder="留空使用 Provider 默认；也可输入自定义值"
                />
                {effortOptions.length > 0 && (
                  <div className="effort-options" aria-label="探测到的思考强度" role="group">
                    <MaterialFilterChip
                      selected={!reasoningEffort}
                      onClick={() => setReasoningEffort("")}
                    >
                      Provider 默认
                    </MaterialFilterChip>
                    {effortOptions.map((item) => (
                      <MaterialFilterChip
                        selected={reasoningEffort === item}
                        key={item}
                        onClick={() => setReasoningEffort(item)}
                      >
                        {item}
                      </MaterialFilterChip>
                    ))}
                  </div>
                )}
              </div>
            </Field>

            <div className="field-grid two-columns">
              <Field label="上下文窗口"><Input inputMode="numeric" value={contextWindow} onChange={(event) => setContextWindow(event.target.value)} /></Field>
              <Field label="最大输出（0 由 Provider 决定）"><Input inputMode="numeric" value={maxOutputTokens} onChange={(event) => setMaxOutputTokens(event.target.value)} /></Field>
            </div>

            {error && <div className="settings-error" role="alert">{error}</div>}
            {saved && <div className="settings-success"><Check /> 已应用配置，Gateway 正在使用新的 Provider。</div>}

            <footer className="panel-footer">
              <span>保存前会发送一条最小真实请求验证模型。</span>
              <MaterialButton onClick={save} disabled={!canSave || saving} loading={saving}>
                {state?.mode === "needs_setup" ? "验证并启动" : "验证并切换"}
              </MaterialButton>
            </footer>
          </section>
        </div>
      </div>
    </main>
  );
}

function Field({ label, hint, children }: { label: string; hint?: string; children: ReactNode }) {
  return (
    <label className="settings-field">
      <span>{label}</span>
      {children}
      {hint && <small>{hint}</small>}
    </label>
  );
}
