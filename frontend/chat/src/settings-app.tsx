import {
  Check,
  ChevronRight,
  KeyRound,
  LoaderCircle,
  Palette,
  Search,
  ShieldCheck,
  X,
} from "lucide-react";
import { useCallback, useMemo, useRef, useState } from "react";
import codexIcon from "./assets/provider-icons/codex.svg";
import deepseekIcon from "./assets/provider-icons/deepseek.svg";
import opencodeIcon from "./assets/provider-icons/opencode.svg";
import { cycleTheme, useTheme } from "../../theme/src/theme-runtime";
import { MemorySettings } from "./memory-settings";
import { SettingsConnectionDialog } from "./settings-connection-dialog";
import {
  groupConnections,
  type ConnectionGroup,
  type ConnectionTemplate,
  type ModelRole,
} from "./settings-data";
import { useSettingsController } from "./use-settings-controller";
import "./settings.css";

const isEmbeddedShell = new URLSearchParams(window.location.search).get("embedded") === "1";

interface ConnectionSelection {
  template: ConnectionTemplate & { icon: string };
  existing?: ConnectionGroup;
}

const PROVIDER_TEMPLATES: Array<ConnectionTemplate & { icon: string }> = [
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

function providerIcon(provider: string): string {
  return PROVIDER_TEMPLATES.find((item) => item.provider === provider)?.icon || "";
}

function ConnectionMark({ provider, name }: { provider: string; name: string }) {
  const icon = providerIcon(provider);
  return <span className="settings-connection-mark" aria-hidden="true">{icon ? <img src={icon} alt="" /> : provider ? name.slice(0, 1).toUpperCase() : <KeyRound size={20} />}</span>;
}

export function SettingsApp() {
  const theme = useTheme();
  const { state, error, setError, notice, setNotice, refresh, updateRole } = useSettingsController();
  const [query, setQuery] = useState("");
  const [selection, setSelection] = useState<ConnectionSelection | null>(null);
  const dialogReturnFocusRef = useRef<HTMLElement | null>(null);
  const connections = useMemo(() => groupConnections(state?.runtimes || [], query), [query, state?.runtimes]);
  const hasConnections = Boolean(state?.runtimes.length);

  const openConnection = useCallback((next: ConnectionSelection) => {
    dialogReturnFocusRef.current = document.activeElement instanceof HTMLElement ? document.activeElement : null;
    setSelection(next);
  }, []);

  const handleConnectionSaved = useCallback(async (firstConnection: boolean, sourceName: string) => {
    await refresh();
    setNotice(firstConnection ? `${sourceName} 已保存，接下来配置记忆` : `${sourceName} 已保存，密钥不会显示在页面中`);
    setSelection(null);
    if (isEmbeddedShell && !firstConnection) {
      window.parent.postMessage({ type: "akashic.settings.applied" }, window.location.origin);
    }
  }, [refresh, setNotice]);

  const handleLoginCompleted = useCallback(async () => {
    await refresh();
    setNotice("Codex 登录已完成，可以发现模型了");
  }, [refresh, setNotice]);

  if (!state && !error) return <div className="settings-loading"><LoaderCircle className="is-spinning" />正在读取模型连接</div>;
  if (state?.mode === "needs_repair") return <main className="settings-page"><section className="settings-repair"><ShieldCheck /><h1>配置需要手动处理</h1><p>{state.error}</p></section></main>;
  if (state?.runtimes.length && !state.memory.configured) return <main className="settings-page">
    <div className="settings-shell settings-shell--onboarding">
      <MemorySettings
        memory={state.memory}
        modelRevision={state.modelRevision}
        onboarding
        onRefresh={async () => (await refresh())?.memory ?? state.memory}
        onError={setError}
        onNotice={setNotice}
        onComplete={(message) => {
          setNotice(message);
          if (isEmbeddedShell) window.parent.postMessage({ type: "akashic.settings.applied" }, window.location.origin);
          window.setTimeout(() => {
            if (isEmbeddedShell) window.parent.location.href = "/";
            else window.location.href = "/";
          }, 350);
        }}
      />
      {error && <p className="settings-inline-error" role="alert">{error}</p>}
    </div>
    <SettingsNotice message={notice} onClose={() => setNotice("")} />
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
            {connections.map((group) => <button type="button" className="settings-connection-card" key={group.sourceId} onClick={() => openConnection({ template: PROVIDER_TEMPLATES[0], existing: group })}>
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
            {PROVIDER_TEMPLATES.map((template) => <button type="button" className="settings-connection-card" key={template.provider} onClick={() => openConnection({ template })}>
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
          onRefresh={async () => (await refresh())?.memory ?? state.memory}
          onError={setError}
          onNotice={setNotice}
          onComplete={async (message) => { setNotice(message); await refresh(); }}
        /> : null}
        {error && !selection && <p className="settings-inline-error" role="alert">{error}</p>}
      </div>

      {selection && state ? <SettingsConnectionDialog
        key={`${selection.template.provider}:${selection.existing?.sourceId ?? "new"}`}
        template={selection.template}
        existing={selection.existing}
        settings={state}
        returnFocusRef={dialogReturnFocusRef}
        onOpenChange={(open) => { if (!open) setSelection(null); }}
        onSaved={handleConnectionSaved}
        onLoginCompleted={handleLoginCompleted}
      /> : null}

      <SettingsNotice message={notice} onClose={() => setNotice("")} />
    </main>
  );
}

function SettingsNotice({ message, onClose }: { message: string; onClose: () => void }) {
  return <div className="settings-toast-region" aria-live="polite" aria-atomic="true">
    {message ? <div className="settings-toast" role="status">
      <Check aria-hidden="true" size={18} />
      <span><strong>{message}</strong></span>
      <button type="button" onClick={onClose} aria-label="关闭通知"><X aria-hidden="true" size={16} /></button>
    </div> : null}
  </div>;
}
