import { Eye, EyeOff, LoaderCircle, RefreshCw, ShieldCheck, X } from "lucide-react";
import { FormEvent, type RefObject, useRef } from "react";
import {
  Dialog,
  DialogClose,
  DialogContent,
  DialogDescription,
  DialogTitle,
} from "./components/ui/dialog";
import type { ConnectionGroup, ConnectionTemplate, SettingsState } from "./settings-data";
import { useSettingsConnection } from "./use-settings-connection";

interface SettingsConnectionDialogProps {
  template: ConnectionTemplate;
  existing?: ConnectionGroup;
  settings: SettingsState;
  returnFocusRef: RefObject<HTMLElement | null>;
  onOpenChange: (open: boolean) => void;
  onSaved: (firstConnection: boolean, sourceName: string) => Promise<void>;
  onLoginCompleted: () => Promise<void>;
}

/** Render one modal connection workflow while its controller owns all transient state. */
export function SettingsConnectionDialog({
  template,
  existing,
  settings,
  returnFocusRef,
  onOpenChange,
  onSaved,
  onLoginCompleted,
}: SettingsConnectionDialogProps) {
  const connection = useSettingsConnection({ template, existing, settings, onSaved, onLoginCompleted });
  const { draft, setDraft, models, discovering, saving, showKey, setShowKey, error, codexLogin } = connection;
  const nameInputRef = useRef<HTMLInputElement>(null);
  const title = connectionDialogTitle(draft.kind, draft.provider, draft.sourceName, Boolean(existing));
  const description = connectionDialogDescription(draft.kind, draft.provider);

  function handleSubmit(event: FormEvent) {
    event.preventDefault();
    void connection.save();
  }

  return (
    <Dialog open onOpenChange={onOpenChange}>
      <DialogContent
        className="settings-dialog"
        overlayClassName="settings-scrim"
        showCloseButton={false}
        onOpenAutoFocus={(event) => {
          event.preventDefault();
          nameInputRef.current?.focus();
        }}
        onCloseAutoFocus={(event) => {
          event.preventDefault();
          returnFocusRef.current?.focus();
        }}
      >
        <header>
          <div>
            <DialogTitle>{title}</DialogTitle>
            <DialogDescription>{description}</DialogDescription>
          </div>
          <DialogClose asChild>
            <button type="button" className="settings-icon-button" aria-label="关闭"><X aria-hidden="true" size={20} /></button>
          </DialogClose>
        </header>
        <form onSubmit={handleSubmit}>
          <div className="settings-dialog-body">
            <div className="settings-form-grid">
              <label className="is-wide">
                <span>连接名称</span>
                <input
                  ref={nameInputRef}
                  aria-label="连接名称"
                  required
                  value={draft.sourceName}
                  onChange={(event) => setDraft({ ...draft, sourceName: event.target.value })}
                  placeholder={draft.provider === "deepseek" ? "例如：DeepSeek 官方" : "例如：公司网关"}
                />
              </label>
              {draft.kind === "codex" ? (
                <div className="settings-login-card is-wide">
                  <ShieldCheck aria-hidden="true" size={20} />
                  <span>
                    <strong>{settings.codexConfigured || codexLogin?.status === "completed" ? "Codex 已登录" : "使用 ChatGPT 订阅登录"}</strong>
                    <small>授权凭据保存在当前 workspace，不会显示在页面中。</small>
                  </span>
                  <button type="button" onClick={() => void connection.beginLogin()}>{settings.codexConfigured ? "重新登录" : "开始登录"}</button>
                </div>
              ) : <>
                {draft.kind === "api" ? <label>
                  <span>Provider ID</span>
                  <input aria-label="Provider ID" required value={draft.provider} onChange={(event) => setDraft({ ...draft, provider: event.target.value })} placeholder="例如：openai" />
                </label> : null}
                <label className={draft.kind === "opencode-go" ? "is-wide" : ""}>
                  <span>Base URL</span>
                  <input aria-label="Base URL" required type="url" value={draft.baseUrl} onChange={(event) => setDraft({ ...draft, baseUrl: event.target.value })} placeholder="https://api.example.com/v1" />
                </label>
                <label className="settings-secret is-wide">
                  <span>API Key{draft.kind === "opencode-go" && settings.localOpenCodeConfigured ? "（可留空使用本机登录）" : ""}</span>
                  <input
                    aria-label="API Key"
                    required={draft.kind === "api" && !existing}
                    type={showKey ? "text" : "password"}
                    value={draft.apiKey}
                    onChange={(event) => setDraft({ ...draft, apiKey: event.target.value })}
                    autoComplete="off"
                    placeholder="sk-…"
                  />
                  <button type="button" onClick={() => setShowKey((value) => !value)} aria-label={showKey ? "隐藏 API Key" : "显示 API Key"}>
                    {showKey ? <EyeOff aria-hidden="true" size={18} /> : <Eye aria-hidden="true" size={18} />}
                  </button>
                </label>
              </>}
            </div>

            {codexLogin?.status === "waiting" && draft.kind === "codex" ? (
              <div className="settings-device-login" role="status">
                <span>验证码</span><strong>{codexLogin.userCode}</strong>
                <a href={codexLogin.verificationUri} target="_blank" rel="noreferrer">打开登录页面</a>
              </div>
            ) : null}

            {draft.kind === "api" ? (
              <section className="settings-model-discovery">
                <header>
                  <div><h3>可用模型</h3><p>先自动检测；服务不提供目录时再手动填写。</p></div>
                  <button type="button" className="settings-quiet-button" onClick={() => void connection.discover()} disabled={discovering}>
                    {discovering ? <LoaderCircle aria-hidden="true" className="is-spinning" size={16} /> : <RefreshCw aria-hidden="true" size={16} />}
                    {discovering ? "检测中" : "检测模型"}
                  </button>
                </header>
                <div className="settings-form-grid">
                  <label className="is-wide">
                    <span>模型名称</span>
                    {models.length ? (
                      <select aria-label="模型名称" required value={draft.model} onChange={(event) => {
                        const model = models.find((item) => item.id === event.target.value);
                        setDraft({ ...draft, model: event.target.value, reasoningEffort: model?.defaultReasoningEffort || draft.reasoningEffort });
                      }}>
                        <option value="">选择模型</option>
                        {models.map((model) => <option value={model.id} key={model.id}>{model.id}</option>)}
                      </select>
                    ) : <input aria-label="模型名称" required value={draft.model} onChange={(event) => setDraft({ ...draft, model: event.target.value })} placeholder={draft.provider === "deepseek" ? "例如：deepseek-chat" : "例如：your-model-name"} />}
                  </label>
                  <ReasoningEffortField models={models} modelId={draft.model} value={draft.reasoningEffort} onChange={(reasoningEffort) => setDraft({ ...draft, reasoningEffort })} />
                </div>
                <p>上下文窗口、多模态、推理能力和用量字段会自动归一化。</p>
              </section>
            ) : <section className="settings-model-discovery settings-model-discovery--automatic"><header><div><h3>模型自动同步</h3><p>保存后读取账号当前可用的全部模型，无需手动选择。</p></div></header></section>}

            {error ? <p className="settings-inline-error" role="alert">{error}</p> : null}
          </div>
          <footer>
            <span><ShieldCheck aria-hidden="true" size={15} />凭据保存后不会显示在页面中</span>
            <button type="submit" className="settings-primary-button" disabled={saving}>
              {saving ? <LoaderCircle aria-hidden="true" className="is-spinning" size={17} /> : null}
              {saving ? "保存中" : draft.kind === "api" ? "保存连接" : "保存并同步模型"}
            </button>
          </footer>
        </form>
      </DialogContent>
    </Dialog>
  );
}

function ReasoningEffortField({ models, modelId, value, onChange }: {
  models: ReturnType<typeof useSettingsConnection>["models"];
  modelId: string;
  value: string;
  onChange: (value: string) => void;
}) {
  const efforts = models.find((item) => item.id === modelId)?.supportedReasoningEfforts || [];
  if (efforts.length === 0) return null;
  return <label className="is-wide">
    <span>默认思考强度</span>
    <select aria-label="默认思考强度" value={value} onChange={(event) => onChange(event.target.value)}>
      {efforts.map((effort) => <option value={effort} key={effort}>{effort}</option>)}
    </select>
  </label>;
}

function connectionDialogTitle(kind: string, provider: string, sourceName: string, existing: boolean) {
  if (existing) return `编辑 ${sourceName}`;
  if (kind === "codex") return "连接 Codex";
  if (kind === "opencode-go") return "连接 OpenCode Go";
  return provider === "deepseek" ? "连接 DeepSeek" : "连接自定义 API";
}

function connectionDialogDescription(kind: string, provider: string) {
  if (kind === "codex") return "授权 ChatGPT 订阅账号，保存后自动同步可用模型。";
  if (kind === "opencode-go") return "使用本机 OpenCode 登录或单独的 API Key，模型会自动同步。";
  if (provider === "deepseek") return "填写 API Key 并选择一个可用模型，其余能力自动识别。";
  return "填写服务地址与凭据；支持模型目录时会自动检测。";
}
