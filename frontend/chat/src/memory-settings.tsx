import { Brain, Check, Database, Eye, EyeOff, LoaderCircle, Pencil, Plus, ShieldCheck, Sparkles, Trash2 } from "lucide-react";
import { FormEvent, useRef, useState } from "react";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogTitle,
} from "./components/ui/dialog";

export interface EmbeddingModelSummary {
  id: string;
  sourceId: string;
  sourceName: string;
  provider: string;
  baseUrl: string;
  model: string;
  dimensions: number;
  credential: { id: string; configured: boolean };
}

export interface MemorySettingsState {
  configured: boolean;
  enabled: boolean;
  engine: "akasha" | "default";
  embeddingModelId: string;
  embeddingModels: EmbeddingModelSummary[];
  changeLocked: boolean;
  revision: string;
}

interface MemorySettingsProps {
  memory: MemorySettingsState;
  modelRevision: number;
  onboarding?: boolean;
  deferRestart?: boolean;
  onRefresh: () => Promise<MemorySettingsState>;
  onNotice: (message: string) => void;
  onComplete: (message: string) => void | Promise<void>;
  onSkip?: () => void | Promise<void>;
  onError: (message: string) => void;
}

interface EmbeddingDraft {
  modelId: string;
  sourceId: string;
  credentialId: string;
  sourceName: string;
  baseUrl: string;
  apiKey: string;
  model: string;
}

const EMPTY_EMBEDDING_DRAFT: EmbeddingDraft = {
  modelId: "",
  sourceId: "",
  credentialId: "",
  sourceName: "向量服务",
  baseUrl: "",
  apiKey: "",
  model: "",
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
  const body = await response.json() as { detail?: string } & T;
  if (!response.ok) throw new Error(body.detail || `请求失败 (${response.status})`);
  return body;
}

export function MemorySettings({ memory, modelRevision, onboarding = false, deferRestart = false, onRefresh, onNotice, onComplete, onSkip, onError }: MemorySettingsProps) {
  const initialMode = memory.enabled ? memory.engine : "off";
  const [mode, setMode] = useState<"akasha" | "default" | "off">(initialMode);
  const [modelId, setModelId] = useState(memory.embeddingModelId);
  const [saving, setSaving] = useState(false);
  const [dialogOpen, setDialogOpen] = useState(false);
  const [showKey, setShowKey] = useState(false);
  const [validationError, setValidationError] = useState("");
  const [dialogError, setDialogError] = useState("");
  const [removeModel, setRemoveModel] = useState<EmbeddingModelSummary | null>(null);
  const modelSelectRef = useRef<HTMLSelectElement>(null);
  const addModelRef = useRef<HTMLButtonElement>(null);
  const [draft, setDraft] = useState<EmbeddingDraft>(EMPTY_EMBEDDING_DRAFT);

  function openAddModel() {
    setDraft(EMPTY_EMBEDDING_DRAFT);
    setDialogError("");
    setDialogOpen(true);
  }

  function openEditModel(model: EmbeddingModelSummary) {
    setDraft({
      modelId: model.id,
      sourceId: model.sourceId,
      credentialId: model.credential.id,
      sourceName: model.sourceName,
      baseUrl: model.baseUrl,
      apiKey: "",
      model: model.model,
    });
    setDialogError("");
    setDialogOpen(true);
  }

  async function saveMemory() {
    if (mode !== "off" && !modelId) {
      setValidationError("启用记忆前，请先添加并选择一个向量模型。");
      (memory.embeddingModels.length ? modelSelectRef.current : addModelRef.current)?.focus();
      return;
    }
    setSaving(true);
    setValidationError("");
    onError("");
    try {
      await requestJson("/api/settings/memory", {
        method: "POST",
        body: JSON.stringify({
          enabled: mode !== "off",
          engine: mode === "default" ? "default" : "akasha",
          embedding_model_id: mode === "off" ? "" : modelId,
          expected_revision: memory.revision,
          defer_restart: deferRestart,
        }),
      });
      await onComplete(mode === "off" ? "已关闭语义记忆" : `${mode === "akasha" ? "Akasha" : "经典记忆"} 已启用`);
    } catch (reason) {
      onError(reason instanceof Error ? reason.message : String(reason));
    } finally {
      setSaving(false);
    }
  }

  async function skipMemory() {
    if (!onSkip) return;
    setSaving(true);
    setValidationError("");
    onError("");
    try {
      await onSkip();
    } catch (reason) {
      onError(reason instanceof Error ? reason.message : String(reason));
    } finally {
      setSaving(false);
    }
  }

  async function saveEmbedding(event: FormEvent) {
    event.preventDefault();
    setSaving(true);
    setDialogError("");
    try {
      const result = await requestJson<{ model: EmbeddingModelSummary }>("/api/settings/embedding-models", {
        method: "POST",
        body: JSON.stringify({
          source_name: draft.sourceName,
          model_id: draft.modelId,
          source_id: draft.sourceId,
          credential_id: draft.credentialId,
          provider: "openai",
          base_url: draft.baseUrl,
          api_key: draft.apiKey,
          model: draft.model,
          expected_revision: modelRevision,
        }),
      });
      setModelId(result.model.id);
      await onRefresh();
      setDialogOpen(false);
      onNotice(`${result.model.model} 已验证并保存，识别为 ${result.model.dimensions} 维`);
    } catch (reason) {
      setDialogError(reason instanceof Error ? reason.message : String(reason));
    } finally {
      setSaving(false);
    }
  }

  async function confirmRemoveModel() {
    if (!removeModel) return;
    setSaving(true);
    setDialogError("");
    try {
      await requestJson(`/api/settings/embedding-models/${encodeURIComponent(removeModel.id)}/remove`, {
        method: "POST",
        body: JSON.stringify({ expected_revision: modelRevision }),
      });
      if (modelId === removeModel.id) setModelId("");
      setRemoveModel(null);
      await onRefresh();
      onNotice(`${removeModel.model} 已移除；恢复备份保留在当前 workspace`);
    } catch (reason) {
      setDialogError(reason instanceof Error ? reason.message : String(reason));
    } finally {
      setSaving(false);
    }
  }

  const needsModel = mode !== "off";
  return <>
    <section className={`settings-memory ${onboarding ? "is-onboarding" : ""}`}>
      <header>
        <div>
          <h2>{onboarding ? "选择是否启用记忆" : "语义记忆"}</h2>
          <p>{onboarding ? "选择一种方式会保存为正式配置；也可以跳过这一步，之后在配置中心设置。" : "记忆引擎与聊天模型独立；向量维度会通过真实请求自动识别。"}</p>
        </div>
        {!onboarding && <Database size={24} aria-hidden="true" />}
      </header>

      <fieldset className="settings-memory-engines" disabled={memory.changeLocked}>
        <legend>记忆方式</legend>
        <label className={mode === "akasha" ? "is-active" : ""}>
          <input type="radio" name="memory-engine" checked={mode === "akasha"} onChange={() => { setMode("akasha"); setValidationError(""); }} />
          <span className="settings-memory-icon" aria-hidden="true"><Sparkles size={19} /></span>
          <span><strong>Akasha</strong><small>推荐 · 语义检索与长期记忆</small></span>
          <Check size={17} aria-hidden="true" />
        </label>
        <label className={mode === "default" ? "is-active" : ""}>
          <input type="radio" name="memory-engine" checked={mode === "default"} onChange={() => { setMode("default"); setValidationError(""); }} />
          <span className="settings-memory-icon" aria-hidden="true"><Brain size={19} /></span>
          <span><strong>经典记忆</strong><small>保留原有记忆流水线</small></span>
          <Check size={17} aria-hidden="true" />
        </label>
        <label className={mode === "off" ? "is-active" : ""}>
          <input type="radio" name="memory-engine" checked={mode === "off"} onChange={() => { setMode("off"); setValidationError(""); }} />
          <span className="settings-memory-icon" aria-hidden="true"><Database size={19} /></span>
          <span><strong>关闭记忆</strong><small>明确保持关闭；仍可正常聊天</small></span>
          <Check size={17} aria-hidden="true" />
        </label>
      </fieldset>

      {needsModel && <section className="settings-embedding-step" aria-labelledby="embedding-step-title">
        <header><div><h3 id="embedding-step-title">向量模型</h3><p>用于检索记忆，不影响聊天模型。</p></div><span className={modelId ? "is-ready" : "is-required"}>{modelId ? "已就绪" : "必需"}</span></header>
        <div className="settings-embedding-picker">
        <label>
          <span>已验证的模型</span>
          <select ref={modelSelectRef} value={modelId} aria-invalid={Boolean(validationError)} aria-describedby={validationError ? "embedding-model-error" : undefined} onChange={(event) => { setModelId(event.target.value); setValidationError(""); }} disabled={memory.changeLocked}>
            <option value="">选择已验证的向量模型</option>
            {memory.embeddingModels.map((model) => <option value={model.id} key={model.id}>{model.model}：{model.sourceName} · {model.dimensions} 维</option>)}
          </select>
        </label>
        <button ref={addModelRef} type="button" className="settings-quiet-button" onClick={openAddModel} disabled={memory.changeLocked}><Plus size={17} />添加向量模型</button>
        </div>
        {memory.embeddingModels.length ? <ul className="settings-embedding-list" aria-label="向量模型连接">
          {memory.embeddingModels.map((model) => {
            const isActive = memory.enabled && memory.embeddingModelId === model.id;
            return <li key={model.id}>
              <div><strong>{model.model}</strong><small>{model.sourceName} · {model.dimensions} 维</small></div>
              <span className={isActive ? "settings-status is-ready" : "settings-status"}>{isActive ? "使用中" : "可用"}</span>
              <div className="settings-row-actions">
                <button type="button" className="settings-icon-button" onClick={() => openEditModel(model)} disabled={memory.changeLocked} aria-label={`编辑 ${model.model}`}><Pencil size={17} aria-hidden="true" /></button>
                <button type="button" className="settings-icon-button settings-icon-button--danger" onClick={() => { setDialogError(""); setRemoveModel(model); }} disabled={isActive || memory.changeLocked} aria-label={`移除 ${model.model}`} title={isActive ? "先关闭记忆或切换向量模型" : "移除向量模型"}><Trash2 size={17} aria-hidden="true" /></button>
              </div>
            </li>;
          })}
        </ul> : null}
        {validationError && <p id="embedding-model-error" className="settings-inline-error" role="alert">{validationError}</p>}
      </section>}

      {memory.changeLocked && <p className="settings-memory-lock"><ShieldCheck size={16} aria-hidden="true" />当前 workspace 已有对话与记忆数据。更换引擎或向量模型需要先执行可恢复的索引迁移。</p>}

      <footer>
        <span>{needsModel ? "向量服务会在添加时验证；API Key 只存入当前 workspace。" : "不会显示 Akasha 或向量模型相关界面，也不会创建新的语义记忆。"}</span>
        <div className="settings-action-row">
          {onboarding && onSkip ? <button type="button" className="settings-quiet-button" onClick={() => void skipMemory()} disabled={saving}>跳过，稍后设置</button> : null}
          <button type="button" className="settings-primary-button" onClick={saveMemory} disabled={saving || memory.changeLocked}>
            {saving && <LoaderCircle className="is-spinning" size={17} />}{onboarding ? "保存选择并继续" : "保存记忆设置"}
          </button>
        </div>
      </footer>
    </section>

    <Dialog open={dialogOpen} onOpenChange={setDialogOpen}>
      <DialogContent className="settings-dialog settings-embedding-dialog" overlayClassName="settings-scrim" aria-describedby="embedding-dialog-description">
        <header><div><DialogTitle id="embedding-dialog-title">{draft.modelId ? "编辑向量模型" : "添加向量模型"}</DialogTitle><DialogDescription id="embedding-dialog-description">兼容 OpenAI `/embeddings` 协议；保存前会重新验证连接和维度。</DialogDescription></div></header>
        <form onSubmit={saveEmbedding}>
          <div className="settings-dialog-body settings-embedding-body">
            <fieldset className="settings-embedding-group">
              <legend>服务连接</legend>
              <div className="settings-form-grid">
                <label className="is-wide"><span>连接名称</span><input required value={draft.sourceName} onChange={(event) => setDraft({ ...draft, sourceName: event.target.value })} placeholder="例如：DashScope 向量" /></label>
                <label className="is-wide"><span>Base URL</span><input required type="url" value={draft.baseUrl} onChange={(event) => setDraft({ ...draft, baseUrl: event.target.value })} placeholder="https://api.example.com/v1" /></label>
              </div>
            </fieldset>
            <fieldset className="settings-embedding-group">
              <legend>模型与凭据</legend>
              <div className="settings-form-grid">
                <label className="settings-secret is-wide"><span>API Key{draft.modelId ? "（留空保留现有凭据）" : ""}</span><input required={!draft.modelId} type={showKey ? "text" : "password"} value={draft.apiKey} onChange={(event) => setDraft({ ...draft, apiKey: event.target.value })} autoComplete="off" placeholder={draft.modelId ? "已保存" : "sk-…"} /><button type="button" onClick={() => setShowKey((value) => !value)} aria-label={showKey ? "隐藏 API Key" : "显示 API Key"}>{showKey ? <EyeOff size={18} /> : <Eye size={18} />}</button></label>
                <label className="is-wide"><span>模型名称</span><input required value={draft.model} onChange={(event) => setDraft({ ...draft, model: event.target.value })} placeholder="例如：text-embedding-v3" /></label>
              </div>
            </fieldset>
            {dialogError && <p className="settings-inline-error" role="alert">{dialogError}</p>}
          </div>
          <footer><span><ShieldCheck size={15} />会发送一条测试文本验证连接</span><button type="submit" className="settings-primary-button" disabled={saving}>{saving ? <LoaderCircle className="is-spinning" size={17} /> : null}{saving ? "验证中" : draft.modelId ? "验证并更新" : "验证并保存"}</button></footer>
        </form>
      </DialogContent>
    </Dialog>

    <Dialog open={Boolean(removeModel)} onOpenChange={(open) => { if (!open) setRemoveModel(null); }}>
      <DialogContent className="settings-dialog" overlayClassName="settings-scrim" showCloseButton={false} aria-describedby="remove-embedding-description">
        <header><div><DialogTitle>移除向量模型</DialogTitle><DialogDescription id="remove-embedding-description">将停用 {removeModel?.model}，但保留可恢复的注册库备份。正在使用的向量模型必须先在记忆设置中切换或关闭。</DialogDescription></div></header>
        {dialogError && <p className="settings-inline-error settings-dialog-inline" role="alert">{dialogError}</p>}
        <div className="settings-dialog-actions"><button type="button" className="settings-quiet-button" onClick={() => setRemoveModel(null)} disabled={saving}>保留模型</button><button type="button" className="settings-danger-button" onClick={() => void confirmRemoveModel()} disabled={saving}>{saving ? <LoaderCircle className="is-spinning" size={17} /> : <Trash2 size={17} />}{saving ? "正在移除" : "移除模型"}</button></div>
      </DialogContent>
    </Dialog>
  </>;
}
