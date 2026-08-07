import { Brain, Check, Database, Eye, EyeOff, LoaderCircle, Plus, ShieldCheck, Sparkles, X } from "lucide-react";
import { FormEvent, useState } from "react";
import { createPortal } from "react-dom";

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
  onRefresh: () => Promise<MemorySettingsState>;
  onNotice: (message: string) => void;
  onComplete: (message: string) => void;
  onError: (message: string) => void;
}

interface EmbeddingDraft {
  sourceName: string;
  baseUrl: string;
  apiKey: string;
  model: string;
}

async function requestJson<T>(url: string, init?: RequestInit): Promise<T> {
  const response = await fetch(url, {
    ...init,
    headers: { "Content-Type": "application/json", "X-Akasic-CSRF": "1", ...init?.headers },
  });
  const body = await response.json() as { detail?: string } & T;
  if (!response.ok) throw new Error(body.detail || `请求失败 (${response.status})`);
  return body;
}

export function MemorySettings({ memory, modelRevision, onboarding = false, onRefresh, onNotice, onComplete, onError }: MemorySettingsProps) {
  const initialMode = memory.enabled ? memory.engine : "off";
  const [mode, setMode] = useState<"akasha" | "default" | "off">(initialMode);
  const [modelId, setModelId] = useState(memory.embeddingModelId);
  const [saving, setSaving] = useState(false);
  const [dialogOpen, setDialogOpen] = useState(false);
  const [showKey, setShowKey] = useState(false);
  const [draft, setDraft] = useState<EmbeddingDraft>({ sourceName: "向量服务", baseUrl: "", apiKey: "", model: "" });

  async function saveMemory() {
    setSaving(true);
    onError("");
    try {
      await requestJson("/api/settings/memory", {
        method: "POST",
        body: JSON.stringify({
          enabled: mode !== "off",
          engine: mode === "default" ? "default" : "akasha",
          embedding_model_id: mode === "off" ? "" : modelId,
          expected_revision: memory.revision,
        }),
      });
      onComplete(mode === "off" ? "已关闭语义记忆" : `${mode === "akasha" ? "Akasha" : "经典记忆"} 已启用`);
    } catch (reason) {
      onError(reason instanceof Error ? reason.message : String(reason));
    } finally {
      setSaving(false);
    }
  }

  async function saveEmbedding(event: FormEvent) {
    event.preventDefault();
    setSaving(true);
    onError("");
    try {
      const result = await requestJson<{ model: EmbeddingModelSummary }>("/api/settings/embedding-models", {
        method: "POST",
        body: JSON.stringify({
          source_name: draft.sourceName,
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
      onNotice(`${result.model.model} 已验证，识别为 ${result.model.dimensions} 维`);
    } catch (reason) {
      onError(reason instanceof Error ? reason.message : String(reason));
    } finally {
      setSaving(false);
    }
  }

  const needsModel = mode !== "off";
  return <>
    <section className={`settings-memory ${onboarding ? "is-onboarding" : ""}`}>
      <header>
        <div>
          <span className="settings-overline">{onboarding ? "第 2 步 · Memory" : "记忆 · Embedding"}</span>
          <h2>{onboarding ? "让记忆真正可用" : "语义记忆"}</h2>
          <p>记忆引擎与聊天模型独立。向量维度会通过真实请求自动识别。</p>
        </div>
        {!onboarding && <Database size={24} aria-hidden="true" />}
      </header>

      <fieldset className="settings-memory-engines" disabled={memory.changeLocked}>
        <legend>记忆引擎</legend>
        <label className={mode === "akasha" ? "is-active" : ""}>
          <input type="radio" name="memory-engine" checked={mode === "akasha"} onChange={() => setMode("akasha")} />
          <span className="settings-memory-icon"><Sparkles size={19} /></span>
          <span><strong>Akasha <i>推荐</i></strong><small>语义检索、稀疏索引与长期记忆</small></span>
          <Check size={17} />
        </label>
        <label className={mode === "default" ? "is-active" : ""}>
          <input type="radio" name="memory-engine" checked={mode === "default"} onChange={() => setMode("default")} />
          <span className="settings-memory-icon"><Brain size={19} /></span>
          <span><strong>经典记忆</strong><small>保留原有记忆流水线</small></span>
          <Check size={17} />
        </label>
        <label className={mode === "off" ? "is-active" : ""}>
          <input type="radio" name="memory-engine" checked={mode === "off"} onChange={() => setMode("off")} />
          <span className="settings-memory-icon"><Database size={19} /></span>
          <span><strong>暂不启用</strong><small>仍可正常聊天，之后再配置</small></span>
          <Check size={17} />
        </label>
      </fieldset>

      {needsModel && <div className="settings-embedding-picker">
        <label>
          <span>向量模型</span>
          <select value={modelId} onChange={(event) => setModelId(event.target.value)} disabled={memory.changeLocked}>
            <option value="">选择已验证的向量模型</option>
            {memory.embeddingModels.map((model) => <option value={model.id} key={model.id}>{model.model}：{model.sourceName} · {model.dimensions} 维</option>)}
          </select>
        </label>
        <button type="button" className="settings-quiet-button" onClick={() => setDialogOpen(true)} disabled={memory.changeLocked}><Plus size={17} />添加向量模型</button>
      </div>}

      {memory.changeLocked && <p className="settings-memory-lock"><ShieldCheck size={16} />当前 workspace 已有对话与记忆数据。更换引擎或向量模型需要先执行可恢复的索引迁移。</p>}

      <footer>
        <span>{needsModel ? "保存前会验证向量服务；API Key 只存入当前 workspace。" : "关闭后不会创建新的语义记忆。"}</span>
        <button type="button" className="settings-primary-button" onClick={saveMemory} disabled={saving || memory.changeLocked || (needsModel && !modelId)}>
          {saving && <LoaderCircle className="is-spinning" size={17} />}{onboarding ? "完成设置并进入对话" : "保存记忆设置"}
        </button>
      </footer>
    </section>

    {dialogOpen && createPortal(<div className="settings-scrim" onMouseDown={(event) => { if (event.target === event.currentTarget) setDialogOpen(false); }}>
      <div className="settings-dialog settings-embedding-dialog" role="dialog" aria-modal="true" aria-labelledby="embedding-dialog-title">
        <header><div><span className="settings-overline">Embedding</span><h2 id="embedding-dialog-title">添加向量模型</h2><p>兼容 OpenAI `/embeddings` 协议；维度不需要手填。</p></div><button type="button" className="settings-icon-button" onClick={() => setDialogOpen(false)} aria-label="关闭"><X size={20} /></button></header>
        <form onSubmit={saveEmbedding}>
          <div className="settings-form-grid">
            <label className="is-wide"><span>连接名称</span><input required value={draft.sourceName} onChange={(event) => setDraft({ ...draft, sourceName: event.target.value })} placeholder="例如：DashScope 向量" /></label>
            <label className="is-wide"><span>Base URL</span><input required type="url" value={draft.baseUrl} onChange={(event) => setDraft({ ...draft, baseUrl: event.target.value })} placeholder="https://api.example.com/v1" /></label>
            <label className="settings-secret is-wide"><span>API Key</span><input required type={showKey ? "text" : "password"} value={draft.apiKey} onChange={(event) => setDraft({ ...draft, apiKey: event.target.value })} autoComplete="off" placeholder="sk-…" /><button type="button" onClick={() => setShowKey((value) => !value)} aria-label={showKey ? "隐藏 API Key" : "显示 API Key"}>{showKey ? <EyeOff size={18} /> : <Eye size={18} />}</button></label>
            <label className="is-wide"><span>模型名称</span><input required value={draft.model} onChange={(event) => setDraft({ ...draft, model: event.target.value })} placeholder="例如：text-embedding-v3" /></label>
          </div>
          <footer><span><ShieldCheck size={15} />将发送一条测试文本验证连接</span><button type="submit" className="settings-primary-button" disabled={saving}>{saving ? <LoaderCircle className="is-spinning" size={17} /> : null}{saving ? "验证中" : "验证并保存"}</button></footer>
        </form>
      </div>
    </div>, document.body)}
  </>;
}
