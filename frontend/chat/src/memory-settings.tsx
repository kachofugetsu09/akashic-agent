import { Brain, Check, Database, Eye, EyeOff, LoaderCircle, Plus, ShieldCheck, Sparkles } from "lucide-react";
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

export function MemorySettings({ memory, modelRevision, onboarding = false, onRefresh, onNotice, onComplete, onError }: MemorySettingsProps) {
  const initialMode = memory.enabled ? memory.engine : "off";
  const [mode, setMode] = useState<"akasha" | "default" | "off">(initialMode);
  const [modelId, setModelId] = useState(memory.embeddingModelId);
  const [saving, setSaving] = useState(false);
  const [dialogOpen, setDialogOpen] = useState(false);
  const [showKey, setShowKey] = useState(false);
  const [validationError, setValidationError] = useState("");
  const [dialogError, setDialogError] = useState("");
  const modelSelectRef = useRef<HTMLSelectElement>(null);
  const addModelRef = useRef<HTMLButtonElement>(null);
  const [draft, setDraft] = useState<EmbeddingDraft>({ sourceName: "向量服务", baseUrl: "", apiKey: "", model: "" });

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
    setDialogError("");
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
          <p>{onboarding ? "可以先关闭，之后随时回来配置。启用记忆时需要一个向量模型。" : "记忆引擎与聊天模型独立；向量维度会通过真实请求自动识别。"}</p>
        </div>
        {!onboarding && <Database size={24} aria-hidden="true" />}
      </header>

      <fieldset className="settings-memory-engines" disabled={memory.changeLocked}>
        <legend>记忆方式</legend>
        <label className={mode === "akasha" ? "is-active" : ""}>
          <input type="radio" name="memory-engine" checked={mode === "akasha"} onChange={() => { setMode("akasha"); setValidationError(""); }} />
          <span className="settings-memory-icon"><Sparkles size={19} /></span>
          <span><strong>Akasha</strong><small>推荐 · 语义检索与长期记忆</small></span>
          <Check size={17} />
        </label>
        <label className={mode === "default" ? "is-active" : ""}>
          <input type="radio" name="memory-engine" checked={mode === "default"} onChange={() => { setMode("default"); setValidationError(""); }} />
          <span className="settings-memory-icon"><Brain size={19} /></span>
          <span><strong>经典记忆</strong><small>保留原有记忆流水线</small></span>
          <Check size={17} />
        </label>
        <label className={mode === "off" ? "is-active" : ""}>
          <input type="radio" name="memory-engine" checked={mode === "off"} onChange={() => { setMode("off"); setValidationError(""); }} />
          <span className="settings-memory-icon"><Database size={19} /></span>
          <span><strong>暂不启用</strong><small>仍可正常聊天，之后再配置</small></span>
          <Check size={17} />
        </label>
      </fieldset>

      {needsModel && <section className="settings-embedding-step" aria-labelledby="embedding-step-title">
        <header><div><h3 id="embedding-step-title">向量模型</h3><p>用于检索记忆，不影响聊天模型。</p></div><span className={modelId ? "is-ready" : "is-required"}>{modelId ? "已就绪" : "必需"}</span></header>
        <div className="settings-embedding-picker">
        <label>
          <span>已验证的模型</span>
          <select ref={modelSelectRef} value={modelId} aria-invalid={Boolean(validationError)} onChange={(event) => { setModelId(event.target.value); setValidationError(""); }} disabled={memory.changeLocked}>
            <option value="">选择已验证的向量模型</option>
            {memory.embeddingModels.map((model) => <option value={model.id} key={model.id}>{model.model}：{model.sourceName} · {model.dimensions} 维</option>)}
          </select>
        </label>
        <button ref={addModelRef} type="button" className="settings-quiet-button" onClick={() => { setDialogError(""); setDialogOpen(true); }} disabled={memory.changeLocked}><Plus size={17} />添加向量模型</button>
        </div>
        {validationError && <p className="settings-inline-error" role="alert">{validationError}</p>}
      </section>}

      {memory.changeLocked && <p className="settings-memory-lock"><ShieldCheck size={16} />当前 workspace 已有对话与记忆数据。更换引擎或向量模型需要先执行可恢复的索引迁移。</p>}

      <footer>
        <span>{needsModel ? "向量服务会在添加时验证；API Key 只存入当前 workspace。" : "不会显示 Akasha 或向量模型相关界面，也不会创建新的语义记忆。"}</span>
        <button type="button" className="settings-primary-button" onClick={saveMemory} disabled={saving || memory.changeLocked}>
          {saving && <LoaderCircle className="is-spinning" size={17} />}{onboarding ? "完成设置并进入对话" : "保存记忆设置"}
        </button>
      </footer>
    </section>

    <Dialog open={dialogOpen} onOpenChange={setDialogOpen}>
      <DialogContent className="settings-dialog settings-embedding-dialog" overlayClassName="settings-scrim" aria-describedby="embedding-dialog-description">
        <header><div><DialogTitle id="embedding-dialog-title">添加向量模型</DialogTitle><DialogDescription id="embedding-dialog-description">兼容 OpenAI `/embeddings` 协议；维度会自动识别。</DialogDescription></div></header>
        <form onSubmit={saveEmbedding}>
          <div className="settings-form-grid">
            <label className="is-wide"><span>连接名称</span><input required value={draft.sourceName} onChange={(event) => setDraft({ ...draft, sourceName: event.target.value })} placeholder="例如：DashScope 向量" /></label>
            <label className="is-wide"><span>Base URL</span><input required type="url" value={draft.baseUrl} onChange={(event) => setDraft({ ...draft, baseUrl: event.target.value })} placeholder="https://api.example.com/v1" /></label>
            <label className="settings-secret is-wide"><span>API Key</span><input required type={showKey ? "text" : "password"} value={draft.apiKey} onChange={(event) => setDraft({ ...draft, apiKey: event.target.value })} autoComplete="off" placeholder="sk-…" /><button type="button" onClick={() => setShowKey((value) => !value)} aria-label={showKey ? "隐藏 API Key" : "显示 API Key"}>{showKey ? <EyeOff size={18} /> : <Eye size={18} />}</button></label>
            <label className="is-wide"><span>模型名称</span><input required value={draft.model} onChange={(event) => setDraft({ ...draft, model: event.target.value })} placeholder="例如：text-embedding-v3" /></label>
          </div>
          {dialogError && <p className="settings-inline-error" role="alert">{dialogError}</p>}
          <footer><span><ShieldCheck size={15} />会发送一条测试文本验证连接</span><button type="submit" className="settings-primary-button" disabled={saving}>{saving ? <LoaderCircle className="is-spinning" size={17} /> : null}{saving ? "验证中" : "验证并保存"}</button></footer>
        </form>
      </DialogContent>
    </Dialog>
  </>;
}
