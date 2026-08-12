import { Brain, Check, Database, LoaderCircle, Plus, ShieldCheck, Sparkles } from "lucide-react";
import { useCallback, useRef, useState } from "react";
import { MemoryEmbeddingDialog } from "./memory-embedding-dialog";
import type { MemorySettingsState } from "./memory-settings-data";
import { useMemorySettings } from "./use-memory-settings";

export type { MemorySettingsState } from "./memory-settings-data";

interface MemorySettingsProps {
  memory: MemorySettingsState;
  modelRevision: number;
  onboarding?: boolean;
  onRefresh: () => Promise<MemorySettingsState>;
  onNotice: (message: string) => void;
  onComplete: (message: string) => void;
  onError: (message: string) => void;
}

export function MemorySettings({ memory, modelRevision, onboarding = false, onRefresh, onNotice, onComplete, onError }: MemorySettingsProps) {
  const [dialogOpen, setDialogOpen] = useState(false);
  const modelSelectRef = useRef<HTMLSelectElement>(null);
  const addModelRef = useRef<HTMLButtonElement>(null);
  const focusValidationTarget = useCallback((hasModels: boolean) => {
    (hasModels ? modelSelectRef.current : addModelRef.current)?.focus();
  }, []);
  const controller = useMemorySettings({ memory, onComplete, onError, onValidationRequired: focusValidationTarget });
  const { mode, setMode, modelId, selectModel, saving, validationError } = controller;

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
          <input type="radio" name="memory-engine" checked={mode === "akasha"} onChange={() => setMode("akasha")} />
          <span className="settings-memory-icon"><Sparkles size={19} /></span>
          <span><strong>Akasha</strong><small>推荐 · 语义检索与长期记忆</small></span>
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

      {needsModel && <section className="settings-embedding-step" aria-labelledby="embedding-step-title">
        <header><div><h3 id="embedding-step-title">向量模型</h3><p>用于检索记忆，不影响聊天模型。</p></div><span className={modelId ? "is-ready" : "is-required"}>{modelId ? "已就绪" : "必需"}</span></header>
        <div className="settings-embedding-picker">
        <label>
          <span>已验证的模型</span>
          <select ref={modelSelectRef} value={modelId} aria-invalid={Boolean(validationError)} onChange={(event) => selectModel(event.target.value)} disabled={memory.changeLocked}>
            <option value="">选择已验证的向量模型</option>
            {memory.embeddingModels.map((model) => <option value={model.id} key={model.id}>{model.model}：{model.sourceName} · {model.dimensions} 维</option>)}
          </select>
        </label>
        <button ref={addModelRef} type="button" className="settings-quiet-button" onClick={() => setDialogOpen(true)} disabled={memory.changeLocked}><Plus size={17} />添加向量模型</button>
        </div>
        {validationError && <p className="settings-inline-error" role="alert">{validationError}</p>}
      </section>}

      {memory.changeLocked && <p className="settings-memory-lock"><ShieldCheck size={16} />当前 workspace 已有对话与记忆数据。更换引擎或向量模型需要先执行可恢复的索引迁移。</p>}

      <footer>
        <span>{needsModel ? "向量服务会在添加时验证；API Key 只存入当前 workspace。" : "不会显示 Akasha 或向量模型相关界面，也不会创建新的语义记忆。"}</span>
        <button type="button" className="settings-primary-button" onClick={() => void controller.save()} disabled={saving || memory.changeLocked}>
          {saving && <LoaderCircle className="is-spinning" size={17} />}{onboarding ? "完成设置并进入对话" : "保存记忆设置"}
        </button>
      </footer>
    </section>

    <MemoryEmbeddingDialog
      open={dialogOpen}
      modelRevision={modelRevision}
      returnFocusRef={addModelRef}
      onOpenChange={setDialogOpen}
      onSaved={async (model) => {
        selectModel(model.id);
        await onRefresh();
        onNotice(`${model.model} 已验证，识别为 ${model.dimensions} 维`);
      }}
    />
  </>;
}
