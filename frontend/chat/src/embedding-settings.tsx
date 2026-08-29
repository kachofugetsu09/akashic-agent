import { Database, LoaderCircle, Plus } from "lucide-react";
import { useRef, useState } from "react";
import { MemoryEmbeddingDialog } from "./memory-embedding-dialog";
import {
  saveDefaultEmbedding,
  type EmbeddingModelSummary,
} from "./memory-settings-data";
import { settingsErrorMessage } from "./settings-http.ts";

interface Props {
  models: EmbeddingModelSummary[];
  selectedModelId: string | null;
  modelRevision: number;
  onRefresh: () => Promise<unknown>;
  onNotice: (message: string) => void;
  onError: (message: string) => void;
}

export function EmbeddingSettings({ models, selectedModelId, modelRevision, onRefresh, onNotice, onError }: Props) {
  const [dialogOpen, setDialogOpen] = useState(false);
  const [saving, setSaving] = useState(false);
  const addModelRef = useRef<HTMLButtonElement>(null);

  async function selectModel(modelId: string) {
    if (!modelId || saving) return;
    const controller = new AbortController();
    setSaving(true);
    onError("");
    try {
      await saveDefaultEmbedding(modelId, modelRevision, controller.signal);
      await onRefresh();
      onNotice("默认向量模型已更新；新的记忆执行会使用新快照");
    } catch (reason) {
      onError(settingsErrorMessage(reason));
    } finally {
      setSaving(false);
    }
  }

  return <>
    <section className="settings-memory">
      <header>
        <div><h2>向量模型</h2><p>Akasha 在需要检索和写入记忆时使用；选择只由 models 插件保存。</p></div>
        <Database size={24} aria-hidden="true" />
      </header>
      <div className="settings-embedding-picker">
        <label>
          <span>默认向量模型</span>
          <select value={selectedModelId || ""} onChange={(event) => void selectModel(event.target.value)} disabled={saving}>
            <option value="">尚未配置</option>
            {models.map((model) => <option value={model.id} key={model.id}>{model.model}：{model.sourceName} · {model.dimensions} 维</option>)}
          </select>
        </label>
        <button ref={addModelRef} type="button" className="settings-quiet-button" onClick={() => setDialogOpen(true)} disabled={saving}>
          {saving ? <LoaderCircle className="is-spinning" size={17} /> : <Plus size={17} />}添加向量模型
        </button>
      </div>
    </section>
    <MemoryEmbeddingDialog
      open={dialogOpen}
      modelRevision={modelRevision}
      returnFocusRef={addModelRef}
      onOpenChange={setDialogOpen}
      onSaved={async (model) => {
        await onRefresh();
        onNotice(`${model.model} 已保存为默认向量模型`);
      }}
    />
  </>;
}
