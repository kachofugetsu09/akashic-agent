import { Eye, EyeOff, LoaderCircle, ShieldCheck } from "lucide-react";
import { FormEvent, type RefObject, useEffect, useRef, useState } from "react";
import { Dialog, DialogContent, DialogDescription, DialogTitle } from "./components/ui/dialog";
import { saveEmbeddingModel, type EmbeddingDraft, type EmbeddingModelSummary } from "./memory-settings-data";
import { settingsErrorMessage } from "./settings-http.ts";

interface Props {
  open: boolean;
  modelRevision: number;
  returnFocusRef: RefObject<HTMLElement | null>;
  onOpenChange: (open: boolean) => void;
  onSaved: (model: EmbeddingModelSummary) => Promise<void>;
}

/** Own vector credential input and its single cancellable validation request. */
export function MemoryEmbeddingDialog({ open, modelRevision, returnFocusRef, onOpenChange, onSaved }: Props) {
  const [draft, setDraft] = useState<EmbeddingDraft>({ sourceName: "向量服务", baseUrl: "", apiKey: "", model: "" });
  const [showKey, setShowKey] = useState(false);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState("");
  const requestRef = useRef<AbortController | null>(null);

  useEffect(() => () => requestRef.current?.abort(), []);

  async function save(event: FormEvent) {
    event.preventDefault();
    if (requestRef.current) return;
    const controller = new AbortController();
    requestRef.current = controller;
    setSaving(true);
    setError("");
    try {
      const result = await saveEmbeddingModel(draft, modelRevision, controller.signal);
      await onSaved(result.model);
      onOpenChange(false);
    } catch (reason) {
      if (!controller.signal.aborted) setError(settingsErrorMessage(reason));
    } finally {
      if (requestRef.current === controller) requestRef.current = null;
      if (!controller.signal.aborted) setSaving(false);
    }
  }

  return <Dialog open={open} onOpenChange={onOpenChange}>
    <DialogContent
      className="settings-dialog settings-embedding-dialog"
      overlayClassName="settings-scrim"
      onCloseAutoFocus={(event) => {
        event.preventDefault();
        returnFocusRef.current?.focus();
      }}
    >
      <header><div><DialogTitle>添加向量模型</DialogTitle><DialogDescription>兼容 OpenAI `/embeddings` 协议；维度会自动识别。</DialogDescription></div></header>
      <form onSubmit={save}>
        <div className="settings-form-grid">
          <label className="is-wide"><span>连接名称</span><input required value={draft.sourceName} onChange={(event) => setDraft({ ...draft, sourceName: event.target.value })} placeholder="例如：DashScope 向量" /></label>
          <label className="is-wide"><span>Base URL</span><input required type="url" value={draft.baseUrl} onChange={(event) => setDraft({ ...draft, baseUrl: event.target.value })} placeholder="https://api.example.com/v1" /></label>
          <label className="settings-secret is-wide"><span>API Key</span><input required type={showKey ? "text" : "password"} value={draft.apiKey} onChange={(event) => setDraft({ ...draft, apiKey: event.target.value })} autoComplete="off" placeholder="sk-…" /><button type="button" onClick={() => setShowKey((value) => !value)} aria-label={showKey ? "隐藏 API Key" : "显示 API Key"}>{showKey ? <EyeOff aria-hidden="true" size={18} /> : <Eye aria-hidden="true" size={18} />}</button></label>
          <label className="is-wide"><span>模型名称</span><input required value={draft.model} onChange={(event) => setDraft({ ...draft, model: event.target.value })} placeholder="例如：text-embedding-v3" /></label>
        </div>
        {error ? <p className="settings-inline-error" role="alert">{error}</p> : null}
        <footer><span><ShieldCheck aria-hidden="true" size={15} />会发送一条测试文本验证连接</span><button type="submit" className="settings-primary-button" disabled={saving}>{saving ? <LoaderCircle aria-hidden="true" className="is-spinning" size={17} /> : null}{saving ? "验证中" : "验证并保存"}</button></footer>
      </form>
    </DialogContent>
  </Dialog>;
}
