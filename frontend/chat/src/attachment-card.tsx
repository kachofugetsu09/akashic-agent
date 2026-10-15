import { Check, Copy, Download, ImageIcon, Pencil, Play, X } from "lucide-react";
import { useEffect, useState, type CSSProperties, type KeyboardEvent as ReactKeyboardEvent, type ReactNode } from "react";
import { Dialog, DialogClose, DialogContent, DialogDescription, DialogTitle } from "@/components/ui/dialog";
import { copyText } from "./copy-text";
import "./attachment-card.css";

// 附件卡（参考 Claude Desktop 的 MessageAttachments，尺寸取紧凑横条）：输入框与已发送消息共用一张卡。
// 文件是 64px 高的横条——左边类型块写扩展名，右边单行文件名（中间省略、保住扩展名）与"大小 · 行数"；
// 输入框里的图片与录屏是 64px 方形缩略图，点开交给媒体查看器。

export type AttachmentCardKind = "image" | "video" | "file" | "paste";

export interface AttachmentCardData {
  id: string;
  name: string;
  kind: AttachmentCardKind;
  mediaType?: string;
  url?: string;
  sizeBytes?: number;
  lines?: number;
  /** 粘贴文本卡的缩影：前 300 字。 */
  previewText?: string;
}

// 粘贴转成的附件按文件名认出来（发出后在消息里也一样），不另存标记。
export const PASTED_TEXT_NAME = "粘贴的文本.txt";
const PASTED_PATTERN = /^粘贴的文本(?:-\d+)?\.txt$/u;

export function attachmentCardKind(mediaType: string | undefined, name: string): AttachmentCardKind {
  if (PASTED_PATTERN.test(name)) return "paste";
  if (mediaType?.startsWith("image/")) return "image";
  if (mediaType?.startsWith("video/")) return "video";
  return "file";
}

// 能按文本读出行数与预览的文件：文本类媒体类型，或常见的纯文本扩展名。
export function isTextLike(mediaType: string | undefined, name: string): boolean {
  if (mediaType?.startsWith("text/") || /json|xml|yaml|toml|javascript|typescript|x-sh/u.test(mediaType ?? "")) return true;
  return /\.(?:log|md|txt|json|jsonl|ya?ml|toml|csv|tsv|py|ts|tsx|js|jsx|sh|sql|ini|cfg|conf|rs|go|java|c|h|cpp)$/iu.test(name);
}

export type MediaKind = "image" | "video";

// 附件媒体按钮的标记：委托查看器据此取地址与类型，不再每个附件各挂一个对话框。
export function mediaAttributes(url: string, label: string, kind: MediaKind): Record<string, string> {
  return { "data-media-url": url, "data-media-label": label, "data-media-kind": kind };
}

export function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  const units = ["KB", "MB", "GB"];
  let value = bytes / 1024;
  let index = 0;
  while (value >= 1024 && index < units.length - 1) {
    value /= 1024;
    index += 1;
  }
  return `${value < 10 ? value.toFixed(1) : Math.round(value)} ${units[index]}`;
}

function extensionOf(name: string): string {
  const dot = name.lastIndexOf(".");
  return dot > 0 && name.length - dot <= 6 ? name.slice(dot + 1) : "";
}

// 类型块的语义色：图片/录屏、文本与代码、表格、PDF、其它。
function toneOf(data: AttachmentCardData, broken: boolean): string {
  const ext = extensionOf(data.name).toLowerCase();
  if (data.kind === "image" || data.kind === "video" || broken) return "media";
  if (ext === "pdf") return "pdf";
  if (/^(?:csv|tsv|xlsx?|ods)$/u.test(ext)) return "data";
  if (data.kind === "paste" || isTextLike(data.mediaType, data.name)) return "text";
  return "other";
}

function describe(data: AttachmentCardData, broken: boolean): string {
  const parts: string[] = [];
  if (data.sizeBytes !== undefined) parts.push(formatBytes(data.sizeBytes));
  if (data.lines !== undefined) parts.push(`${data.lines.toLocaleString("zh-CN")} 行`);
  if (broken) parts.push("无法预览");
  return parts.join(" · ");
}

// 文件名中间省略：最后 6 个字（按字形簇，组合 emoji 不拆）固定不截，前面放不下的部分以省略号收尾。
const graphemes = new Intl.Segmenter(undefined, { granularity: "grapheme" });

function FileName({ name }: { name: string }) {
  const chars = Array.from(graphemes.segment(name), (part) => part.segment);
  const cut = Math.max(0, chars.length - 6);
  return <span className="ak-attachment__name">
    <bdi className="ak-attachment__head">{chars.slice(0, cut).join("")}</bdi>
    <bdi className="ak-attachment__tail">{chars.slice(cut).join("")}</bdi>
  </span>;
}

export interface AttachmentCardProps {
  data: AttachmentCardData;
  /** 已发送但还在上传：整卡骨架闪烁，不可点。 */
  pending?: boolean;
  /** 点开：媒体走查看器（mediaTrigger 写 data-media-*），文件交给调用方的预览。 */
  onOpen?: () => void;
  mediaTrigger?: Record<string, string>;
  onRemove?: () => void;
  /** 粘贴卡还原成输入框里的文字。 */
  onEdit?: () => void;
  onSurfaceKeyDown?: (event: ReactKeyboardEvent<HTMLButtonElement>) => void;
  style?: CSSProperties;
}

export function AttachmentCard({ data, pending = false, onOpen, mediaTrigger, onRemove, onEdit, onSurfaceKeyDown, style }: AttachmentCardProps) {
  // 浏览器解不了的图（HEIC、损坏文件）按文件卡显示，写"无法预览"，不露破图。
  const [broken, setBroken] = useState(false);
  const media = (data.kind === "image" || data.kind === "video") && Boolean(data.url) && !broken;
  const label = data.kind === "paste" ? `粘贴的文本${data.lines !== undefined ? `，${data.lines} 行` : ""}` : data.name;
  const description = describe(data, broken);
  return <div className={`ak-attachment ${media ? "is-media" : "is-file"}`} data-kind={data.kind} data-tone={toneOf(data, broken)}
    aria-busy={pending || undefined} style={style}>
    <button type="button" className="ak-attachment__surface" disabled={pending} title={data.name}
      aria-label={`${media ? "查看 " : onOpen ? "预览 " : ""}${label}${description ? `（${description}）` : ""}`}
      data-static={!media && !onOpen ? "" : undefined}
      onClick={onOpen} onKeyDown={onSurfaceKeyDown} {...(media ? mediaTrigger : undefined)}>
      {media ? <AttachmentThumb data={data} onBroken={() => setBroken(true)} /> : <>
        <span className="ak-attachment__tile" aria-hidden="true">
          {data.kind === "paste" && data.previewText ? <span className="ak-attachment__paste">{data.previewText}</span>
            : broken ? <ImageIcon size={18} /> : extensionOf(data.name) || "文件"}
        </span>
        <span className="ak-attachment__meta">
          {data.kind === "paste" ? <span className="ak-attachment__name"><bdi className="ak-attachment__head">粘贴的文本</bdi></span> : <FileName name={data.name} />}
          {description ? <span className="ak-attachment__desc">{description}</span> : null}
        </span>
      </>}
    </button>
    {pending ? <span className="ak-attachment__skeleton" aria-hidden="true" /> : null}
    {onEdit ? <button type="button" className="ak-attachment__edit" aria-label={`把 ${label} 还原成文字`} title="还原成文字" onClick={onEdit}>
      <Pencil size={13} aria-hidden="true" />
    </button> : null}
    {onRemove ? <button type="button" className="ak-attachment__remove" aria-label={`移除 ${label}`} title="移除" onClick={onRemove}>
      <X size={11} strokeWidth={3} aria-hidden="true" />
    </button> : null}
  </div>;
}

function AttachmentThumb({ data, onBroken }: { data: AttachmentCardData; onBroken: () => void }) {
  if (data.kind === "video") {
    return <>
      <video className="ak-attachment__image" src={data.url} muted playsInline preload="metadata" onError={onBroken} />
      <span className="ak-attachment__play" aria-hidden="true"><Play size={10} fill="currentColor" /></span>
    </>;
  }
  // 加载完成后 200ms 淡入（Desktop 同款），先露底色而不是半张图逐行刷出来。
  return <img className="ak-attachment__image" src={data.url} alt="" draggable={false} decoding="async"
    onLoad={(event) => event.currentTarget.setAttribute("data-loaded", "")} onError={onBroken} />;
}

// —— 文本内容：行数、粘贴缩影、预览 ——

export interface TextFacts {
  lines: number;
  preview: string;
}

// 只读开头 2MB：再大的日志也只为数行数与预览，不把整份读进内存；超出部分不计行数。
const TEXT_READ_LIMIT = 2 * 1024 * 1024;
const factsCache = new Map<string, Promise<TextFacts | null>>();

function countLines(text: string): number {
  if (!text) return 0;
  let lines = 1;
  for (let index = text.indexOf("\n"); index >= 0; index = text.indexOf("\n", index + 1)) lines += 1;
  return text.endsWith("\n") ? lines - 1 : lines;
}

async function readText(url: string, limit = TEXT_READ_LIMIT): Promise<{ text: string; complete: boolean }> {
  const response = await fetch(url);
  if (!response.ok) throw new Error(`读取附件失败: ${response.status}`);
  const blob = await response.blob();
  return { text: await blob.slice(0, limit).text(), complete: blob.size <= limit };
}

function loadFacts(url: string): Promise<TextFacts | null> {
  let pending = factsCache.get(url);
  if (!pending) {
    pending = readText(url).then(({ text, complete }) => complete ? { lines: countLines(text), preview: text.slice(0, 300) } : null,
      () => null);
    factsCache.set(url, pending);
  }
  return pending;
}

// 必要 effect：行数要读文件内容，读完才知道；同一地址只读一次。
export function useTextFacts(url: string | undefined, enabled: boolean): TextFacts | null {
  const [facts, setFacts] = useState<{ url: string; value: TextFacts | null } | null>(null);
  useEffect(() => {
    if (!enabled || !url) return;
    let live = true;
    void loadFacts(url).then((value) => { if (live) setFacts({ url, value }); });
    return () => { live = false; };
  }, [url, enabled]);
  return enabled && facts && facts.url === url ? facts.value : null;
}

export async function readAttachmentText(url: string): Promise<string> {
  return (await readText(url, Number.POSITIVE_INFINITY)).text;
}

// 预览最多排 5000 行：再多的行交给下载，避免一次挂上几十万个节点。
const PREVIEW_LINE_LIMIT = 5000;

type PreviewTarget = { name: string; url: string; sizeBytes?: number };
type PreviewState = { target: PreviewTarget; text: string | null; error: string };

// 文本附件预览（参考 Desktop 的文本附件面板）：标题下写"大小 • 行数"，等宽带行号，可复制全文、下载；
// CSV/TSV 先列出列名。媒体不走这里，交给媒体查看器。
export function useAttachmentPreview(): { open: (target: PreviewTarget) => void; dialog: ReactNode } {
  const [state, setState] = useState<PreviewState | null>(null);
  const open = (target: PreviewTarget) => {
    setState({ target, text: null, error: "" });
    readAttachmentText(target.url).then(
      (text) => setState((current) => current?.target === target ? { ...current, text } : current),
      (error: unknown) => setState((current) => current?.target === target
        ? { ...current, error: error instanceof Error ? error.message : "读取失败" } : current));
  };
  const dialog = <Dialog open={state !== null} onOpenChange={(next) => { if (!next) setState(null); }}>
    {state ? <AttachmentPreviewContent state={state} /> : null}
  </Dialog>;
  return { open, dialog };
}

function AttachmentPreviewContent({ state }: { state: PreviewState }) {
  const { target, text, error } = state;
  const [copied, setCopied] = useState(false);
  const lines = text === null ? [] : text.replace(/\n$/u, "").split("\n");
  const meta = [target.sizeBytes !== undefined ? formatBytes(target.sizeBytes) : "", text !== null ? `${lines.length.toLocaleString("zh-CN")} 行` : ""]
    .filter(Boolean).join(" • ");
  const delimiter = /\.csv$/iu.test(target.name) ? "," : /\.tsv$/iu.test(target.name) ? "\t" : "";
  const columns = delimiter && text ? lines[0].split(delimiter) : [];
  const copy = () => {
    if (text === null) return;
    void copyText(text).then(() => {
      setCopied(true);
      window.setTimeout(() => setCopied(false), 2000);
    });
  };
  return <DialogContent className="attachment-preview" showCloseButton={false}>
    <header className="attachment-preview__bar">
      <div className="attachment-preview__title">
        <DialogTitle><bdi>{target.name}</bdi></DialogTitle>
        <DialogDescription>{meta || "\u00a0"}</DialogDescription>
      </div>
      <button type="button" onClick={copy} disabled={text === null} aria-label={copied ? "已复制" : "复制全文"} title="复制全文">
        {copied ? <Check size={16} aria-hidden="true" /> : <Copy size={16} aria-hidden="true" />}
      </button>
      <a href={target.url} download={target.name} aria-label={`下载 ${target.name}`} title="下载"><Download size={16} aria-hidden="true" /></a>
      <DialogClose aria-label="关闭" title="关闭（Esc）"><X size={18} aria-hidden="true" /></DialogClose>
    </header>
    {columns.length > 1 ? <section className="attachment-preview__columns" aria-label="表格结构">
      <ul>{columns.map((column, index) => <li key={index}><bdi>{column.trim() || `第 ${index + 1} 列`}</bdi></li>)}</ul>
      <p>{columns.length} 列，{Math.max(0, lines.length - 1).toLocaleString("zh-CN")} 行</p>
    </section> : null}
    <div className="attachment-preview__body" tabIndex={0} aria-label={`${target.name} 内容`}>
      {error ? <p className="attachment-preview__note" role="alert">{error}</p>
        : text === null ? <p className="attachment-preview__note" role="status">正在读取…</p>
        : <pre>{lines.slice(0, PREVIEW_LINE_LIMIT).map((line, index) => <span key={index}>{line || " "}</span>)}</pre>}
      {lines.length > PREVIEW_LINE_LIMIT ? <p className="attachment-preview__note">只显示前 {PREVIEW_LINE_LIMIT} 行，完整内容请下载。</p> : null}
    </div>
  </DialogContent>;
}
