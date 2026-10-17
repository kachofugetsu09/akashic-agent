import { Copy, Download, ImageIcon, Pencil, Play, X } from "lucide-react";
import { useEffect, useRef, useState, type CSSProperties, type KeyboardEvent as ReactKeyboardEvent, type ReactNode } from "react";
import { Dialog, DialogClose, DialogContent, DialogDescription, DialogTitle } from "@/components/ui/dialog";

import { CheckGlyph, IconSwap } from "./icon-swap";
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
  /** 读屏名里点开动作的动词：预览文本、下载其它文件。 */
  openVerb?: string;
  mediaTrigger?: Record<string, string>;
  onRemove?: () => void;
  /** 粘贴卡还原成输入框里的文字。 */
  onEdit?: () => void;
  onSurfaceKeyDown?: (event: ReactKeyboardEvent<HTMLButtonElement>) => void;
  style?: CSSProperties;
}

export function AttachmentCard({ data, pending = false, onOpen, openVerb = "预览", mediaTrigger, onRemove, onEdit, onSurfaceKeyDown, style }: AttachmentCardProps) {
  // 浏览器解不了的图（HEIC、损坏文件）按文件卡显示，写"无法预览"，不露破图。
  const [broken, setBroken] = useState(false);
  const media = (data.kind === "image" || data.kind === "video") && Boolean(data.url) && !broken;
  const label = data.kind === "paste" ? `粘贴的文本${data.lines !== undefined ? `，${data.lines} 行` : ""}` : data.name;
  const description = describe(data, broken);
  return <div className={`ak-attachment ${media ? "is-media" : "is-file"}`} data-kind={data.kind} data-tone={toneOf(data, broken)}
    aria-busy={pending || undefined} style={style}>
    <button type="button" className="ak-attachment__surface" disabled={pending} title={data.name}
      aria-label={`${media ? "查看 " : onOpen ? `${openVerb} ` : ""}${label}${description ? `（${description}）` : ""}`}
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
    {onEdit ? <button type="button" className="ak-attachment__edit" aria-label={`把 ${label} 还原成文字`} data-tip="还原成文字" onClick={onEdit}>
      <Pencil size={13} aria-hidden="true" />
    </button> : null}
    {onRemove ? <button type="button" className="ak-attachment__remove" aria-label={`移除 ${label}`} data-tip="移除" onClick={onRemove}>
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

// —— 已发送的附件 ——

export interface SentAttachmentItem {
  id: string;
  name: string;
  mediaType?: string;
  url: string;
  sizeBytes?: number;
}

const isMediaKind = (kind: AttachmentCardKind): kind is MediaKind => kind === "image" || kind === "video";

// 已发送附件条（参考 Desktop）：用户消息里放在气泡上方、右对齐，回复里左对齐；图片与录屏排前面并保持原比例
// （单张最大 480×320，多张每张最宽 240、最高 160，不裁成方块），之后另起一行排文件卡。
export function SentAttachments({ items, pending = false }: { items: SentAttachmentItem[]; pending?: boolean }) {
  const preview = useAttachmentPreview();
  const kinds = items.map((item) => attachmentCardKind(item.mediaType, item.name));
  const media = items.filter((_, index) => isMediaKind(kinds[index]));
  const files = items.filter((_, index) => !isMediaKind(kinds[index]));
  const card = (item: SentAttachmentItem) => <SentAttachment key={item.id} item={item} pending={pending} onPreview={preview.open} />;
  return <div className="ak-attachment-strip" data-multi={media.length > 1 ? "" : undefined} aria-busy={pending || undefined}>
    {media.map(card)}
    {media.length && files.length ? <span className="ak-attachment-strip__break" aria-hidden="true" /> : null}
    {files.map(card)}
    {preview.dialog}
  </div>;
}

function SentAttachment({ item, pending, onPreview }: {
  item: SentAttachmentItem;
  pending: boolean;
  onPreview: ReturnType<typeof useAttachmentPreview>["open"];
}) {
  const kind = attachmentCardKind(item.mediaType, item.name);
  const media = isMediaKind(kind);
  const textual = kind === "paste" || (!media && isTextLike(item.mediaType, item.name));
  // 不能预览的文件点开就下载：卡片上已写全大小与类型，再弹一层说明只是多点一下。
  const download = () => {
    const link = document.createElement("a");
    link.href = item.url;
    link.download = item.name;
    link.click();
  };
  return <AttachmentCard pending={pending}
    data={{ id: item.id, name: item.name, kind, mediaType: item.mediaType, url: item.url, sizeBytes: item.sizeBytes }}
    mediaTrigger={media ? mediaAttributes(item.url, item.name, kind) : undefined}
    onOpen={media ? undefined : textual ? () => onPreview({ name: item.name, url: item.url, sizeBytes: item.sizeBytes }) : download}
    openVerb={textual ? "预览" : "下载"} />;
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

// 预览只读开头 512KB、最多排 5000 行：大日志不整份解码、不拆成千万个数组项，其余交给下载。
const PREVIEW_LINE_LIMIT = 5000;
const PREVIEW_BYTE_LIMIT = 512 * 1024;
// 复制全文要整份读进内存再写剪贴板：太大的文件只给下载。
const COPY_BYTE_LIMIT = 8 * 1024 * 1024;

type PreviewTarget = { name: string; url: string; sizeBytes?: number };
type PreviewText = { lines: string[]; truncated: boolean; size: number; full?: string; totalLines?: number; table?: { columns: string[]; rows?: number } };
type PreviewState = { target: PreviewTarget; text: PreviewText | null; error: string };

// 1. 只解码前缀；2. 截断时丢掉最后半行；3. 行数沿用卡片的 2MB 统计；4. CSV/TSV 按引号规则读表头与记录数。
async function readPreview(target: PreviewTarget): Promise<PreviewText> {
  const response = await fetch(target.url);
  if (!response.ok) throw new Error(`读取附件失败: ${response.status}`);
  const blob = await response.blob();
  const prefix = await blob.slice(0, PREVIEW_BYTE_LIMIT).text();
  const lines: string[] = [];
  let from = 0;
  while (lines.length <= PREVIEW_LINE_LIMIT && from <= prefix.length) {
    const end = prefix.indexOf("\n", from);
    lines.push(prefix.slice(from, end < 0 ? prefix.length : end));
    from = end < 0 ? prefix.length + 1 : end + 1;
  }
  const bytesCut = blob.size > PREVIEW_BYTE_LIMIT;
  if (bytesCut && lines.length > 1 && lines.length <= PREVIEW_LINE_LIMIT) lines.pop();
  if (!bytesCut && from > prefix.length && lines.at(-1) === "") lines.pop();
  const truncated = bytesCut || lines.length > PREVIEW_LINE_LIMIT;
  const delimiter = /\.csv$/iu.test(target.name) ? "," : /\.tsv$/iu.test(target.name) ? "\t" : "";
  const totalLines = (await loadFacts(target.url))?.lines;
  return { lines: lines.slice(0, PREVIEW_LINE_LIMIT), truncated, size: blob.size, full: bytesCut ? undefined : prefix, totalLines,
    table: delimiter ? readTable(prefix, delimiter, !bytesCut) : undefined };
}

// 前缀放不下的文件按下复制时才读：剪贴板接受读取中的 Promise，读完前不丢用户手势；不支持时读完再走普通复制。
async function copyLarge(url: string): Promise<void> {
  const blob = fetch(url).then((response) => {
    if (!response.ok) throw new Error(`读取附件失败: ${response.status}`);
    return response.blob();
  }).then((raw) => new Blob([raw], { type: "text/plain" }));
  if (navigator.clipboard?.write && typeof ClipboardItem !== "undefined") {
    try {
      await navigator.clipboard.write([new ClipboardItem({ "text/plain": blob })]);
      return;
    } catch {
      // 权限或类型不被接受时走普通复制。
    }
  }
  await copyText(await (await blob).text());
}

// RFC 4180：引号字段里可以有分隔符、换行和成对的 ""。只数前缀里的记录，截断时不报行数。
function readTable(text: string, delimiter: string, complete: boolean): { columns: string[]; rows?: number } {
  const columns: string[] = [];
  let field = "", quoted = false, started = false, records = 0;
  const endField = () => {
    if (records === 0) columns.push(field);
    field = "";
  };
  for (let index = 0; index < text.length; index += 1) {
    const character = text[index];
    if (quoted) {
      if (character === '"' && text[index + 1] === '"') { field += '"'; index += 1; }
      else if (character === '"') quoted = false;
      else field += character;
    } else if (character === '"') {
      quoted = started = true;
    } else if (character === delimiter) {
      endField();
      started = true;
    } else if (character === "\n" || character === "\r") {
      if (character === "\r" && text[index + 1] === "\n") continue;
      // 空行不算记录。
      if (started) { endField(); records += 1; }
      started = false;
    } else {
      field += character;
      started = true;
    }
  }
  if (started) { endField(); records += 1; }
  return { columns, rows: complete ? Math.max(0, records - 1) : undefined };
}

// 文本附件预览（参考 Desktop 的文本附件面板）：标题下写"大小 • 行数"，等宽带行号，可复制全文、下载；
// CSV/TSV 先列出列名。媒体不走这里，交给媒体查看器。关闭后焦点回到打开它的卡，卡已不在时回到正文框。
export function useAttachmentPreview(): { open: (target: PreviewTarget) => void; dialog: ReactNode } {
  const [state, setState] = useState<PreviewState | null>(null);
  const opener = useRef<HTMLElement | null>(null);
  const open = (target: PreviewTarget) => {
    opener.current = document.activeElement instanceof HTMLElement ? document.activeElement : null;
    setState({ target, text: null, error: "" });
    readPreview(target).then(
      (text) => setState((current) => current?.target === target ? { ...current, text } : current),
      (error: unknown) => setState((current) => current?.target === target
        ? { ...current, error: error instanceof Error ? error.message : "读取失败" } : current));
  };
  const restoreFocus = (event: Event) => {
    event.preventDefault();
    const target = opener.current?.isConnected ? opener.current : document.querySelector<HTMLElement>("textarea[name=message]");
    target?.focus();
  };
  const dialog = <Dialog open={state !== null} onOpenChange={(next) => { if (!next) setState(null); }}>
    {state ? <AttachmentPreviewContent state={state} onCloseAutoFocus={restoreFocus} /> : null}
  </Dialog>;
  return { open, dialog };
}

function AttachmentPreviewContent({ state, onCloseAutoFocus }: { state: PreviewState; onCloseAutoFocus: (event: Event) => void }) {
  const { target, text, error } = state;
  const [copied, setCopied] = useState(false);
  const [copyError, setCopyError] = useState("");
  const lines = text?.lines ?? [];
  const size = target.sizeBytes ?? text?.size;
  const meta = [size !== undefined ? formatBytes(size) : "", text?.totalLines !== undefined ? `${text.totalLines.toLocaleString("zh-CN")} 行` : ""]
    .filter(Boolean).join(" • ");
  const columns = text?.table?.columns ?? [];
  const copyable = text !== null && text.size <= COPY_BYTE_LIMIT;
  const copy = () => {
    if (!copyable) return;
    setCopied(false);
    setCopyError("");
    void (text.full !== undefined ? copyText(text.full) : copyLarge(target.url)).then(() => {
      setCopied(true);
      window.setTimeout(() => setCopied(false), 2000);
    }, (error: unknown) => setCopyError(error instanceof Error ? `复制失败：${error.message}` : "复制失败，请重试或下载文件。"));
  };
  return <DialogContent className="attachment-preview" showCloseButton={false} onCloseAutoFocus={onCloseAutoFocus}>
    <header className="attachment-preview__bar">
      <div className="attachment-preview__title">
        <DialogTitle><bdi>{target.name}</bdi></DialogTitle>
        <DialogDescription>{meta || "\u00a0"}</DialogDescription>
      </div>
      <button type="button" onClick={copy} disabled={!copyable} aria-label={copied ? "已复制" : "复制全文"}
        data-tip={text && !copyable ? "文件较大，请下载后复制" : "复制全文"}>
        <IconSwap state={copied} a={<Copy size={16} aria-hidden="true" />} b={<CheckGlyph />} />
      </button>
      <a href={target.url} download={target.name} aria-label={`下载 ${target.name}`} data-tip="下载"><Download size={16} aria-hidden="true" /></a>
      <DialogClose aria-label="关闭" data-tip="关闭（Esc）"><X size={18} aria-hidden="true" /></DialogClose>
    </header>
    {copyError ? <p className="attachment-preview__note" role="alert">{copyError}</p> : null}
    {columns.length > 1 ? <section className="attachment-preview__columns" aria-label="表格结构">
      <ul>{columns.map((column, index) => <li key={index}><bdi>{column.trim() || `第 ${index + 1} 列`}</bdi></li>)}</ul>
      <p>{columns.length} 列{text?.table?.rows !== undefined ? `，${text.table.rows.toLocaleString("zh-CN")} 行` : ""}</p>
    </section> : null}
    <div className="attachment-preview__body" tabIndex={0} aria-label={`${target.name} 内容`}>
      {error ? <p className="attachment-preview__note" role="alert">{error}</p>
        : text === null ? <p className="attachment-preview__note" role="status">正在读取…</p>
        : <pre>{lines.map((line, index) => <span key={index}>{line || " "}</span>)}</pre>}
      {text?.truncated ? <p className="attachment-preview__note">只显示前 {lines.length.toLocaleString("zh-CN")} 行，完整内容请下载。</p> : null}
    </div>
  </DialogContent>;
}
