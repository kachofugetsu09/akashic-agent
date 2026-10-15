import { ImageIcon, Play, X } from "lucide-react";
import { useEffect, useState, type CSSProperties, type KeyboardEvent as ReactKeyboardEvent } from "react";
import "./attachment-card.css";

// 附件卡（参考 Claude Desktop 的 MessageAttachments，尺寸取紧凑横条）：输入框与已发送消息共用一张卡。
// 文件是 64px 高的横条——左边类型块写扩展名，右边单行文件名（中间省略、保住扩展名）与"大小 · 行数"；
// 输入框里的图片与录屏是 64px 方形缩略图，点开交给媒体查看器。

export type AttachmentCardKind = "image" | "video" | "file";

export interface AttachmentCardData {
  id: string;
  name: string;
  kind: AttachmentCardKind;
  mediaType?: string;
  url?: string;
  sizeBytes?: number;
  lines?: number;
}

export function attachmentCardKind(mediaType: string | undefined): AttachmentCardKind {
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
  if (isTextLike(data.mediaType, data.name)) return "text";
  return "other";
}

function describe(data: AttachmentCardData, broken: boolean): string {
  const parts: string[] = [];
  if (data.sizeBytes !== undefined) parts.push(formatBytes(data.sizeBytes));
  if (data.lines !== undefined) parts.push(`${data.lines.toLocaleString("zh-CN")} 行`);
  if (broken) parts.push("无法预览");
  return parts.join(" · ");
}

// 文件名中间省略：最后 6 个字符（含扩展名）固定不截，前面放不下的部分以省略号收尾。
function FileName({ name }: { name: string }) {
  const chars = Array.from(name);
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
  onSurfaceKeyDown?: (event: ReactKeyboardEvent<HTMLButtonElement>) => void;
  style?: CSSProperties;
}

export function AttachmentCard({ data, pending = false, onOpen, mediaTrigger, onRemove, onSurfaceKeyDown, style }: AttachmentCardProps) {
  // 浏览器解不了的图（HEIC、损坏文件）按文件卡显示，写"无法预览"，不露破图。
  const [broken, setBroken] = useState(false);
  const media = (data.kind === "image" || data.kind === "video") && Boolean(data.url) && !broken;
  const label = data.name;
  const description = describe(data, broken);
  return <div className={`ak-attachment ${media ? "is-media" : "is-file"}`} data-kind={data.kind} data-tone={toneOf(data, broken)}
    aria-busy={pending || undefined} style={style}>
    <button type="button" className="ak-attachment__surface" disabled={pending} title={data.name}
      aria-label={`${media ? "查看 " : onOpen ? "预览 " : ""}${label}${description ? `（${description}）` : ""}`}
      data-static={!media && !onOpen ? "" : undefined}
      onClick={onOpen} onKeyDown={onSurfaceKeyDown} {...(media ? mediaTrigger : undefined)}>
      {media ? <AttachmentThumb data={data} onBroken={() => setBroken(true)} /> : <>
        <span className="ak-attachment__tile" aria-hidden="true">
          {broken ? <ImageIcon size={18} /> : extensionOf(data.name) || "文件"}
        </span>
        <span className="ak-attachment__meta">
          <FileName name={data.name} />
          {description ? <span className="ak-attachment__desc">{description}</span> : null}
        </span>
      </>}
    </button>
    {pending ? <span className="ak-attachment__skeleton" aria-hidden="true" /> : null}
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

// —— 文本内容：行数 ——

export interface TextFacts {
  lines: number;
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
    pending = readText(url).then(({ text, complete }) => complete ? { lines: countLines(text) } : null,
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
