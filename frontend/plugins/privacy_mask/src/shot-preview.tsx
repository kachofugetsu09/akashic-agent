import { useEffect, useRef, useState } from "react";
import { Check, ChevronDown, ChevronLeft, ChevronRight, Download, Share, SlidersHorizontal } from "lucide-react";
import { render, type MessageSelection, type PluginUiHostContext } from "@akashic/chat-ui-v1";
import type { PrivacyMaskEngine } from "./privacy-engine";
import { PII_LABEL, renderShot, type PiiKind, type ShotOptions, type ShotResult } from "./shot-render";

// 截图对话：宿主选范围 → 预览即编辑（打码三档、宽度、包含项收进「调整」）→ 分享/保存/复制。

type Initial = { around: string } | { from: string; to: string };

// 消息操作的入口：选好范围后在 body 上挂预览；「范围」回到选择并带着上次的范围。
export async function startScreenshot(context: Pick<PluginUiHostContext, "selectMessages">, engine: PrivacyMaskEngine, initial: Initial): Promise<void> {
  const selection = await context.selectMessages({ title: "截图", confirmLabel: "预览", initial });
  if (!selection) return;
  const host = document.createElement("div");
  document.body.append(host);
  // 全屏预览期间背后的聊天不进 Tab 序；关掉后焦点回到原处。
  const returnFocus = document.activeElement instanceof HTMLElement ? document.activeElement : null;
  const background = [...document.body.children].filter((element): element is HTMLElement => element instanceof HTMLElement && element !== host && !element.inert);
  for (const element of background) element.inert = true;
  let unmount = () => {};
  const release = () => {
    unmount();
    host.remove();
    for (const element of background) element.inert = false;
  };
  const close = () => { release(); returnFocus?.focus(); };
  const reselect = () => {
    release();
    void startScreenshot(context, engine, { from: selection.messageIds[0], to: selection.messageIds[selection.messageIds.length - 1] });
  };
  unmount = render(host, <ShotPreview selection={selection} screenMasked={engine.getSnapshot().enabled} onClose={close} onReselect={reselect} />);
}

const touch = () => window.matchMedia("(pointer: coarse)").matches;
const fileName = (index: number, total: number) => {
  const day = new Date().toISOString().slice(0, 10);
  return `akashic-${day}${total > 1 ? `-${index + 1}` : ""}.png`;
};

function ShotPreview({ selection, screenMasked, onClose, onReselect }: {
  selection: MessageSelection;
  screenMasked: boolean;
  onClose(): void;
  onReselect(): void;
}) {
  // 1. 打码默认值：屏幕已开截屏打码 → 全部模糊；否则先遮敏感信息，识别不到再说
  const [options, setOptions] = useState<ShotOptions>({
    mask: screenMasked ? "all" : "pii", unmasked: {}, wide: !touch(), process: false, time: true, footer: true,
  });
  const [result, setResult] = useState<ShotResult | null>(null);
  const [urls, setUrls] = useState<string[]>([]);
  const [busy, setBusy] = useState(true);
  const [saving, setSaving] = useState(false);
  const [failure, setFailure] = useState("");
  const [sheet, setSheet] = useState(false);
  const [toast, setToast] = useState("");
  const root = useRef<HTMLDivElement>(null);

  // 选项变了或预览关了，这一代就作废：晚到的结果不建 URL、不回写。
  useEffect(() => {
    let live = true;
    setBusy(true);
    renderShot(selection, options, () => live).then((next) => {
      if (!next || !live) return;
      setResult(next);
      setUrls(next.images.map((image) => URL.createObjectURL(image.blob)));
      setFailure("");
      setBusy(false);
    }).catch((error: unknown) => {
      if (!live) return;
      setFailure(error instanceof Error ? error.message : String(error));
      setBusy(false);
    });
    return () => { live = false; };
  }, [options, selection]);
  useEffect(() => () => urls.forEach((url) => URL.revokeObjectURL(url)), [urls]);
  useEffect(() => {
    root.current?.focus();
    const onKey = (event: KeyboardEvent) => {
      if (event.key !== "Escape") return;
      event.stopPropagation();
      if (sheet) setSheet(false);
      else onClose();
    };
    document.addEventListener("keydown", onKey, true);
    return () => document.removeEventListener("keydown", onKey, true);
  }, [onClose, sheet]);
  useEffect(() => {
    if (!toast) return;
    const timer = window.setTimeout(() => setToast(""), 2200);
    return () => window.clearTimeout(timer);
  }, [toast]);

  const images = result?.images ?? [];
  // 生成中或失败时旧图和当前选项对不上，导出入口一律停用。
  const ready = images.length > 0 && !busy && !failure;
  const act = shotActions(images, urls, setToast, setSaving);
  return <div ref={root} className="privacy-shot-preview" role="dialog" aria-modal="true" aria-label="截图预览" tabIndex={-1}>
    <header className="privacy-shot-preview__top">
      <button type="button" onClick={onReselect}><ChevronLeft size={18} aria-hidden="true" />范围</button>
      <strong>预览</strong>
      <button type="button" onClick={onClose}>关闭</button>
    </header>
    <div className="privacy-shot-preview__body" data-busy={busy || undefined}>
      <button type="button" className="privacy-shot-summary" onClick={() => setSheet(true)}>
        {summaryText(options, result)}<ChevronDown size={14} aria-hidden="true" />
      </button>
      <p className="privacy-shot-meta" role="status">{busy ? "正在生成…" : failure ? `没能生成图片：${failure}` : metaText(images)}</p>
      <div className="privacy-shot-images">
        {images.map((image, index) => <figure key={urls[index]}>
          <img src={urls[index]} width={image.width} height={image.height} alt={`对话截图 ${index + 1}`} />
          {images.length > 1 ? <figcaption>第 {index + 1}/{images.length} 张</figcaption> : null}
        </figure>)}
      </div>
    </div>
    <footer className="privacy-shot-preview__actions">
      <button type="button" onClick={() => setSheet(true)}><SlidersHorizontal size={16} aria-hidden="true" />调整</button>
      <span />
      <button type="button" disabled={!ready || saving} onClick={act.save}><Download size={16} aria-hidden="true" />{saving ? "准备中…" : "保存"}</button>
      {"share" in navigator ? <button type="button" disabled={!ready} className={touch() ? "is-primary" : ""} onClick={act.share}><Share size={16} aria-hidden="true" />分享</button> : null}
      {touch() ? null : <button type="button" disabled={!ready} className="is-primary" onClick={act.copy}>复制图片</button>}
    </footer>
    {sheet ? <ShotSheet options={options} found={result?.found ?? {}} onChange={setOptions} onClose={() => setSheet(false)} /> : null}
    {toast ? <p className="privacy-shot-toast" role="status">{toast}</p> : null}
  </div>;
}

function maskedCount(options: ShotOptions, result: ShotResult | null): number {
  return Object.entries(result?.found ?? {}).reduce((sum, [kind, count]) => sum + (options.unmasked[kind as PiiKind] ? 0 : count ?? 0), 0);
}

function summaryText(options: ShotOptions, result: ShotResult | null): string {
  const mask = options.mask === "all" ? "全部模糊" : options.mask === "off" ? "不打码"
    : maskedCount(options, result) ? `遮了 ${maskedCount(options, result)} 处` : "没有要遮的";
  return [mask, options.wide ? "宽屏" : "手机宽", options.process ? "含过程" : ""].filter(Boolean).join(" · ");
}

function metaText(images: ShotResult["images"]): string {
  if (!images.length) return "";
  const kb = Math.round(images.reduce((sum, image) => sum + image.blob.size, 0) / 1024);
  const size = kb > 1024 ? `${(kb / 1024).toFixed(1)} MB` : `${kb} KB`;
  return images.length > 1 ? `超过手机画布上限，按消息切成 ${images.length} 张 · 共 ${size}`
    : `${images[0].width} × ${images[0].height.toLocaleString()} px · ${size}`;
}

// 0.5.5 及更早版本的 Android Shell 仅支持 http(s) 下载，遇到 blob: 会提示“暂不支持保存这种文件”。
function needsHttpDownload(): boolean {
  const match = /\bAkashicShell\/(\d+)\.(\d+)\.(\d+)/.exec(navigator.userAgent);
  if (!match) return false;
  const major = Number(match[1]);
  const minor = Number(match[2]);
  const patch = Number(match[3]);
  return major === 0 && (minor < 5 || (minor === 5 && patch <= 5));
}

function triggerDownload(url: string, name: string): void {
  const link = document.createElement("a");
  link.href = url;
  link.download = name;
  link.click();
}

// 保存逐张下载；分享走系统面板（iOS 可存进相册）；剪贴板一次只放得下一张。
function shotActions(
  images: ShotResult["images"],
  urls: string[],
  toast: (text: string) => void,
  setSaving: (saving: boolean) => void,
) {
  const save = () => {
    if (!needsHttpDownload()) {
      urls.forEach((url, index) => triggerDownload(url, fileName(index, urls.length)));
      toast(urls.length > 1 ? `已保存 ${urls.length} 张` : "已保存");
      return;
    }
    setSaving(true);
    void (async () => {
      try {
        const prepared: Array<{ url: string; name: string }> = [];
        for (let index = 0; index < images.length; index += 1) {
          const image = images[index];
          const name = fileName(index, images.length);
          const response = await fetch(`/api/chat/downloads?filename=${encodeURIComponent(name)}`, {
            method: "POST",
            headers: { "Content-Type": image.blob.type || "image/png" },
            body: image.blob,
          });
          if (!response.ok) throw new Error(`HTTP ${response.status}`);
          const payload = await response.json() as { url?: unknown };
          if (typeof payload.url !== "string" || !payload.url.startsWith("/api/chat/downloads/")) {
            throw new Error("下载链接无效");
          }
          prepared.push({ url: payload.url, name });
        }
        prepared.forEach(({ url, name }) => triggerDownload(url, name));
        toast(prepared.length > 1 ? `已开始下载 ${prepared.length} 张` : "已开始下载");
      } catch (error: unknown) {
        toast(`保存失败：${error instanceof Error ? error.message : String(error)}`);
      } finally {
        setSaving(false);
      }
    })();
  };
  return {
    save,
    share: async () => {
      const files = images.map((image, index) => new File([image.blob], fileName(index, images.length), { type: "image/png" }));
      if (!navigator.canShare?.({ files })) { toast("这个浏览器不能分享图片，改为保存"); save(); return; }
      try { await navigator.share({ files }); } catch { /* 用户取消分享 */ }
    },
    copy: async () => {
      try {
        await navigator.clipboard.write([new ClipboardItem({ "image/png": images[0].blob })]);
        toast(images.length > 1 ? `复制了第 1 张（共 ${images.length} 张，全部请用保存）` : "已复制图片");
      } catch {
        toast("复制失败：浏览器没有给剪贴板写图片的权限");
      }
    },
  };
}

// 调整面板：打码放最上面；按类型取消再折一层；样式与包含项各一行。
function ShotSheet({ options, found, onChange, onClose }: {
  options: ShotOptions;
  found: Partial<Record<PiiKind, number>>;
  onChange(next: ShotOptions): void;
  onClose(): void;
}) {
  const set = (patch: Partial<ShotOptions>) => onChange({ ...options, ...patch });
  const kinds = Object.entries(found) as [PiiKind, number][];
  const seg = <K extends "mask" | "wide">(key: K, value: ShotOptions[K], label: string) => <button type="button"
    aria-pressed={options[key] === value} onClick={() => set({ [key]: value } as Partial<ShotOptions>)}>{label}</button>;
  const toggle = (key: "process" | "time" | "footer", label: string) => <button type="button" className="privacy-shot-check"
    aria-pressed={options[key]} onClick={() => set({ [key]: !options[key] })}><span><Check size={12} aria-hidden="true" /></span>{label}</button>;
  return <>
    <div className="privacy-shot-scrim" onClick={onClose} />
    <div className="privacy-shot-sheet" role="group" aria-label="调整截图">
      <div className="privacy-shot-sheet__row"><strong>打码</strong>
        <div className="privacy-shot-seg">{seg("mask", "off", "不打码")}{seg("mask", "pii", "敏感信息")}{seg("mask", "all", "全部模糊")}</div></div>
      {options.mask === "pii" && kinds.length ? <details className="privacy-shot-kinds">
        <summary>按类型：{kinds.map(([kind, count]) => `${PII_LABEL[kind]} ${count}`).join("、")}<ChevronRight size={14} aria-hidden="true" /></summary>
        <div>{kinds.map(([kind, count]) => <button key={kind} type="button" className="privacy-shot-check" aria-pressed={!options.unmasked[kind]}
          onClick={() => set({ unmasked: { ...options.unmasked, [kind]: !options.unmasked[kind] } })}>
          <span><Check size={12} aria-hidden="true" /></span>{PII_LABEL[kind]} {count}</button>)}</div>
      </details> : null}
      <p className="privacy-shot-sheet__hint">{options.mask === "all" ? "和截屏打码一样：消息内容全部模糊，只留版式。"
        : options.mask === "pii" && !kinds.length ? "选中范围里没有识别到手机号、邮箱、用户名或密钥。" : ""}</p>
      <div className="privacy-shot-sheet__row"><strong>宽度</strong>
        <div className="privacy-shot-seg">{seg("wide", false, "手机")}{seg("wide", true, "宽屏")}</div></div>
      <div className="privacy-shot-sheet__row"><strong>包含</strong>
        <div className="privacy-shot-sheet__checks">{toggle("process", "过程")}{toggle("time", "时间")}{toggle("footer", "页脚")}</div></div>
      <button type="button" className="privacy-shot-sheet__done" onClick={onClose}>好了</button>
    </div>
  </>;
}
