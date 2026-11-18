import { useEffect, useRef, useState } from "react";
import { ChevronLeft, Download, Share } from "lucide-react";
import { render, type MessageSelection, type PluginUiHostContext } from "@akashic/chat-ui-v1";
import type { PrivacyMaskEngine } from "./privacy-engine";
import { renderShot, type PiiKind, type ShotOptions, type ShotResult } from "./shot-render";

// 截图对话：宿主选范围 → 预览（默认遮敏感信息；截屏打码开着时全部模糊）→ 分享/保存/复制。

type Initial = { around: string } | { from: string; to: string };

// 消息操作的入口：选好范围后在 body 上挂预览；「范围」回到选择并带着上次的范围。
export async function startScreenshot(context: Pick<PluginUiHostContext, "selectMessages">, engine: PrivacyMaskEngine, initial: Initial): Promise<void> {
  const selection = await context.selectMessages({ title: "截图", confirmLabel: "预览", initial });
  if (!selection) return;
  const host = document.createElement("div");
  document.body.append(host);
  let unmount = () => {};
  const close = () => { unmount(); host.remove(); };
  const reselect = () => {
    close();
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
  const [options] = useState<ShotOptions>(() => ({
    mask: screenMasked ? "all" : "pii", unmasked: {}, wide: !touch(), process: false, time: true, footer: true,
  }));
  const [result, setResult] = useState<ShotResult | null>(null);
  const [urls, setUrls] = useState<string[]>([]);
  const [busy, setBusy] = useState(true);
  const [failure, setFailure] = useState("");
  const [toast, setToast] = useState("");
  const generation = useRef(0);
  const root = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const ticket = ++generation.current;
    setBusy(true);
    renderShot(selection, options, () => generation.current === ticket).then((next) => {
      if (!next) return;
      setResult(next);
      setUrls(next.images.map((image) => URL.createObjectURL(image.blob)));
      setFailure("");
      setBusy(false);
    }).catch((error: unknown) => {
      if (generation.current !== ticket) return;
      setFailure(error instanceof Error ? error.message : String(error));
      setBusy(false);
    });
  }, [options, selection]);
  useEffect(() => () => urls.forEach((url) => URL.revokeObjectURL(url)), [urls]);
  useEffect(() => {
    root.current?.focus();
    const onKey = (event: KeyboardEvent) => {
      if (event.key !== "Escape") return;
      event.stopPropagation();
      onClose();
    };
    document.addEventListener("keydown", onKey, true);
    return () => document.removeEventListener("keydown", onKey, true);
  }, [onClose]);
  useEffect(() => {
    if (!toast) return;
    const timer = window.setTimeout(() => setToast(""), 2200);
    return () => window.clearTimeout(timer);
  }, [toast]);

  const images = result?.images ?? [];
  const act = shotActions(images, urls, setToast);
  return <div ref={root} className="privacy-shot-preview" role="dialog" aria-modal="true" aria-label="截图预览" tabIndex={-1}>
    <header className="privacy-shot-preview__top">
      <button type="button" onClick={onReselect}><ChevronLeft size={18} aria-hidden="true" />范围</button>
      <strong>预览</strong>
      <button type="button" onClick={onClose}>关闭</button>
    </header>
    <div className="privacy-shot-preview__body" data-busy={busy || undefined}>
      <p className="privacy-shot-summary">{summaryText(options, result)}</p>
      <p className="privacy-shot-meta" role="status">{busy ? "正在生成…" : failure ? `没能生成图片：${failure}` : metaText(images)}</p>
      <div className="privacy-shot-images">
        {images.map((image, index) => <figure key={urls[index]}>
          <img src={urls[index]} width={image.width} height={image.height} alt={`对话截图 ${index + 1}`} />
          {images.length > 1 ? <figcaption>第 {index + 1}/{images.length} 张</figcaption> : null}
        </figure>)}
      </div>
    </div>
    <footer className="privacy-shot-preview__actions">
      <span />
      <button type="button" disabled={!images.length} onClick={act.save}><Download size={16} aria-hidden="true" />保存</button>
      {"share" in navigator ? <button type="button" disabled={!images.length} className={touch() ? "is-primary" : ""} onClick={act.share}><Share size={16} aria-hidden="true" />分享</button> : null}
      {touch() ? null : <button type="button" disabled={!images.length} className="is-primary" onClick={act.copy}>复制图片</button>}
    </footer>
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

// 保存逐张下载；分享走系统面板（iOS 可存进相册）；剪贴板一次只放得下一张。
function shotActions(images: ShotResult["images"], urls: string[], toast: (text: string) => void) {
  const save = () => {
    urls.forEach((url, index) => {
      const link = document.createElement("a");
      link.href = url;
      link.download = fileName(index, urls.length);
      link.click();
    });
    toast(urls.length > 1 ? `已保存 ${urls.length} 张` : "已保存");
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
