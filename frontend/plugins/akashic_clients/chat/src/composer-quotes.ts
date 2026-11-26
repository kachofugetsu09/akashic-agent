import { useSyncExternalStore } from "react";
import type { WorkbenchQuote, WorkbenchQuoteAnchor, WorkbenchQuoteImage } from "./shell-workbench";

export const MAX_COMPOSER_QUOTES = 8;

export interface ComposerQuoteItem {
  readonly id: string;
  readonly source: string;
  readonly label: string;
  readonly preview: string;
  readonly text: string;
  readonly anchor?: WorkbenchQuoteAnchor;
}

const store = new Map<string, readonly ComposerQuoteItem[]>();
const listeners = new Set<() => void>();
const EMPTY: readonly ComposerQuoteItem[] = [];
let seq = 0;

function emit(): void {
  listeners.forEach((fn) => fn());
}

function subscribe(listener: () => void): () => void {
  listeners.add(listener);
  return () => listeners.delete(listener);
}

function shortSourceLabel(source: string, anchor?: WorkbenchQuoteAnchor): string {
  const path = anchor?.path;
  if (path) {
    const base = path.split("/").filter(Boolean).pop() || path;
    if (anchor?.page) return `${base} · p.${anchor.page}`;
    if (anchor?.sheet && anchor?.range) return `${base} · ${anchor.sheet}!${anchor.range}`;
    if (anchor?.sheet) return `${base} · ${anchor.sheet}`;
    if (anchor?.line) return `${base}:${anchor.line}`;
    return base;
  }
  const cleaned = source.replace(/^(文档区域|文档|文件|改动)\s*·\s*/u, "").trim();
  const slash = cleaned.lastIndexOf("/");
  return slash >= 0 ? cleaned.slice(slash + 1) : cleaned || source;
}

function summarizePreview(text: string): string {
  const plain = text
    .replace(/^>\s*\[[^\]]*\]\s*/gmu, "")
    .replace(/^>\s?/gmu, "")
    .replace(/```[\s\S]*?```/gu, (block) => block.replace(/```[^\n]*\n?/gu, "").trim())
    .replace(/\s+/gu, " ")
    .trim();
  return plain.length > 64 ? `${plain.slice(0, 64)}…` : plain;
}

export function composerQuotes(draftKey: string): readonly ComposerQuoteItem[] {
  return store.get(draftKey) ?? EMPTY;
}

export function useComposerQuotes(draftKey: string): readonly ComposerQuoteItem[] {
  return useSyncExternalStore(
    subscribe,
    () => composerQuotes(draftKey),
    () => EMPTY
  );
}

export function addComposerQuote(draftKey: string, quote: WorkbenchQuote): "added" | "duplicate" | "full" {
  const current = composerQuotes(draftKey);
  if (current.some((item) => item.source === quote.source && item.text === quote.text)) {
    return "duplicate";
  }
  if (current.length >= MAX_COMPOSER_QUOTES) {
    return "full";
  }
  const item: ComposerQuoteItem = {
    id: `q:${++seq}`,
    source: quote.source,
    label: shortSourceLabel(quote.source, quote.anchor),
    preview: summarizePreview(quote.text),
    text: quote.text,
    anchor: quote.anchor,
  };
  store.set(draftKey, [...current, item]);
  emit();
  return "added";
}

export function removeComposerQuote(draftKey: string, id: string): void {
  const current = composerQuotes(draftKey);
  const next = current.filter((item) => item.id !== id);
  if (next.length === current.length) return;
  if (next.length === 0) store.delete(draftKey);
  else store.set(draftKey, next);
  emit();
}

export function takeComposerQuotes(draftKey: string): readonly ComposerQuoteItem[] {
  const current = composerQuotes(draftKey);
  if (current.length === 0) return EMPTY;
  store.delete(draftKey);
  emit();
  return current;
}

export function restoreComposerQuotes(draftKey: string, items: readonly ComposerQuoteItem[]): void {
  if (items.length === 0) return;
  const current = composerQuotes(draftKey);
  store.set(draftKey, [...items, ...current].slice(0, MAX_COMPOSER_QUOTES));
  emit();
}

export function adoptComposerQuotes(fromKey: string, toKey: string): void {
  const current = composerQuotes(fromKey);
  if (current.length === 0) return;
  store.delete(fromKey);
  const existing = composerQuotes(toKey);
  store.set(toKey, [...existing, ...current].slice(0, MAX_COMPOSER_QUOTES));
  emit();
}

export function formatQuotesForMessage(quotes: readonly ComposerQuoteItem[], userText: string): string {
  if (quotes.length === 0) return userText;
  const blocks = quotes.map((q) => {
    const header = `> 引用：${q.source.replace(/\s+/gu, " ")}`;
    return q.text.trim() ? `${header}\n\n${q.text.trim()}` : header;
  });
  const body = userText.trim();
  return body ? `${blocks.join("\n\n---\n\n")}\n\n${body}` : blocks.join("\n\n---\n\n");
}

/** 把框选截图的 dataUrl 转为标准 File，直接接入聊天输入框的多模态图片附件管线。 */
export function quoteImageToFile(image: WorkbenchQuoteImage): File | null {
  const match = /^data:(image\/(?:png|jpeg|webp));base64,([A-Za-z0-9+/=]+)$/.exec(image.dataUrl);
  if (!match) return null;
  const mime = match[1];
  const raw = window.atob(match[2]);
  const bytes = new Uint8Array(raw.length);
  for (let i = 0; i < raw.length; i += 1) {
    bytes[i] = raw.charCodeAt(i);
  }
  const ext = mime === "image/jpeg" ? "jpg" : mime === "image/webp" ? "webp" : "png";
  const name = image.filename?.trim() || `snip.${ext}`;
  return new File([bytes], name, { type: mime });
}
