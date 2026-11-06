export interface PrivacyMaskState {
  enabled: boolean;
  redactTitles: boolean;
  blurPii: boolean;
  pickMode: boolean;
}

export interface PrivacyMaskSnapshot extends PrivacyMaskState {
  piiCount: number;
  pickedCount: number;
}

export interface PrivacyMaskEngine {
  getSnapshot(): PrivacyMaskSnapshot;
  subscribe(listener: () => void): () => void;
  toggle(force?: boolean): void;
  setScope(patch: Partial<Omit<PrivacyMaskState, "enabled">>): void;
  clearPicked(): void;
  release(): void;
}

const STORAGE_KEY = "akashic.privacy.mask.v1";
const HIGHLIGHT_NAME = "akashic-privacy-pii";
const MAX_HIGHLIGHT_RANGES = 800;

const DEFAULT_STATE: PrivacyMaskState = {
  enabled: false,
  redactTitles: true,
  blurPii: true,
  pickMode: false,
};

const SKIP_TAGS = new Set(["SCRIPT", "STYLE", "NOSCRIPT", "TEXTAREA", "INPUT", "SELECT", "OPTION"]);

const TITLE_REDACT_TARGETS = [
  ".conversation-session",
  ".conversation-session__title",
  ".navigation-group__open",
  ".navigation-group__name",
  ".group-session",
  ".conversation-heading__title-text",
  ".header-meta__project",
  "[data-privacy-redact]",
].join(", ");

export const CANDIDATE_SELECTOR = [
  ".chat-markdown pre",
  ".chat-markdown blockquote",
  ".chat-markdown table",
  ".chat-markdown img",
  ".chat-markdown li",
  ".chat-markdown p",
  ".tool-step",
  ".attachment-card",
  ".user-message",
  ".message-bubble",
  ".shell-settings-entry",
].join(", ");

// 匹配常见敏感词面：邮箱、Unix/macOS/Windows 本机路径、API Key/Token、IPv4 地址、11 位手机号。
const PII_PATTERN = new RegExp(
  [
    // 1. 邮箱（含部分掩码邮箱）
    "[\\w.+*•-]+@[\\w*•-]+(?:\\.[\\w*•-]+)+",
    // 2. Unix / macOS 常见敏感路径与 ~ 路径
    "(?:~|\\/(?:home|Users|srv|root|var|etc|opt|tmp|mnt|private))\\/[A-Za-z0-9._~\\/@:-]+",
    // 3. Windows 盘符路径
    "\\b[A-Za-z]:\\\\[A-Za-z0-9._~\\\\/-]+",
    // 4. 常见密钥与令牌（sk- / ghp_ / github_pat_ / AIza / JWT / Bearer）
    "\\b(?:sk-[A-Za-z0-9_-]{8,}|gh[pousr]_[A-Za-z0-9_]{10,}|github_pat_[A-Za-z0-9_]{12,}|AIza[0-9A-Za-z_-]{16,}|eyJ[A-Za-z0-9_-]{8,}\\.[A-Za-z0-9_-]{8,}\\.[A-Za-z0-9_-]{8,})",
    // 5. IPv4 地址（含可选端口）
    "\\b(?:(?:25[0-5]|2[0-4]\\d|1?\\d?\\d)\\.){3}(?:25[0-5]|2[0-4]\\d|1?\\d?\\d)(?::\\d{2,5})?\\b",
    // 6. 中国大陆 11 位手机号
    "(?<!\\d)1[3-9]\\d{9}(?!\\d)",
  ].join("|"),
  "g",
);

function loadState(): PrivacyMaskState {
  try {
    const raw = window.localStorage.getItem(STORAGE_KEY);
    if (!raw) return { ...DEFAULT_STATE };
    const parsed = JSON.parse(raw) as Partial<PrivacyMaskState>;
    return {
      enabled: typeof parsed.enabled === "boolean" ? parsed.enabled : DEFAULT_STATE.enabled,
      redactTitles: typeof parsed.redactTitles === "boolean" ? parsed.redactTitles : DEFAULT_STATE.redactTitles,
      blurPii: typeof parsed.blurPii === "boolean" ? parsed.blurPii : DEFAULT_STATE.blurPii,
      pickMode: typeof parsed.pickMode === "boolean" ? parsed.pickMode : false,
    };
  } catch {
    return { ...DEFAULT_STATE };
  }
}

function saveState(state: PrivacyMaskState): void {
  try {
    window.localStorage.setItem(STORAGE_KEY, JSON.stringify(state));
  } catch {
    // 隐私窗口或存储配额受限时退回内存状态。
  }
}

function cssHighlights(): Map<string, unknown> | null {
  const css = (globalThis as unknown as { CSS?: { highlights?: Map<string, unknown> } }).CSS;
  return css?.highlights ?? null;
}

function createHighlight(ranges: Range[]): unknown | null {
  const HighlightCtor = (globalThis as unknown as { Highlight?: new (...items: Range[]) => unknown }).Highlight;
  if (typeof HighlightCtor !== "function") return null;
  return new HighlightCtor(...ranges);
}

interface InternalEngine extends PrivacyMaskEngine {
  refs: number;
}

declare global {
  interface Window {
    __akashicPrivacyMask?: InternalEngine;
  }
}

export function acquirePrivacyMaskEngine(): PrivacyMaskEngine {
  if (window.__akashicPrivacyMask) {
    window.__akashicPrivacyMask.refs += 1;
    return window.__akashicPrivacyMask;
  }

  let state: PrivacyMaskState = loadState();
  let piiCount = 0;
  let pickedCount = 0;
  let snapshot: PrivacyMaskSnapshot = { ...state, piiCount, pickedCount };
  const listeners = new Set<() => void>();
  let scanFrame = 0;
  let transitionTimer = 0;
  let hoveredCandidate: HTMLElement | null = null;
  let applyingAttributes = false;

  const notify = () => {
    snapshot = { ...state, piiCount, pickedCount };
    for (const listener of listeners) listener();
  };

  const clearHover = () => {
    if (hoveredCandidate) {
      hoveredCandidate.removeAttribute("data-privacy-hover");
      hoveredCandidate = null;
    }
  };

  const restoreTooltips = () => {
    for (const el of document.querySelectorAll<HTMLElement>("[data-privacy-raw-title]")) {
      const raw = el.getAttribute("data-privacy-raw-title");
      if (raw !== null) el.setAttribute("title", raw);
      el.removeAttribute("data-privacy-raw-title");
    }
    for (const el of document.querySelectorAll<HTMLElement>("[data-privacy-raw-tip]")) {
      const raw = el.getAttribute("data-privacy-raw-tip");
      if (raw !== null) el.setAttribute("data-tip", raw);
      el.removeAttribute("data-privacy-raw-tip");
    }
  };

  const maskTooltips = () => {
    restoreTooltips();
    if (!state.enabled) return;

    if (state.redactTitles) {
      for (const el of document.querySelectorAll<HTMLElement>(TITLE_REDACT_TARGETS)) {
        if (el.closest("[data-privacy-ui]")) continue;
        const title = el.getAttribute("title");
        if (title) {
          el.setAttribute("data-privacy-raw-title", title);
          el.removeAttribute("title");
        }
        const tip = el.getAttribute("data-tip");
        if (tip) {
          el.setAttribute("data-privacy-raw-tip", tip);
          el.removeAttribute("data-tip");
        }
      }
    }

    if (state.blurPii) {
      for (const el of document.querySelectorAll<HTMLElement>("[title], [data-tip]")) {
        if (el.closest("[data-privacy-ui]")) continue;
        const title = el.getAttribute("title");
        if (title && !el.hasAttribute("data-privacy-raw-title")) {
          PII_PATTERN.lastIndex = 0;
          if (PII_PATTERN.test(title)) {
            PII_PATTERN.lastIndex = 0;
            el.setAttribute("data-privacy-raw-title", title);
            el.setAttribute("title", title.replace(PII_PATTERN, (m) => "•".repeat(Math.min(12, m.length))));
          }
        }
        const tip = el.getAttribute("data-tip");
        if (tip && !el.hasAttribute("data-privacy-raw-tip")) {
          PII_PATTERN.lastIndex = 0;
          if (PII_PATTERN.test(tip)) {
            PII_PATTERN.lastIndex = 0;
            el.setAttribute("data-privacy-raw-tip", tip);
            el.setAttribute("data-tip", tip.replace(PII_PATTERN, (m) => "•".repeat(Math.min(12, m.length))));
          }
        }
      }
    }
  };

  const runScan = () => {
    scanFrame = 0;
    const highlights = cssHighlights();
    applyingAttributes = true;
    try {
      maskTooltips();
    } finally {
      applyingAttributes = false;
    }

    if (!state.enabled || !state.blurPii || !document.body) {
      highlights?.delete(HIGHLIGHT_NAME);
      if (piiCount !== 0) {
        piiCount = 0;
        notify();
      }
      return;
    }

    const ranges: Range[] = [];
    const walker = document.createTreeWalker(document.body, NodeFilter.SHOW_TEXT);
    for (let node = walker.nextNode(); node; node = walker.nextNode()) {
      if (ranges.length >= MAX_HIGHLIGHT_RANGES) break;
      const textNode = node as Text;
      const value = textNode.nodeValue;
      if (!value || value.length < 4) continue;
      const parent = textNode.parentElement;
      if (!parent || SKIP_TAGS.has(parent.tagName)) continue;
      if (parent.closest("[data-privacy-ui], [inert], [aria-hidden='true'], svg")) continue;
      if (state.redactTitles && parent.closest(TITLE_REDACT_TARGETS)) continue;

      PII_PATTERN.lastIndex = 0;
      for (const match of value.matchAll(PII_PATTERN)) {
        if (ranges.length >= MAX_HIGHLIGHT_RANGES) break;
        const start = match.index ?? 0;
        const end = start + match[0].length;
        if (end <= start) continue;
        const range = new Range();
        range.setStart(textNode, start);
        range.setEnd(textNode, end);
        ranges.push(range);
      }
    }

    if (highlights) {
      if (ranges.length > 0) {
        const highlight = createHighlight(ranges);
        if (highlight) highlights.set(HIGHLIGHT_NAME, highlight);
      } else {
        highlights.delete(HIGHLIGHT_NAME);
      }
    }

    const nextPickedCount = document.querySelectorAll("[data-privacy-picked='true']").length;
    if (piiCount !== ranges.length || pickedCount !== nextPickedCount) {
      piiCount = ranges.length;
      pickedCount = nextPickedCount;
      notify();
    }
  };

  const scheduleScan = () => {
    if (scanFrame) return;
    scanFrame = window.requestAnimationFrame(runScan);
  };

  const syncDomDataset = (animate = false) => {
    const root = document.documentElement;
    applyingAttributes = true;
    try {
      if (state.enabled) {
        root.setAttribute("data-privacy-mask", "true");
        if (state.redactTitles) root.setAttribute("data-privacy-titles", "true");
        else root.removeAttribute("data-privacy-titles");
        if (state.blurPii) root.setAttribute("data-privacy-pii", "true");
        else root.removeAttribute("data-privacy-pii");
        if (state.pickMode) root.setAttribute("data-privacy-pick", "true");
        else {
          root.removeAttribute("data-privacy-pick");
          clearHover();
        }
      } else {
        root.removeAttribute("data-privacy-mask");
        root.removeAttribute("data-privacy-titles");
        root.removeAttribute("data-privacy-pii");
        root.removeAttribute("data-privacy-pick");
        clearHover();
      }
      if (animate) {
        root.setAttribute("data-privacy-transition", "true");
        window.clearTimeout(transitionTimer);
        transitionTimer = window.setTimeout(() => {
          root.removeAttribute("data-privacy-transition");
        }, 260);
      }
    } finally {
      applyingAttributes = false;
    }
    runScan();
  };

  const observer = new MutationObserver(() => {
    if (applyingAttributes || !state.enabled) return;
    scheduleScan();
  });

  if (document.body) {
    observer.observe(document.body, {
      subtree: true,
      childList: true,
      characterData: true,
    });
  }

  const findCandidate = (target: EventTarget | null): HTMLElement | null => {
    if (!(target instanceof Element)) return null;
    if (target.closest("[data-privacy-ui]")) return null;
    const candidate = target.closest<HTMLElement>(CANDIDATE_SELECTOR);
    if (!candidate || candidate.closest("[data-privacy-ui]")) return null;
    const rect = candidate.getBoundingClientRect();
    if (rect.width < 20 || rect.height < 12) return null;
    return candidate;
  };

  const onPointerMove = (event: PointerEvent) => {
    if (!state.enabled || !state.pickMode) return;
    const next = findCandidate(event.target);
    if (next === hoveredCandidate) return;
    clearHover();
    if (next) {
      hoveredCandidate = next;
      hoveredCandidate.setAttribute("data-privacy-hover", "true");
    }
  };

  const onClickCapture = (event: MouseEvent) => {
    if (!state.enabled) return;
    const target = event.target;
    if (!(target instanceof Element) || target.closest("[data-privacy-ui]")) return;

    // 1. 点选打码模式下：单击任意候选块切换局部模糊。
    if (state.pickMode) {
      const candidate = findCandidate(target);
      if (!candidate) return;
      event.preventDefault();
      event.stopPropagation();
      if (candidate.getAttribute("data-privacy-picked") === "true") {
        candidate.removeAttribute("data-privacy-picked");
      } else {
        candidate.setAttribute("data-privacy-picked", "true");
      }
      pickedCount = document.querySelectorAll("[data-privacy-picked='true']").length;
      notify();
      return;
    }

    // 2. 非点选模式下：若某区块已被手动打码，按住 Alt 单击可直接解除该块打码。
    if (event.altKey) {
      const picked = target.closest<HTMLElement>("[data-privacy-picked='true']");
      if (picked) {
        event.preventDefault();
        event.stopPropagation();
        picked.removeAttribute("data-privacy-picked");
        pickedCount = document.querySelectorAll("[data-privacy-picked='true']").length;
        notify();
      }
    }
  };

  const onKeyDown = (event: KeyboardEvent) => {
    if (event.isComposing || event.defaultPrevented) return;
    // Alt+M：一键切换截屏隐私打码。
    if (event.altKey && !event.ctrlKey && !event.metaKey && !event.shiftKey && event.code === "KeyM") {
      event.preventDefault();
      engine.toggle();
      return;
    }
    // 点选模式下按 Esc 先退出点选模式，不误关会话。
    if (event.key === "Escape" && state.enabled && state.pickMode) {
      if (document.querySelector("[data-privacy-ui] [aria-expanded='true']")) return;
      event.preventDefault();
      event.stopPropagation();
      engine.setScope({ pickMode: false });
    }
  };

  const onStorage = (event: StorageEvent) => {
    if (event.key !== STORAGE_KEY) return;
    state = loadState();
    syncDomDataset(true);
    notify();
  };

  document.addEventListener("pointermove", onPointerMove, { passive: true });
  document.addEventListener("click", onClickCapture, true);
  window.addEventListener("keydown", onKeyDown);
  window.addEventListener("storage", onStorage);

  const engine: InternalEngine = {
    refs: 1,
    getSnapshot() {
      return snapshot;
    },
    subscribe(listener: () => void) {
      listeners.add(listener);
      return () => {
        listeners.delete(listener);
      };
    },
    toggle(force?: boolean) {
      const nextEnabled = typeof force === "boolean" ? force : !state.enabled;
      state = {
        ...state,
        enabled: nextEnabled,
        // 重新开启时若三个子维度全关，自动恢复默认两项，避免"开了但什么都没打码"的空状态。
        redactTitles: nextEnabled && !state.redactTitles && !state.blurPii && !state.pickMode ? true : state.redactTitles,
        blurPii: nextEnabled && !state.redactTitles && !state.blurPii && !state.pickMode ? true : state.blurPii,
        pickMode: nextEnabled ? state.pickMode : false,
      };
      saveState(state);
      syncDomDataset(true);
      notify();
    },
    setScope(patch: Partial<Omit<PrivacyMaskState, "enabled">>) {
      const next = { ...state, ...patch };
      // 在面板中直接开启任一子维度时，若总开关未开则一并开启。
      const enablingSubScope =
        (patch.redactTitles === true || patch.blurPii === true || patch.pickMode === true) && !state.enabled;
      state = {
        ...next,
        enabled: enablingSubScope ? true : state.enabled,
      };
      saveState(state);
      syncDomDataset(true);
      notify();
    },
    clearPicked() {
      for (const el of document.querySelectorAll<HTMLElement>("[data-privacy-picked='true']")) {
        el.removeAttribute("data-privacy-picked");
      }
      clearHover();
      pickedCount = 0;
      notify();
    },
    release() {
      engine.refs -= 1;
      if (engine.refs > 0) return;
      window.clearTimeout(transitionTimer);
      if (scanFrame) window.cancelAnimationFrame(scanFrame);
      observer.disconnect();
      document.removeEventListener("pointermove", onPointerMove);
      document.removeEventListener("click", onClickCapture, true);
      window.removeEventListener("keydown", onKeyDown);
      window.removeEventListener("storage", onStorage);
      clearHover();
      restoreTooltips();
      for (const el of document.querySelectorAll<HTMLElement>("[data-privacy-picked='true']")) {
        el.removeAttribute("data-privacy-picked");
      }
      const root = document.documentElement;
      root.removeAttribute("data-privacy-mask");
      root.removeAttribute("data-privacy-titles");
      root.removeAttribute("data-privacy-pii");
      root.removeAttribute("data-privacy-pick");
      root.removeAttribute("data-privacy-transition");
      cssHighlights()?.delete(HIGHLIGHT_NAME);
      delete window.__akashicPrivacyMask;
    },
  };

  window.__akashicPrivacyMask = engine;
  syncDomDataset(false);
  return engine;
}
