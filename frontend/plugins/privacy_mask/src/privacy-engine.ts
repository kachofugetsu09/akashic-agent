export interface PrivacyMaskState {
  enabled: boolean;
  redactTitles: boolean;
  blurContent: boolean;
}

export interface PrivacyMaskEngine {
  getSnapshot(): PrivacyMaskState;
  subscribe(listener: () => void): () => void;
  toggle(force?: boolean): void;
  setScope(patch: Partial<Omit<PrivacyMaskState, "enabled">>): void;
  release(): void;
}

const STORAGE_KEY = "akashic.privacy.mask.v1";

const DEFAULT_STATE: PrivacyMaskState = {
  enabled: false,
  redactTitles: true,
  blurContent: true,
};

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

const CONTENT_BLUR_TARGETS = [
  ".conversation-content [title]",
  ".conversation-content [data-tip]",
  ".header-meta__path [title]",
  ".header-meta__path [data-tip]",
].join(", ");

function loadState(): PrivacyMaskState {
  try {
    const raw = window.localStorage.getItem(STORAGE_KEY);
    if (!raw) return { ...DEFAULT_STATE };
    const parsed = JSON.parse(raw) as Partial<PrivacyMaskState>;
    return {
      enabled: typeof parsed.enabled === "boolean" ? parsed.enabled : DEFAULT_STATE.enabled,
      redactTitles: typeof parsed.redactTitles === "boolean" ? parsed.redactTitles : DEFAULT_STATE.redactTitles,
      blurContent: typeof parsed.blurContent === "boolean" ? parsed.blurContent : DEFAULT_STATE.blurContent,
    };
  } catch {
    return { ...DEFAULT_STATE };
  }
}

function saveState(state: PrivacyMaskState): void {
  try {
    window.localStorage.setItem(STORAGE_KEY, JSON.stringify(state));
  } catch {
    // 隐私窗口或存储受限时退回内存状态。
  }
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
  const listeners = new Set<() => void>();
  let scanFrame = 0;
  let applyingAttributes = false;

  const notify = () => {
    for (const listener of listeners) listener();
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

  const stripElementTooltips = (selector: string) => {
    for (const el of document.querySelectorAll<HTMLElement>(selector)) {
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
  };

  const syncTooltips = () => {
    scanFrame = 0;
    applyingAttributes = true;
    try {
      restoreTooltips();
      if (!state.enabled) return;
      if (state.redactTitles) stripElementTooltips(TITLE_REDACT_TARGETS);
      if (state.blurContent) stripElementTooltips(CONTENT_BLUR_TARGETS);
    } finally {
      applyingAttributes = false;
    }
  };

  const scheduleTooltipSync = () => {
    if (scanFrame) return;
    scanFrame = window.requestAnimationFrame(syncTooltips);
  };

  const syncDomDataset = () => {
    const root = document.documentElement;
    applyingAttributes = true;
    try {
      if (state.enabled) {
        root.setAttribute("data-privacy-mask", "true");
        if (state.redactTitles) root.setAttribute("data-privacy-titles", "true");
        else root.removeAttribute("data-privacy-titles");
        if (state.blurContent) root.setAttribute("data-privacy-content", "true");
        else root.removeAttribute("data-privacy-content");
      } else {
        root.removeAttribute("data-privacy-mask");
        root.removeAttribute("data-privacy-titles");
        root.removeAttribute("data-privacy-content");
      }
    } finally {
      applyingAttributes = false;
    }
    syncTooltips();
  };

  const observer = new MutationObserver(() => {
    if (applyingAttributes || !state.enabled) return;
    scheduleTooltipSync();
  });

  if (document.body) {
    observer.observe(document.body, {
      subtree: true,
      childList: true,
    });
  }

  const onKeyDown = (event: KeyboardEvent) => {
    if (event.isComposing || event.defaultPrevented) return;
    // Alt+M：一键切换截屏隐私打码。
    if (event.altKey && !event.ctrlKey && !event.metaKey && !event.shiftKey && event.code === "KeyM") {
      event.preventDefault();
      engine.toggle();
    }
  };

  const onStorage = (event: StorageEvent) => {
    if (event.key !== STORAGE_KEY) return;
    state = loadState();
    syncDomDataset();
    notify();
  };

  window.addEventListener("keydown", onKeyDown);
  window.addEventListener("storage", onStorage);

  const engine: InternalEngine = {
    refs: 1,
    getSnapshot() {
      return state;
    },
    subscribe(listener: () => void) {
      listeners.add(listener);
      return () => {
        listeners.delete(listener);
      };
    },
    toggle(force?: boolean) {
      const nextEnabled = typeof force === "boolean" ? force : !state.enabled;
      const bothOff = !state.redactTitles && !state.blurContent;
      state = {
        enabled: nextEnabled,
        redactTitles: nextEnabled && bothOff ? true : state.redactTitles,
        blurContent: nextEnabled && bothOff ? true : state.blurContent,
      };
      saveState(state);
      syncDomDataset();
      notify();
    },
    setScope(patch: Partial<Omit<PrivacyMaskState, "enabled">>) {
      const next = { ...state, ...patch };
      const anyOn = next.redactTitles || next.blurContent;
      const enablingSubScope = (patch.redactTitles === true || patch.blurContent === true) && !state.enabled;
      state = {
        ...next,
        enabled: enablingSubScope ? true : anyOn ? state.enabled : false,
      };
      saveState(state);
      syncDomDataset();
      notify();
    },
    release() {
      engine.refs -= 1;
      if (engine.refs > 0) return;
      if (scanFrame) window.cancelAnimationFrame(scanFrame);
      observer.disconnect();
      window.removeEventListener("keydown", onKeyDown);
      window.removeEventListener("storage", onStorage);
      restoreTooltips();
      const root = document.documentElement;
      root.removeAttribute("data-privacy-mask");
      root.removeAttribute("data-privacy-titles");
      root.removeAttribute("data-privacy-content");
      delete window.__akashicPrivacyMask;
    },
  };

  window.__akashicPrivacyMask = engine;
  syncDomDataset();
  return engine;
}
