type NativeTransport = {
  postMessage(message: string): void;
};

type NativeBridgeMethod = (...args: unknown[]) => void;

export const MOBILE_NATIVE_METHODS = [
  "requestSnapshot", "selectSession", "removeUnavailableSession", "createSession",
  "restartPairing", "reloadFromServer", "exportDiagnostics", "openSettings",
  "chooseAttachments", "removeAttachment", "retryAttachment", "continueMeteredTransfer",
  "retryFailedMessage", "saveReadingPosition", "markSessionReadThrough", "navigationTargetHandled",
  "retryDownloadedAttachment", "touchDownloadedAttachment", "openDownloadedAttachment",
  "shareDownloadedAttachment", "saveDownloadedAttachment", "setWebHistoryActive", "dismissError",
  "shareText", "saveComposerDraft", "commitSharedText", "rejectSharedText", "sendMessage",
  "copyText", "performActionHaptic", "sendCommand", "refreshRuntimeInspection",
  "openRuntimeDocument", "openRuntimeMcp", "openRuntimeJob", "clearRuntimeInspectionDetail",
  "stopTurn", "queryPluginUi", "cancelPluginUiOwner", "setTheme", "setModelSelection", "reportHealthy",
] as const;

export type MobileNativeMethod = (typeof MOBILE_NATIVE_METHODS)[number];

const METHOD_ARITY: Record<MobileNativeMethod, number> = {
  requestSnapshot: 0, selectSession: 1, removeUnavailableSession: 1, createSession: 0,
  restartPairing: 0, reloadFromServer: 0, exportDiagnostics: 0, openSettings: 0,
  chooseAttachments: 0, removeAttachment: 1, retryAttachment: 1, continueMeteredTransfer: 0,
  retryFailedMessage: 1, saveReadingPosition: 3, markSessionReadThrough: 2, navigationTargetHandled: 1,
  retryDownloadedAttachment: 1, touchDownloadedAttachment: 1, openDownloadedAttachment: 1,
  shareDownloadedAttachment: 1, saveDownloadedAttachment: 1, setWebHistoryActive: 1, dismissError: 0,
  shareText: 2, saveComposerDraft: 4, commitSharedText: 4, rejectSharedText: 2, sendMessage: 6,
  copyText: 1, performActionHaptic: 0, sendCommand: 1, refreshRuntimeInspection: 0,
  openRuntimeDocument: 1, openRuntimeMcp: 2, openRuntimeJob: 1, clearRuntimeInspectionDetail: 0,
  stopTurn: 0, queryPluginUi: 10, cancelPluginUiOwner: 1, setTheme: 1, setModelSelection: 2,
  reportHealthy: 0,
};

export function installMobileBridge(): void {
  const transport = window.AkashicNativeTransport as NativeTransport | undefined;
  if (!transport || typeof transport.postMessage !== "function") return;
  const url = new URL(window.location.href);
  const generationId = url.searchParams.get("generation_id");
  const nonce = url.searchParams.get("nonce");
  if (!generationId || !nonce) {
    console.error("[mobile] native transport missing generation-specific URL identity");
    return;
  }
  const bridge: Record<string, NativeBridgeMethod> = {};
  for (const method of MOBILE_NATIVE_METHODS) {
    bridge[method] = (...args: unknown[]) => {
      if (args.length !== METHOD_ARITY[method]) {
        throw new TypeError(`[mobile] ${method} expects ${METHOD_ARITY[method]} args`);
      }
      transport.postMessage(JSON.stringify({
        v: 1,
        generation_id: generationId,
        nonce,
        method,
        args,
      }));
    };
  }
  window.AkashicNative = bridge as unknown as Window["AkashicNative"];
}
