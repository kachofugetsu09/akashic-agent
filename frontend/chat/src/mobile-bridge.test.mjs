import assert from "node:assert/strict";
import test from "node:test";

import { installMobileBridge } from "./mobile-bridge.ts";

const EXPECTED_METHODS = [
  "requestSnapshot", "selectSession", "removeUnavailableSession", "createSession",
  "restartPairing", "reloadFromServer", "exportDiagnostics", "openSettings",
  "chooseAttachments", "removeAttachment", "retryAttachment", "continueMeteredTransfer",
  "retryFailedMessage", "saveReadingPosition", "markSessionReadThrough", "navigationTargetHandled",
  "retryDownloadedAttachment", "touchDownloadedAttachment", "openDownloadedAttachment",
  "shareDownloadedAttachment", "saveDownloadedAttachment", "setWebHistoryActive", "dismissError",
  "shareText", "saveComposerDraft", "commitSharedText", "rejectSharedText", "sendMessage",
  "copyText", "performActionHaptic", "sendCommand", "refreshRuntimeInspection",
  "openRuntimeDocument", "openRuntimeMcp", "openRuntimeJob", "clearRuntimeInspectionDetail",
  "stopTurn", "queryPluginUi", "cancelPluginUiOwner", "reportHealthy",
];

function installFor(url) {
  const messages = [];
  globalThis.window = {
    location: { href: url },
    AkashicNativeTransport: { postMessage: (message) => messages.push(JSON.parse(message)) },
  };
  installMobileBridge();
  return { bridge: window.AkashicNative, messages };
}

test("embedded and remote WebUI install one generation-bound native surface", () => {
  const remote = installFor("https://mobile.invalid/mobile.html?generation_id=remote-gen&nonce=remote-nonce");
  const embedded = installFor("file:///android_asset/mobile.html?generation_id=embedded&nonce=baseline");
  assert.deepEqual(Object.keys(remote.bridge).sort(), [...EXPECTED_METHODS].sort());
  assert.deepEqual(Object.keys(embedded.bridge).sort(), [...EXPECTED_METHODS].sort());

  assert.throws(() => remote.bridge.selectSession(), /expects 1 args/);
  remote.bridge.requestSnapshot();
  embedded.bridge.reportHealthy();
  assert.deepEqual(remote.messages[0], {
    v: 1,
    generation_id: "remote-gen",
    nonce: "remote-nonce",
    method: "requestSnapshot",
    args: [],
  });
  assert.deepEqual(embedded.messages[0], {
    v: 1,
    generation_id: "embedded",
    nonce: "baseline",
    method: "reportHealthy",
    args: [],
  });
  delete globalThis.window;
});
