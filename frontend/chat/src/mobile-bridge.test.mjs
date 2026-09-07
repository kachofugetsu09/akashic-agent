import assert from "node:assert/strict";
import test from "node:test";

import { installMobileBridge } from "./mobile-bridge.ts";

test("session stop bridge keeps the explicit owner and rejects dropped arguments", () => {
  const messages = [];
  globalThis.window = {
    location: new URL("https://mobile.example.test/?generation_id=generation&nonce=nonce"),
    AkashicNativeTransport: {
      postMessage(value) {
        messages.push(JSON.parse(value));
      },
    },
  };
  try {
    installMobileBridge();
    window.AkashicNative.sendSessionCommand("akashic:target", "/stop");
    assert.deepEqual(messages[0], {
      v: 1,
      generation_id: "generation",
      nonce: "nonce",
      method: "sendSessionCommand",
      args: ["akashic:target", "/stop"],
    });
    assert.throws(
      () => window.AkashicNative.sendSessionCommand("akashic:target"),
      /sendSessionCommand expects 2 args/u,
    );
    assert.throws(
      () => window.AkashicNative.sendCommand("/stop", "akashic:target"),
      /sendCommand expects 1 args/u,
    );
  } finally {
    delete globalThis.window;
  }
});
