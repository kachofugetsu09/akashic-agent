import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

import { parsePairedDevice, parsePairingOffer, parsePairingStatus } from "./mobile-pairing-data.ts";

const app = await readFile(new URL("./desktop-chat-view.tsx", import.meta.url), "utf8");
const dialog = await readFile(new URL("./mobile-pairing-dialog.tsx", import.meta.url), "utf8");
const controller = await readFile(new URL("./use-mobile-pairing.ts", import.meta.url), "utf8");

const offer = {
  protocol_version: 1,
  server_id: "server",
  server_application_key_fingerprint: "fingerprint",
  server_application_public_key: "public-key",
  lan_endpoints: ["wss://192.0.2.1/ws"],
  tunnel_endpoints: [],
  tls_spki_pins: ["pin"],
  pairing_id: "pairing",
  one_time_secret: "secret",
  expires_at: "2026-08-12T12:00:00.000Z",
};

test("pairing protocol validates every external response at its boundary", () => {
  assert.deepEqual(parsePairingOffer(offer), offer);
  assert.equal(parsePairingStatus({ pairing_id: "pairing", status: "waiting_for_phone" }), null);
  assert.equal(parsePairingStatus({
    pairing_id: "pairing", status: "waiting_for_desktop_confirmation",
    device_name: "Pixel 7", confirmation_code: "358864", capabilities: ["chat"],
  })?.confirmation_code, "358864");
  assert.deepEqual(parsePairedDevice({ device_id: "pixel-7", display_name: "Pixel 7" }), {
    device_id: "pixel-7", display_name: "Pixel 7",
  });
  assert.throws(() => parsePairingOffer({ ...offer, protocol_version: 2 }), /无效二维码数据/u);
  assert.throws(() => parsePairingStatus({ pairing_id: "pairing", status: "waiting_for_desktop_confirmation", confirmation_code: "123" }), /无效设备确认信息/u);
});

test("pairing view, controller, and protocol have separate owners", () => {
  assert.match(dialog, /useMobilePairing/);
  assert.doesNotMatch(dialog, /\bfetch\b|toDataURL|parsePairingOffer/);
  assert.match(controller, /new AbortController\(\)/);
  assert.match(controller, /actionRef\.current\?\.abort\(\)/);
  assert.match(app, /const LazyMobilePairingDialog = lazy/);
  assert.doesNotMatch(app, /import \{ MobilePairingDialog \} from/);
});

test("pairing countdown is visible without announcing every second", () => {
  assert.match(dialog, /role="timer"/);
  assert.doesNotMatch(dialog, /mobile-pairing-countdown" aria-live="polite"/);
  assert.match(dialog, /useReducedMotion\(\)/);
});
