import assert from "node:assert/strict";
import test from "node:test";

import { parsePairedDevice, parsePairingOffer, parsePairingStatus } from "./mobile-pairing-data.ts";

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
