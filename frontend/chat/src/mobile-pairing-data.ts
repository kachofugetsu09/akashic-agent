export interface MobilePairingOffer {
  protocol_version: number;
  server_id: string;
  server_application_key_fingerprint: string;
  server_application_public_key: string;
  lan_endpoints: string[];
  tunnel_endpoints: string[];
  tls_spki_pins: string[];
  pairing_id: string;
  one_time_secret: string;
  expires_at: string;
}

export interface PendingClaim {
  pairing_id: string;
  status: "waiting_for_desktop_confirmation";
  device_name: string;
  confirmation_code: string;
  capabilities: string[];
}

export interface PairedDevice { device_id: string; display_name: string }

export type PairingState =
  | { stage: "creating" }
  | { stage: "waiting"; offer: MobilePairingOffer; qrDataUrl: string }
  | { stage: "confirming"; offer: MobilePairingOffer; claim: PendingClaim; approving: boolean }
  | { stage: "connected"; device: PairedDevice }
  | { stage: "error"; message: string };

export async function createPairingOffer(signal: AbortSignal) {
  return parsePairingOffer(await requestPairingJson("/api/chat/mobile-pairing", { method: "POST", signal }));
}

export async function loadPairingClaim(pairingId: string, signal: AbortSignal) {
  return parsePairingStatus(await requestPairingJson(`/api/chat/mobile-pairing/${encodeURIComponent(pairingId)}`, { signal }));
}

export async function approvePairing(offer: MobilePairingOffer, claim: PendingClaim, signal: AbortSignal) {
  return parsePairedDevice(await requestPairingJson(`/api/chat/mobile-pairing/${encodeURIComponent(offer.pairing_id)}/approve`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ confirmation_code: claim.confirmation_code }),
    signal,
  }));
}

async function requestPairingJson(url: string, options: RequestInit): Promise<unknown> {
  const response = await fetch(url, options);
  const text = await response.text();
  const payload: unknown = text ? JSON.parse(text) : null;
  if (!response.ok) {
    const body = recordValue(payload);
    const detail = typeof body?.detail === "string" ? body.detail : "";
    if (response.status === 404) throw new Error("移动网关尚未启用，请先开启 mobile_realtime 配置");
    throw new Error(detail || `配对请求失败: ${response.status}`);
  }
  if (payload === null) throw new Error("配对服务返回空响应");
  return payload;
}

export function parsePairingOffer(payload: unknown): MobilePairingOffer {
  const body = recordValue(payload);
  if (!body || body.protocol_version !== 1 || !isNonEmptyString(body.server_id)
    || !isNonEmptyString(body.server_application_key_fingerprint) || !isNonEmptyString(body.server_application_public_key)
    || !isStringArray(body.lan_endpoints) || !isStringArray(body.tunnel_endpoints) || !isStringArray(body.tls_spki_pins)
    || !isNonEmptyString(body.pairing_id) || !isNonEmptyString(body.one_time_secret)
    || !isNonEmptyString(body.expires_at) || !Number.isFinite(Date.parse(body.expires_at))) {
    throw new Error("配对服务返回了无效二维码数据");
  }
  return body as unknown as MobilePairingOffer;
}

export function parsePairingStatus(payload: unknown): PendingClaim | null {
  const body = recordValue(payload);
  if (!body || !isNonEmptyString(body.pairing_id) || !isNonEmptyString(body.status)) throw new Error("配对服务返回了无效状态");
  if (body.status === "waiting_for_phone") return null;
  if (body.status !== "waiting_for_desktop_confirmation" || !isNonEmptyString(body.device_name)
    || typeof body.confirmation_code !== "string" || !/^\d{6}$/.test(body.confirmation_code)
    || !isStringArray(body.capabilities)) throw new Error("配对服务返回了无效设备确认信息");
  return body as unknown as PendingClaim;
}

export function parsePairedDevice(payload: unknown): PairedDevice {
  const body = recordValue(payload);
  if (!body || !isNonEmptyString(body.device_id) || !isNonEmptyString(body.display_name)) throw new Error("配对服务返回了无效设备信息");
  return body as unknown as PairedDevice;
}

export function pairingErrorMessage(error: unknown) {
  if (error instanceof SyntaxError) return "配对服务返回了无效 JSON";
  return error instanceof Error ? error.message : String(error);
}

function recordValue(value: unknown): Record<string, unknown> | undefined {
  if (!value || typeof value !== "object" || Array.isArray(value)) return undefined;
  return value as Record<string, unknown>;
}

function isNonEmptyString(value: unknown): value is string { return typeof value === "string" && value.length > 0 }
function isStringArray(value: unknown): value is string[] { return Array.isArray(value) && value.every(isNonEmptyString) }
