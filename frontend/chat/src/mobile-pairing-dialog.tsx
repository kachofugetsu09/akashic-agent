import { useCallback, useEffect, useRef, useState } from "react";
import { AnimatePresence, motion } from "motion/react";
import {
  Check,
  CircleAlert,
  LoaderCircle,
  RefreshCw,
  ScanLine,
  ShieldCheck,
  Smartphone,
} from "lucide-react";
import { toDataURL } from "qrcode";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";

const POLL_INTERVAL_MS = 1_250;

interface MobilePairingOffer {
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

interface PendingClaim {
  pairing_id: string;
  status: "waiting_for_desktop_confirmation";
  device_name: string;
  confirmation_code: string;
  capabilities: string[];
}

interface PairedDevice {
  device_id: string;
  display_name: string;
}

type PairingState =
  | { stage: "creating" }
  | { stage: "waiting"; offer: MobilePairingOffer; qrDataUrl: string }
  | { stage: "confirming"; offer: MobilePairingOffer; claim: PendingClaim; approving: boolean }
  | { stage: "connected"; device: PairedDevice }
  | { stage: "error"; message: string };

interface MobilePairingDialogProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
}

export function MobilePairingDialog({ open, onOpenChange }: MobilePairingDialogProps) {
  const [state, setState] = useState<PairingState>({ stage: "creating" });
  const [now, setNow] = useState(0);
  const requestGenerationRef = useRef(0);

  const startPairing = useCallback(async () => {
    const generation = ++requestGenerationRef.current;
    setState({ stage: "creating" });
    try {
      const offer = parsePairingOffer(await requestJson("/api/chat/mobile-pairing", { method: "POST" }));
      const qrDataUrl = await toDataURL(JSON.stringify(offer), {
        errorCorrectionLevel: "M",
        margin: 4,
        width: 480,
        color: { dark: "#111111ff", light: "#ffffffff" },
      });
      if (requestGenerationRef.current !== generation) return;
      setNow(Date.now());
      setState({ stage: "waiting", offer, qrDataUrl });
    } catch (error) {
      if (requestGenerationRef.current !== generation) return;
      setState({ stage: "error", message: pairingErrorMessage(error) });
    }
  }, []);

  useEffect(() => {
    if (!open) {
      requestGenerationRef.current += 1;
      return;
    }
    void startPairing();
  }, [open, startPairing]);

  useEffect(() => {
    if (!open || state.stage !== "waiting") return;
    const offer = state.offer;
    let active = true;
    let timeoutId = 0;

    const poll = async () => {
      try {
        const payload = await requestJson(`/api/chat/mobile-pairing/${encodeURIComponent(offer.pairing_id)}`);
        if (!active) return;
        const claim = parsePairingStatus(payload);
        if (claim !== null) {
          setState({ stage: "confirming", offer, claim, approving: false });
          return;
        }
        timeoutId = window.setTimeout(() => void poll(), POLL_INTERVAL_MS);
      } catch (error) {
        if (!active) return;
        setState({ stage: "error", message: pairingErrorMessage(error) });
      }
    };

    timeoutId = window.setTimeout(() => void poll(), POLL_INTERVAL_MS);
    return () => {
      active = false;
      window.clearTimeout(timeoutId);
    };
  }, [open, state]);

  useEffect(() => {
    if (!open || state.stage !== "waiting") return;
    const expiresAt = Date.parse(state.offer.expires_at);
    const intervalId = window.setInterval(() => {
      const current = Date.now();
      setNow(current);
      if (current >= expiresAt) {
        setState({ stage: "error", message: "二维码已过期，请生成新的二维码" });
      }
    }, 1_000);
    return () => window.clearInterval(intervalId);
  }, [open, state]);

  const approve = useCallback(async () => {
    if (state.stage !== "confirming" || state.approving) return;
    const { offer, claim } = state;
    setState({ stage: "confirming", offer, claim, approving: true });
    try {
      const device = parsePairedDevice(await requestJson(
        `/api/chat/mobile-pairing/${encodeURIComponent(offer.pairing_id)}/approve`,
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ confirmation_code: claim.confirmation_code }),
        },
      ));
      setState({ stage: "connected", device });
    } catch (error) {
      setState({ stage: "error", message: pairingErrorMessage(error) });
    }
  }, [state]);

  const handleOpenChange = useCallback((nextOpen: boolean) => {
    if (!nextOpen) requestGenerationRef.current += 1;
    onOpenChange(nextOpen);
  }, [onOpenChange]);

  return (
    <Dialog open={open} onOpenChange={handleOpenChange}>
      <DialogContent className="mobile-pairing-dialog">
        <DialogHeader className="mobile-pairing-header">
          <div className="mobile-pairing-symbol" aria-hidden="true">
            <Smartphone size={24} strokeWidth={1.8} />
          </div>
          <div>
            <DialogTitle>连接 Android 手机</DialogTitle>
            <DialogDescription>扫码后在两端核对确认码，设备密钥会安全保存在手机上。</DialogDescription>
          </div>
        </DialogHeader>

        <AnimatePresence initial={false} mode="wait">
          <motion.div
            key={state.stage}
            className="mobile-pairing-stage"
            initial={{ opacity: 0, y: 10, filter: "blur(4px)" }}
            animate={{ opacity: 1, y: 0, filter: "blur(0px)" }}
            exit={{ opacity: 0, y: -6, filter: "blur(4px)" }}
            transition={{ type: "spring", duration: 0.3, bounce: 0 }}
          >
            {state.stage === "creating" ? <CreatingStage /> : null}
            {state.stage === "waiting" ? (
              <WaitingStage state={state} secondsLeft={secondsUntil(state.offer.expires_at, now)} />
            ) : null}
            {state.stage === "confirming" ? <ConfirmingStage state={state} /> : null}
            {state.stage === "connected" ? <ConnectedStage device={state.device} /> : null}
            {state.stage === "error" ? <ErrorStage message={state.message} /> : null}
          </motion.div>
        </AnimatePresence>

        <DialogFooter className="mobile-pairing-footer">
          {state.stage === "confirming" ? (
            <button
              className="mobile-pairing-button primary"
              type="button"
              disabled={state.approving}
              onClick={() => void approve()}
            >
              {state.approving ? <LoaderCircle className="mobile-pairing-spinner" size={18} /> : <ShieldCheck size={18} />}
              {state.approving ? "正在批准" : "确认并连接"}
            </button>
          ) : null}
          {state.stage === "error" ? (
            <button className="mobile-pairing-button primary" type="button" onClick={() => void startPairing()}>
              <RefreshCw size={18} />
              重新生成
            </button>
          ) : null}
          {state.stage === "connected" ? (
            <button className="mobile-pairing-button primary" type="button" onClick={() => handleOpenChange(false)}>
              完成
            </button>
          ) : null}
          {state.stage !== "connected" ? (
            <button className="mobile-pairing-button quiet" type="button" onClick={() => handleOpenChange(false)}>
              取消
            </button>
          ) : null}
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}

function CreatingStage() {
  return (
    <div className="mobile-pairing-centered" role="status">
      <LoaderCircle className="mobile-pairing-spinner" size={28} />
      <p>正在创建一次性连接…</p>
    </div>
  );
}

function WaitingStage({ state, secondsLeft }: { state: Extract<PairingState, { stage: "waiting" }>; secondsLeft: number }) {
  return (
    <div className="mobile-pairing-waiting">
      <div className="mobile-pairing-qr-wrap">
        <img className="mobile-pairing-qr" src={state.qrDataUrl} alt="Android 手机配对二维码" />
      </div>
      <div className="mobile-pairing-instructions">
        <div className="mobile-pairing-step active">
          <span className="mobile-pairing-step-node"><ScanLine size={16} /></span>
          <div><strong>用手机扫描二维码</strong><span>打开 Akashic Android 客户端并选择“扫描电脑”。</span></div>
        </div>
        <div className="mobile-pairing-step">
          <span className="mobile-pairing-step-node"><ShieldCheck size={16} /></span>
          <div><strong>核对确认码</strong><span>手机完成安全验证后，这里会显示 6 位数字。</span></div>
        </div>
        <p className="mobile-pairing-countdown" aria-live="polite">二维码将在 <strong>{formatCountdown(secondsLeft)}</strong> 后失效</p>
      </div>
    </div>
  );
}

function ConfirmingStage({ state }: { state: Extract<PairingState, { stage: "confirming" }> }) {
  return (
    <div className="mobile-pairing-confirm">
      <div className="mobile-pairing-device-row">
        <span className="mobile-pairing-device-icon"><Smartphone size={21} /></span>
        <div><span>等待连接的设备</span><strong>{state.claim.device_name}</strong></div>
      </div>
      <p className="mobile-pairing-code-label">确认手机显示的是同一组数字</p>
      <div className="mobile-pairing-code" aria-label={`确认码 ${state.claim.confirmation_code}`}>
        {state.claim.confirmation_code.split("").map((digit, index) => (
          <span key={`${index}-${digit}`}>{digit}</span>
        ))}
      </div>
      <p className="mobile-pairing-security-note"><ShieldCheck size={17} />数字不一致时不要批准，关闭窗口并重新扫码。</p>
    </div>
  );
}

function ConnectedStage({ device }: { device: PairedDevice }) {
  return (
    <div className="mobile-pairing-centered success" role="status">
      <span className="mobile-pairing-result-icon"><Check size={28} strokeWidth={2.4} /></span>
      <h3>手机已连接</h3>
      <p>{device.display_name} 已保存设备身份，之后无需重复扫码。</p>
    </div>
  );
}

function ErrorStage({ message }: { message: string }) {
  return (
    <div className="mobile-pairing-centered error" role="alert">
      <span className="mobile-pairing-result-icon"><CircleAlert size={27} /></span>
      <h3>连接没有完成</h3>
      <p>{message}</p>
    </div>
  );
}

async function requestJson(url: string, options: RequestInit = {}): Promise<unknown> {
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

function parsePairingOffer(payload: unknown): MobilePairingOffer {
  const body = recordValue(payload);
  if (
    !body
    || body.protocol_version !== 1
    || !isNonEmptyString(body.server_id)
    || !isNonEmptyString(body.server_application_key_fingerprint)
    || !isNonEmptyString(body.server_application_public_key)
    || !isStringArray(body.lan_endpoints)
    || !isStringArray(body.tunnel_endpoints)
    || !isStringArray(body.tls_spki_pins)
    || !isNonEmptyString(body.pairing_id)
    || !isNonEmptyString(body.one_time_secret)
    || !isNonEmptyString(body.expires_at)
    || !Number.isFinite(Date.parse(body.expires_at))
  ) {
    throw new Error("配对服务返回了无效二维码数据");
  }
  return body as unknown as MobilePairingOffer;
}

function parsePairingStatus(payload: unknown): PendingClaim | null {
  const body = recordValue(payload);
  if (!body || !isNonEmptyString(body.pairing_id) || !isNonEmptyString(body.status)) {
    throw new Error("配对服务返回了无效状态");
  }
  if (body.status === "waiting_for_phone") return null;
  if (
    body.status !== "waiting_for_desktop_confirmation"
    || !isNonEmptyString(body.device_name)
    || typeof body.confirmation_code !== "string"
    || !/^\d{6}$/.test(body.confirmation_code)
    || !isStringArray(body.capabilities)
  ) {
    throw new Error("配对服务返回了无效设备确认信息");
  }
  return body as unknown as PendingClaim;
}

function parsePairedDevice(payload: unknown): PairedDevice {
  const body = recordValue(payload);
  if (!body || !isNonEmptyString(body.device_id) || !isNonEmptyString(body.display_name)) {
    throw new Error("配对服务返回了无效设备信息");
  }
  return body as unknown as PairedDevice;
}

function secondsUntil(expiresAt: string, now: number): number {
  return Math.max(0, Math.ceil((Date.parse(expiresAt) - now) / 1_000));
}

function formatCountdown(seconds: number): string {
  return `${Math.floor(seconds / 60)}:${String(seconds % 60).padStart(2, "0")}`;
}

function pairingErrorMessage(error: unknown): string {
  if (error instanceof SyntaxError) return "配对服务返回了无效 JSON";
  return error instanceof Error ? error.message : String(error);
}

function recordValue(value: unknown): Record<string, unknown> | undefined {
  if (!value || typeof value !== "object" || Array.isArray(value)) return undefined;
  return value as Record<string, unknown>;
}

function isNonEmptyString(value: unknown): value is string {
  return typeof value === "string" && value.length > 0;
}

function isStringArray(value: unknown): value is string[] {
  return Array.isArray(value) && value.every(isNonEmptyString);
}
