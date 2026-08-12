import { useCallback } from "react";
import { AnimatePresence, motion, useReducedMotion } from "motion/react";
import {
  Check,
  CircleAlert,
  LoaderCircle,
  RefreshCw,
  ScanLine,
  ShieldCheck,
  Smartphone,
} from "lucide-react";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import type { PairedDevice, PairingState } from "./mobile-pairing-data";
import { useMobilePairing } from "./use-mobile-pairing";

interface MobilePairingDialogProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
}

export function MobilePairingDialog({ open, onOpenChange }: MobilePairingDialogProps) {
  const { state, now, start, approve, cancel } = useMobilePairing(open);
  const prefersReducedMotion = useReducedMotion();

  const handleOpenChange = useCallback((nextOpen: boolean) => {
    if (!nextOpen) cancel();
    onOpenChange(nextOpen);
  }, [cancel, onOpenChange]);

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
            initial={prefersReducedMotion ? false : { opacity: 0, y: 10, filter: "blur(4px)" }}
            animate={{ opacity: 1, y: 0, filter: "blur(0px)" }}
            exit={prefersReducedMotion ? undefined : { opacity: 0, y: -6, filter: "blur(4px)" }}
            transition={prefersReducedMotion ? { duration: 0 } : { type: "spring", duration: 0.3, bounce: 0 }}
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
              {state.approving
                ? <LoaderCircle aria-hidden="true" className="mobile-pairing-spinner" size={18} />
                : <ShieldCheck aria-hidden="true" size={18} />}
              {state.approving ? "正在批准" : "确认并连接"}
            </button>
          ) : null}
          {state.stage === "error" ? (
            <button className="mobile-pairing-button primary" type="button" onClick={() => void start()}>
              <RefreshCw aria-hidden="true" size={18} />
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
      <LoaderCircle aria-hidden="true" className="mobile-pairing-spinner" size={28} />
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
          <span className="mobile-pairing-step-node"><ScanLine aria-hidden="true" size={16} /></span>
          <div><strong>用手机扫描二维码</strong><span>打开 Akashic Android 客户端并选择“扫描电脑”。</span></div>
        </div>
        <div className="mobile-pairing-step">
          <span className="mobile-pairing-step-node"><ShieldCheck aria-hidden="true" size={16} /></span>
          <div><strong>核对确认码</strong><span>手机完成安全验证后，这里会显示 6 位数字。</span></div>
        </div>
        <p className="mobile-pairing-countdown" role="timer">二维码将在 <strong>{formatCountdown(secondsLeft)}</strong> 后失效</p>
      </div>
    </div>
  );
}

function ConfirmingStage({ state }: { state: Extract<PairingState, { stage: "confirming" }> }) {
  return (
    <div className="mobile-pairing-confirm">
      <div className="mobile-pairing-device-row">
        <span className="mobile-pairing-device-icon"><Smartphone aria-hidden="true" size={21} /></span>
        <div><span>等待连接的设备</span><strong>{state.claim.device_name}</strong></div>
      </div>
      <p className="mobile-pairing-code-label">确认手机显示的是同一组数字</p>
      <div className="mobile-pairing-code" aria-label={`确认码 ${state.claim.confirmation_code}`}>
        {state.claim.confirmation_code.split("").map((digit, index) => (
          <span key={`${index}-${digit}`}>{digit}</span>
        ))}
      </div>
      <p className="mobile-pairing-security-note"><ShieldCheck aria-hidden="true" size={17} />数字不一致时不要批准，关闭窗口并重新扫码。</p>
    </div>
  );
}

function ConnectedStage({ device }: { device: PairedDevice }) {
  return (
    <div className="mobile-pairing-centered success" role="status">
      <span className="mobile-pairing-result-icon"><Check aria-hidden="true" size={28} strokeWidth={2.4} /></span>
      <h3>手机已连接</h3>
      <p>{device.display_name} 已保存设备身份，之后无需重复扫码。</p>
    </div>
  );
}

function ErrorStage({ message }: { message: string }) {
  return (
    <div className="mobile-pairing-centered error" role="alert">
      <span className="mobile-pairing-result-icon"><CircleAlert aria-hidden="true" size={27} /></span>
      <h3>连接没有完成</h3>
      <p>{message}</p>
    </div>
  );
}

function secondsUntil(expiresAt: string, now: number): number {
  return Math.max(0, Math.ceil((Date.parse(expiresAt) - now) / 1_000));
}

function formatCountdown(seconds: number): string {
  return `${Math.floor(seconds / 60)}:${String(seconds % 60).padStart(2, "0")}`;
}
