import { useCallback, useEffect, useRef, useState } from "react";
import { toDataURL } from "qrcode";
import {
  approvePairing, createPairingOffer, loadPairingClaim, pairingErrorMessage,
  type PairingState,
} from "./mobile-pairing-data";

const POLL_INTERVAL_MS = 1_250;

/** Own the cancellable pairing lifecycle independently from dialog presentation. */
export function useMobilePairing(open: boolean) {
  const [state, setState] = useState<PairingState>({ stage: "creating" });
  const [now, setNow] = useState(0);
  const generationRef = useRef(0);
  const actionRef = useRef<AbortController | null>(null);

  const cancel = useCallback(() => {
    generationRef.current += 1;
    actionRef.current?.abort();
    actionRef.current = null;
  }, []);

  const start = useCallback(async () => {
    cancel();
    const generation = generationRef.current;
    const controller = new AbortController();
    actionRef.current = controller;
    setState({ stage: "creating" });
    try {
      const offer = await createPairingOffer(controller.signal);
      const qrDataUrl = await toDataURL(JSON.stringify(offer), {
        errorCorrectionLevel: "M", margin: 4, width: 480,
        color: { dark: "#111111ff", light: "#ffffffff" },
      });
      if (generationRef.current !== generation) return;
      setNow(Date.now());
      setState({ stage: "waiting", offer, qrDataUrl });
    } catch (error) {
      if (controller.signal.aborted || generationRef.current !== generation) return;
      setState({ stage: "error", message: pairingErrorMessage(error) });
    } finally {
      if (actionRef.current === controller) actionRef.current = null;
    }
  }, [cancel]);

  useEffect(() => {
    if (!open) return cancel;
    void start();
    return cancel;
  }, [cancel, open, start]);

  useEffect(() => {
    if (!open || state.stage !== "waiting") return;
    const controller = new AbortController();
    let timeoutId = 0;
    const poll = async () => {
      try {
        const claim = await loadPairingClaim(state.offer.pairing_id, controller.signal);
        if (claim) setState({ stage: "confirming", offer: state.offer, claim, approving: false });
        else timeoutId = window.setTimeout(() => void poll(), POLL_INTERVAL_MS);
      } catch (error) {
        if (!controller.signal.aborted) setState({ stage: "error", message: pairingErrorMessage(error) });
      }
    };
    timeoutId = window.setTimeout(() => void poll(), POLL_INTERVAL_MS);
    return () => { controller.abort(); window.clearTimeout(timeoutId); };
  }, [open, state]);

  useEffect(() => {
    if (!open || state.stage !== "waiting") return;
    const expiresAt = Date.parse(state.offer.expires_at);
    const intervalId = window.setInterval(() => {
      const current = Date.now();
      setNow(current);
      if (current >= expiresAt) setState({ stage: "error", message: "二维码已过期，请生成新的二维码" });
    }, 1_000);
    return () => window.clearInterval(intervalId);
  }, [open, state]);

  const approve = useCallback(async () => {
    if (state.stage !== "confirming" || state.approving || actionRef.current) return;
    const controller = new AbortController();
    actionRef.current = controller;
    const { offer, claim } = state;
    setState({ stage: "confirming", offer, claim, approving: true });
    try {
      setState({ stage: "connected", device: await approvePairing(offer, claim, controller.signal) });
    } catch (error) {
      if (!controller.signal.aborted) setState({ stage: "error", message: pairingErrorMessage(error) });
    } finally {
      if (actionRef.current === controller) actionRef.current = null;
    }
  }, [state]);

  return { state, now, start, approve, cancel };
}
