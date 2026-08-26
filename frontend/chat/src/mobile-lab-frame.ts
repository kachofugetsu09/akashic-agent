import { installMobileBridge } from "./mobile-bridge";
import { startMobileNativeApp } from "./mobile-native-mount";

window.AkashicNativeTransport = {
  postMessage(message: string): void {
    window.parent.postMessage({ type: "akashic.mobile-lab.bridge", payload: message }, window.location.origin);
  },
};

startMobileNativeApp(installMobileBridge);
