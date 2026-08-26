window.AkashicNativeTransport = {
  postMessage(message: string): void {
    window.parent.postMessage({ type: "akashic.mobile-lab.bridge", payload: message }, window.location.origin);
  },
};

void import("./mobile-native");
