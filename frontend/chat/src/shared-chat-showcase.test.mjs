import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

const showcaseSource = await readFile(
  new URL("./shared-chat-showcase.tsx", import.meta.url),
  "utf8",
);
const messageViewCss = await readFile(
  new URL("./message-view.css", import.meta.url),
  "utf8",
);
const messageViewSource = await readFile(
  new URL("./message-view.tsx", import.meta.url),
  "utf8",
);
const mobileNativeSource = await readFile(
  new URL("./mobile-native.tsx", import.meta.url),
  "utf8",
);

test("offline showcase preserves the P0 boundary fixtures", () => {
  const gifMatch = showcaseSource.match(
    /const PREVIEW_GIF_URL = "data:image\/gif;base64,([^"]+)";/,
  );
  assert.ok(gifMatch, "showcase must embed an offline GIF fixture");

  const gif = Buffer.from(gifMatch[1], "base64");
  assert.equal(gif.subarray(0, 6).toString("ascii"), "GIF89a");
  assert.equal(gif.includes(Buffer.from("NETSCAPE2.0")), true);
  assert.match(showcaseSource, /https:\/\/preview\.akashic\.local\/validation\/shared-webui/);
  assert.match(showcaseSource, /<ChatMessageView message=\{message\} \/>/);
  assert.match(showcaseSource, /mediaType: "image\/gif"/);
  assert.match(showcaseSource, /data-preview-state=/);
  assert.match(showcaseSource, /checksum: "sha256:[0-9a-f]{64}"/);
});

test("reduced motion disables the process trigger transition", () => {
  const reducedMotionBlock = messageViewCss.slice(
    messageViewCss.indexOf("@media (prefers-reduced-motion: reduce)"),
  );
  assert.match(reducedMotionBlock, /\.process-trigger,/);
  assert.match(reducedMotionBlock, /transition-duration: 0ms;/);
});

test("growing flow and echo motion is shared by desktop and mobile process traces", () => {
  assert.match(messageViewCss, /\.process-line\s*\{[^}]*transition: height 420ms/s);
  assert.match(messageViewCss, /\.process-line::after\s*\{/);
  assert.match(messageViewCss, /animation: shared-trace-flow 1\.8s/);
  assert.match(messageViewCss, /animation: shared-trace-node-arrive 180ms/);
  assert.match(messageViewCss, /animation: shared-trace-core 1\.8s/);
  assert.match(messageViewCss, /animation: shared-trace-echo 1\.8s/);
  assert.match(messageViewSource, /new ResizeObserver\(scheduleLineHeight\)/);
  assert.match(messageViewSource, /line\.style\.height = `\$\{nextHeight\}px`/);
  assert.match(mobileNativeSource, /import "\.\/message-view\.css";/);

  const reducedMotionBlock = messageViewCss.slice(
    messageViewCss.indexOf("@media (prefers-reduced-motion: reduce)"),
  );
  assert.match(reducedMotionBlock, /\.process-line::after,/);
  assert.match(reducedMotionBlock, /\.process-line\s*\{\s*transition-duration: 0ms;/);
  assert.match(reducedMotionBlock, /\.process-item\.active \.process-node::after/);
});
