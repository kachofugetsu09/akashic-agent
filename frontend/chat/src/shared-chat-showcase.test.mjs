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
const messageElementSource = await readFile(
  new URL("./components/ai-elements/message.tsx", import.meta.url),
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

test("full and deferred user messages share the semantic container colors", () => {
  assert.match(messageViewCss, /\.user-bubble\s*\{[^}]*color:\s*var\(--ak-color-on-action-container\);/s);
  assert.match(messageViewCss, /\.user-bubble\s*\{[^}]*background:\s*var\(--ak-color-action-container\);/s);
  assert.doesNotMatch(messageElementSource, /group-\[\.is-user\]:bg-/);
  assert.doesNotMatch(messageElementSource, /group-\[\.is-user\]:text-/);
});

test("process motion avoids per-delta layout animation on desktop and mobile", () => {
  assert.match(messageViewCss, /\.process-line\s*\{[^}]*bottom: 0;/s);
  assert.doesNotMatch(messageViewCss, /transition: height/);
  assert.match(messageViewCss, /\.process-flow::after\s*\{/);
  assert.match(messageViewCss, /animation: shared-trace-flow 1\.8s/);
  assert.match(messageViewCss, /animation: shared-trace-node-arrive 180ms/);
  assert.match(messageViewCss, /animation: shared-trace-core 1\.8s/);
  assert.doesNotMatch(messageViewCss, /shared-trace-echo/);
  assert.doesNotMatch(messageViewSource, /ResizeObserver/);
  assert.doesNotMatch(messageViewSource, /style\.height/);
  assert.match(messageViewSource, /const frontierItem = processItems\[Math\.max\(0, activeItemIndex - 1\)\]/);
  assert.match(messageViewSource, /flow\.style\.bottom =/);
  assert.match(messageViewSource, /flow\.dataset\.active = "true"/);
  assert.match(mobileNativeSource, /import "\.\/message-view\.css";/);

  const reducedMotionBlock = messageViewCss.slice(
    messageViewCss.indexOf("@media (prefers-reduced-motion: reduce)"),
  );
  assert.match(reducedMotionBlock, /\.process-flow::after,/);
  assert.match(reducedMotionBlock, /\.process-item\.active \.process-node/);
});
