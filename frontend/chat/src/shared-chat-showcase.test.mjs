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
