import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

const platformStyles = await readFile(
  new URL("./mobile-native.css", import.meta.url),
  "utf8",
);
const sharedStyles = await readFile(
  new URL("./message-view.css", import.meta.url),
  "utf8",
);
const desktopStyles = await readFile(
  new URL("./styles.css", import.meta.url),
  "utf8",
);
const desktopSource = await readFile(
  new URL("./main.tsx", import.meta.url),
  "utf8",
);
const mobileSource = await readFile(
  new URL("./mobile-native.tsx", import.meta.url),
  "utf8",
);

test("process plugin slots align with thinking and tool content", () => {
  assert.match(
    sharedStyles,
    /\.process-item\s*\{[\s\S]*?grid-template-columns:\s*18px minmax\(0, 1fr\);[\s\S]*?column-gap:\s*12px;/,
  );
  assert.match(
    platformStyles,
    /\.mobile-plugin-slot\[data-slot="turn\.before_reasoning"\],[\s\S]*?margin-inline-start:\s*30px;/,
  );
});

test("desktop and mobile keep one shared conversation owner", () => {
  assert.match(sharedStyles, /\.tool-step-disclosure\s*\{/);
  assert.match(sharedStyles, /\.message-reply-reference\s*\{/);
  assert.match(sharedStyles, /\.agent-content ul\s*\{[\s\S]*?list-style:\s*disc;/);
  assert.match(sharedStyles, /\.agent-content ol\s*\{[\s\S]*?list-style:\s*decimal;/);
  assert.doesNotMatch(platformStyles, /\.tool-step-disclosure\s*\{/);
  assert.doesNotMatch(platformStyles, /\.agent-content (?:ul|ol)\s*\{/);
  assert.doesNotMatch(desktopStyles, /\.tool-step-disclosure\s*\{/);
  assert.match(desktopSource, /import "\.\/message-view\.css";/);
  assert.match(mobileSource, /import "\.\/message-view\.css";/);
  assert.match(desktopSource, /<ConversationNavigation/);
  assert.match(mobileSource, /<ConversationNavigation/);
  assert.match(desktopSource, /<SharedMessageActions/);
  assert.match(mobileSource, /<SharedMessageActions/);
  assert.doesNotMatch(mobileSource, /mobile-message-actions/);
  assert.doesNotMatch(mobileSource, /SwipeToReply|useMotionValue/);
});

test("mobile attachment previews preserve intrinsic aspect ratios", () => {
  assert.match(
    platformStyles,
    /\.message-attachment-preview\s*\{[^}]*display:\s*grid;[^}]*min-block-size:\s*44px;[^}]*place-items:\s*center;/,
  );
  assert.match(
    platformStyles,
    /\.message-attachment-preview img\s*\{[^}]*inline-size:\s*auto;[^}]*max-inline-size:\s*100%;[^}]*block-size:\s*auto;[^}]*object-fit:\s*contain;/,
  );
  assert.doesNotMatch(
    platformStyles,
    /\.message-attachment-preview img\s*\{[^}]*\bwidth:\s*100%;/,
  );
});
