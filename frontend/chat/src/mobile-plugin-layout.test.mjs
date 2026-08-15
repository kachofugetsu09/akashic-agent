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
const themeStyles = await readFile(
  new URL("./theme.css", import.meta.url),
  "utf8",
);
const desktopStyles = await readFile(
  new URL("./styles.css", import.meta.url),
  "utf8",
);
const dashboardStyles = await readFile(
  new URL("../../dashboard/src/styles.css", import.meta.url),
  "utf8",
);
const desktopSource = await readFile(
  new URL("./desktop-chat-view.tsx", import.meta.url),
  "utf8",
);
const desktopAppSource = await readFile(
  new URL("./desktop-chat-app.tsx", import.meta.url),
  "utf8",
);
const desktopControllerSource = await readFile(
  new URL("./use-desktop-chat-controller.ts", import.meta.url),
  "utf8",
);
const desktopConversationSource = await readFile(
  new URL("./desktop-conversation.tsx", import.meta.url),
  "utf8",
);
const desktopSidebarSource = await readFile(
  new URL("./desktop-sidebar.tsx", import.meta.url),
  "utf8",
);
const mobileSource = await readFile(
  new URL("./mobile-native.tsx", import.meta.url),
  "utf8",
);
const pluginRuntimeSource = await readFile(
  new URL("./mobile-plugin-runtime.tsx", import.meta.url),
  "utf8",
);
const sharedMessageSource = await readFile(
  new URL("./message-view.tsx", import.meta.url),
  "utf8",
);
const navigationSource = await readFile(
  new URL("./conversation-navigation.tsx", import.meta.url),
  "utf8",
);
const navigationStyles = await readFile(
  new URL("./conversation-navigation.css", import.meta.url),
  "utf8",
);
const runtimeDashboardSource = await readFile(
  new URL("./runtime-dashboard.tsx", import.meta.url),
  "utf8",
);
const runtimeDashboardStyles = await readFile(
  new URL("./runtime-dashboard.css", import.meta.url),
  "utf8",
);

test("process plugin slots align with thinking and tool content", () => {
  assert.match(
    sharedStyles,
    /\.process-item\s*\{[\s\S]*?grid-template-columns:\s*var\(--process-rail-width\) minmax\(0, 1fr\);[\s\S]*?column-gap:\s*12px;/,
  );
  assert.match(
    sharedStyles,
    /\.mobile-plugin-slot\[data-slot="turn\.before_reasoning"\],[\s\S]*?margin-inline-start:\s*30px;/,
  );
  assert.doesNotMatch(platformStyles, /\.mobile-plugin-slot\[data-slot="turn\.before_reasoning"\]/);
  assert.match(
    sharedStyles,
    /\.process-line\s*\{[\s\S]*?bottom:\s*0;[\s\S]*?left:\s*var\(--process-content-inset\);[\s\S]*?width:\s*var\(--process-rail-width\);/,
  );
  assert.match(
    sharedStyles,
    /\.process-line::before\s*\{[^}]*top:\s*0;[^}]*bottom:\s*0;[^}]*width:\s*1px;/,
  );
  assert.doesNotMatch(sharedMessageSource, /ResizeObserver/);
  assert.doesNotMatch(sharedStyles, /transition:\s*height/);
  assert.match(
    sharedStyles,
    /\.process-node\.diamond\s*\{[^}]*width:\s*8px;[^}]*height:\s*8px;/,
  );
});

test("streaming thinking uses the shared Streamdown renderer", () => {
  assert.match(
    sharedMessageSource,
    /function ThinkingStep[\s\S]*?<LazyMessageResponse isAnimating=\{active\}>\{block\.content\}<\/LazyMessageResponse>/,
  );
  assert.match(
    sharedStyles,
    /\.process-markdown\s*\{[^}]*white-space:\s*normal;/,
  );
  assert.match(
    sharedStyles,
    /\.process-markdown-fallback\s*\{[^}]*white-space:\s*pre-wrap;/,
  );
});

test("desktop shares plugin shell slots without exposing mobile dashboards", () => {
  assert.match(desktopControllerSource, /import \{ loadWebPluginCatalog \} from "\.\/mobile-plugin-runtime";/);
  assert.match(desktopConversationSource, /import \{ MobilePluginSlot \} from "\.\/mobile-plugin-runtime";/);
  assert.match(desktopConversationSource, /name="turn\.before_reasoning"/);
  assert.match(desktopConversationSource, /name="turn\.before_tool"/);
  assert.match(desktopConversationSource, /name="turn\.after_answer"/);
  assert.doesNotMatch(desktopControllerSource, /MobilePluginDashboard|useMobilePluginDashboards/);
  assert.doesNotMatch(desktopConversationSource, /MobilePluginDashboard|useMobilePluginDashboards/);
  assert.match(pluginRuntimeSource, /fetch\("\/api\/chat\/plugin-ui\/catalog"/);
  assert.match(pluginRuntimeSource, /fetch\("\/api\/chat\/plugin-ui\/query"/);
  assert.match(pluginRuntimeSource, /slot === "dashboard\.main"/);
});

test("desktop and mobile keep one shared conversation owner", () => {
  assert.match(sharedStyles, /\.tool-step-disclosure\s*\{/);
  assert.match(sharedStyles, /\.message-reply-reference\s*\{/);
  assert.match(sharedStyles, /\.agent-content ul\s*\{[\s\S]*?list-style:\s*disc;/);
  assert.match(sharedStyles, /\.agent-content ol\s*\{[\s\S]*?list-style:\s*decimal;/);
  assert.doesNotMatch(platformStyles, /\.tool-step-disclosure\s*\{/);
  assert.doesNotMatch(platformStyles, /\.agent-content (?:ul|ol)\s*\{/);
  assert.doesNotMatch(desktopStyles, /\.tool-step-disclosure\s*\{/);
  assert.match(desktopAppSource, /import "\.\/message-view\.css";/);
  assert.match(mobileSource, /import "\.\/message-view\.css";/);
  assert.match(desktopSidebarSource, /<ConversationNavigation/);
  assert.match(mobileSource, /<ConversationNavigation/);
  assert.match(desktopConversationSource, /<SharedMessageActions/);
  assert.match(mobileSource, /<SharedMessageActions/);
  assert.doesNotMatch(mobileSource, /mobile-message-actions/);
  assert.doesNotMatch(mobileSource, /SwipeToReply|useMotionValue/);
});

test("shared navigation keeps the compact mobile drawer language", () => {
  assert.doesNotMatch(navigationSource, /对话与知识/);
  assert.doesNotMatch(navigationSource, />Akashic</);
  assert.match(navigationSource, /conversation-navigation__heading">会话/);
  assert.match(navigationSource, /featuredDestinations/);
  assert.match(mobileSource, /label: "知识与运行",[\s\S]*?featured: true,/);
  assert.match(
    navigationStyles,
    /\.conversation-destination__icon\s*\{[^}]*width:\s*24px;[^}]*background:\s*transparent;/,
  );
  assert.match(
    navigationStyles,
    /\.conversation-navigation__action\.primary\s*\{[^}]*width:\s*fit-content;[^}]*border-radius:\s*var\(--md-sys-shape-corner-full\);/,
  );
  assert.match(
    navigationStyles,
    /\.conversation-session-list\s*\{[^}]*min-height:\s*0;[^}]*flex:\s*1;[^}]*grid-auto-rows:\s*min-content;[^}]*align-content:\s*start;[^}]*overflow-y:\s*auto;/,
  );
  assert.match(
    navigationSource,
    /<section className="conversation-navigation__sessions">[\s\S]*?<\/section>[\s\S]*?conversation-navigation__auxiliary/,
  );
  assert.match(
    navigationStyles,
    /\.conversation-navigation__auxiliary\s*\{[^}]*position:\s*relative;[^}]*height:\s*80px;[^}]*flex:\s*0 0 80px;/,
  );
  assert.match(
    sharedStyles,
    /\.mobile-plugin-slot\[data-slot="drawer\.panel"\]\s*\{[^}]*position:\s*absolute;[^}]*inset-block-end:\s*0;[^}]*background:\s*var\(--ak-color-bg-canvas\);/,
  );
  assert.match(
    navigationStyles,
    /\.conversation-destination\.featured\s*\{[^}]*min-height:\s*68px;[^}]*border-radius:\s*22px;[^}]*background:\s*var\(--ak-color-action-primary\);[^}]*box-shadow:\s*none;/,
  );
  assert.match(
    platformStyles,
    /\.mobile-drawer\s*\{[^}]*box-shadow:\s*4px 0 12px rgb\(var\(--md-sys-color-shadow-rgb\) \/ 0\.16\);/,
  );
});

test("Material shadow roles always declare opacity at use sites", () => {
  for (const styles of [themeStyles, platformStyles, desktopStyles, navigationStyles, dashboardStyles]) {
    assert.doesNotMatch(styles, /var\(--ak-color-shadow\)/);
  }
});

test("runtime metrics keep values and categories on one compact baseline", () => {
  assert.match(
    runtimeDashboardSource,
    /<div><small>\{label\}<\/small><strong>\{value\}<\/strong><\/div>/,
  );
  assert.match(
    runtimeDashboardStyles,
    /\.runtime-metric div\s*\{[^}]*display:\s*flex;[^}]*align-items:\s*baseline;/,
  );
  assert.match(
    runtimeDashboardStyles,
    /\.runtime-metric strong\s*\{[^}]*order:\s*-1;/,
  );
});

test("mobile attachment previews preserve intrinsic aspect ratios", () => {
  assert.match(
    platformStyles,
    /\.message-attachment-preview\s*\{[^}]*display:\s*grid;[^}]*min-block-size:\s*44px;[^}]*place-items:\s*center;/,
  );
  assert.match(
    platformStyles,
    /\.message-attachment-preview img\s*\{[^}]*width:\s*auto;[^}]*max-width:\s*100%;[^}]*height:\s*auto;[^}]*max-height:\s*260px;[^}]*object-fit:\s*contain;/,
  );
  assert.doesNotMatch(
    platformStyles,
    /\.message-attachment-preview img\s*\{[^}]*(?:max-block-size|min\(40vh)/,
  );
});

test("mobile scroll control is anchored outside the virtual scroll plane", () => {
  assert.match(
    mobileSource,
    /<div className="mobile-conversation-frame">[\s\S]*?<div ref=\{scrollRef\} className="mobile-conversation mobile-virtual-conversation"[\s\S]*?<\/div>\s*<MobileScrollButton/,
  );
  assert.match(
    platformStyles,
    /\.mobile-conversation-frame\s*\{[^}]*position:\s*relative;[^}]*flex:\s*1;[^}]*overflow:\s*hidden;/,
  );
  assert.match(
    platformStyles,
    /\.mobile-conversation\s*\{[^}]*height:\s*100%;[^}]*overflow-y:\s*auto;/,
  );
});

test("same-role divider stays centered in the inter-message gap", () => {
  assert.match(
    platformStyles,
    /\.mobile-role-divider\s*\{[^}]*height:\s*1px;[^}]*margin-block:\s*-14px 13px;/,
  );
  assert.doesNotMatch(platformStyles, /\.mobile-role-divider\s*\{[^}]*margin-block:\s*-7px;/);
});

test("virtual search highlight waits for its target row to mount", () => {
  assert.match(
    mobileSource,
    /const register = \(\) => \{[\s\S]*attempts < 4[\s\S]*requestAnimationFrame\(register\)/,
  );
});

test("dynamic message measurement stays in the ResizeObserver frame", () => {
  assert.doesNotMatch(mobileSource, /useAnimationFrameWithResizeObserver:\s*true/);
});

test("full native snapshots commit without waiting for another animation frame", () => {
  const receiver = mobileSource.match(/receiveSnapshot\(next\) \{[\s\S]*?\n[ ]{6}\},\n[ ]{6}receiveStreamPatch/);
  assert.ok(receiver, "mobile snapshot receiver must remain discoverable");
  assert.match(receiver[0], /nextSnapshot = parseMobileSnapshot\(next\)/);
  assert.match(receiver[0], /setSnapshot\(nextSnapshot\)/);
  assert.doesNotMatch(receiver[0], /requestAnimationFrame|startTransition/);
});

test("stream patches publish by frame while terminal remains immediate", () => {
  const receiver = mobileSource.match(/receiveStreamPatch\(next\) \{[\s\S]*?\n[ ]{6}\},\n[ ]{6}receiveStatePatch/);
  assert.ok(receiver, "mobile stream receiver must remain discoverable");
  assert.match(receiver[0], /streamSnapshotRef\.current = nextSnapshot/);
  assert.match(receiver[0], /nextMessage\.streaming\) streamStore\.publishFrame\(/);
  assert.match(receiver[0], /else streamStore\.publishImmediate\(/);
  assert.doesNotMatch(receiver[0], /requestAnimationFrame/);
  assert.doesNotMatch(receiver[0], /startTransition/);
});

test("streaming redraws only dynamic message subtrees", () => {
  assert.match(mobileSource, /useSyncExternalStore\(subscribe, getSnapshot, getSnapshot\)/);
  assert.match(mobileSource, /const MessageMeta = React\.memo/);
  assert.match(mobileSource, /blocks: toCachedAgentBlocks\(message\.blocks\)/);
  assert.match(sharedMessageSource, /const MessageBody = memo/);
  assert.match(sharedMessageSource, /const MessageAttachments = memo/);
  assert.match(sharedMessageSource, /const ProcessTrace = memo/);
});

test("user message bubble uses a defined secondary container token", () => {
  assert.match(platformStyles, /\.mobile-plain-message-view\.user[\s\S]*?background:\s*var\(--ak-color-action-soft\)/);
  assert.doesNotMatch(themeStyles, /--m-secondary-container:/);
});

test("fixed mobile chrome stays opaque over the native window", () => {
  assert.match(
    platformStyles,
    /\.mobile-topbar\s*\{[^}]*background:\s*var\(--md-sys-color-surface\);/,
  );
  assert.match(
    platformStyles,
    /\.mobile-composer-zone\s*\{[^}]*background:\s*var\(--md-sys-color-surface\);/,
  );
  assert.doesNotMatch(
    platformStyles,
    /\.(?:mobile-topbar|mobile-composer-zone)\s*\{[^}]*background:\s*color-mix\([^}]*transparent/,
  );
});

test("cached images retry once and degrade to an openable file instead of a blank card", () => {
  assert.match(mobileSource, /\^image\\\/\/i\.test\(attachment\.contentType\.trim\(\)\)/);
  assert.match(mobileSource, /if \(imageRetry === 0\) setImageRetry\(1\);[\s\S]*else setImageUnavailable\(true\);/);
  assert.match(mobileSource, /imageUrl && !imageUnavailable/);
});
