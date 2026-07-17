import assert from "node:assert/strict";
import test from "node:test";

import {
  advanceMobileProjectionBaseline,
  advanceMobileUnreadTracking,
  allMobileAttachmentsReady,
  captureMobileComposerDraftWrite,
  flushMobileComposerBeforePairing,
  formatMobileSelectionCopyText,
  isMobileImageViewerHistoryState,
  mobileMessageCanReply,
  mobileComposerDraftHydration,
  mobileComposerDraftMatches,
  MOBILE_COMPOSER_DRAFT_MAX_LENGTH,
  normalizeMobileComposerDraftText,
  reconcileMobileMessageSelection,
  reconcileAssistantMessageIds,
  resolveMobileComposerDraft,
  selectableMobileMessages,
  shouldClearAcceptedMobileComposerDraft,
  updateMobileUnreadMessageIds,
  updateMobileSearchIndex,
} from "./mobile-message-state.ts";

function selectableMessage(id, role, content, streaming = false) {
  return {
    id,
    sessionId: "mobile-current",
    role,
    content,
    streaming,
    replyable: true,
    createdAt: 1_000,
    attachments: [],
  };
}

test("message selection preserves conversation order and excludes streaming turns", () => {
  const first = selectableMessage("first", "user", "第一条");
  const streaming = selectableMessage("streaming", "assistant", "还在生成", true);
  const last = selectableMessage("last", "assistant", "最后一条");

  const selected = selectableMobileMessages(
    [first, streaming, last],
    new Set(["last", "streaming", "first"]),
  );

  assert.deepEqual(selected.map((item) => item.id), ["first", "last"]);
});

test("message selection drops vanished and newly streaming identities", () => {
  const selected = new Set(["kept", "vanished", "became-streaming"]);
  const reconciled = reconcileMobileMessageSelection(selected, [
    selectableMessage("kept", "user", "保留"),
    selectableMessage("became-streaming", "assistant", "变化中", true),
  ]);

  assert.deepEqual([...reconciled], ["kept"]);
});

test("single selection copies exact body and attachment names without transcript labels", () => {
  const source = {
    ...selectableMessage("one", "user", "  正文  "),
    attachments: [{ filename: "report.csv" }],
  };

  assert.equal(
    formatMobileSelectionCopyText([source], () => "不应出现"),
    "正文\n[附件] report.csv",
  );
});

test("multiple selection copies only copyable messages in conversation order", () => {
  const messages = [
    selectableMessage("first", "user", "问题"),
    selectableMessage("empty", "assistant", ""),
    selectableMessage("last", "assistant", "回答"),
  ];

  assert.equal(
    formatMobileSelectionCopyText(messages, () => "昨天 11:02"),
    "你 · 昨天 11:02\n问题\n\nAkashic · 昨天 11:02\n回答",
  );
});

test("reply actions only target messages owned by the selected mobile session", () => {
  const current = selectableMessage("current", "assistant", "当前会话");
  const historical = { ...current, id: "historical", sessionId: "mobile-previous" };

  assert.equal(mobileMessageCanReply(current, "mobile-current"), true);
  assert.equal(mobileMessageCanReply(historical, "mobile-current"), false);
  assert.equal(mobileMessageCanReply({ ...current, replyable: false }, "mobile-current"), false);
});

test("composer waits until every attachment is ready", () => {
  assert.equal(allMobileAttachmentsReady([]), true);
  assert.equal(allMobileAttachmentsReady([{ state: "ready" }, { state: "ready" }]), true);
  assert.equal(allMobileAttachmentsReady([{ state: "ready" }, { state: "failed" }]), false);
  assert.equal(allMobileAttachmentsReady([{ state: "ready" }, { state: "uploading" }]), false);
});

test("session draft restores text and its visible reply target together", () => {
  const target = selectableMessage("answer", "assistant", "回答");

  const resolved = resolveMobileComposerDraft(
    { text: "继续追问", replyToMessageId: target.id },
    [target],
    "mobile-current",
  );

  assert.equal(resolved.text, "继续追问");
  assert.equal(resolved.replyTarget, target);
  assert.equal(resolved.cleanedDraft, undefined);
});

test("missing draft reply stays hidden and asks the owner to preserve only text", () => {
  const resolved = resolveMobileComposerDraft(
    { text: "仍需保留", replyToMessageId: "vanished" },
    [],
    "mobile-current",
  );

  assert.equal(resolved.replyTarget, null);
  assert.deepEqual(resolved.cleanedDraft, { text: "仍需保留" });
});

test("debounced draft write keeps the session captured when it was scheduled", () => {
  const scheduled = captureMobileComposerDraftWrite("mobile-old", "旧会话草稿", "message-old");
  const selectedSessionId = "mobile-new";

  assert.equal(selectedSessionId, "mobile-new");
  assert.deepEqual(scheduled, {
    sessionId: "mobile-old",
    text: "旧会话草稿",
    replyToMessageId: "message-old",
  });
});

test("snapshot owner acknowledgement compares text and reply identity", () => {
  assert.equal(mobileComposerDraftMatches({ text: "" }, { text: "" }), true);
  assert.equal(
    mobileComposerDraftMatches(
      { text: "继续", replyToMessageId: "one" },
      { text: "继续", replyToMessageId: "two" },
    ),
    false,
  );
});

test("composer text is bounded at the native persistence boundary", () => {
  const oversized = "x".repeat(MOBILE_COMPOSER_DRAFT_MAX_LENGTH + 1);
  assert.equal(normalizeMobileComposerDraftText(oversized).length, MOBILE_COMPOSER_DRAFT_MAX_LENGTH);
  assert.equal(
    normalizeMobileComposerDraftText("保留原文"),
    "保留原文",
  );
});

test("optimistic clear wins until the native owner acknowledges it", () => {
  const staleOwner = { text: "已经发送" };
  const optimistic = { sessionId: "mobile-current", text: "" };

  assert.deepEqual(mobileComposerDraftHydration(staleOwner, optimistic), {
    draft: optimistic,
    ownerAcknowledged: false,
  });
  assert.deepEqual(mobileComposerDraftHydration({ text: "" }, optimistic), {
    draft: { text: "" },
    ownerAcknowledged: true,
  });
});

test("pairing restart is enqueued only after the pending draft flush", () => {
  const calls = [];
  flushMobileComposerBeforePairing(
    () => calls.push("flush"),
    () => calls.push("restart"),
  );
  assert.deepEqual(calls, ["flush", "restart"]);
});

test("accepted send clears its inactive session or the unchanged active draft", () => {
  const sent = {
    sessionId: "mobile-current",
    text: "发送内容",
    replyToMessageId: "answer",
  };

  assert.equal(shouldClearAcceptedMobileComposerDraft({ ...sent }, sent), true);
  assert.equal(
    shouldClearAcceptedMobileComposerDraft({ ...sent, text: "发送后新输入" }, sent),
    false,
  );
  assert.equal(
    shouldClearAcceptedMobileComposerDraft({ ...sent, replyToMessageId: undefined }, sent),
    false,
  );
  assert.equal(
    shouldClearAcceptedMobileComposerDraft({ ...sent, sessionId: "mobile-other" }, sent),
    true,
  );
  assert.equal(shouldClearAcceptedMobileComposerDraft(null, sent), true);
});

test("image viewer owns only its matching history entry", () => {
  assert.equal(isMobileImageViewerHistoryState({ akashicImageViewer: "image-1" }, "image-1"), true);
  assert.equal(isMobileImageViewerHistoryState({ akashicImageViewer: "image-2" }, "image-1"), false);
  assert.equal(isMobileImageViewerHistoryState(null, "image-1"), false);
});

function message(id, role, createdAt) {
  return { id, role, createdAt, sessionId: "mobile:test" };
}

test("canonical assistant identity keeps one unread turn and a live anchor", () => {
  const streaming = message("assistant:turn-1", "assistant", 1_000);
  const final = message("mobile:test:2", "assistant", 1_000);

  const result = reconcileAssistantMessageIds(
    new Map([[streaming.id, streaming]]),
    [streaming.id],
    [final],
  );

  assert.deepEqual(result.unseenMessageIds, [final.id]);
  assert.deepEqual(result.newlySeenAssistants, []);
  assert.deepEqual([...result.knownMessages.keys()], [final.id]);
});

test("thinking and tool snapshots do not create another assistant turn", () => {
  const streaming = message("assistant:turn-1", "assistant", 1_000);

  const result = reconcileAssistantMessageIds(
    new Map([[streaming.id, streaming]]),
    [streaming.id],
    [{ ...streaming }],
  );

  assert.deepEqual(result.unseenMessageIds, [streaming.id]);
  assert.deepEqual(result.newlySeenAssistants, []);
});

test("a later assistant turn is reported exactly once", () => {
  const first = message("mobile:test:2", "assistant", 1_000);
  const second = message("assistant:turn-2", "assistant", 2_000);

  const result = reconcileAssistantMessageIds(
    new Map([[first.id, first]]),
    [],
    [first, second],
  );

  assert.deepEqual(result.newlySeenAssistants.map((item) => item.id), [second.id]);
});

test("an escaped scroll lock wins over a transient at-bottom measurement", () => {
  const result = updateMobileUnreadMessageIds(
    [],
    ["assistant:turn-1"],
    { escapedFromLock: true, isAtBottom: true, suspended: false },
  );

  assert.deepEqual(result, ["assistant:turn-1"]);
});

test("returning to bottom under the scroll lock clears unread messages", () => {
  const result = updateMobileUnreadMessageIds(
    ["assistant:turn-1"],
    [],
    { escapedFromLock: false, isAtBottom: true, suspended: false },
  );

  assert.deepEqual(result, []);
});

test("streaming search reuses stable history and only rematches changed revisions", () => {
  const history = { id: "history", content: "旧消息里的 fitbit", searchRevision: 1, attachments: [] };
  const streaming = { id: "streaming", content: "正在", searchRevision: 1, attachments: [] };
  const initial = updateMobileSearchIndex(new Map(), [history, streaming], "fitbit", true);

  const updated = updateMobileSearchIndex(
    initial,
    [history, { ...streaming, content: "正在读取 fitbit", searchRevision: 2 }],
    "fitbit",
    false,
  );

  assert.equal(updated.get("history"), initial.get("history"));
  assert.equal(updated.get("streaming")?.matches, true);
});

test("query changes rematch cached normalized text without rebuilding it", () => {
  const source = { id: "message", content: "Fitbit 睡眠", searchRevision: 1, attachments: [] };
  const initial = updateMobileSearchIndex(new Map(), [source], "fitbit", true);
  const updated = updateMobileSearchIndex(initial, [source], "睡眠", true);

  assert.equal(updated.get("message")?.searchable, initial.get("message")?.searchable);
  assert.equal(updated.get("message")?.matches, true);
});

test("attachment filenames participate in conversation search", () => {
  const source = {
    id: "attachment-message",
    content: "正文不包含查询词",
    searchRevision: 1,
    attachments: [{ filename: "cycle2-file-probe.md" }],
  };

  const result = updateMobileSearchIndex(new Map(), [source], "file-probe", true);

  assert.equal(result.get(source.id)?.matches, true);
});

test("same-session history rebuild establishes a read baseline", () => {
  const history = message("mobile:test:2", "assistant", 1_000);
  const viewport = { escapedFromLock: true, isAtBottom: false, suspended: false };
  const emptied = advanceMobileUnreadTracking(
    new Map([[history.id, history]]),
    [],
    [],
    viewport,
    true,
  );
  const rebuilt = advanceMobileUnreadTracking(
    emptied.knownMessages,
    emptied.unseenMessageIds,
    [history],
    viewport,
    true,
  );

  assert.deepEqual(rebuilt.unseenMessageIds, []);
  assert.deepEqual([...rebuilt.knownMessages.keys()], [history.id]);
});

test("canonical migration preserves the visited unread anchor identity", () => {
  const streaming = message("assistant:turn-1", "assistant", 1_000);
  const final = message("mobile:test:2", "assistant", 1_000);

  const reconciled = reconcileAssistantMessageIds(
    new Map([[streaming.id, streaming]]),
    [streaming.id],
    [final],
  );

  assert.equal(reconciled.messageIdMigrations.get(streaming.id), final.id);
});

test("ordinary reconnect does not reset unread while destructive rebuild does", () => {
  const ordinary = advanceMobileProjectionBaseline(
    { generation: 3, rebuilding: false },
    3,
    true,
  );
  const rebuilding = advanceMobileProjectionBaseline(ordinary.state, 4, true);
  const completed = advanceMobileProjectionBaseline(rebuilding.state, 4, false);
  const stable = advanceMobileProjectionBaseline(completed.state, 4, false);

  assert.equal(ordinary.resetBaseline, false);
  assert.equal(rebuilding.resetBaseline, true);
  assert.equal(completed.resetBaseline, true);
  assert.equal(stable.resetBaseline, false);
});
