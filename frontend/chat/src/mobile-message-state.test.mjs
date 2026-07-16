import assert from "node:assert/strict";
import test from "node:test";

import {
  advanceMobileProjectionBaseline,
  advanceMobileUnreadTracking,
  allMobileAttachmentsReady,
  isMobileImageViewerHistoryState,
  reconcileAssistantMessageIds,
  updateMobileUnreadMessageIds,
  updateMobileSearchIndex,
} from "./mobile-message-state.ts";

test("composer waits until every attachment is ready", () => {
  assert.equal(allMobileAttachmentsReady([]), true);
  assert.equal(allMobileAttachmentsReady([{ state: "ready" }, { state: "ready" }]), true);
  assert.equal(allMobileAttachmentsReady([{ state: "ready" }, { state: "failed" }]), false);
  assert.equal(allMobileAttachmentsReady([{ state: "ready" }, { state: "uploading" }]), false);
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
