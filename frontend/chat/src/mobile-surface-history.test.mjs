import assert from "node:assert/strict";
import test from "node:test";

import {
  pushMobileSurface,
  readMobileSurfaceHistoryState,
  replaceMobileSurface,
} from "./mobile-surface-history.ts";

class FakeHistory {
  entries = [];
  index = -1;

  replaceState(data) {
    if (this.index < 0) {
      this.entries.push(data);
      this.index = 0;
    } else {
      this.entries[this.index] = data;
    }
  }

  pushState(data) {
    this.entries.splice(this.index + 1);
    this.entries.push(data);
    this.index += 1;
  }

  back() {
    this.index -= 1;
    return this.entries[this.index];
  }
}

test("dashboard back stack returns to directory then chat without losing plugin identity", () => {
  const history = new FakeHistory();
  replaceMobileSurface(history, { kind: "chat" });
  pushMobileSurface(history, { kind: "plugins" });
  pushMobileSurface(history, { kind: "dashboard", pluginId: "status_commands" });

  assert.deepEqual(readMobileSurfaceHistoryState(history.entries[history.index]), {
    kind: "dashboard",
    pluginId: "status_commands",
  });
  assert.deepEqual(readMobileSurfaceHistoryState(history.back()), { kind: "plugins" });
  assert.deepEqual(readMobileSurfaceHistoryState(history.back()), { kind: "chat" });
});

test("foreign or malformed history state returns to chat", () => {
  assert.deepEqual(readMobileSurfaceHistoryState(null), { kind: "chat" });
  assert.deepEqual(
    readMobileSurfaceHistoryState({
      akashicMobileSurface: true,
      surface: { kind: "dashboard", pluginId: "" },
    }),
    { kind: "chat" },
  );
});
