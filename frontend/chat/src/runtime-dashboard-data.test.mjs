import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

import { runtimeItems } from "./runtime-dashboard-data.ts";

const overview = {
  documents: [{ id: "doc", title: "文档", relative_path: "docs/doc.md", group: "core", description: "说明", available: true }],
  jobs: [{ id: "job", name: "任务", trigger: "schedule", tier: "routine", fire_at: "2026-08-12T00:00:00Z", timezone: "Asia/Shanghai", enabled: true, run_count: 1 }],
  capabilities: { snapshot_id: "one", plugins: [], skills: [], mcp_servers: [{ owner_id: "core", name: "filesystem", tool_count: 4 }] },
};

test("runtime directory projection stays independent from React presentation", () => {
  assert.equal(runtimeItems("documents", overview)[0].icon, "documents");
  assert.equal(runtimeItems("mcp", overview)[0].key, "core\u0000filesystem");
  assert.equal(runtimeItems("jobs", overview)[0].status, "启用");
});

test("runtime controller derives selection before detail effects", async () => {
  const controller = await readFile(new URL("./use-runtime-dashboard.ts", import.meta.url), "utf8");
  assert.match(controller, /const selectedKey = useMemo/);
  assert.doesNotMatch(controller, /const \[selectedKey, setSelectedKey\]/);
});
