import assert from "node:assert/strict";
import test from "node:test";
import { present, type PanelFacts } from "./panel-state.ts";

const watching: PanelFacts = {
  runtime: "ready", runtimeError: "", link: "connected", linkError: "", control: "none",
  agentActive: false, targetKind: "desktop", targetTitle: "", wakeRequested: false,
};

test("只读观看：主操作是接管，外框保持安静", () => {
  const view = present(watching);
  assert.equal(view.title, "只读观看");
  assert.equal(view.action?.id, "take");
  assert.equal(view.overlay, null);
  assert.equal(view.frame.kind, "none");
});

test("匿名浏览器用外框说明身份；接管时控制权优先", () => {
  const anonymous = present({ ...watching, targetKind: "browser", targetTitle: "结账测试" });
  assert.equal(anonymous.frame.kind, "anonymous");
  assert.match(anonymous.detail, /^结账测试 · /);
  const mine = present({ ...watching, targetKind: "browser", control: "mine" });
  assert.equal(mine.frame.kind, "control");
  assert.match(mine.frame.legend, /匿名浏览器/);
  assert.equal(mine.action?.id, "release");
});

test("另一个窗口在操作：按钮不变，原因写在按钮下方", () => {
  const view = present({ ...watching, control: "other" });
  assert.equal(view.action?.label, "接管操作");
  assert.equal(view.action?.blockedBy, "等对方交还后可接管");
});

test("没有画面时只有遮罩里的操作，状态条不重复放按钮", () => {
  for (const facts of [
    { ...watching, runtime: "sleeping", link: "idle" },
    { ...watching, runtime: "sleeping", runtimeError: "boom", link: "idle" },
    { ...watching, link: "ended" },
    { ...watching, link: "stale" },
    { ...watching, link: "lost" },
  ] as PanelFacts[]) {
    const view = present(facts);
    assert.equal(view.action, null);
    assert.ok(view.overlay && view.overlay.actions.length > 0, view.title);
  }
});

test("唤醒失败带详情，运行环境异常优先于其它状态", () => {
  assert.equal(present({ ...watching, runtime: "sleeping", runtimeError: "boom", link: "idle" }).overlay?.details, "boom");
  assert.equal(present({ ...watching, runtime: "failed", runtimeError: "exit", link: "ended" }).title, "电脑出错了");
  assert.equal(present({ ...watching, runtime: "sleeping", link: "idle", wakeRequested: true }).title, "正在唤醒");
});

test("接管中可以取消", () => {
  const view = present({ ...watching, link: "connecting", control: "taking" });
  assert.equal(view.title, "正在接管…");
  assert.deepEqual(view.overlay?.actions.map(action => action.id), ["cancel"]);
});
