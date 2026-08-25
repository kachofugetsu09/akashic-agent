import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

const source = await readFile(new URL("./chat-product-variants.tsx", import.meta.url), "utf8");
const styles = await readFile(new URL("./chat-product-variants.css", import.meta.url), "utf8");
const main = await readFile(new URL("./main.tsx", import.meta.url), "utf8");

test("liked study keeps alternating thinking/tool and waku model popover", () => {
  assert.match(source, /id: "liked"/);
  assert.match(source, /useState<ChatProductVariantId>\("liked"\)/);
  assert.match(source, /liked-trace/);
  assert.match(source, /思考与工具交替/);
  assert.match(source, /ModelPopover/);
  assert.match(source, /streamdown/);
  assert.match(source, /markdown-veil/);
  assert.doesNotMatch(source, /from "\.\/desktop-sidebar"|from "\.\/message-view"|from "\.\/model-capsule-picker"/);
});

test("liked composer is a restrained oval and process trace stays borderless", () => {
  assert.match(styles, /\.liked-composer\.empty\s*\{[\s\S]*?min-height:\s*44px/);
  assert.match(styles, /\.liked-composer\s*\{[\s\S]*?border-radius:\s*999px/);
  assert.match(styles, /\.liked-composer\.has-text\s*\{[\s\S]*?22px/);
  assert.match(styles, /\.liked-think\s*\{[\s\S]*?font-size:\s*0\.8125rem/);
  assert.match(styles, /\.liked-tool__head\s*\{[\s\S]*?background:\s*transparent/);
  assert.doesNotMatch(styles, /\.liked-tool\s*\{[\s\S]*?#f3e6d4/);
  assert.match(styles, /\.liked-model__pop\s*\{/);
  assert.match(styles, /\.liked-trace::before/);
});

test("preview entry routes only the product study showcase", () => {
  assert.match(main, /preview === "chat-product"/);
  assert.match(main, /ChatProductVariants/);
});
