import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

const styles = await readFile(
  new URL("./mobile-native.css", import.meta.url),
  "utf8",
);

test("process plugin slots align with thinking and tool content", () => {
  assert.match(
    styles,
    /\.process-item\s*\{[\s\S]*?grid-template-columns:\s*18px minmax\(0, 1fr\);[\s\S]*?column-gap:\s*12px;/,
  );
  assert.match(
    styles,
    /\.mobile-plugin-slot\[data-slot="turn\.before_reasoning"\],[\s\S]*?margin-inline-start:\s*30px;/,
  );
});
