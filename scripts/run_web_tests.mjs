import { spawn } from "node:child_process";

const expectedTests = 62;
const files = [
  "frontend/chat/src/mobile-message-state.test.mjs",
  "frontend/chat/src/mobile-pairing.test.mjs",
  "frontend/chat/src/web-chat-transport.test.mjs",
  "tests/test_akasha_mobile_ui.mjs",
];

const child = spawn(
  process.execPath,
  ["--experimental-strip-types", "--test", ...files],
  { stdio: ["inherit", "pipe", "inherit"] },
);

let output = "";
child.stdout.setEncoding("utf8");
child.stdout.on("data", (chunk) => {
  output += chunk;
  process.stdout.write(chunk);
});

child.on("error", (error) => {
  console.error(`无法运行 Web 测试: ${error.message}`);
  process.exitCode = 1;
});

child.on("close", (code) => {
  if (code !== 0) {
    process.exitCode = code ?? 1;
    return;
  }
  const match = output.match(/^# tests (\d+)$/m);
  const actual = match ? Number(match[1]) : null;
  if (actual !== expectedTests) {
    console.error(`Web 测试必须恰好为 ${expectedTests} 项: actual=${actual}`);
    process.exitCode = 1;
  }
});
