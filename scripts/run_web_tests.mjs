import { readdir } from "node:fs/promises";
import { spawn } from "node:child_process";
import { join } from "node:path";
import { fileURLToPath } from "node:url";

const repoRoot = fileURLToPath(new URL("..", import.meta.url));
const testRoots = ["frontend", "plugins", "scripts", "tests"];

async function findTestFiles(directory) {
  const entries = await readdir(directory, { withFileTypes: true });
  const files = await Promise.all(
    entries.sort((left, right) => left.name.localeCompare(right.name)).map((entry) => {
      const path = join(directory, entry.name);
      if (entry.isDirectory()) return findTestFiles(path);
      return entry.isFile() && entry.name.endsWith(".test.mjs") ? [path] : [];
    }),
  );
  return files.flat();
}

const files = (
  await Promise.all(testRoots.map((root) => findTestFiles(`${repoRoot}/${root}`)))
).flat().sort();

if (files.length === 0) {
  throw new Error("未找到 Web 测试文件");
}

const child = spawn(
  process.execPath,
  ["--experimental-strip-types", "--test", ...files],
  { stdio: ["inherit", "pipe", "inherit"] },
);

child.stdout.on("data", (chunk) => {
  process.stdout.write(chunk);
});

child.on("error", (error) => {
  console.error(`无法运行 Web 测试: ${error.message}`);
  process.exitCode = 1;
});

child.on("close", (code) => {
  if (code !== 0) {
    process.exitCode = code ?? 1;
  }
});
