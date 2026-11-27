// 插件前端里带独立 tsconfig 的子项目（如设计编辑器）按目录存在才检查：插件迁出到 extra 仓库后，Core 自己的检查不依赖它们，
// 合成目录里它们存在，同一条 npm run typecheck 照常覆盖。同时兼容 plugins/<name>/web/<subproject> 与历史 frontend/plugins/<name>/<subproject>。
import { readdirSync, existsSync } from "node:fs";
import { spawnSync } from "node:child_process";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const root = dirname(dirname(fileURLToPath(import.meta.url)));

function discoverSubprojects(baseDir, subDirName) {
  if (!existsSync(baseDir)) return [];
  const found = [];
  for (const entry of readdirSync(baseDir, { withFileTypes: true })) {
    if (!entry.isDirectory()) continue;
    const container = subDirName ? join(baseDir, entry.name, subDirName) : join(baseDir, entry.name);
    if (!existsSync(container)) continue;
    for (const child of readdirSync(container, { withFileTypes: true })) {
      if (!child.isDirectory() || child.name === "node_modules" || child.name.startsWith(".")) continue;
      const tsconfig = join(container, child.name, "tsconfig.json");
      if (existsSync(tsconfig)) found.push(tsconfig);
    }
  }
  return found;
}

const projects = [
  ...discoverSubprojects(join(root, "plugins"), "web"),
  ...discoverSubprojects(join(root, "frontend/plugins"), ""),
].sort();

for (const project of projects) {
  const result = spawnSync(resolve(root, "node_modules/.bin/tsc"), ["--noEmit", "-p", project], { cwd: root, stdio: "inherit" });
  if (result.status !== 0) process.exit(result.status ?? 1);
}
