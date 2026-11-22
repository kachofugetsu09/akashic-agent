// 插件前端里带独立 tsconfig 的子项目（如设计编辑器）按目录存在才检查：插件迁出到 extra 仓库后，Core 自己的检查不依赖它们，
// 合成目录里它们存在，同一条 npm run typecheck 照常覆盖。
import { readdirSync, existsSync } from "node:fs";
import { spawnSync } from "node:child_process";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const root = dirname(dirname(fileURLToPath(import.meta.url)));
const plugins = join(root, "frontend/plugins");
const projects = readdirSync(plugins, { withFileTypes: true }).filter((entry) => entry.isDirectory())
  .flatMap((entry) => readdirSync(join(plugins, entry.name), { withFileTypes: true })
    .filter((child) => child.isDirectory() && existsSync(join(plugins, entry.name, child.name, "tsconfig.json")))
    .map((child) => join(plugins, entry.name, child.name, "tsconfig.json")));
for (const project of projects.sort()) {
  const result = spawnSync(resolve(root, "node_modules/.bin/tsc"), ["--noEmit", "-p", project], { cwd: root, stdio: "inherit" });
  if (result.status !== 0) process.exit(result.status ?? 1);
}
