// 外壳应用的产物归各自的插件：发行版只取插件目录，不再由 Core 点名谁拥有哪个应用。
// 聊天构建会改写外壳 index.html 里的聊天版本，所以在 npm run build 全部完成后统一放置。
import { cpSync, existsSync, readdirSync, rmSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const repoRoot = dirname(dirname(fileURLToPath(import.meta.url)));
const owners = { dashboard: "ui", chat: "akashic_clients" };

for (const [app, plugin] of Object.entries(owners)) {
  const built = resolve(repoRoot, "static", app);
  if (!existsSync(built) || readdirSync(built).length === 0) {
    throw new Error(`没有 ${app} 的构建产物：${built}`);
  }
  const target = resolve(repoRoot, "plugins", plugin, "static", app);
  rmSync(target, { recursive: true, force: true });
  cpSync(built, target, { recursive: true });
}
