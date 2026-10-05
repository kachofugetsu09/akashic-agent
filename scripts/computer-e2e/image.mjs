import { readFile } from "node:fs/promises";
import { resolve } from "node:path";

// 验证默认运行插件声明的发布镜像；源码挂载只用于明确的开发模式。
const root = resolve(import.meta.dirname, "../..");
const source = await readFile(resolve(root, "plugins/computer/plugin.py"), "utf8");
const pin = source.match(/_IMAGE = \(\s*"([^"\n]+@)"\s*"(sha256:[a-f0-9]{64})"\s*\)/);
if (!pin) throw new Error("Computer plugin must declare one fixed image digest");
export const image = process.env.COMPUTER_E2E_IMAGE ?? pin[1] + pin[2];
export const sourceMounts = process.env.COMPUTER_E2E_SOURCE === "1" ? [
  "-v", `${root}/docker/computer/driver:/opt/computer/driver:ro`,
  "-v", `${root}/docker/computer/gateway.mjs:/opt/computer/gateway.mjs:ro`,
] : [];
