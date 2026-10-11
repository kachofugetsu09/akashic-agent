/**
 * 上游 mock 的同一个 app，外加逐请求记录（时刻、请求体字节数、messages）。
 * 用法：bun mock.ts <wire.jsonl> --script <剧本> --port <端口> --exhausted <策略>
 */
import { appendFileSync } from "node:fs";
import { createApp } from "../../src/app";
import { loadConfig } from "../../src/config";
import { ScriptPlayer } from "../../src/script";

const [wirePath, ...args] = process.argv.slice(2);
if (wirePath === undefined) throw new Error("缺少 wire.jsonl 路径");
const config = loadConfig(args);
const player = ScriptPlayer.fromFile(config.scriptPath, { policy: config.exhausted });
const app = createApp({ player, config });
Bun.serve({
    hostname: "127.0.0.1",
    port: config.port,
    async fetch(request) {
        if (request.method === "POST" && new URL(request.url).pathname === "/v1/chat/completions") {
            const raw = await request.clone().text();
            const body = JSON.parse(raw);
            appendFileSync(wirePath, JSON.stringify({
                epoch_ms: Date.now(), bytes: Buffer.byteLength(raw), messages: body.messages,
            }) + "\n");
        }
        return app.fetch(request);
    },
});
