"""只供可信评测端使用的 CDP 页面初始化与判分。"""
import asyncio
import base64
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import subprocess
import threading

import httpx

CDP = r"""
let input = ''; for await (const chunk of process.stdin) input += chunk;
const args = JSON.parse(input);
const base = 'http://127.0.0.1:9222';
if (args.url) {
  const response = await fetch(base + '/json/new?' + encodeURIComponent(args.url), {method: 'PUT'});
  if (!response.ok) throw Error(await response.text());
  console.log(JSON.stringify(await response.json()));
} else {
  const targets = await (await fetch(base + '/json/list')).json();
  const target = targets.find(t => t.id === args.target);
  if (!target) throw Error('Task tab disappeared');
  const ws = new WebSocket(target.webSocketDebuggerUrl);
  const timer = setTimeout(() => { throw Error('CDP deadline'); }, 20000);
  ws.onopen = () => ws.send(JSON.stringify({id: 1, method: args.method, params: args.params}));
  ws.onmessage = ({data}) => {
    const value = JSON.parse(data); if (value.id !== 1) return;
    if (value.error || value.result.exceptionDetails) throw Error(JSON.stringify(value));
    console.log(JSON.stringify(value.result)); clearTimeout(timer); ws.close();
  };
}
"""


class Page:
    def __init__(self, container: str, target: str):
        self.container, self.target = container, target

    @staticmethod
    async def command(container: str, **args):
        result = await asyncio.to_thread(subprocess.run,
            ["docker", "exec", "-i", container, "node", "--input-type=module", "-e", CDP],
            input=json.dumps(args), text=True, capture_output=True, timeout=30, check=False)
        if result.returncode:
            raise RuntimeError(result.stderr.strip())
        return json.loads(result.stdout)

    async def evaluate(self, expression: str):
        result = await self.command(self.container, target=self.target, method="Runtime.evaluate",
            params={"expression": expression, "returnByValue": True, "awaitPromise": True})
        return result["result"]["value"]

    async def screenshot(self, path: Path):
        result = await self.command(self.container, target=self.target, method="Page.captureScreenshot", params={})
        path.write_bytes(base64.b64decode(result["data"]))

    async def score(self):
        return await self.evaluate("({ ...(__miniwobAudit.first || "
            "{done:false,raw:0,reward:0,reason:null,episode:0}), "
            "starts:__miniwobAudit.starts, elapsed_ms:performance.now()-__miniwobAudit.started, "
            "final_episode:WOB_EPISODE_ID })")


async def prepare(identity: dict, html: Path, task: str, seed: int, timeout: int):
    """使用固定 seed 开始一个 episode；服务器由调用方关闭。"""
    server = ThreadingHTTPServer((identity["host"], 0), partial(SimpleHTTPRequestHandler, directory=str(html)))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        async with httpx.AsyncClient(timeout=90, trust_env=False) as client:
            response = await client.post(identity["gateway"] + "/wake", json={})
            response.raise_for_status()
        url = f'http://{identity["host"]}:{server.server_port}/miniwob/{task}.html'
        target = await Page.command(identity["container"], url=url)
        page = Page(identity["container"], target["id"])
        async with asyncio.timeout(60):
            while not await page.evaluate("document.readyState === 'complete' && typeof core !== 'undefined' && WOB_TASK_READY"):
                await asyncio.sleep(.1)
        # 首次结算是本题成绩；继续观察 Agent 是否正常收尾，不把重开后的分数覆盖进来。
        await page.evaluate("""(() => {
          const audit = window.__miniwobAudit = {starts:0,first:null,started:performance.now()};
          const start = core.startEpisodeReal, end = core.endEpisode;
          core.startEpisodeReal = function(...args) {
            audit.starts++; return start.apply(this,args);
          };
          core.endEpisode = function(...args) {
            const result = end.apply(this,args);
            if (WOB_DONE_GLOBAL && audit.first === null) audit.first = {
              done:true,raw:WOB_RAW_REWARD_GLOBAL,reward:WOB_REWARD_GLOBAL,
              reason:WOB_REWARD_REASON,episode:WOB_EPISODE_ID,
              done_ms:performance.now()-audit.started
            };
            return result;
          };
          return true;
        })()""")
        instruction = await page.evaluate(f"Math.seedrandom({json.dumps(str(seed))}); "
            f"core.EPISODE_MAX_TIME={timeout * 1000}; core.startEpisodeReal(); document.querySelector('#query').innerText")
        return server, page, url, instruction
    except BaseException:
        server.shutdown()
        server.server_close()
        raise
