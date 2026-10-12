import { image, sourceMounts } from "./image.mjs";
import assert from "node:assert/strict";
import { execFileSync } from "node:child_process";
import { mkdtemp, chmod, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { randomUUID } from "node:crypto";

// 真实 gateway、driver 和 Chromium 验证输出及按住鼠标的连续移动。
const artifacts = await mkdtemp(join(tmpdir(), "akashic-computer-input-"));
await chmod(artifacts, 0o777);
const container = `akashic-input-${randomUUID().slice(0, 8)}`;
const docker = (...args) => execFileSync("docker", args, { encoding: "utf8", timeout: 60000 }).trim();
const fixture = `<!doctype html><style>
  #drag {position:absolute;left:20px;top:20px;width:80px;height:80px;background:orange}
  #drop {position:absolute;left:200px;top:20px;width:80px;height:80px;background:lightblue}
</style><div id="drag">Drag</div><div id="drop">Drop here</div><output id="events"></output>
<script>
let active=false; const events=[];
document.addEventListener('mousedown', e => { active=e.target.id==='drag'; });
document.addEventListener('mousemove', e => {
  events.push(e.buttons); document.querySelector('#events').textContent=JSON.stringify(events);
  if(active && e.buttons!==1) active=false;
});
document.addEventListener('mouseup', e => {
  if(active && e.target.id==='drop') e.target.textContent='Dropped'; active=false;
});
</script>`;
let started = false;
const checks = [];
try {
  // 1. 独立桌面、数据目录和本地页面；不接触正式 workspace。
  docker("run", "-d", "--name", container, "--security-opt", "seccomp=unconfined",
    "--shm-size=512m", "-v", `${artifacts}:/data`, ...sourceMounts, image);
  started = true;
  const address = docker("inspect", container, "--format", "{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}");
  const gateway = `http://${address}:8080`;
  const deadline = Date.now() + 30000;
  while (true) {
    try { if ((await fetch(gateway + "/health")).ok) break; }
    catch (error) { if (error.cause?.code !== "ECONNREFUSED") throw error; }
    if (Date.now() > deadline) throw new Error("Computer startup timed out");
    await new Promise(resolve => setTimeout(resolve, 100));
  }
  const post = async (path, body) => {
    const response = await fetch(gateway + path, {method:"POST", headers:{"Content-Type":"application/json"}, body:JSON.stringify(body)});
    const result = await response.json();
    assert.equal(response.ok, true, JSON.stringify(result));
    return result;
  };
  const action = code => post("/driver/run", {context:{session_id:"input-e2e",turn_id:"one",call_id:randomUUID()},code});
  const texts = result => result.content.filter(item => item.type === "text").map(item => item.text);
  await post("/wake", {});
  docker("exec", "-d", container, "node", "-e",
    `require('http').createServer((req,res)=>{res.setHeader('Content-Type','text/html');res.end(${JSON.stringify(fixture)})}).listen(8989,'127.0.0.1')`);
  await action("nodeRepl.write(await browser.documentation());");
  // 2. 主浏览器和匿名 Context 都经过相同公开拖拽入口。
  for (const mode of ["main", "anonymous"]) {
    await action(`var owner=${mode === "main" ? "browser" : "await agent.browsers.create()"};
      var tab=await owner.tabs.new(); await tab.goto('http://127.0.0.1:8989/');`);
    const stored = await action("var frames=[]; for(var i=0;i<3;i++) frames.push(await tab.screenshot({})); frames=frames;");
    assert.deepEqual(stored.content, []);
    const explicit = await action("nodeRepl.write('saved'); await nodeRepl.emitImage(frames[0]);");
    assert.deepEqual(texts(explicit), ["saved"]);
    assert.equal(explicit.content.filter(item => item.type === "image").length, 1);
    await action(`await tab.cua.drag({path:[{x:60,y:60},{x:90,y:60},{x:120,y:60},{x:160,y:60},{x:210,y:60},{x:240,y:60}]});`);
    const result = texts(await action(`nodeRepl.write(await tab.playwright.evaluate(() => ({
      label:document.querySelector('#drop').textContent, events:document.querySelector('#events').textContent
    })));`))[0];
    const state = JSON.parse(result);
    assert.equal(state.label, "Dropped");
    const moves = JSON.parse(state.events), pressed = moves.indexOf(1);
    assert.ok(pressed >= 0);
    assert.ok(moves.slice(pressed).every(buttons => buttons === 1));
    await action("await tab.cua.move({x:300,y:60});");
    const after = JSON.parse(texts(await action("nodeRepl.write(await tab.playwright.evaluate(() => document.querySelector('#events').textContent));"))[0]);
    assert.equal(after.at(-1), 0);
    checks.push(`${mode}: explicit image output, held drag, released hover`);
    if (mode === "anonymous") await action("await owner.close();");
  }
  await writeFile(join(artifacts, "result.json"), JSON.stringify({image,sourceMounts,checks},null,2));
  console.log(JSON.stringify({artifacts,checks}));
} finally {
  if (started) {
    await writeFile(join(artifacts, "computer.log"), docker("logs", container));
    docker("rm", "-f", container);
  }
}
