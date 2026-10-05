import { image, sourceMounts } from "./image.mjs";
import assert from "node:assert/strict";
import { execFileSync } from "node:child_process";
import { mkdtemp, mkdir, chmod, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";
import { randomUUID } from "node:crypto";

// 真实 gateway 与 Chromium，加速同一资源 owner 的闲置时钟，不替换协议或进程。
const root = resolve(import.meta.dirname, "../..");
const artifacts = await mkdtemp(join(tmpdir(), "akashic-computer-idle-"));
const name = `akashic-computer-idle-${randomUUID().slice(0,8)}`;
const docker = (...args) => execFileSync("docker", args, {encoding:"utf8", timeout:60000, stdio:["ignore","pipe","pipe"]}).trim();

let started = false;
async function until(read, check, label) {
  const end = Date.now()+25000;
  let value;
  while (Date.now()<end) {
    value = await read();
    if(check(value)) return value;
    await new Promise(resolve=>setTimeout(resolve,80));
  }
  throw new Error(`${label}: ${JSON.stringify(value)}`);
}
try {
  const data = join(artifacts,"data"); await mkdir(data); await chmod(data,0o777);
  docker("run","-d","--name",name,"--security-opt","seccomp=unconfined","--shm-size=512m",
    "-e","COMPUTER_ANONYMOUS_IDLE_MS=1200","-e","COMPUTER_IDLE_MS=1200","-p","127.0.0.1::8080","-v",`${data}:/data`,
    ...sourceMounts,image);
  started = true;
  const port=JSON.parse(docker("inspect",name,"--format","{{json .NetworkSettings.Ports}}"))["8080/tcp"][0].HostPort;
  const origin=`http://127.0.0.1:${port}`;
  const post = async(path,body,expected=200)=>{
    const response=await fetch(origin+path,{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(body)});
    const value=await response.json(); assert.equal(response.status,expected,JSON.stringify(value)); return value;
  };
  const action=code=>post("/driver/run",{context:{session_id:"idle",turn_id:"turn",call_id:randomUUID()},code});
  const targets=async()=> (await (await fetch(origin+"/targets")).json()).targets;
  await until(async()=>{try{return(await fetch(origin+"/health")).ok;}catch(error){if(!["ECONNREFUSED","UND_ERR_SOCKET"].includes(error.cause?.code))throw error;return false;}},Boolean,"startup");
  docker("cp",join(root,"scripts/computer-e2e/remote-page.mjs"),`${name}:/tmp/remote-page.mjs`);
  docker("exec","-d",name,"node","/tmp/remote-page.mjs");
  await until(async()=>docker("exec",name,"node","-e","fetch('http://127.0.0.1:8989').then(r=>console.log(r.status)).catch(()=>console.log('starting'))"),v=>v==="200","fixture");
  await post("/wake",{});
  await action("var anon=await agent.browsers.create(); var tab=await anon.tabs.new(); await tab.goto('http://127.0.0.1:8989/slow');");
  let target=(await targets()).find(t=>t.kind==="browser"); assert.ok(target);
  console.log("PASS active navigation survives more than the idle limit");
  const owner=randomUUID(); await post("/control/take",{id:owner,target:target.id});
  // 必须越过真实闲置期限，证明接管持有资源；帧读取继续保持只读。
  await new Promise(resolve=>setTimeout(resolve,2200));
  assert.ok((await targets()).some(t=>t.id===target.id));
  await post("/control/release",{id:owner});
  console.log("PASS human takeover prevents idle disposal");
  await until(async()=>{
    const values=await targets();
    if(values.some(t=>t.id===target.id)) await fetch(origin+"/view/frame?target="+encodeURIComponent(target.id));
    return values;
  },v=>v.length===1,"idle disposal despite watching");
  const failed=await post("/driver/run",{context:{session_id:"idle",turn_id:"turn",call_id:randomUUID()},code:"await agent.browsers.get(anon.browserId);"},500);
  assert.match(failed.error,/released after idle timeout/); assert.match(failed.error,/bindings and browser pages were kept/);
  console.log("PASS idle disposal removes targets and reports expired identity without selecting main");
  await action("console.log(await browser.tabs.list());");
  await post("/driver/run",{context:{session_id:"idle",turn_id:"turn",call_id:randomUUID()},endTurn:true});
  const desktopOwner=randomUUID(); await post("/control/take",{id:desktopOwner});
  await new Promise(resolve=>setTimeout(resolve,2200));
  const state=(await (await fetch(origin+"/activity")).json()).browser;
  assert.equal(state.turns,0); assert.equal(state.state,"ready");
  await post("/control/release",{id:desktopOwner});
  await until(async()=>(await (await fetch(origin+"/activity")).json()).browser.state,v=>v==="sleeping","desktop sleeps after control release");
  console.log("PASS takeover keeps desktop awake without a Turn and release permits real idle shutdown");
  await writeFile(join(artifacts,"result.json"),JSON.stringify({image,sourceMounted:sourceMounts.length>0,checks:4,artifacts},null,2));
} finally {
  if(started) {await writeFile(join(artifacts,"computer.log"),docker("logs",name)); docker("rm","-f",name);}
  console.log(`Artifacts: ${artifacts}`);
}
