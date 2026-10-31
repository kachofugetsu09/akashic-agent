// 只用于 --auth 新 fixture；真实登录表单与真实退出 API，不注入 cookie、不读正式凭据。
import {chromium} from 'playwright-core';
import assert from 'node:assert/strict';
import {randomBytes,createHash} from 'node:crypto';
import {readFileSync,writeFileSync,mkdirSync,realpathSync,lstatSync,readdirSync,utimesSync} from 'node:fs';
import {execFileSync} from 'node:child_process';
import {join,relative,resolve} from 'node:path';
import {parseArgs} from 'node:util';
import {fileURLToPath} from 'node:url';
const {values}=parseArgs({options:{state:{type:'string'},baseline:{type:'string'},candidate:{type:'string'},browser:{type:'string'},output:{type:'string'}}});
for(const name of ['state','baseline','candidate','browser','output'])assert.ok(values[name],'需要 --'+name);
const state=realpathSync(values.state),output=resolve(values.output);
const marker=JSON.parse(readFileSync(join(state,'.akashic-oneoff-fixture.json')));
assert.equal(marker.kind,'akashic-attachment-oneoff');assert.equal(marker.state,state);
assert.equal(marker.workspace,join(state,'workspace'));assert.equal(marker.authEnabled,true);
assert.equal(marker.root,realpathSync(new URL('../..',import.meta.url)));
assert.ok(Number.isSafeInteger(marker.pid)&&marker.pid>0);
assert.ok(Number.isSafeInteger(marker.port)&&marker.port>0&&marker.port<65536);
const url=`http://127.0.0.1:${marker.port}`;
const hash=bytes=>createHash('sha256').update(bytes).digest('hex');
function owner(){
  assert.equal(realpathSync(marker.workspace),join(state,'workspace'));
  assert.equal(realpathSync(marker.chat),join(state,'plugins/akashic_clients/static/chat'));
  assert.ok(readFileSync(`/proc/${marker.pid}/cmdline`).includes(Buffer.from(state)));
  assert.ok(execFileSync('ss',['-ltnpH',`sport = :${marker.port}`],{encoding:'utf8'}).includes(`pid=${marker.pid},`));
}
owner();
const initialPath=join(state,'workspace/web-auth/initial-password');
assert.equal(realpathSync(initialPath),initialPath,'只读本次 fixture 的 regular credential file');
assert.ok(lstatSync(initialPath).isFile());assert.equal(lstatSync(initialPath).mode&0o777,0o600);
let password=readFileSync(initialPath,'utf8').trim();assert.ok(password.length>=10);
const replacement=randomBytes(24).toString('base64url');
const secrets=[password,replacement];
const redact=value=>secrets.reduce((text,secret)=>text.split(secret).join('[REDACTED]'),String(value));
const oracle=JSON.parse(readFileSync(join(state,'attachment-oracle.json')));
const sample=oracle.samples.find(item=>item.name==='medium.txt');assert.ok(sample);
const original=readFileSync(join(state,'originals',sample.name));assert.equal(hash(original),sample.sha256);
const artifactPath=`/api/chat/artifacts/${sample.id}`,chatPath='/?session='+encodeURIComponent(oracle.session);
const verify=()=>execFileSync('python3',
  [fileURLToPath(new URL('serve-attachment-runtime.py',import.meta.url)),'--state',state,'--verify'],{encoding:'utf8'});
verify();mkdirSync(output,{mode:0o700});
const report={complete:false,flows:[],builds:[],harnessSha256:hash(readFileSync(import.meta.filename)),
  fixtureSha256:hash(readFileSync(new URL('serve-attachment-runtime.py',import.meta.url))),
  oracleSha256:hash(readFileSync(join(state,'attachment-oracle.json'))),
  scope:'真实一次性 web-auth；账号、密码更新、cookie 与会话注销仅属于新 fixture；不记录凭据、cookie、请求体或认证响应体'};
const save=()=>writeFileSync(join(output,'report.json'),JSON.stringify(report,null,2),{mode:0o600});
function readBuild(root){
  const hashes={};
  function visit(directory){
    for(const name of readdirSync(directory).sort()){
      const path=join(directory,name),stat=lstatSync(path);assert.ok(!stat.isSymbolicLink());
      if(stat.isDirectory())visit(path);
      else{assert.ok(stat.isFile());hashes[relative(root,path)]=hash(readFileSync(path));}
    }
  }
  visit(root);return hashes;
}
// 登录与首个附件操作之前同时固定两侧清单，禁止每轮接受新制品。
const frozenBuilds=Object.fromEntries(['baseline','candidate'].map(variant=>{
  const source=realpathSync(values[variant]),hashes=readBuild(source);
  assert.ok(hashes['chat-app.js'],'缺少聊天入口制品');
  return [variant,{source,hashes}];
}));
report.frozenBuilds=frozenBuilds;save();
function install(variant){
  owner();const {source,hashes}=frozenBuilds[variant],target=realpathSync(marker.chat);
  assert.ok(source!==target&&!source.startsWith(target+'/')&&!target.startsWith(source+'/'));
  assert.deepEqual(readBuild(source),hashes,'源制品偏离首次冻结清单');
  execFileSync('rsync',['-a','--checksum','--delete','--chmod=u+rwX',source+'/',target+'/']);
  assert.deepEqual(readBuild(target),hashes,'安装制品偏离首次冻结清单');
  for(const [name,sha] of Object.entries(hashes)){
    const dest=join(target,name);
    const stamp=1767225600+parseInt(sha.slice(0,10),16)/1e6;utimesSync(dest,stamp,stamp);
  }
  const build={variant,hashes,httpVerified:false};report.builds.push(build);save();return build;
}
async function verifyHttpBuild(context,installed){
  for(const [name,sha] of Object.entries(frozenBuilds[installed.variant].hashes).filter(([name])=>/\.(js|css)$/.test(name))){
    const response=await context.request.get(url+'/assets/'+name);
    assert.equal(response.status(),200);assert.equal(hash(await response.body()),sha,'HTTP 制品偏离冻结值: '+name);
  }
  installed.httpVerified=true;save();
}
async function login(page,first){
  // 浏览器真实表单获得服务端 Set-Cookie；首次强制改密也走真实表单。
  await page.goto(url+'/_auth/login?next='+encodeURIComponent(chatPath));
  await page.locator('#login').waitFor({state:'visible'});
  await page.locator('#username').fill('akashic');await page.locator('#password').fill(password);
  const response=page.waitForResponse(r=>new URL(r.url()).pathname==='/_auth/login'&&r.request().method()==='POST');
  await page.locator('#login button[type=submit]').click();assert.equal((await response).status(),200);
  if(first){
    await page.locator('#change').waitFor({state:'visible'});
    await page.locator('#new1').fill(replacement);await page.locator('#new2').fill(replacement);
    const changed=page.waitForResponse(r=>new URL(r.url()).pathname==='/_auth/password');
    await page.locator('#change button[type=submit]').click();assert.equal((await changed).status(),200);password=replacement;
  }
  await page.waitForURL(url+chatPath);
  await page.locator(`[data-message-id="${sample.messageId}"] .ak-attachment__surface`).waitFor({state:'visible'});
}
async function shown(page){
  await page.waitForFunction(expected=>{
    const dialog=document.querySelector('.attachment-preview'),pre=dialog?.querySelector('pre');
    if(!pre)return false;
    const description=dialog.querySelector('.attachment-preview__title p')?.textContent||'';
    const lineMatch=description.match(/(?:^| • )([\d,]+) 行$/u);
    return JSON.stringify([...pre.children].map(node=>node.textContent))===JSON.stringify(expected.preview)
      &&lineMatch?.[1]===expected.lines.toLocaleString('zh-CN');
  },sample);
}
let browser;
try{
  browser=await chromium.launch({executablePath:values.browser});report.browserVersion=browser.version();
  for(const [index,variant] of ['baseline','candidate'].entries()){
    const installed=install(variant);
    const context=await browser.newContext({viewport:{width:1440,height:900},serviceWorkers:'block'});
    await context.addInitScript(()=>localStorage.setItem('akashic.onboarding.seen','1'));
    const page=await context.newPage(),flow={variant,requests:[],success:false};report.flows.push(flow);save();
    let stage='initial-login';
    page.on('response',response=>{
      const path=new URL(response.url()).pathname;
      if(path===artifactPath)flow.requests.push({stage,status:response.status(),method:response.request().method(),path});
    });
    let failed=false;
    try{
      // 独立、无凭据客户端在首次登录前也必须被拒绝，直接读正文不是跳到登录页的成功。
      const anonymous=await fetch(url+artifactPath,{redirect:'manual',headers:{Accept:'application/octet-stream'}});
      assert.equal(anonymous.status,401);flow.beforeLoginStatus=401;
      await login(page,index===0);flow.realLogin=true;
      await verifyHttpBuild(context,installed);
      const response=await context.request.get(url+artifactPath);assert.equal(response.status(),200);
      assert.ok((await response.body()).equals(original));flow.authorizedBytesExact=true;
      const card=page.locator(`[data-message-id="${sample.messageId}"] .ak-attachment__surface`);
      stage='authorized-open';await card.click();await shown(page);
      await page.getByRole('button',{name:'关闭',exact:true}).click();await page.locator('.attachment-preview').waitFor({state:'detached'});
      const documentStart=await page.evaluate(()=>performance.timeOrigin);
      // 退出走真实同源 HTTP API，不清 cookie 伪装注销；主页面不导航、不刷新。
      stage='logout';
      const loggedOut=await page.evaluate(async()=>{
        const response=await fetch('/_auth/logout',{method:'POST',credentials:'same-origin'});
        return {status:response.status,body:await response.json()};
      });
      assert.equal(loggedOut.status,200);assert.equal(loggedOut.body.signedIn,false);flow.realLogout=true;
      const authState=await context.request.get(url+'/_auth/state');assert.equal((await authState.json()).signedIn,false);
      stage='denied-hot-reopen';
      const denied=page.waitForResponse(r=>new URL(r.url()).pathname===artifactPath);
      await card.click();assert.equal((await denied).status(),401);
      await page.locator('.attachment-preview [role=alert]').filter({hasText:'读取附件失败: 401'}).waitFor();
      assert.equal(await page.locator('.attachment-preview pre').count(),0);
      assert.equal(await page.evaluate(()=>performance.timeOrigin),documentStart,'退出后必须仍是原文档与热缓存');
      flow.hotReopenDenied=true;flow.sameDocumentAfterLogout=true;
      await page.getByRole('button',{name:'关闭',exact:true}).click();await page.locator('.attachment-preview').waitFor({state:'detached'});
      const afterLogout=await fetch(url+artifactPath,{redirect:'manual',headers:{Accept:'application/octet-stream'}});
      assert.equal(afterLogout.status,401);flow.afterLogoutAnonymousStatus=401;
      // 新 tab 通过真实登录表单恢复同一 browser context；原聊天页仍不刷新。
      stage='relogin';const authTab=await context.newPage();await login(authTab,false);await authTab.close();
      await page.bringToFront();assert.equal(await page.evaluate(()=>performance.timeOrigin),documentStart);
      stage='restored-hot-reopen';await card.click();await shown(page);
      const restored=await context.request.get(url+artifactPath);assert.equal(restored.status(),200);assert.ok((await restored.body()).equals(original));
      flow.reloginRestored=true;flow.sameDocumentAfterRelogin=true;
      await page.getByRole('button',{name:'关闭',exact:true}).click();
      const statuses=flow.requests.filter(item=>item.stage==='denied-hot-reopen');
      assert.equal(statuses.length,1);assert.equal(statuses[0].status,401);
      const recovered=flow.requests.filter(item=>item.stage==='restored-hot-reopen');
      assert.equal(recovered.length,1);assert.equal(recovered[0].status,200);
      // 清理本次登录 session 也用真实 endpoint；账号与审计文件保留在一次性 state。
      const cleanup=await page.evaluate(async()=>{const r=await fetch('/_auth/logout',{method:'POST',credentials:'same-origin'});return r.status;});
      assert.equal(cleanup,200);flow.finalLogout=true;flow.success=true;save();
    }catch(error){failed=true;throw error;}
    finally{
      try{await context.close();}
      catch(error){
        (report.contextCloseErrors??=[]).push(redact(error));
        if(!failed)throw error;
      }
    }
  }
  report.integrity=verify();report.complete=true;
}catch(error){report.error=redact(error);process.exitCode=1;}
finally{
  // 保留主失败；浏览器关闭失败也继续恢复制品并核对持久事实。
  try{await browser?.close();}
  catch(error){report.complete=false;report.browserCloseError=redact(error);process.exitCode=1;}
  try{install('baseline');report.restoredBaselineOnDisk=true;}
  catch(error){report.complete=false;report.restoreError=redact(error);process.exitCode=1;}
  try{report.finalIntegrity=verify();}
  catch(error){report.complete=false;report.integrityError=redact(error);process.exitCode=1;}
  save();
}
