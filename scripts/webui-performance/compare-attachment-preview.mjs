// 真实附件 API + 浏览器 AB/BA；与其他构建或基准串行运行。
import {chromium} from 'playwright-core';
import assert from 'node:assert/strict';
import {createHash} from 'node:crypto';
import {readFileSync, writeFileSync, mkdirSync, readdirSync, realpathSync, lstatSync, utimesSync} from 'node:fs';
import {execFileSync} from 'node:child_process';
import {join, relative, resolve} from 'node:path';
import {parseArgs} from 'node:util';
import {fileURLToPath} from 'node:url';
const {values} = parseArgs({options: {
  state:{type:'string'}, baseline:{type:'string'}, candidate:{type:'string'}, browser:{type:'string'},
  output:{type:'string'}, runs:{type:'string',default:'7'}, mode:{type:'string',default:'performance'},
}});
for (const name of ['state','baseline','candidate','browser','output']) assert.ok(values[name], '需要 --'+name);
assert.ok(['performance','correctness'].includes(values.mode));
const state=realpathSync(values.state), output=resolve(values.output), rounds=Number(values.runs);
assert.ok(Number.isSafeInteger(rounds)&&rounds>0);
const marker=JSON.parse(readFileSync(join(state,'.akashic-oneoff-fixture.json')));
assert.equal(marker.kind,'akashic-attachment-oneoff'); assert.equal(marker.state,state);
assert.equal(marker.workspace,join(state,'workspace'));
assert.equal(marker.root,realpathSync(new URL('../..',import.meta.url)));
assert.equal(marker.authEnabled,false); // 本组不冒充登录/退出鉴权验收。
assert.ok(Number.isSafeInteger(marker.pid)&&marker.pid>0);
assert.ok(Number.isSafeInteger(marker.port)&&marker.port>0&&marker.port<65536);
const url=`http://127.0.0.1:${marker.port}`;
const oracle=JSON.parse(readFileSync(join(state,'attachment-oracle.json')));
const hash=bytes=>createHash('sha256').update(bytes).digest('hex');
const verifyState=()=>execFileSync('python3',
  [fileURLToPath(new URL('serve-attachment-runtime.py',import.meta.url)),'--state',state,'--verify'],{encoding:'utf8'});
function owner() {
  assert.equal(realpathSync(marker.chat),join(state,'plugins/akashic_clients/static/chat'));
  assert.ok(readFileSync(`/proc/${marker.pid}/cmdline`).includes(Buffer.from(state)),'服务 PID 不再拥有 state');
  assert.ok(execFileSync('ss',['-ltnpH',`sport = :${marker.port}`],{encoding:'utf8'}).includes(`pid=${marker.pid},`),'端口 PID 不一致');
}
owner(); verifyState(); mkdirSync(output,{mode:0o700});
const report={complete:false,mode:values.mode,rounds,samples:[],builds:[],
  harnessSha256:hash(readFileSync(import.meta.filename)),
  fixtureSha256:hash(readFileSync(new URL('serve-attachment-runtime.py',import.meta.url))),
  oracleSha256:hash(readFileSync(join(state,'attachment-oracle.json'))),
  plan:{targetMs:300,profiles:{lan:{latency:20,bps:100000000},weak:{latency:800,bps:1000000}},
    gates:{weakCold:{minimumMedianGainFraction:0.20,minimumMedianGainMs:600},
      otherCells:{maximumMedianRegressionMs:25,maximumMedianRegressionFraction:0.10},
      metrics:['contentMs','settledMs'],absoluteTargetMs:300,absoluteTargetIsIndependent:true},
    timing:'真实 pointerdown → 正文逐行与元数据正确 → 两帧；另等有限动画结束及两帧',
    auth:'未覆盖：此隔离 fixture 明确关闭 web-auth，须另做真实登录/退出验收',
    plannedSamples:values.mode==='performance'?rounds*2*2*2*2:oracle.samples.length*2*2},
};
const save=()=>writeFileSync(join(output,'report.json'),JSON.stringify(report,null,2),{mode:0o600});
function score() {
  // 门槛先于首个样本冻结；每种文件、温度、网络和两个计时各自判定。
  const stats=rows=>{
    const sorted=[...rows].sort((a,b)=>a-b);
    return {count:sorted.length,min:sorted[0],max:sorted.at(-1),
      median:sorted.length%2?sorted[(sorted.length-1)/2]:(sorted[sorted.length/2-1]+sorted[sorted.length/2])/2,
      p95:sorted[Math.ceil(sorted.length*0.95)-1]};
  };
  const cells=[];
  for(const name of oracle.samples.filter(s=>s.group==='performance').map(s=>s.name))
    for(const profile of ['lan','weak'])for(const temperature of ['cold','hot'])for(const metric of ['contentMs','settledMs']) {
      const sides={};
      for(const variant of ['baseline','candidate']) {
        const selected=report.samples.filter(s=>s.name===name&&s.profile===profile&&s.temperature===temperature&&s.variant===variant);
        assert.equal(selected.length,rounds);assert.ok(selected.every(s=>s.success&&Number.isFinite(s[metric])));
        sides[variant]=selected.sort((a,b)=>a.round-b.round);
      }
      const baseline=stats(sides.baseline.map(s=>s[metric])),candidate=stats(sides.candidate.map(s=>s[metric]));
      const gainMs=baseline.median-candidate.median,gainFraction=gainMs/baseline.median;
      const weakCold=profile==='weak'&&temperature==='cold';
      const allowedRegressionMs=Math.max(25,0.10*baseline.median);
      const passed=weakCold?gainMs>=600&&gainFraction>=0.20:-gainMs<=allowedRegressionMs;
      cells.push({name,profile,temperature,metric,baseline,candidate,gainMs,gainFraction,allowedRegressionMs,
        pairedDeltasMs:sides.baseline.map((s,i)=>s[metric]-sides.candidate[i][metric]),passed,
        baselineMedianWithin300ms:baseline.median<=300,candidateMedianWithin300ms:candidate.median<=300});
    }
  return {passed:cells.every(cell=>cell.passed),cells};
}
function readBuild(root) {
  const hashes={};
  function visit(directory) {
    for(const name of readdirSync(directory).sort()) {
      const path=join(directory,name),stat=lstatSync(path);
      assert.ok(!stat.isSymbolicLink(),'制品不得使用软链接');
      if(stat.isDirectory())visit(path);
      else {assert.ok(stat.isFile());hashes[relative(root,path)]=hash(readFileSync(path));}
    }
  }
  visit(root);return hashes;
}
// 两侧完整文件清单必须先于首个样本固定，后续轮次和恢复共用这一个事实。
const frozenBuilds=Object.fromEntries(['baseline','candidate'].map(variant=>{
  const source=realpathSync(values[variant]),hashes=readBuild(source);
  assert.ok(hashes['chat-app.js'],'缺少聊天入口制品');
  return [variant,{source,hashes}];
}));
report.frozenBuilds=frozenBuilds;save();
async function install(variant) {
  // 只有 marker、PID、监听端口和 canonical path 同时成立才允许覆盖临时制品。
  owner();
  const {source,hashes}=frozenBuilds[variant],target=realpathSync(marker.chat);
  assert.ok(source!==target&&!source.startsWith(target+'/')&&!target.startsWith(source+'/'));
  assert.deepEqual(readBuild(source),hashes,'源制品偏离首次冻结清单');
  execFileSync('rsync',['-a','--checksum','--delete','--chmod=u+rwX',source+'/',target+'/']);
  assert.deepEqual(readBuild(target),hashes,'安装制品偏离首次冻结清单');
  for(const [name,sha] of Object.entries(hashes)) {
    const dest=join(target,name);
    const stamp=1767225600+parseInt(sha.slice(0,10),16)/1e6; utimesSync(dest,stamp,stamp);
    if(/\.(js|css)$/.test(name)) {
      const response=await fetch(url+'/assets/'+name,{headers:{'Accept-Encoding':'gzip'}});
      assert.equal(response.status,200);assert.equal(hash(Buffer.from(await response.arrayBuffer())),sha,'HTTP 制品偏离冻结值: '+name);
    }
  }
  report.builds.push({variant,hashes}); save();
}

async function arm(page,sample) {
  await page.evaluate(expected=>{
    window.__attachmentDone=new Promise((resolve,reject)=>{
      const timer=setTimeout(()=>reject(Error('附件正文或元数据未满足 oracle')),45000);
      document.addEventListener('pointerdown',()=>{
        const start=performance.now();
        function check() {
          const dialog=document.querySelector('.attachment-preview');
          const pre=dialog?.querySelector('.attachment-preview__body pre');
          if(!pre){requestAnimationFrame(check);return;}
          const actual=[...pre.children].map(node=>node.textContent);
          const wanted=expected.preview.map(text=>text||' ');
          const description=dialog.querySelector('.attachment-preview__title p')?.textContent||'';
          const lineMatch=description.match(/(?:^| • )([\d,]+) 行$/u);
          const lineLabel=expected.lines===null?null:expected.lines.toLocaleString('zh-CN');
          const columns=[...dialog.querySelectorAll('.attachment-preview__columns li')].map(node=>node.textContent);
          const tableMeta=dialog.querySelector('.attachment-preview__columns p')?.textContent;
          const truncated=Boolean(dialog.querySelector('.attachment-preview__body > .attachment-preview__note'));
          const ready=JSON.stringify(actual)===JSON.stringify(wanted)
            &&(lineLabel===null?lineMatch===null:lineMatch?.[1]===lineLabel)
            &&document.fonts.status==='loaded'&&dialog.getBoundingClientRect().height>0
            &&truncated===expected.truncated
            &&(expected.table?JSON.stringify(columns)===JSON.stringify(expected.table.columns)
                &&tableMeta===`${expected.table.columns.length} 列，${expected.table.rows.toLocaleString('zh-CN')} 行`:columns.length===0);
          window.__attachmentDiagnostic={actualLines:actual.length,wantedLines:wanted.length,description,columns,tableMeta,truncated};
          if(!ready){requestAnimationFrame(check);return;}
          requestAnimationFrame(()=>requestAnimationFrame(async()=>{
            const contentMs=performance.now()-start;
            try {
              await Promise.all(dialog.getAnimations({subtree:true}).filter(a=>a.timeline instanceof DocumentTimeline&&a.effect?.getComputedTiming().iterations!==Infinity).map(a=>a.finished));
              let previous='',stable=0;
              function settle(){
                const geometry=JSON.stringify([dialog,pre,dialog.querySelector('.attachment-preview__title')].map(e=>{const r=e.getBoundingClientRect();return [r.x,r.y,r.width,r.height];}));
                stable=geometry===previous?stable+1:0;previous=geometry;
                if(stable<2){requestAnimationFrame(settle);return;}
                clearTimeout(timer);resolve({contentMs,settledMs:performance.now()-start});
              }
              requestAnimationFrame(settle);
            }catch(error){clearTimeout(timer);reject(error);}
          }));
        }
        requestAnimationFrame(check);
      },{capture:true,once:true});
    });
  },sample);
}

let browser, current;
async function measure(variant,profile,round,sample) {
  // 每个冷首开使用独立 browser context；热重开只关闭同一弹窗后再点开。
  const context=await browser.newContext({viewport:{width:1440,height:900},permissions:['clipboard-read','clipboard-write'],serviceWorkers:'block',acceptDownloads:true});
  const page=await context.newPage();current=page;const errors=[];page.on('pageerror',e=>errors.push(String(e)));
  await context.addInitScript(()=>localStorage.setItem('akashic.onboarding.seen','1'));
  const cdp=await context.newCDPSession(page);await cdp.send('Network.enable');
  await page.goto(url+'/?session='+encodeURIComponent(oracle.session));
  const card=page.locator(`[data-message-id="${sample.messageId}"] .ak-attachment__surface`);
  await card.waitFor({state:'visible'});await card.scrollIntoViewIfNeeded();
  await page.evaluate(()=>document.fonts.ready);await page.waitForLoadState('networkidle');
  const settings=report.plan.profiles[profile];
  await cdp.send('Network.emulateNetworkConditions',{offline:false,latency:settings.latency,downloadThroughput:settings.bps/8,uploadThroughput:settings.bps/8});
  let failed=false;
  try {
    for(const temperature of ['cold','hot']) {
      const row={variant,profile,round,name:sample.name,temperature,success:false,requests:[]};report.samples.push(row);
      const requests=new Map(), start=performance.now();
      const onRequest=event=>{
        if(new URL(event.request.url).pathname!==`/api/chat/artifacts/${sample.id}`)return;
        const item={id:event.requestId,method:event.request.method,startMs:performance.now()-start};
        requests.set(event.requestId,item);row.requests.push(item);
      };
      const onResponse=event=>{const item=requests.get(event.requestId);if(item)Object.assign(item,{status:event.response.status,fromDiskCache:event.response.fromDiskCache,fromServiceWorker:event.response.fromServiceWorker});};
      const onExtra=event=>{const item=requests.get(event.requestId);if(item)item.wireStatus=event.statusCode;};
      const onEnd=event=>{const item=requests.get(event.requestId);if(item)Object.assign(item,{endMs:performance.now()-start,encodedDataLength:event.encodedDataLength});};
      const onFail=event=>{const item=requests.get(event.requestId);if(item)item.error=event.errorText;};
      cdp.on('Network.requestWillBeSent',onRequest);cdp.on('Network.responseReceived',onResponse);cdp.on('Network.responseReceivedExtraInfo',onExtra);cdp.on('Network.loadingFinished',onEnd);cdp.on('Network.loadingFailed',onFail);
      try {
        await arm(page,sample);await card.click();Object.assign(row,await page.evaluate(()=>window.__attachmentDone));
        await page.waitForLoadState('networkidle');
        assert.equal(errors.length,0,errors.join('\n'));
        assert.equal(row.requests.length,temperature==='cold'&&variant==='baseline'?2:1,'冗余请求数量不符合基线/候选合同');
        for(const request of row.requests){assert.equal(request.method,'GET');assert.ok(!request.error);assert.ok([200,304].includes(request.status));assert.equal(typeof request.endMs,'number');}
        row.success=true;
      }catch(error){row.diagnostic=await page.evaluate(()=>window.__attachmentDiagnostic);throw error;}
      finally {
        cdp.off('Network.requestWillBeSent',onRequest);cdp.off('Network.responseReceived',onResponse);cdp.off('Network.responseReceivedExtraInfo',onExtra);cdp.off('Network.loadingFinished',onEnd);cdp.off('Network.loadingFailed',onFail);save();
      }
      if(values.mode==='correctness'&&temperature==='hot') {
        const original=readFileSync(join(state,'originals',sample.name));assert.equal(hash(original),sample.sha256);
        const copy=page.getByRole('button',{name:'复制全文',exact:true});
        assert.equal(await copy.isEnabled(),sample.copyable);
        if(sample.copyable){await copy.click();await page.getByRole('button',{name:'已复制',exact:true}).waitFor();assert.equal(await page.evaluate(()=>navigator.clipboard.readText()),original.toString('utf8'));row.copyExact=true;}
        else row.copyDisabled=true;
        const pending=page.waitForEvent('download');await page.getByRole('link',{name:'下载 '+sample.name,exact:true}).click();
        const download=await pending;assert.equal(await download.failure(),null);assert.ok(readFileSync(await download.path()).equals(original));row.downloadExact=true;save();
      }
      await page.getByRole('button',{name:'关闭',exact:true}).click();
      await page.locator('.attachment-preview').waitFor({state:'detached'});
    }
  }catch(error){failed=true;throw error;}
  finally {
    try{await context.close();}
    catch(error){
      (report.contextCloseErrors??=[]).push(String(error));
      if(!failed)throw error;
    }
  }
}

try {
  browser=await chromium.launch({executablePath:values.browser});report.browserVersion=browser.version();
  // 原文 hash 与真实 HTTP 完整字节先核对；发生在计时外且不共享浏览器缓存。
  report.artifactChecks=[];
  for(const sample of oracle.samples){
    const bytes=readFileSync(join(state,'originals',sample.name));assert.equal(hash(bytes),sample.sha256);assert.equal(bytes.length,sample.bytes);
    const response=await fetch(`${url}/api/chat/artifacts/${sample.id}`);assert.equal(response.status,200);
    assert.ok(Buffer.from(await response.arrayBuffer()).equals(bytes));assert.equal(response.headers.get('cache-control'),'private, no-cache');
    report.artifactChecks.push({name:sample.name,bytes:bytes.length,sha256:sample.sha256});
  }
  const missing=await fetch(url+'/api/chat/artifacts/parallel-attachment-missing');assert.equal(missing.status,404);report.missingArtifactStatus=404;
  const selected=oracle.samples.filter(sample=>values.mode==='correctness'||sample.group==='performance');
  for(const profile of values.mode==='performance'?['lan','weak']:['lan']) {
    for(let round=0;round<(values.mode==='performance'?rounds:1);round++) {
      for(const variant of round%2?['candidate','baseline']:['baseline','candidate']) {
        await install(variant);
        for(const sample of selected)await measure(variant,profile,round,sample);
      }
    }
  }
  assert.equal(report.samples.length,report.plan.plannedSamples);
  if(values.mode==='performance'){
    report.performanceGates=score();save();
    assert.ok(report.performanceGates.passed,'冻结性能门槛未通过，不能声称验收完成');
  }
  report.integrity=verifyState();report.complete=true;
}catch(error){
  report.error=String(error);process.exitCode=1;
  try{if(current&&!current.isClosed())await current.screenshot({path:join(output,'failure.png')});}
  catch(screenshotError){report.screenshotError=String(screenshotError);}
}
finally {
  // 关闭、恢复与完整性核对各自尝试；任何一步失败都不能跳过下一步。
  try{await browser?.close();}
  catch(error){report.complete=false;report.browserCloseError=String(error);process.exitCode=1;}
  try{await install('baseline');report.restoredBaseline=true;}
  catch(error){report.complete=false;report.restoreError=String(error);process.exitCode=1;}
  try{report.finalIntegrity=verifyState();}
  catch(error){report.complete=false;report.integrityError=String(error);process.exitCode=1;}
  save();
}
