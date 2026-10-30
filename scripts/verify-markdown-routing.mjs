import assert from 'node:assert/strict';
import { mkdtemp, writeFile, readFile, mkdir } from 'node:fs/promises';
import { resolve } from 'node:path';
import { createServer } from 'node:http';
import { build } from 'esbuild';
import { chromium } from 'playwright-core';
const repo=process.cwd();
const root=await mkdtemp(resolve(repo,'node_modules/.cache-md-routing-'));
const evidence=resolve(process.env.EVIDENCE_DIR||'../evidence');await mkdir(evidence,{recursive:true});
const source=JSON.stringify;
await writeFile(root+'/entry.tsx',`
import React from 'react';import{createRoot}from'react-dom/client';
import{StaticMessageResponse}from ${source(repo+'/frontend/chat/src/static-message-response.tsx')};
import{ReadingInlineCode,ReadingLink}from ${source(repo+'/frontend/chat/src/markdown-reading-nodes.tsx')};
import{connectHost}from ${source(repo+'/frontend/chat/src/chat-host.ts')};
import{setWorkbenchSession,setWorkbenchIntents,FILE_INTENT,DOCUMENT_INTENT}from ${source(repo+'/frontend/chat/src/shell-workbench.ts')};
import{activate}from ${source(repo+'/frontend/plugins/markdown_reader/src/index.tsx')};
window.calls=[];window.reads=[];
const ctx={http:{request:async value=>{const u=new URL(value,location.href);reads.push(u.pathname+u.search);return {ok:true,json:async()=>u.pathname.endsWith('/files')?{status:'available',path:'/repo',root:'/repo',items:[]}:{status:'available',path:u.searchParams.get('path'),text:'# Opened document\\n'+u.searchParams.get('path')}}}},ui:{inject:(_,fn)=>fn({register:entry=>window.entry=entry})}};
activate(ctx);
entry.render(document.querySelector('#reader'),null,{session:'s',active:false,onSessionChange:()=>()=>{},onActiveChange:()=>()=>{},onIntent:fn=>{window.intent=fn;return()=>{}}});
connectHost({requestIntent:async(name,session,payload)=>{calls.push({name,session,payload});return name===DOCUMENT_INTENT?await intent({name,payload}):'handled'},reportSession(){},reportOverlay(){},activateRailAction(){}});
setWorkbenchSession('s');setWorkbenchIntents([FILE_INTENT,DOCUMENT_INTENT]);window.setCaps=setWorkbenchIntents;
const markdown='\`docs/guide.md\` \`/outside/中文 note.MD\` \`../outside/guide.md:12\` \`src/main.ts:4\` [linked](../outside/guide.md) [encoded](/outside/a%20b.md) [web](https://example.com/README.md)';
const root=createRoot(document.querySelector('#root'));
window.render=()=>root.render(<><div id="static"><StaticMessageResponse>{markdown}</StaticMessageResponse></div><div id="stream"><StaticMessageResponse streaming>{markdown}</StaticMessageResponse></div><div id="reading"><ReadingInlineCode node={{type:'inline_code',code:'/outside/中文 note.MD'}}/><ReadingLink ctx={{}} renderNode={()=> 'dynamic link'} node={{type:'link',href:'../outside/guide.md',children:[{type:'text',content:'dynamic link'}],content:'dynamic link'}}/></div></>);render();
`);
await build({entryPoints:[root+'/entry.tsx'],outfile:root+'/bundle.js',bundle:true,external:['@terrastruct/d2','mermaid'],format:'esm',jsx:'automatic',loader:{'.css':'empty'},define:{'process.env.NODE_ENV':'"production"'},logLevel:'silent'});
const server=createServer(async(req,res)=>{res.setHeader('Content-Type',req.url==='/bundle.js'?'text/javascript':'text/html');res.end(req.url==='/bundle.js'?await readFile(root+'/bundle.js'):'<div id="root"></div><div id="reader"></div><script type="module" src="/bundle.js"></script>');});
await new Promise(r=>server.listen(0,'127.0.0.1',r));
const browser=await chromium.launch({executablePath:process.env.CHROMIUM_PATH||'/usr/bin/chromium',args:['--no-sandbox']});
try{const p=await browser.newPage();const errors=[];p.on('pageerror',e=>errors.push(e.message));await p.goto('http://127.0.0.1:'+server.address().port);
for(const area of ['static','stream']){
 await p.locator('#'+area+' button.reading-file-link').first().waitFor();
 for(const path of ['docs/guide.md','/outside/中文 note.MD','../outside/guide.md']){
  await p.locator('#'+area+' button').filter({hasText:path}).click();
  await p.waitForFunction(path=>document.querySelector('.md-name').title===path,path);
  assert.equal(await p.evaluate(()=>calls.at(-1).name),'workspace.document.open.v1');
 }
 await p.locator('#'+area+' button').filter({hasText:'src/main.ts'}).click();assert.equal(await p.evaluate(()=>calls.at(-1).name),'workspace.file.open.v1');
 for(const label of ['linked','encoded']){await p.locator('#'+area).getByRole('link',{name:label,exact:true}).click();assert.equal(await p.evaluate(()=>calls.at(-1).name),'workspace.document.open.v1');}
 assert.equal(await p.locator('#'+area+' a button').count(),0);
}
await p.locator('#reading button').click();assert.equal(await p.evaluate(()=>calls.at(-1).payload.path),'/outside/中文 note.MD');
await p.locator('#reading a').click();assert.equal(await p.evaluate(()=>calls.at(-1).name),'workspace.document.open.v1');
await p.evaluate(()=>setCaps(['workspace.document.open.v1']));await p.waitForFunction(()=>document.querySelectorAll('#static button').length===3);
await p.evaluate(()=>setCaps(['workspace.file.open.v1']));await p.waitForFunction(()=>document.querySelectorAll('#static button').length===1);
assert.match(await p.locator('#static button').innerText(),/main.ts/);
assert.deepEqual(errors,[]);await p.screenshot({path:evidence+'/routing.png'});console.log(JSON.stringify(await p.evaluate(()=>({calls,reads})),null,2));
}finally{await browser.close();server.close();}
