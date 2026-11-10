// 用法：node scripts/command_suggestions_scenario.mjs <base/commands-sample.json> <distribution/sources/commands/plugin_ui.js>
// Production hook and plugin module; backend payload is captured from an actual bundle runtime.
import { mkdtempSync, symlinkSync, writeFileSync, readFileSync, copyFileSync } from 'node:fs';
import { join } from 'node:path';
import { tmpdir } from 'node:os';
import { pathToFileURL } from 'node:url';
import { build } from 'esbuild';
import { JSDOM } from 'jsdom';
const repo = process.cwd();
const temp = mkdtempSync(join(tmpdir(), 'akashic-command-suggest-'));
symlinkSync(repo + '/node_modules', join(temp, 'node_modules'));
const dom = new JSDOM('<div id="root"></div>', { url: 'http://suggest.test', pretendToBeVisual: true });
const { window } = dom;
for (const key of ['window','document','HTMLElement','Element','Node','Event','MouseEvent','KeyboardEvent','MutationObserver','localStorage','sessionStorage']) {
 Object.defineProperty(globalThis, key, {value:key==='window'?window:window[key],configurable:true});
}
Object.defineProperty(globalThis,'navigator',{value:window.navigator, configurable:true});
window.HTMLElement.prototype.scrollIntoView = function() {};
globalThis.IS_REACT_ACT_ENVIRONMENT = true;
const sample = JSON.parse(readFileSync(process.argv[2], 'utf8'));
globalThis.fetch = async (url, options) => {
 if(url !== '/api/chat/plugin-ui/query' || JSON.parse(options.body).method !== 'commands.list') throw Error('unexpected HTTP request');
 return Response.json(sample);
};
writeFileSync(join(temp,'entry.tsx'), `
import React, {useRef,useState} from 'react';
import {useComposerSuggest} from '${repo}/frontend/plugins/akashic_clients/chat/src/composer-suggest.tsx';
export {receivePluginUiCatalog} from '${repo}/frontend/plugins/akashic_clients/chat/src/plugin-ui-runtime.tsx';
export function Harness() {
 const [value,setValue]=useState(''); const ref=useRef<HTMLTextAreaElement>(null);
 const suggest=useComposerSuggest({value,scope:'test',textareaRef:ref,setValue});
 return <><textarea ref={ref} value={value} onChange={e=>{setValue(e.target.value);suggest.sync()}} onSelect={suggest.sync}
 onFocus={suggest.sync} onKeyDown={suggest.onKeyDown} {...suggest.inputProps}/>{suggest.menu}</>;
}`);
await build({entryPoints:[join(temp,'entry.tsx')],bundle:true,platform:'node',format:'esm',packages:'external',
 alias:{'@':repo+'/frontend/plugins/akashic_clients/chat/src'},define:{'import.meta.env.DEV':'false'},outfile:join(temp,'entry.mjs')});
copyFileSync(process.argv[3], join(temp,'commands.mjs'));
const React = await import('react');
const {createRoot} = await import('react-dom/client');
const {Harness,receivePluginUiCatalog} = await import(pathToFileURL(join(temp,'entry.mjs')));
await receivePluginUiCatalog({catalogRevision:'commands-scenario',updating:false,plugins:[{id:'commands@release',revision:'live',moduleUrl:pathToFileURL(join(temp,'commands.mjs')).href,slots:['composer.suggest']}]});
const root = createRoot(document.getElementById('root'));
await React.act(async()=>{root.render(React.createElement(Harness));});
const field = document.querySelector('textarea');
async function type(value) {
 await React.act(async()=>{
  field.focus();
  Object.getOwnPropertyDescriptor(window.HTMLTextAreaElement.prototype,'value').set.call(field,value);
  field.setSelectionRange(value.length,value.length);
  field.dispatchEvent(new window.Event('input',{bubbles:true}));
  field.dispatchEvent(new window.Event('select',{bubbles:true}));
 });
}
await type('/go');
if(!document.querySelector('[role=option]')?.textContent.includes('goal')) throw Error('prefix did not show goal');
await type('/goal');
const option=document.querySelector('[role=option]');
if(!option || !option.textContent.includes('pause')) throw Error('exact command lost its usage hint');
await React.act(async()=>{field.dispatchEvent(new window.KeyboardEvent('keydown',{key:'Tab',bubbles:true}));});
if(field.value!=='/goal ' || document.querySelector('[role=listbox]')) throw Error('selection did not insert command and close: '+JSON.stringify({value:field.value,body:document.body.innerHTML}));
await type('/goal');
await React.act(async()=>{field.dispatchEvent(new window.KeyboardEvent('keydown',{key:'Escape',bubbles:true}));});
if(document.querySelector('[role=listbox]')) throw Error('Escape did not dismiss');
await React.act(async()=>{root.unmount();});
dom.window.close();
console.log(JSON.stringify({prefix:true,exact_name_hint:true,tab_inserts:true,escape_dismisses:true,boundary:'Actual production React and command plugin in JSDOM with captured runtime query; no layout or browser screenshot'}));
