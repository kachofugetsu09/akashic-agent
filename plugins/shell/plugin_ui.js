import{forwardRef as O,createElement as H}from"react";var i=(...a)=>a.filter((e,t,u)=>!!e&&e.trim()!==""&&u.indexOf(e)===t).join(" ").trim();var C=a=>a.replace(/([a-z0-9])([A-Z])/g,"$1-$2").toLowerCase();var S=a=>a.replace(/^([A-Z])|[\s-_]+(\w)/g,(e,t,u)=>u?u.toUpperCase():t.toLowerCase());var m=a=>{let e=S(a);return e.charAt(0).toUpperCase()+e.slice(1)};import{forwardRef as U,createElement as k}from"react";var c={xmlns:"http://www.w3.org/2000/svg",width:24,height:24,viewBox:"0 0 24 24",fill:"none",stroke:"currentColor",strokeWidth:2,strokeLinecap:"round",strokeLinejoin:"round"};var h=a=>{for(let e in a)if(e.startsWith("aria-")||e==="role"||e==="title")return!0;return!1};import{createContext as q,useContext as y,useMemo as ea,createElement as ta}from"react";var b=q({});var g=()=>y(b);var P=U(({color:a,size:e,strokeWidth:t,absoluteStrokeWidth:u,className:l="",children:o,iconNode:p,...I},A)=>{let{size:L=24,strokeWidth:x=2,absoluteStrokeWidth:B=!1,color:M="currentColor",className:D=""}=g()??{},F=u??B?Number(t??x)*24/Number(e??L):t??x;return k("svg",{ref:A,...c,width:e??L??c.width,height:e??L??c.height,stroke:a??M,strokeWidth:F,className:i("lucide",D,l),...!o&&!h(I)&&{"aria-hidden":"true"},...I},[...p.map(([R,T])=>k(R,T)),...Array.isArray(o)?o:[o]])});var n=(a,e)=>{let t=O(({className:u,...l},o)=>H(P,{ref:o,iconNode:e,className:i(`lucide-${C(m(a))}`,`lucide-${a}`,u),...l}));return t.displayName=m(a),t};var v=[["path",{d:"M20 6 9 17l-5-5",key:"1gmf2c"}]],r=n("check",v);var G=[["path",{d:"M18 6 6 18",key:"1bl5f8"}],["path",{d:"m6 6 12 12",key:"d8bk6v"}]],s=n("x",G);import{formatDuration as W,render as V,ToolData as E,ToolText as z}from"@akashic/chat-ui-v1";import{Fragment as K,jsx as d,jsxs as f}from"react/jsx-runtime";function w(a){if(a===null||typeof a!="object"||Array.isArray(a))return null;let{command:e,output:t,exit_code:u,wall_time_ms:l,process_status:o,...p}=a;return typeof e!="string"||typeof t!="string"?null:{command:e,output:t,exitCode:typeof u=="number"?u:void 0,wallTimeMs:typeof l=="number"?l:void 0,status:typeof o=="string"?o:void 0,rest:{...p,...typeof o=="string"?{process_status:o}:{}}}}function X({result:a}){let e=a.exitCode!==void 0&&a.exitCode!==0;return f(K,{children:[f("div",{className:"shell-result",children:[f("div",{className:"shell-result-command",children:[d("span",{"aria-hidden":"true",children:"$"}),d("span",{children:a.command})]}),a.output.trim()?d(z,{text:a.output,tail:!0,className:"tool-result"}):d("p",{className:"shell-result-empty",children:"\u65E0\u8F93\u51FA"}),f("div",{className:"shell-result-foot",children:[a.exitCode===void 0?d("span",{children:a.status??"\u8FD0\u884C\u4E2D"}):f("span",{className:`shell-result-exit${e?" is-failed":""}`,children:[e?d(s,{size:13,"aria-hidden":"true"}):d(r,{size:13,"aria-hidden":"true"}),e?`\u9000\u51FA\u7801 ${a.exitCode}`:"\u6210\u529F"]}),a.wallTimeMs!==void 0?d("span",{children:W(a.wallTimeMs)}):null]})]}),Object.keys(a.rest).length?f("details",{className:"shell-result-raw",children:[d("summary",{children:"\u539F\u59CB\u6570\u636E"}),d(E,{value:a.rest})]}):null]})}var N={slots:{"tool.result":{accepts:a=>w(a.value)!==null,mount(a,e){let t=w(e.block.value);return t?V(a,d(X,{result:t})):void 0}}}},Da=N;export{Da as default};
/*! Bundled license information:

lucide-react/dist/esm/shared/src/utils/mergeClasses.mjs:
lucide-react/dist/esm/shared/src/utils/toKebabCase.mjs:
lucide-react/dist/esm/shared/src/utils/toCamelCase.mjs:
lucide-react/dist/esm/shared/src/utils/toPascalCase.mjs:
lucide-react/dist/esm/defaultAttributes.mjs:
lucide-react/dist/esm/shared/src/utils/hasA11yProp.mjs:
lucide-react/dist/esm/context.mjs:
lucide-react/dist/esm/Icon.mjs:
lucide-react/dist/esm/createLucideIcon.mjs:
lucide-react/dist/esm/icons/check.mjs:
lucide-react/dist/esm/icons/x.mjs:
lucide-react/dist/esm/lucide-react.mjs:
  (**
   * @license lucide-react v1.28.0 - ISC
   *
   * This source code is licensed under the ISC license.
   * See the LICENSE file in the root directory of this source tree.
   *)
*/
