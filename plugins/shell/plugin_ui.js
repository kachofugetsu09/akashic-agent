import{forwardRef as U,createElement as O}from"react";var s=(...a)=>a.filter((e,t,u)=>!!e&&e.trim()!==""&&u.indexOf(e)===t).join(" ").trim();var I=a=>a.replace(/([a-z0-9])([A-Z])/g,"$1-$2").toLowerCase();var x=a=>a.replace(/^([A-Z])|[\s-_]+(\w)/g,(e,t,u)=>u?u.toUpperCase():t.toLowerCase());var p=a=>{let e=x(a);return e.charAt(0).toUpperCase()+e.slice(1)};import{forwardRef as b,createElement as h}from"react";var i={xmlns:"http://www.w3.org/2000/svg",width:24,height:24,viewBox:"0 0 24 24",fill:"none",stroke:"currentColor",strokeWidth:2,strokeLinecap:"round",strokeLinejoin:"round"};var C=a=>{for(let e in a)if(e.startsWith("aria-")||e==="role"||e==="title")return!0;return!1};import{createContext as T,useContext as q,useMemo as $,createElement as aa}from"react";var y=T({});var S=()=>q(y);var g=b(({color:a,size:e,strokeWidth:t,absoluteStrokeWidth:u,className:l="",children:o,iconNode:c,...L},w)=>{let{size:n=24,strokeWidth:m=2,absoluteStrokeWidth:A=!1,color:B="currentColor",className:M=""}=S()??{},D=u??A?Number(t??m)*24/Number(e??n):t??m;return h("svg",{ref:w,...i,width:e??n??i.width,height:e??n??i.height,stroke:a??B,strokeWidth:D,className:s("lucide",M,l),...!o&&!C(L)&&{"aria-hidden":"true"},...L},[...c.map(([F,R])=>h(F,R)),...Array.isArray(o)?o:[o]])});var k=(a,e)=>{let t=U(({className:u,...l},o)=>O(g,{ref:o,iconNode:e,className:s(`lucide-${I(p(a))}`,`lucide-${a}`,u),...l}));return t.displayName=p(a),t};var H=[["path",{d:"M18 6 6 18",key:"1bl5f8"}],["path",{d:"m6 6 12 12",key:"d8bk6v"}]],r=k("x",H);import{formatDuration as v,render as G,ToolData as W,ToolText as V}from"@akashic/chat-ui-v1";import{Fragment as X,jsx as d,jsxs as f}from"react/jsx-runtime";function P(a){if(a===null||typeof a!="object"||Array.isArray(a))return null;let{command:e,output:t,exit_code:u,wall_time_ms:l,process_status:o,...c}=a;return typeof e!="string"||typeof t!="string"?null:{command:e,output:t,exitCode:typeof u=="number"?u:void 0,wallTimeMs:typeof l=="number"?l:void 0,status:typeof o=="string"?o:void 0,rest:{...c,...typeof o=="string"?{process_status:o}:{}}}}function E({result:a}){let e=a.exitCode!==void 0&&a.exitCode!==0,t=a.exitCode===void 0?d("span",{children:a.status??"\u8FD0\u884C\u4E2D"}):e?f("span",{className:"shell-result-exit",children:[d(r,{size:13,"aria-hidden":"true"}),"\u9000\u51FA\u7801 ",a.exitCode]}):null,u=a.wallTimeMs!==void 0?d("span",{children:v(a.wallTimeMs)}):null;return f(X,{children:[f("div",{className:"shell-result",children:[f("div",{className:"shell-result-command",children:[d("span",{"aria-hidden":"true",children:"$"}),d("span",{children:a.command})]}),a.output.trim()?d(V,{text:a.output,tail:!0,className:"tool-result"}):d("p",{className:"shell-result-empty",children:"\u65E0\u8F93\u51FA"}),t||u?f("div",{className:"shell-result-foot",children:[t,u]}):null]}),Object.keys(a.rest).length?f("details",{className:"shell-result-raw",children:[d("summary",{children:"\u539F\u59CB\u6570\u636E"}),d(W,{value:a.rest})]}):null]})}var z={slots:{"tool.result":{accepts:a=>P(a.value)!==null,mount(a,e){let t=P(e.block.value);return t?G(a,d(E,{result:t})):void 0}}}},Pa=z;export{Pa as default};
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
lucide-react/dist/esm/icons/x.mjs:
lucide-react/dist/esm/lucide-react.mjs:
  (**
   * @license lucide-react v1.28.0 - ISC
   *
   * This source code is licensed under the ISC license.
   * See the LICENSE file in the root directory of this source tree.
   *)
*/
