var g=`<!doctype html>
<html data-theme="light">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <meta http-equiv="Content-Security-Policy" content="default-src 'none'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; img-src data:; font-src data:; connect-src 'none'; frame-src 'none'; object-src 'none'; base-uri 'none'; form-action 'none'">
  <style>
    :root {
      color-scheme: light dark;
      --ak-type-ui: system-ui, -apple-system, "PingFang SC", "Noto Sans CJK SC", "Microsoft YaHei", sans-serif;
      font-family: var(--ak-type-reading, "Source Serif 4"), "Noto Serif SC", "Noto Serif CJK SC", serif;
      color: var(--ak-color-text-primary);
      background: transparent;
    }
    * { box-sizing: border-box; }
    body { margin: 0; padding: 0; overflow-wrap: anywhere; font-size: 16px; line-height: 1.65; }
    h1, h2, h3, p { margin: 0 0 12px; }
    h2 { padding-inline-end: 44px; font-size: 1.0625rem; font-weight: 600; margin-bottom: 2px; }
    p { line-height: 1.65; }
    figure { margin: 0; }
    /* \u9ED8\u8BA4\u6392\u7248\u7EC4\u4EF6\uFF1A\u6A21\u578B\u76F4\u63A5\u5199\u539F\u751F HTML\uFF0C\u4E5F\u548C\u804A\u5929\u6B63\u6587\u540C\u4E00\u5957\u89C2\u611F\u3002 */
    .caption { color: var(--ak-color-text-secondary); font-size: .875rem; margin-bottom: 16px; }
    .note { color: var(--ak-color-text-muted); font-size: .8125rem; margin: 12px 0 0; }
    .readout { color: var(--ak-color-text-secondary); font-size: .875rem; margin: 8px 0 0; font-variant-numeric: tabular-nums; }
    svg, canvas { display: block; max-width: 100%; }
    .chart { width: 100%; height: 280px; margin-top: 8px; }
    svg text { font: 12px/1 var(--ak-type-ui); font-variant-numeric: tabular-nums; }
    svg text:not([fill]) { fill: var(--ak-color-text-muted); }
    .grid { stroke: var(--ak-color-border-default); stroke-dasharray: 2 5; fill: none; }
    .axis { stroke: var(--ak-color-border-default); fill: none; }
    .legend { display: flex; justify-content: center; flex-wrap: wrap; gap: 6px 24px; margin-top: 8px; font-size: .875rem; }
    .legend span { display: inline-flex; align-items: center; gap: 8px; }
    .legend i { width: 9px; height: 9px; border-radius: 50%; background: var(--series, var(--ak-chart-1)); }
    /* \u63A7\u4EF6\uFF1A\u53C2\u6570\u884C\u3001\u6309\u94AE\u7EC4\u3001\u6ED1\u5757 */
    .param { display: grid; grid-template-columns: 1fr auto; align-items: center; gap: 0 12px; margin: 16px 0 0; font-size: .875rem; }
    .param input { grid-column: 1 / -1; }
    .param output { font: 600 .9375rem var(--ak-type-ui); }
    .controls { display: flex; flex-wrap: wrap; align-items: center; gap: 8px 12px; margin: 16px 0 0; }
    output { font-variant-numeric: tabular-nums; }
    button, input, select { font: inherit; accent-color: var(--ak-color-action-primary); }
    button, select {
      min-height: 36px;
      padding: 6px 14px;
      color: var(--ak-color-text-primary);
      background: transparent;
      border: 1px solid var(--ak-color-border-default);
      border-radius: 6px;
      font-size: .875rem;
      cursor: pointer;
    }
    button:hover, select:hover { border-color: var(--ak-color-text-muted); }
    button.primary { color: var(--ak-color-bg-canvas); background: var(--ak-color-action-primary); border-color: var(--ak-color-action-primary); }
    button:disabled { opacity: .5; cursor: default; }
    @media (pointer: coarse) { button, select { min-height: 44px; } }
    label { display: block; }
    input[type=range] {
      --fill: 50%;
      width: 100%; min-height: 44px; margin: 0; cursor: pointer; appearance: none; -webkit-appearance: none; background: transparent;
    }
    input[type=range]::-webkit-slider-runnable-track {
      height: 4px; border-radius: 2px;
      background: linear-gradient(to right, var(--ak-color-action-primary) var(--fill), var(--ak-color-border-default) var(--fill));
    }
    input[type=range]::-webkit-slider-thumb {
      -webkit-appearance: none; width: 18px; height: 18px; margin-top: -7px; border-radius: 50%;
      background: var(--ak-color-bg-canvas); border: 2px solid var(--ak-color-action-primary);
    }
    input[type=range]::-moz-range-track { height: 4px; border-radius: 2px; background: var(--ak-color-border-default); }
    input[type=range]::-moz-range-progress { height: 4px; border-radius: 2px; background: var(--ak-color-action-primary); }
    input[type=range]::-moz-range-thumb {
      width: 14px; height: 14px; border-radius: 50%; background: var(--ak-color-bg-canvas); border: 2px solid var(--ak-color-action-primary);
    }
    button:focus-visible, select:focus-visible {
      outline: 2px solid var(--ak-color-action-primary);
      outline-offset: 2px;
    }
    input[type=range]:focus-visible { outline: none; }
    input[type=range]:focus-visible::-webkit-slider-thumb { box-shadow: 0 0 0 3px var(--ak-color-bg-canvas), 0 0 0 5px var(--ak-color-action-primary); }
    input[type=range]:focus-visible::-moz-range-thumb { box-shadow: 0 0 0 3px var(--ak-color-bg-canvas), 0 0 0 5px var(--ak-color-action-primary); }
    /* \u7F16\u8F91\u5F0F\u6570\u636E\u56FE\uFF1A\u7ED3\u8BBA\u6807\u9898\u3001\u5927\u53F7\u6570\u5B57\u3001\u76F4\u63A5\u6CE8\u91CA\u7684\u56FE\u3001\u660E\u7EC6\u8868\u4E0E\u811A\u6CE8\uFF1B\u6570\u5B57\u548C\u56FE\u5185\u6587\u5B57\u7528\u754C\u9762\u65E0\u886C\u7EBF\u5B57\u3002 */
    :root {
      --k1: var(--ak-chart-1); --k2: var(--ak-chart-5); --k3: var(--ak-chart-3);
      --k4: var(--ak-chart-2); --k5: var(--ak-chart-4); --k6: var(--ak-chart-6);
      --k-hot: var(--ak-chart-2); --k-gray: color-mix(in oklab, var(--ak-color-text-muted) 55%, var(--ak-color-bg-surface));
    }
    .headline { margin: 0 0 4px; font: 700 1.375rem/1.25 var(--ak-type-ui); letter-spacing: -.01em; text-wrap: balance; }
    .headline em { font-style: normal; color: color-mix(in oklab, var(--k-hot) 82%, var(--ak-color-text-primary)); }
    .dek { margin: 0 0 4px; font: .9375rem/1.5 var(--ak-type-ui); color: var(--ak-color-text-secondary); text-wrap: pretty; }
    .dek b { color: var(--ak-color-text-primary); font-weight: 600; }
    .hero { display: grid; grid-template-columns: repeat(auto-fit, minmax(128px, 1fr)); gap: 16px 24px; margin: 18px 0 22px; font-family: var(--ak-type-ui); }
    .hero .v { display: block; font: 700 2.25rem/1.05 var(--ak-type-ui); letter-spacing: -.02em; font-variant-numeric: lining-nums tabular-nums; }
    .hero .v small { margin-inline-start: 2px; font-size: .55em; font-weight: 600; letter-spacing: 0; }
    .hero .k { display: block; margin-top: 8px; font-size: .875rem; font-weight: 600; line-height: 1.35; }
    .hero .d { display: block; font-size: .8125rem; line-height: 1.4; color: var(--ak-color-text-secondary); }
    .chart-title { margin: 0 0 6px; font: 600 .9375rem/1.4 var(--ak-type-ui); }
    .chart-sub { margin: -4px 0 8px; font: .8125rem/1.45 var(--ak-type-ui); color: var(--ak-color-text-secondary); }
    .split { display: grid; grid-template-columns: minmax(0, 1.35fr) minmax(0, 1fr); gap: 20px 28px; align-items: start; }
    @media (max-width: 600px) { .split { grid-template-columns: minmax(0, 1fr); } }
    .dtable { width: 100%; border-collapse: collapse; font: .8125rem/1.45 var(--ak-type-ui); color: var(--ak-color-text-secondary); }
    .dtable th { padding: 0 10px 6px 0; text-align: start; font-weight: 500; color: var(--ak-color-text-muted); border-bottom: .75px solid var(--ak-color-border-default); }
    .dtable td { padding: 5px 10px 5px 0; vertical-align: baseline; }
    .dtable td:first-child { color: var(--ak-color-text-primary); font-weight: 600; }
    .dtable :is(th, td):last-child { padding-inline-end: 0; }
    .dtable .n { text-align: end; color: var(--ak-color-text-primary); font-variant-numeric: lining-nums tabular-nums; white-space: nowrap; }
    .dtable tfoot td { border-top: .75px solid var(--ak-color-border-strong); color: var(--ak-color-text-primary); font-weight: 600; }
    .tag { margin-inline-end: 4px; font-weight: 700; color: var(--ak-color-text-primary); }
    .foot { margin: 16px 0 0; font: .75rem/1.55 var(--ak-type-ui); color: var(--ak-color-text-muted); text-wrap: pretty; }
    .ak-chart { position: relative; margin: 2px 0 8px; font-family: var(--ak-type-ui); }
    .ak-chart svg { width: 100%; height: auto; overflow: visible; }
    .ak-chart svg:focus-visible { outline: 2px solid var(--ak-color-action-primary); outline-offset: 4px; }
    .ak-chart svg text { fill: var(--ak-color-text-secondary); font-variant-numeric: lining-nums tabular-nums; }
    .ak-chart svg .ak-b { font-weight: 600; fill: var(--ak-color-text-primary); }
    .ak-chart svg .ak-m { fill: var(--ak-color-text-muted); }
    .ak-grid { stroke: var(--ak-color-border-default); }
    .ak-base { stroke: var(--ak-color-border-strong); stroke-width: .75; }
    .ak-ref { stroke: var(--ak-color-text-primary); stroke-width: 1.5; }
    .ak-lead { stroke: var(--ak-color-text-muted); stroke-width: .75; stroke-dasharray: 2 3; }
    .ak-cross { stroke: var(--ak-color-text-muted); }
    .ak-sep { stroke: var(--ak-color-bg-canvas); }
    .ak-legend { display: flex; flex-wrap: wrap; gap: 4px 16px; margin: 0 0 8px; font: .8125rem/1.5 var(--ak-type-ui); color: var(--ak-color-text-secondary); }
    .ak-legend i { display: inline-block; width: 10px; height: 10px; margin-inline-end: 6px; border-radius: 2px; vertical-align: -1px; }
    .ak-legend b { color: var(--ak-color-text-primary); font-weight: 600; font-variant-numeric: tabular-nums; }
    .ak-tip { position: absolute; z-index: 2; pointer-events: none; padding: 6px 10px; font: 12px/1.55 var(--ak-type-ui); color: var(--ak-color-text-secondary);
      background: var(--ak-color-bg-surface); border: .75px solid var(--ak-color-border-strong); border-radius: 6px; white-space: nowrap; font-variant-numeric: tabular-nums; }
    .ak-tip b { color: var(--ak-color-text-primary); font-weight: 600; }
    .ak-tip i { display: inline-block; width: 8px; height: 8px; margin-inline-end: 6px; border-radius: 2px; }
    .ak-sr { position: absolute; width: 1px; height: 1px; overflow: hidden; clip-path: inset(50%); white-space: nowrap; }
    @media (prefers-reduced-motion: reduce) {
      *, *::before, *::after {
        animation: none !important;
        transition: none !important;
        scroll-behavior: auto !important;
      }
    }
  </style>
  <style id="visual-theme">/*visual-theme*/</style>
  <script>"use strict";(()=>{var se='system-ui, -apple-system, "Segoe UI", "PingFang SC", "Noto Sans CJK SC", "Microsoft YaHei", sans-serif',P=/^(\\d{4})-(\\d{2})-(\\d{2})/,K=document.createElement("canvas").getContext("2d"),oe=0,f=e=>Math.round(e*10)/10,F=(e,t,n)=>Math.max(t,Math.min(n,e)),v=e=>String(e??"").replace(/[&<>"']/g,t=>\`&#\${t.charCodeAt(0)};\`),E=(e,t=12,n=400)=>K?(K.font=\`\${n} \${t}px \${se}\`,K.measureText(String(e)).width):String(e).length*t*.6,J=e=>{let t=P.exec(e);return t?\`\${Number(t[2])}/\${Number(t[3])}\`:e};function X(e,t){return e==="gray"?"var(--k-gray)":e==="hot"?"var(--k-hot)":\`var(--k\${(typeof e=="number"&&Number.isInteger(e)&&e>0?e-1:t)%6+1})\`}function Q(e,t=""){if(typeof e=="function")return e;let n=e==="wan"?[[1e8,"\\u4EBF",1],[1e4,"\\u4E07",1]]:e==="si"?[[1e9,"B",1],[1e6,"M",1],[1e3,"K",10]]:[];return r=>{let s=Math.abs(r);if(e==="%")return\`\${s>=10||Number.isInteger(r)?Math.round(r):f(r)}%\`;let a=n.find(([u,,c],h)=>Math.abs(f(r/u))>=c&&(h<n.length-1||Math.abs(Number(r.toFixed(1)))>=u*c));if(a)return\`\${f(r/a[0])}\${a[1]}\${t}\`;let o=Number.isInteger(r)?r:Number(r.toPrecision(s<1?2:Math.max(2,Math.floor(Math.log10(s))+2)));return(o===0?0:o).toLocaleString("en-US",{maximumFractionDigits:6})+t}}function Z(e,t,n){t>e||(t=e+(Math.abs(e)*.1||1));let r=(t-e)/n,s=10**Math.floor(Math.log10(r)),a=r/s,o=(a<=1?1:a<=2?2:a<=2.5?2.5:a<=5?5:10)*s,u=Math.floor(e/o+1e-9)*o,c=Math.ceil(t/o-1e-9)*o,h=Math.round((c-u)/o);if(![o,u,c,h].every(Number.isFinite)||!(o>0)||h<1||h>20||u+o===u)throw new Error("ak: \\u6570\\u503C\\u8303\\u56F4\\u8D85\\u51FA\\u53EF\\u7ED8\\u5236\\u7CBE\\u5EA6\\uFF0C\\u8BF7\\u5148\\u6362\\u7B97\\u5355\\u4F4D\\uFF08\\u5982\\u4E07\\u3001\\u4EBF\\uFF09\\u518D\\u753B");let m=Array.from({length:h+1},(g,M)=>Number((u+M*o).toPrecision(12)));return{low:u,high:c,ticks:m}}function ie(e,t){if(e.length>35&&e.every(a=>P.test(a))){let a=new Set(e.map(g=>g.slice(0,4))).size>1,o=[],u="";e.forEach((g,M)=>{let b=P.exec(g),x=b[1]+b[2];x!==u&&(M>0||Number(b[3])<=3)&&o.push({index:M,text:a&&b[2]==="01"?b[1]:\`\${Number(b[2])}\\u6708\`}),u=x});let c=Math.max(...o.map(g=>E(g.text)))+12,h=t/e.length,m=[];for(let g of o)(!m.length||(g.index-m[m.length-1].index)*h>=c)&&m.push(g);return m}let n=e.map(J),r=Math.max(0,...n.map(a=>E(a)))+12,s=Math.max(1,Math.ceil(r/(t/Math.max(1,e.length))));return n.map((a,o)=>({index:o,text:a})).filter(a=>a.index%s===0)}function le(e,t,n,r){if(!Array.isArray(t)||!t.length)throw new Error(\`ak.\${e}: series \\u81F3\\u5C11\\u9700\\u8981\\u4E00\\u4E2A\\u7CFB\\u5217\`);return t.map((s,a)=>{let o=s?.values??[];if(o.length!==n)throw new Error(\`ak.\${e}: series[\${a}].values \\u6709 \${o.length} \\u4E2A\\u503C\\uFF0Cx \\u6709 \${n} \\u4E2A\`);return{name:String(s.name??\`\\u7CFB\\u5217 \${a+1}\`),color:X(s.color,a),values:o.map((u,c)=>{if(u==null&&r)return null;if(typeof u!="number"||!Number.isFinite(u))throw new Error(\`ak.\${e}: series[\${a}].values[\${c}] \\u4E0D\\u662F\\u6709\\u9650\\u6570\\u5B57\`);return u})}})}function ce(e,t){let n=e.indexOf(String(t));return n>=0?n:typeof t=="number"&&Number.isInteger(t)&&t>=0&&t<e.length?t:-1}var D=class{constructor(t,n){this.width=t;this.height=n}boxes=[];block(t){this.boxes.push(t)}place(t,n=!1){let s=t.map(c=>({...c,x:F(c.x,0,Math.max(0,this.width-c.w))})).filter(c=>c.y>=0&&c.y+c.h<=this.height).find(c=>!this.boxes.some(h=>c.x<h.x+h.w+2&&h.x<c.x+c.w+2&&c.y<h.y+h.h+1&&h.y<c.y+c.h+1)),a=t[0],o=a&&{...a,x:F(a.x,0,Math.max(0,this.width-a.w)),y:F(a.y,0,Math.max(0,this.height-a.h))},u=s??(n?o??null:null);return u&&this.boxes.push(u),u}};function G(e,t,n=400){let r=[],s="";for(let a of Array.from(e))if(s&&E(s+a,12,n)>t){let o=s.lastIndexOf(" ");r.push(o>0&&a!==" "?s.slice(0,o):s),s=o>0&&a!==" "?s.slice(o+1)+a:a.trimStart()}else s+=a;return s?[...r,s]:r}function ue(e,t,n,r,s=8){let a=String(t.text??t.label??"");if(!a)return"";let o=Math.max(48,Math.min(240,e.width-4)),u=G(a,o,600),c=t.sub?G(t.sub,o):[],h=Math.max(...u.map(x=>E(x,12,600)),...c.map(x=>E(x))),m=16*(u.length+c.length),g=(x,w)=>({x,y:w,w:h,h:m}),M=e.place([g(n-h/2,r-s-m),g(n-h+8,r-s-m),g(n-8,r-s-m),g(n+s,r-m/2),g(n-s-h,r-m/2),g(n-h/2,r+s),g(n-h/2,r-s-m-20),g(n-h-s,r-s-m),g(n+s,r-s-m)],!0);if(!M)return"";let b=f(M.x+h/2);return[...u.map(x=>[x,' class="ak-b"']),...c.map(x=>[x,""])].map(([x,w],y)=>\`<text\${w} x="\${b}" y="\${f(M.y+12+y*16)}" text-anchor="middle">\${v(x)}</text>\`).join("")}var ee=(e,t,n)=>\`<svg viewBox="0 0 \${e} \${t}" width="\${e}" height="\${t}" role="img" aria-label="\${v(n||"\\u56FE\\u8868")}">\`,U=e=>\`<div class="ak-legend">\${e.map(([t,n,r])=>\`<span><i style="background:\${t}"></i>\${n==null?"":\`<b>\${v(n)}</b> \`}\${v(r)}</span>\`).join("")}</div>\`,te=(e,t,n)=>\`<b>\${v(e)}</b>\${n==null?"":\` \\xB7 \\u5408\\u8BA1 <b>\${v(n)}</b>\`}\`+t.map(([r,s,a])=>\`<br><i style="background:\${a}"></i>\${v(r)} <b>\${v(s)}</b>\`).join(""),me=(e,t,n,r,s)=>e.map(a=>\`<line class="ak-grid" x1="\${n}" x2="\${r}" y1="\${f(t(a))}" y2="\${f(t(a))}"/><text x="\${n-8}" y="\${f(t(a))+4}" text-anchor="end">\${v(s(a))}</text>\`).join(""),j=(e,t,n,r)=>{let s=E(n)/2;return\`<text x="\${f(F(e,s,r-s))}" y="\${t}" text-anchor="middle">\${v(n)}</text>\`},Y=new WeakMap;function he(e,t){let n=typeof e=="string"?document.querySelector(e):e;if(!(n instanceof HTMLElement))throw new Error(\`ak: \\u627E\\u4E0D\\u5230\\u5BB9\\u5668 \${String(e)}\`);n.classList.add("ak-chart");let r=0,s=()=>{let o=Math.max(160,Math.floor(n.clientWidth||640));if(o===r)return;r=o;let u=t(o,\`ak\${oe+=1}\`);n.innerHTML=u.html;let c=n.querySelector("svg");c&&u.hits.length&&ge(n,c,u.hits)};s(),Y.get(n)?.disconnect();let a=new ResizeObserver(s);return a.observe(n),Y.set(n,a),{redraw:()=>{r=0,s()}}}function ge(e,t,n){let r=document.createElement("div"),s=document.createElement("span"),a=t.querySelector(".ak-cross");r.className="ak-tip",r.hidden=!0,s.className="ak-sr",s.setAttribute("role","status"),e.append(r,s);let o=-1,u=()=>t.getBoundingClientRect().width/(t.viewBox.baseVal.width||1),c=()=>{r.hidden=!0,o=-1,a?.style.setProperty("visibility","hidden")},h=(m,g,M)=>{let b=n[m];o=m,r.innerHTML=b.html,r.hidden=!1,s.textContent=r.textContent;let x=e.getBoundingClientRect(),w=t.getBoundingClientRect(),y=u(),p=w.left-x.left+(g??(b.x0+b.x1)/2)*y,A=w.top-x.top+(M??b.y0)*y,C=p+14+r.offsetWidth>e.clientWidth?Math.max(0,p-r.offsetWidth-14):p+14;r.style.left=\`\${C}px\`,r.style.top=\`\${Math.max(0,A-r.offsetHeight-10)}px\`,a&&b.cx!=null&&(a.setAttribute("x1",String(b.cx)),a.setAttribute("x2",String(b.cx)),a.style.visibility="visible")};t.tabIndex=0,t.addEventListener("pointermove",m=>{let g=t.getBoundingClientRect(),M=u(),b=(m.clientX-g.left)/M,x=(m.clientY-g.top)/M,w=n.findIndex(y=>b>=y.x0&&b<=y.x1&&x>=y.y0&&x<=y.y1);w<0?c():h(w,b,x)}),t.addEventListener("pointerleave",c),t.addEventListener("blur",c),t.addEventListener("keydown",m=>{if(m.key==="Escape")return c();let g={ArrowRight:1,ArrowDown:1,ArrowLeft:-1,ArrowUp:-1}[m.key];g&&(m.preventDefault(),h(o<0?g>0?0:n.length-1:(o+g+n.length)%n.length))})}function be(e,t){let n=(t?.x??[]).map(String);if(!n.length)throw new Error("ak.bars: x \\u4E0D\\u80FD\\u4E3A\\u7A7A");let r=le("bars",t.series,n.length,!1),s=Q(t.fmt,t.unit);return t.horizontal&&fe(t,n.length,r),he(e,(a,o)=>t.horizontal?ye(a,o,t,n,r,s):xe(a,t,n,r,s))}function fe(e,t,n){if(n.some(r=>r.values.some(s=>(s??0)<0)))throw new Error("ak.bars: \\u6A2A\\u5411\\u6761\\u4E0D\\u652F\\u6301\\u8D1F\\u6570\\uFF0C\\u8BF7\\u6539\\u7528\\u7AD6\\u5411\\u67F1");(e.range??[]).slice(0,t).forEach((r,s)=>{if(r!=null){if(![r.lo,r.hi].every(a=>typeof a=="number"&&Number.isFinite(a)))throw new Error(\`ak.bars: range[\${s}] \\u7684 lo\\u3001hi \\u5FC5\\u987B\\u662F\\u6709\\u9650\\u6570\\u5B57\`);if(r.lo<0||r.lo>r.hi)throw new Error(\`ak.bars: range[\${s}] \\u9700\\u8981 0 \\u2264 lo \\u2264 hi\`)}})}function xe(e,t,n,r,s){let a=t.stack!==!1,o=n.length,u=d=>r.map(i=>i.values[d]??0),c=n.map((d,i)=>u(i).reduce((l,$)=>l+Math.max(0,$),0)),h=n.map((d,i)=>u(i).reduce((l,$)=>l+Math.min(0,$),0)),m=n.map((d,i)=>a?c[i]:Math.max(0,...u(i))),g=Math.max(0,...m,t.refLine?.value??0),M=Math.min(0,...a?h:r.flatMap(d=>d.values)),b=t.height??(e<480?220:260),x=t.annotate?.length?t.annotate.some(d=>d.sub)?44:28:12,w=24,y=Z(M,g,b<240?3:4),p=Math.max(...y.ticks.map(d=>E(s(d))))+10,A=e-p-4,C=A/o,k=Math.max(1,Math.min(C*(o>40?.8:.66),44)),S=d=>x+(b-x-w)-(d-y.low)/(y.high-y.low)*(b-x-w),O=new D(e,b);O.block({x:0,y:0,w:p-4,h:b}),O.block({x:0,y:b-w+4,w:e,h:w});let N="",V=n.map((d,i)=>{let l=p+C*(i+.5),$=0,T=0;return r.forEach((B,L)=>{let H=B.values[i]??0,I=a?k:k/r.length,re=a?l-k/2:l-k/2+L*I,_=a?H>0?$:T:0,W=_+H;a&&(H>0?$=W:T=W);let q=S(Math.max(_,W)),ae=S(Math.min(_,W));N+=\`<rect x="\${f(re)}" y="\${f(q)}" width="\${f(Math.max(1,I-(a?0:1)))}" height="\${f(Math.max(.5,ae-q))}" style="fill:\${B.color}"\${a&&r.length>1&&k>6?' class="ak-sep"':""}/>\`}),O.block({x:l-k/2,y:S(m[i]),w:k,h:S(0)-S(m[i])}),{x0:l-C/2,x1:l+C/2,y0:x,y1:b-w,html:te(J(d),[...r].reverse().map(B=>[B.name,s(B.values[i]??0),B.color]),a&&r.length>1?s(c[i]+h[i]):void 0)}}),z=me(y.ticks,S,p,e-4,s)+\`<line class="ak-base" x1="\${p}" x2="\${e-4}" y1="\${f(S(0))}" y2="\${f(S(0))}"/>\`;ie(n,A).forEach(d=>{z+=j(p+C*(d.index+.5),b-6,d.text,e)}),N+=de(O,t.refLine,s,{x1:p,x2:e-4,y:S(t.refLine?.value??0)});for(let d of t.annotate??[]){let i=ce(n,d.at);i>=0&&(N+=ue(O,d,p+C*(i+.5),S(m[i]),6))}return{html:(r.length>1&&t.legend!==!1?U(r.map(d=>[d.color,s(d.values.reduce((i,l)=>i+(l??0),0)),d.name])):"")+ee(e,b,t.label)+z+N+"</svg>",hits:V}}function ne(e,t){let n=t(e.value);return e.label?e.label.includes(n)?e.label:\`\${n} \\xB7 \${e.label}\`:n}function de(e,t,n,r){if(!t)return"";let s=ne(t,n),a=E(s,12,600),o=e.place([{x:r.x2-a,y:r.y-18,w:a,h:14},{x:r.x2-a,y:r.y+4,w:a,h:14},{x:r.x1+4,y:r.y-18,w:a,h:14},{x:r.x1+4,y:r.y+4,w:a,h:14}],!0);return\`<line class="ak-ref" x1="\${r.x1}" x2="\${r.x2}" y1="\${f(r.y)}" y2="\${f(r.y)}"/>\`+(o?\`<text class="ak-b" x="\${f(o.x)}" y="\${f(o.y+11)}">\${v(s)}</text>\`:"")}function ye(e,t,n,r,s,a){let o=n.range??[],u=n.sub??[],c=e<420,h=s.length===1,m=r.map((i,l)=>s.reduce(($,T)=>$+Math.max(0,T.values[l]??0),0)),g=r.map((i,l)=>{let $=o[l];return $?\`\${a($.lo)}\\u2013\${a($.hi)}\`:a(m[l])}),M=Math.max(0,n.refLine?.value??0,...r.map((i,l)=>Math.max(m[l],o[l]?.hi??0))),b=Z(0,M,3),x=c?0:Math.min(e*.34,Math.max(...r.map(i=>E(i,13,600)),...u.map(i=>E(i)))+14),w=Math.max(...g.map(i=>E(i,13,600)))+12,y=18,p=y+(c?30:u.some(Boolean)?22:14),A=Math.max(40,e-x-w-4),C=4+r.length*p+24,k=i=>x+F(i,0,b.high)/b.high*A,S=i=>n.colors?X(n.colors[i],i):s[0].color,O="",N="",V=r.map((i,l)=>{let $=4+l*p+(c?22:(p-y)/2),T=$+y/2,B=o[l];if(N+=c?\`<text class="ak-b" x="0" y="\${f($-7)}">\${v(i)}\${u[l]?\`<tspan class="ak-m" font-weight="400"> \${v(u[l])}</tspan>\`:""}</text>\`:\`<text class="ak-b" x="0" y="\${f(u[l]?T-2:T+4)}">\${v(i)}</text>\`+(u[l]?\`<text class="ak-m" x="0" y="\${f(T+13)}">\${v(u[l])}</text>\`:""),B){let L=S(l),H=Math.min(B.lo,B.hi),I=Math.max(B.lo,B.hi);O+=\`<pattern id="\${t}h\${l}" patternUnits="userSpaceOnUse" width="6" height="6" patternTransform="rotate(45)"><rect width="6" height="6" style="fill:\${L};opacity:.14"/><rect width="2.2" height="6" style="fill:\${L}"/></pattern>\`,N+=\`<rect x="\${x}" y="\${f($)}" width="\${f(k(H)-x)}" height="\${y}" style="fill:\${L}"/><rect x="\${f(k(H))}" y="\${f($)}" width="\${f(k(I)-k(H))}" height="\${y}" style="fill:url(#\${t}h\${l})"/>\`}else{let L=0;s.forEach(H=>{let I=Math.max(0,H.values[l]??0);N+=\`<rect x="\${f(k(L))}" y="\${f($)}" width="\${f(k(L+I)-k(L))}" height="\${y}" style="fill:\${h?S(l):H.color}"\${h?"":' class="ak-sep"'}/>\`,L+=I})}return N+=\`<text class="ak-b" x="\${e}" y="\${f(T+5)}" text-anchor="end">\${v(g[l])}</text>\`,{x0:0,x1:e,y0:4+l*p,y1:4+(l+1)*p,html:te(i,B?[[n.rangeLabels?.[1]??"\\u4F30\\u8BA1\\u533A\\u95F4",g[l],S(l)]]:s.map(L=>[L.name,a(L.values[l]??0),h?S(l):L.color]))}}),z=new D(e,C),R=4+r.length*p+16;if(n.refLine){let i=k(n.refLine.value),l=ne(n.refLine,a);N+=\`<line class="ak-ref" x1="\${f(i)}" x2="\${f(i)}" y1="2" y2="\${R-12}"/>\`+j(i,R,l,e).replace("<text",'<text class="ak-b"'),z.block({x:i-E(l,12,600)/2-6,y:R-12,w:E(l,12,600)+12,h:16})}return b.ticks.forEach(i=>{let l=k(i),$=E(a(i))/2;z.place([{x:F(l-$,0,e-2*$),y:R-12,w:2*$,h:16}])&&(N+=j(l,R,a(i),e-w))}),{html:(o.some(Boolean)?U([["var(--ak-color-text-muted)",null,n.rangeLabels?.[0]??"\\u4E0B\\u9650\\u4F30\\u8BA1"],["repeating-linear-gradient(45deg, var(--ak-color-text-muted) 0 2px, transparent 2px 5px)",null,n.rangeLabels?.[1]??"\\u4F30\\u8BA1\\u533A\\u95F4"]]):s.length>1&&n.legend!==!1?U(s.map(i=>[i.color,null,i.name])):"")+ee(e,C,n.label)+\`<defs>\${O}</defs>\`+N+"</svg>",hits:V}}var $e={bars:be,fmt:(e,t,n)=>Q(t,n)(e)};window.ak=$e;})();<\/script>
  <script>
    (() => {
      const send = (type, value) => parent.postMessage({channel: 'akashic.visual.v1', type, value}, '*');
      let queued = false, last = 0;
      const measure = () => {
        queued = false;
        const height = Math.ceil(document.body.getBoundingClientRect().height);
        if (height !== last) { last = height; send('height', height); }
      };
      const resize = () => {
        if (!queued) { queued = true; requestAnimationFrame(measure); }
      };
      addEventListener('error', event => send('error', event.message));
      addEventListener('unhandledrejection', event => send('error', String(event.reason)));
      addEventListener('message', event => {
        if (event.source !== parent || event.data?.channel !== 'akashic.visual.v1' || event.data.type !== 'theme') return;
        document.getElementById('visual-theme').textContent = event.data.value;
        dispatchEvent(new CustomEvent('akashic:theme'));
        resize();
      });
      // \u6ED1\u5757\u5DF2\u586B\u5145\u6BB5\u8DDF\u968F\u6570\u503C\uFF0C\u6A21\u578B\u811A\u672C\u65E0\u9700\u81EA\u5DF1\u7EF4\u62A4\u3002
      const fill = input => {
        if (!(input instanceof HTMLInputElement) || input.type !== 'range') return;
        const min = Number(input.min || 0), max = Number(input.max || 100);
        input.style.setProperty('--fill', \`\${max > min ? (input.valueAsNumber - min) / (max - min) * 100 : 0}%\`);
      };
      addEventListener('input', event => fill(event.target), true);
      addEventListener('DOMContentLoaded', () => {
        document.querySelectorAll('input[type=range]').forEach(fill);
        new ResizeObserver(resize).observe(document.body);
        measure();
        // \u9690\u85CF\u6216\u5C4F\u5916 iframe \u7684 rAF \u53EF\u6682\u505C\uFF1B\u542F\u52A8\u786E\u8BA4\u4E0D\u80FD\u7B49\u5F85\u4E0B\u4E00\u5E27\u3002
        send('ready', true);
      });
    })();
  <\/script>
</head>
<body><!--visual-content--></body>
</html>
`;var u="akashic.visual.v1",y=["type-reading","color-action-primary","color-text-muted","color-text-primary","color-text-secondary","color-bg-canvas","color-bg-surface","color-border-default","color-border-strong",...[1,2,3,4,5,6].map(r=>`chart-${r}`)];function v(r){if(!r||typeof r!="object")return!1;let a=r;return a.origin?.kind==="tool_result"&&a.origin.tool==="render_visual"&&a.kind==="file"&&a.media_type==="application/json"&&!!a.filename?.endsWith(".akashic-visual.json")}function f(r){let a=getComputedStyle(r);return`:root{${y.map(o=>`--ak-${o}:${a.getPropertyValue(`--ak-${o}`)};`).join("")}
    color-scheme:${a.colorScheme};}`}function k(r,a){let o=a.block,l=new AbortController,i=document.createElement("p");i.setAttribute("role","status"),i.textContent="\u6B63\u5728\u6253\u5F00\u4EA4\u4E92\u56FE\u2026";let t=document.createElement("iframe");t.sandbox.add("allow-scripts"),t.referrerPolicy="no-referrer",t.style.cssText="display:block;width:100%;height:320px;border:0;visibility:hidden",r.append(i);let c=!1,m=window.setTimeout(()=>{c||s("\u52A0\u8F7D\u8D85\u65F6")},1e4),s=e=>{window.clearTimeout(m),l.abort(),t.remove(),i.textContent="",a.reportError(e)},d=e=>{e.source!==t.contentWindow||e.data?.channel!==u||(e.data.type==="ready"?(c=!0,window.clearTimeout(m),t.style.visibility="visible",i.textContent=""):e.data.type==="height"&&Number.isFinite(e.data.value)?t.style.height=`${Math.max(80,Math.min(1600,e.data.value))}px`:e.data.type==="error"&&typeof e.data.value=="string"&&(c?i.textContent=`\u672C\u6B21\u4EA4\u4E92\u8BA1\u7B97\u9047\u5230\u95EE\u9898\uFF1A${e.data.value.slice(0,300)}\u3002\u53EF\u4EE5\u8C03\u6574\u53C2\u6570\u518D\u8BD5\u3002`:s(e.data.value.slice(0,300))))};window.addEventListener("message",d);let h=0,p="";t.addEventListener("load",()=>{h+=1,h===1?(t.contentWindow?.postMessage({channel:u,type:"init",document:p},"*"),p=""):h>2&&s("\u6587\u6863\u53D1\u751F\u4E86\u4E0D\u5141\u8BB8\u7684\u5BFC\u822A")});let x=new MutationObserver(()=>t.contentWindow?.postMessage({channel:u,type:"theme",value:f(r)},"*"));return x.observe(document.documentElement,{attributes:!0,attributeFilter:["data-theme","style","class"]}),(async()=>{try{if(o.size_bytes>15e5)throw new Error("\u9644\u4EF6\u8D85\u8FC7\u5927\u5C0F\u9650\u5236");let e=await fetch(`/api/chat/artifacts/${encodeURIComponent(o.artifact_id)}`,{signal:l.signal});if(!e.ok)throw new Error(`\u8BFB\u53D6\u5931\u8D25 (${e.status})`);let n=await e.json();if(!n||n.schema!==u||typeof n.title!="string"||typeof n.summary!="string"||typeof n.html!="string"||Array.from(n.html).length>2e5)throw new Error("\u4E0D\u652F\u6301\u7684\u6587\u6863\u683C\u5F0F");if(l.signal.aborted)return;t.title=n.title,p=g.replace("/*visual-theme*/",()=>f(r)).replace("<!--visual-content-->",()=>n.html),t.src="/visuals/frame.html",r.append(t)}catch(e){l.signal.aborted||s(e instanceof Error?e.message:String(e))}})(),()=>{l.abort(),window.clearTimeout(m),x.disconnect(),window.removeEventListener("message",d),t.remove(),i.remove()}}var M={slots:{"message.attachment":{accepts:v,mount:k}}};export{M as default};
