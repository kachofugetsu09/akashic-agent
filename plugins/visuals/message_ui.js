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
  <script>"use strict";(()=>{var re='system-ui, -apple-system, "Segoe UI", "PingFang SC", "Noto Sans CJK SC", "Microsoft YaHei", sans-serif',U=/^(\\d{4})-(\\d{2})-(\\d{2})/,z=document.createElement("canvas").getContext("2d"),se=0,u=t=>Math.round(t*10)/10,V=(t,n,e)=>Math.max(n,Math.min(e,t)),p=t=>String(t??"").replace(/[&<>"']/g,n=>\`&#\${n.charCodeAt(0)};\`),B=(t,n=12,e=400)=>z?(z.font=\`\${e} \${n}px \${re}\`,z.measureText(String(t)).width):String(t).length*n*.6,j=t=>{let n=U.exec(t);return n?\`\${Number(n[2])}/\${Number(n[3])}\`:t};function Y(t,n){return t==="gray"?"var(--k-gray)":t==="hot"?"var(--k-hot)":\`var(--k\${(typeof t=="number"&&Number.isInteger(t)&&t>0?t-1:n)%6+1})\`}function J(t,n=""){return typeof t=="function"?t:e=>{let r=Math.abs(e);if(t==="%")return\`\${r>=10||Number.isInteger(e)?Math.round(e):u(e)}%\`;let a;return t==="wan"&&r>=1e8?a=\`\${u(e/1e8)}\\u4EBF\`:t==="wan"&&r>=1e4?a=\`\${u(e/1e4)}\\u4E07\`:t==="si"&&r>=1e9?a=\`\${u(e/1e9)}B\`:t==="si"&&r>=1e6?a=\`\${u(e/1e6)}M\`:t==="si"&&r>=1e4?a=\`\${u(e/1e3)}K\`:a=(Number.isInteger(e)?e:Number(e.toFixed(r<1?2:1))).toLocaleString("en-US"),a+n}}function X(t,n,e){n>t||(n=t+1);let r=(n-t)/e,a=10**Math.floor(Math.log10(r)),s=r/a,o=(s<=1?1:s<=2?2:s<=2.5?2.5:s<=5?5:10)*a,c=Math.floor(t/o+1e-9)*o,m=Math.ceil(n/o-1e-9)*o,x=[];for(let h=c;h<=m+o/2;h+=o)x.push(Number(h.toPrecision(12)));return{low:c,high:m,ticks:x}}function ae(t,n){if(t.length>35&&t.every(s=>U.test(s))){let s=[],o="";return t.forEach((c,m)=>{let x=U.exec(c),h=x[1]+x[2];h!==o&&(m>0||Number(x[3])<=3)&&s.push({index:m,text:\`\${Number(x[2])}\\u6708\`}),o=h}),s}let e=t.map(j),r=Math.max(0,...e.map(s=>B(s)))+12,a=Math.max(1,Math.ceil(r/(n/Math.max(1,t.length))));return e.map((s,o)=>({index:o,text:s})).filter(s=>s.index%a===0)}function oe(t,n,e,r){if(!Array.isArray(n)||!n.length)throw new Error(\`ak.\${t}: series \\u81F3\\u5C11\\u9700\\u8981\\u4E00\\u4E2A\\u7CFB\\u5217\`);return n.map((a,s)=>{let o=a?.values??[];if(o.length!==e)throw new Error(\`ak.\${t}: series[\${s}].values \\u6709 \${o.length} \\u4E2A\\u503C\\uFF0Cx \\u6709 \${e} \\u4E2A\`);return{name:String(a.name??\`\\u7CFB\\u5217 \${s+1}\`),color:Y(a.color,s),values:o.map((c,m)=>{if(c==null&&r)return null;if(typeof c!="number"||!Number.isFinite(c))throw new Error(\`ak.\${t}: series[\${s}].values[\${m}] \\u4E0D\\u662F\\u6709\\u9650\\u6570\\u5B57\`);return c})}})}function le(t,n){let e=t.indexOf(String(n));return e>=0?e:typeof n=="number"&&Number.isInteger(n)&&n>=0&&n<t.length?n:-1}var D=class{constructor(n,e){this.width=n;this.height=e}boxes=[];block(n){this.boxes.push(n)}place(n,e=!1){let r=n.map(o=>({...o,x:V(o.x,0,Math.max(0,this.width-o.w))})).filter(o=>o.y>=0&&o.y+o.h<=this.height),s=r.find(o=>!this.boxes.some(c=>o.x<c.x+c.w+2&&c.x<o.x+o.w+2&&o.y<c.y+c.h+1&&c.y<o.y+o.h+1))??(e?r[0]??null:null);return s&&this.boxes.push(s),s}};function ie(t,n,e,r,a=8){let s=String(n.text??n.label??"");if(!s)return"";let o=Math.max(B(s,12,600),n.sub?B(n.sub):0),c=n.sub?32:16,m=(w,N)=>({x:w,y:N,w:o,h:c}),x=t.place([m(e-o/2,r-a-c),m(e-o+8,r-a-c),m(e-8,r-a-c),m(e+a,r-c/2),m(e-a-o,r-c/2),m(e-o/2,r+a),m(e-o/2,r-a-c-20),m(e-o-a,r-a-c),m(e+a,r-a-c)],!0);if(!x)return"";let h=x.x+o/2;return\`<text class="ak-b" x="\${u(h)}" y="\${u(x.y+12)}" text-anchor="middle">\${p(s)}</text>\`+(n.sub?\`<text x="\${u(h)}" y="\${u(x.y+28)}" text-anchor="middle">\${p(n.sub)}</text>\`:"")}var Q=(t,n,e)=>\`<svg viewBox="0 0 \${t} \${n}" width="\${t}" height="\${n}" role="img" aria-label="\${p(e||"\\u56FE\\u8868")}">\`,q=t=>\`<div class="ak-legend">\${t.map(([n,e,r])=>\`<span><i style="background:\${n}"></i>\${e==null?"":\`<b>\${p(e)}</b> \`}\${p(r)}</span>\`).join("")}</div>\`,Z=(t,n,e)=>\`<b>\${p(t)}</b>\${e==null?"":\` \\xB7 \\u5408\\u8BA1 <b>\${p(e)}</b>\`}\`+n.map(([r,a,s])=>\`<br><i style="background:\${s}"></i>\${p(r)} <b>\${p(a)}</b>\`).join(""),ce=(t,n,e,r,a)=>t.map(s=>\`<line class="ak-grid" x1="\${e}" x2="\${r}" y1="\${u(n(s))}" y2="\${u(n(s))}"/><text x="\${e-8}" y="\${u(n(s))+4}" text-anchor="end">\${p(a(s))}</text>\`).join(""),G=(t,n,e,r)=>{let a=B(e)/2;return\`<text x="\${u(V(t,a,r-a))}" y="\${n}" text-anchor="middle">\${p(e)}</text>\`};function ue(t,n){let e=typeof t=="string"?document.querySelector(t):t;if(!(e instanceof HTMLElement))throw new Error(\`ak: \\u627E\\u4E0D\\u5230\\u5BB9\\u5668 \${String(t)}\`);e.classList.add("ak-chart");let r=0,a=()=>{let s=Math.max(160,Math.floor(e.clientWidth||640));if(s===r)return;r=s;let o=n(s,\`ak\${se+=1}\`);e.innerHTML=o.html;let c=e.querySelector("svg");c&&o.hits.length&&me(e,c,o.hits)};return a(),new ResizeObserver(a).observe(e),{redraw:()=>{r=0,a()}}}function me(t,n,e){let r=document.createElement("div"),a=document.createElement("span"),s=n.querySelector(".ak-cross");r.className="ak-tip",r.hidden=!0,a.className="ak-sr",a.setAttribute("role","status"),t.append(r,a);let o=-1,c=()=>n.getBoundingClientRect().width/(n.viewBox.baseVal.width||1),m=()=>{r.hidden=!0,o=-1,s?.style.setProperty("visibility","hidden")},x=(h,w,N)=>{let g=e[h];o=h,r.innerHTML=g.html,r.hidden=!1,a.textContent=r.textContent;let k=t.getBoundingClientRect(),v=n.getBoundingClientRect(),d=c(),$=v.left-k.left+(w??(g.x0+g.x1)/2)*d,R=v.top-k.top+(N??g.y0)*d,C=$+14+r.offsetWidth>t.clientWidth?Math.max(0,$-r.offsetWidth-14):$+14;r.style.left=\`\${C}px\`,r.style.top=\`\${Math.max(0,R-r.offsetHeight-10)}px\`,s&&g.cx!=null&&(s.setAttribute("x1",String(g.cx)),s.setAttribute("x2",String(g.cx)),s.style.visibility="visible")};n.tabIndex=0,n.addEventListener("pointermove",h=>{let w=n.getBoundingClientRect(),N=c(),g=(h.clientX-w.left)/N,k=(h.clientY-w.top)/N,v=e.findIndex(d=>g>=d.x0&&g<=d.x1&&k>=d.y0&&k<=d.y1);v<0?m():x(v,g,k)}),n.addEventListener("pointerleave",m),n.addEventListener("blur",m),n.addEventListener("keydown",h=>{if(h.key==="Escape")return m();let w={ArrowRight:1,ArrowDown:1,ArrowLeft:-1,ArrowUp:-1}[h.key];w&&(h.preventDefault(),x(o<0?w>0?0:e.length-1:(o+w+e.length)%e.length))})}function he(t,n){let e=(n?.x??[]).map(String);if(!e.length)throw new Error("ak.bars: x \\u4E0D\\u80FD\\u4E3A\\u7A7A");let r=oe("bars",n.series,e.length,!1),a=J(n.fmt,n.unit);return ue(t,(s,o)=>n.horizontal?xe(s,o,n,e,r,a):ge(s,n,e,r,a))}function ge(t,n,e,r,a){let s=n.stack!==!1,o=e.length,c=b=>r.map(l=>l.values[b]??0),m=e.map((b,l)=>c(l).reduce((i,f)=>i+Math.max(0,f),0)),x=e.map((b,l)=>c(l).reduce((i,f)=>i+Math.min(0,f),0)),h=e.map((b,l)=>s?m[l]:Math.max(0,...c(l))),w=Math.max(0,...h,n.refLine?.value??0),N=Math.min(0,...s?x:r.flatMap(b=>b.values)),g=n.height??(t<480?220:260),k=n.annotate?.length?n.annotate.some(b=>b.sub)?44:28:12,v=24,d=X(N,w,g<240?3:4),$=Math.max(...d.ticks.map(b=>B(a(b))))+10,R=t-$-4,C=R/o,y=Math.max(1,Math.min(C*(o>40?.8:.66),44)),M=b=>k+(g-k-v)-(b-d.low)/(d.high-d.low)*(g-k-v),I=new D(t,g);I.block({x:0,y:0,w:$-4,h:g}),I.block({x:0,y:g-v+4,w:t,h:v});let E="",K=e.map((b,l)=>{let i=$+C*(l+.5),f=0,T=0;return r.forEach((L,S)=>{let H=L.values[l]??0,O=s?y:y/r.length,te=s?i-y/2:i-y/2+S*O,_=s?H>0?f:T:0,W=_+H;s&&(H>0?f=W:T=W);let P=M(Math.max(_,W)),ne=M(Math.min(_,W));E+=\`<rect x="\${u(te)}" y="\${u(P)}" width="\${u(Math.max(1,O-(s?0:1)))}" height="\${u(Math.max(.5,ne-P))}" style="fill:\${L.color}"\${s&&r.length>1&&y>6?' class="ak-sep"':""}/>\`}),I.block({x:i-y/2,y:M(h[l]),w:y,h:M(0)-M(h[l])}),{x0:i-C/2,x1:i+C/2,y0:k,y1:g-v,html:Z(j(b),[...r].reverse().map(L=>[L.name,a(L.values[l]??0),L.color]),s&&r.length>1?a(m[l]+x[l]):void 0)}}),F=ce(d.ticks,M,$,t-4,a)+\`<line class="ak-base" x1="\${$}" x2="\${t-4}" y1="\${u(M(0))}" y2="\${u(M(0))}"/>\`;ae(e,R).forEach(b=>{F+=G($+C*(b.index+.5),g-6,b.text,t)}),E+=be(I,n.refLine,a,{x1:$,x2:t-4,y:M(n.refLine?.value??0)});for(let b of n.annotate??[]){let l=le(e,b.at);l>=0&&(E+=ie(I,b,$+C*(l+.5),M(h[l]),6))}return{html:(r.length>1&&n.legend!==!1?q(r.map(b=>[b.color,a(b.values.reduce((l,i)=>l+(i??0),0)),b.name])):"")+Q(t,g,n.label)+F+E+"</svg>",hits:K}}function ee(t,n){let e=n(t.value);return t.label?t.label.includes(e)?t.label:\`\${e} \\xB7 \${t.label}\`:e}function be(t,n,e,r){if(!n)return"";let a=ee(n,e),s=B(a,12,600),o=t.place([{x:r.x2-s,y:r.y-18,w:s,h:14},{x:r.x2-s,y:r.y+4,w:s,h:14},{x:r.x1+4,y:r.y-18,w:s,h:14},{x:r.x1+4,y:r.y+4,w:s,h:14}],!0);return\`<line class="ak-ref" x1="\${r.x1}" x2="\${r.x2}" y1="\${u(r.y)}" y2="\${u(r.y)}"/>\`+(o?\`<text class="ak-b" x="\${u(o.x)}" y="\${u(o.y+11)}">\${p(a)}</text>\`:"")}function xe(t,n,e,r,a,s){let o=e.range??[],c=e.sub??[],m=t<420,x=a.length===1,h=r.map((l,i)=>a.reduce((f,T)=>f+Math.max(0,T.values[i]??0),0)),w=r.map((l,i)=>{let f=o[i];return f?\`\${s(f.lo)}\\u2013\${s(f.hi)}\`:s(h[i])}),N=Math.max(0,e.refLine?.value??0,...r.map((l,i)=>Math.max(h[i],o[i]?.hi??0))),g=X(0,N,3),k=m?0:Math.min(t*.34,Math.max(...r.map(l=>B(l,13,600)),...c.map(l=>B(l)))+14),v=Math.max(...w.map(l=>B(l,13,600)))+12,d=18,$=d+(m?30:c.some(Boolean)?22:14),R=Math.max(40,t-k-v-4),C=4+r.length*$+24,y=l=>k+V(l,0,g.high)/g.high*R,M=l=>e.colors?Y(e.colors[l],l):a[0].color,I="",E="",K=r.map((l,i)=>{let f=4+i*$+(m?22:($-d)/2),T=f+d/2,L=o[i];if(E+=m?\`<text class="ak-b" x="0" y="\${u(f-7)}">\${p(l)}\${c[i]?\`<tspan class="ak-m" font-weight="400"> \${p(c[i])}</tspan>\`:""}</text>\`:\`<text class="ak-b" x="0" y="\${u(c[i]?T-2:T+4)}">\${p(l)}</text>\`+(c[i]?\`<text class="ak-m" x="0" y="\${u(T+13)}">\${p(c[i])}</text>\`:""),L){let S=M(i),H=Math.min(L.lo,L.hi),O=Math.max(L.lo,L.hi);I+=\`<pattern id="\${n}h\${i}" patternUnits="userSpaceOnUse" width="6" height="6" patternTransform="rotate(45)"><rect width="6" height="6" style="fill:\${S};opacity:.14"/><rect width="2.2" height="6" style="fill:\${S}"/></pattern>\`,E+=\`<rect x="\${k}" y="\${u(f)}" width="\${u(y(H)-k)}" height="\${d}" style="fill:\${S}"/><rect x="\${u(y(H))}" y="\${u(f)}" width="\${u(y(O)-y(H))}" height="\${d}" style="fill:url(#\${n}h\${i})"/>\`}else{let S=0;a.forEach(H=>{let O=Math.max(0,H.values[i]??0);E+=\`<rect x="\${u(y(S))}" y="\${u(f)}" width="\${u(y(S+O)-y(S))}" height="\${d}" style="fill:\${x?M(i):H.color}"\${x?"":' class="ak-sep"'}/>\`,S+=O})}return E+=\`<text class="ak-b" x="\${t}" y="\${u(T+5)}" text-anchor="end">\${p(w[i])}</text>\`,{x0:0,x1:t,y0:4+i*$,y1:4+(i+1)*$,html:Z(l,L?[[e.rangeLabels?.[1]??"\\u4F30\\u8BA1\\u533A\\u95F4",w[i],M(i)]]:a.map(S=>[S.name,s(S.values[i]??0),x?M(i):S.color]))}}),F=new D(t,C),A=4+r.length*$+16;if(e.refLine){let l=y(e.refLine.value),i=ee(e.refLine,s);E+=\`<line class="ak-ref" x1="\${u(l)}" x2="\${u(l)}" y1="2" y2="\${A-12}"/>\`+G(l,A,i,t).replace("<text",'<text class="ak-b"'),F.block({x:l-B(i,12,600)/2-6,y:A-12,w:B(i,12,600)+12,h:16})}return g.ticks.forEach(l=>{let i=y(l),f=B(s(l))/2;F.place([{x:V(i-f,0,t-2*f),y:A-12,w:2*f,h:16}])&&(E+=G(i,A,s(l),t-v))}),{html:(o.some(Boolean)?q([["var(--ak-color-text-muted)",null,e.rangeLabels?.[0]??"\\u4E0B\\u9650\\u4F30\\u8BA1"],["repeating-linear-gradient(45deg, var(--ak-color-text-muted) 0 2px, transparent 2px 5px)",null,e.rangeLabels?.[1]??"\\u4F30\\u8BA1\\u533A\\u95F4"]]):a.length>1&&e.legend!==!1?q(a.map(l=>[l.color,null,l.name])):"")+Q(t,C,e.label)+\`<defs>\${I}</defs>\`+E+"</svg>",hits:K}}var fe={bars:he,fmt:(t,n,e)=>J(n,e)(t)};window.ak=fe;})();<\/script>
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
`;var c="akashic.visual.v1",v=["type-reading","color-action-primary","color-text-muted","color-text-primary","color-text-secondary","color-bg-canvas","color-bg-surface","color-border-default","color-border-strong",...[1,2,3,4,5,6].map(r=>`chart-${r}`)];function y(r){if(!r||typeof r!="object")return!1;let a=r;return a.origin?.kind==="tool_result"&&a.origin.tool==="render_visual"&&a.kind==="file"&&a.media_type==="application/json"&&!!a.filename?.endsWith(".akashic-visual.json")}function f(r){let a=getComputedStyle(r);return`:root{${v.map(o=>`--ak-${o}:${a.getPropertyValue(`--ak-${o}`)};`).join("")}
    color-scheme:${a.colorScheme};}`}function k(r,a){let o=a.block,l=new AbortController,i=document.createElement("p");i.setAttribute("role","status"),i.textContent="\u6B63\u5728\u6253\u5F00\u4EA4\u4E92\u56FE\u2026";let t=document.createElement("iframe");t.sandbox.add("allow-scripts"),t.referrerPolicy="no-referrer",t.style.cssText="display:block;width:100%;height:320px;border:0;visibility:hidden",r.append(i);let u=!1,m=window.setTimeout(()=>{u||s("\u52A0\u8F7D\u8D85\u65F6")},1e4),s=e=>{window.clearTimeout(m),l.abort(),t.remove(),i.textContent="",a.reportError(e)},h=e=>{e.source!==t.contentWindow||e.data?.channel!==c||(e.data.type==="ready"?(u=!0,window.clearTimeout(m),t.style.visibility="visible",i.textContent=""):e.data.type==="height"&&Number.isFinite(e.data.value)?t.style.height=`${Math.max(80,Math.min(1600,e.data.value))}px`:e.data.type==="error"&&typeof e.data.value=="string"&&(u?i.textContent=`\u672C\u6B21\u4EA4\u4E92\u8BA1\u7B97\u9047\u5230\u95EE\u9898\uFF1A${e.data.value.slice(0,300)}\u3002\u53EF\u4EE5\u8C03\u6574\u53C2\u6570\u518D\u8BD5\u3002`:s(e.data.value.slice(0,300))))};window.addEventListener("message",h);let p=0,d="";t.addEventListener("load",()=>{p+=1,p===1?(t.contentWindow?.postMessage({channel:c,type:"init",document:d},"*"),d=""):p>2&&s("\u6587\u6863\u53D1\u751F\u4E86\u4E0D\u5141\u8BB8\u7684\u5BFC\u822A")});let x=new MutationObserver(()=>t.contentWindow?.postMessage({channel:c,type:"theme",value:f(r)},"*"));return x.observe(document.documentElement,{attributes:!0,attributeFilter:["data-theme","style","class"]}),(async()=>{try{if(o.size_bytes>15e5)throw new Error("\u9644\u4EF6\u8D85\u8FC7\u5927\u5C0F\u9650\u5236");let e=await fetch(`/api/chat/artifacts/${encodeURIComponent(o.artifact_id)}`,{signal:l.signal});if(!e.ok)throw new Error(`\u8BFB\u53D6\u5931\u8D25 (${e.status})`);let n=await e.json();if(!n||n.schema!==c||typeof n.title!="string"||typeof n.summary!="string"||typeof n.html!="string"||Array.from(n.html).length>2e5)throw new Error("\u4E0D\u652F\u6301\u7684\u6587\u6863\u683C\u5F0F");if(l.signal.aborted)return;t.title=n.title,d=g.replace("/*visual-theme*/",()=>f(r)).replace("<!--visual-content-->",()=>n.html),t.src="/visuals/frame.html",r.append(t)}catch(e){l.signal.aborted||s(e instanceof Error?e.message:String(e))}})(),()=>{l.abort(),window.clearTimeout(m),x.disconnect(),window.removeEventListener("message",h),t.remove(),i.remove()}}var M={slots:{"message.attachment":{accepts:y,mount:k}}};export{M as default};
