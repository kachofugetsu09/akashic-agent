var f=`<!doctype html>
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
    @media (prefers-reduced-motion: reduce) {
      *, *::before, *::after {
        animation: none !important;
        transition: none !important;
        scroll-behavior: auto !important;
      }
    }
  </style>
  <style id="visual-theme">/*visual-theme*/</style>
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
`;var c="akashic.visual.v1",y=["type-reading","color-action-primary","color-text-muted","color-text-primary","color-text-secondary","color-bg-canvas","color-bg-surface","color-border-default",...[1,2,3,4,5,6].map(r=>`chart-${r}`)];function x(r){if(!r||typeof r!="object")return!1;let a=r;return a.origin?.kind==="tool_result"&&a.origin.tool==="render_visual"&&a.kind==="file"&&a.media_type==="application/json"&&!!a.filename?.endsWith(".akashic-visual.json")}function v(r){let a=getComputedStyle(r);return`:root{${y.map(o=>`--ak-${o}:${a.getPropertyValue(`--ak-${o}`)};`).join("")}
    color-scheme:${a.colorScheme};}`}function k(r,a){let o=a.block,s=new AbortController,i=document.createElement("p");i.setAttribute("role","status"),i.textContent="\u6B63\u5728\u6253\u5F00\u4EA4\u4E92\u56FE\u2026";let t=document.createElement("iframe");t.sandbox.add("allow-scripts"),t.referrerPolicy="no-referrer",t.style.cssText="display:block;width:100%;height:320px;border:0;visibility:hidden",r.append(i);let p=!1,u=window.setTimeout(()=>{p||l("\u52A0\u8F7D\u8D85\u65F6")},1e4),l=e=>{window.clearTimeout(u),s.abort(),t.remove(),i.textContent="",a.reportError(e)},g=e=>{e.source!==t.contentWindow||e.data?.channel!==c||(e.data.type==="ready"?(p=!0,window.clearTimeout(u),t.style.visibility="visible",i.textContent=""):e.data.type==="height"&&Number.isFinite(e.data.value)?t.style.height=`${Math.max(80,Math.min(1600,e.data.value))}px`:e.data.type==="error"&&typeof e.data.value=="string"&&(p?i.textContent=`\u672C\u6B21\u4EA4\u4E92\u8BA1\u7B97\u9047\u5230\u95EE\u9898\uFF1A${e.data.value.slice(0,300)}\u3002\u53EF\u4EE5\u8C03\u6574\u53C2\u6570\u518D\u8BD5\u3002`:l(e.data.value.slice(0,300))))};window.addEventListener("message",g);let d=0,m="";t.addEventListener("load",()=>{d+=1,d===1?(t.contentWindow?.postMessage({channel:c,type:"init",document:m},"*"),m=""):d>2&&l("\u6587\u6863\u53D1\u751F\u4E86\u4E0D\u5141\u8BB8\u7684\u5BFC\u822A")});let h=new MutationObserver(()=>t.contentWindow?.postMessage({channel:c,type:"theme",value:v(r)},"*"));return h.observe(document.documentElement,{attributes:!0,attributeFilter:["data-theme","style","class"]}),(async()=>{try{if(o.size_bytes>15e5)throw new Error("\u9644\u4EF6\u8D85\u8FC7\u5927\u5C0F\u9650\u5236");let e=await fetch(`/api/chat/artifacts/${encodeURIComponent(o.artifact_id)}`,{signal:s.signal});if(!e.ok)throw new Error(`\u8BFB\u53D6\u5931\u8D25 (${e.status})`);let n=await e.json();if(!n||n.schema!==c||typeof n.title!="string"||typeof n.summary!="string"||typeof n.html!="string"||Array.from(n.html).length>2e5)throw new Error("\u4E0D\u652F\u6301\u7684\u6587\u6863\u683C\u5F0F");if(s.signal.aborted)return;t.title=n.title,m=f.replace("/*visual-theme*/",()=>v(r)).replace("<!--visual-content-->",()=>n.html),t.src="/visuals/frame.html",r.append(t)}catch(e){s.signal.aborted||l(e instanceof Error?e.message:String(e))}})(),()=>{s.abort(),window.clearTimeout(u),h.disconnect(),window.removeEventListener("message",g),t.remove(),i.remove()}}var C={slots:{"message.attachment":{accepts:x,mount:k}}};export{C as default};
