var b=`<!doctype html>
<html data-theme="light">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <meta http-equiv="Content-Security-Policy" content="default-src 'none'; script-src 'unsafe-inline' __VISUAL_RUNTIME_CSP__; style-src 'unsafe-inline'; img-src data:; font-src data:; connect-src 'none'; frame-src 'none'; object-src 'none'; base-uri 'none'; form-action 'none'">
  <style>
    :root {
      color-scheme: light dark;
      font-family: var(--ak-type-reading, "Source Serif 4"), "Noto Serif SC", "Noto Serif CJK SC", serif;
      color: var(--ak-color-text-primary);
      background: transparent;
    }
    * { box-sizing: border-box; }
    body { margin: 0; padding: 0; overflow-wrap: anywhere; font-size: 16px; line-height: 1.65; }
    h1, h2, h3, p { margin: 0 0 12px; }
    h2 { padding-inline-end: 44px; font-size: 1.0625rem; font-weight: 600; margin-bottom: 2px; }
    .caption { color: var(--ak-color-text-secondary); font-size: .875rem; margin-bottom: 16px; }
    p { line-height: 1.65; }
    svg, canvas { display: block; max-width: 100%; }
    svg text { font-family: inherit; font-size: 12px; }
    svg text:not([fill]) { fill: currentColor; }
    [aria-label$="-grid"] line { stroke: var(--ak-color-border-default); stroke-opacity: 1; stroke-dasharray: 2 5; }
    figure { margin: 0; }
    svg { --plot-background: var(--ak-color-bg-canvas) !important; }
    [class*="-swatches"] { justify-content: center; font-size: .875rem; color: var(--ak-color-text-primary); }
    [class*="-swatches"] svg { width: 9px; height: 9px; border-radius: 50%; }
    button, input, select { font: inherit; accent-color: var(--ak-color-action-primary); }
    button, select, input[type=range] { min-height: 44px; }
    button, select {
      color: inherit;
      background: var(--ak-color-bg-surface);
      border: 1px solid var(--ak-color-border-default);
      border-radius: 4px;
      padding: 8px 12px;
    }
    label { display: block; }
    input[type=range] { width: 100%; margin: 0; cursor: pointer; }
    .ak-param { display: grid; grid-template-columns: 1fr auto; align-items: center; gap: 0 12px; margin-top: 16px; font-size: .875rem; }
    .ak-param output { font-weight: 600; }
    .ak-param input { grid-column: 1 / -1; appearance: none; background: transparent; }
    .ak-param input::-webkit-slider-runnable-track { height: 4px; border-radius: 2px;
      background: linear-gradient(to right, var(--ak-color-action-primary) var(--range-fill), var(--ak-color-border-default) var(--range-fill)); }
    .ak-param input::-webkit-slider-thumb { appearance: none; width: 18px; height: 18px; margin-top: -7px; border-radius: 50%;
      background: var(--ak-color-bg-canvas); border: 2px solid var(--ak-color-action-primary); }
    .ak-param input::-moz-range-track { height: 4px; border-radius: 2px; background: var(--ak-color-border-default); }
    .ak-param input::-moz-range-progress { height: 4px; background: var(--ak-color-action-primary); }
    .ak-param input::-moz-range-thumb { width: 14px; height: 14px; border-radius: 50%; border: 2px solid var(--ak-color-action-primary); background: var(--ak-color-bg-canvas); }
    output { font-variant-numeric: tabular-nums; }
    button:focus-visible, input:focus-visible, select:focus-visible {
      outline: 2px solid var(--ak-color-action-primary);
      outline-offset: 3px;
    }
    @media (prefers-reduced-motion: reduce) {
      *, *::before, *::after {
        animation: none !important;
        transition: none !important;
        scroll-behavior: auto !important;
      }
    }
  </style>
  <style id="visual-theme">/*visual-theme*/</style>
  <script src="__VISUAL_RUNTIME_URL__"><\/script>
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
      addEventListener('DOMContentLoaded', () => {
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
`;var u="akashic.visual.v1",w=["type-reading","color-action-primary","color-text-muted","color-text-primary","color-text-secondary","color-bg-canvas","color-bg-surface","color-border-default",...[1,2,3,4,5,6].map(r=>`chart-${r}`)];function _(r){if(!r||typeof r!="object")return!1;let a=r;return a.origin?.kind==="tool_result"&&a.origin.tool==="render_visual"&&a.kind==="file"&&a.media_type==="application/json"&&!!a.filename?.endsWith(".akashic-visual.json")}function y(r){let a=getComputedStyle(r);return`:root{${w.map(o=>`--ak-${o}:${a.getPropertyValue(`--ak-${o}`)};`).join("")}
    color-scheme:${a.colorScheme};}`}function E(r,a){let o=a.block,s=new AbortController,n=document.createElement("p");n.setAttribute("role","status"),n.textContent="\u6B63\u5728\u6253\u5F00\u4EA4\u4E92\u56FE\u2026";let t=document.createElement("iframe");t.sandbox.add("allow-scripts"),t.referrerPolicy="no-referrer",t.style.cssText="display:block;width:100%;height:320px;border:0;visibility:hidden",r.append(n);let c=!1,d=window.setTimeout(()=>{c||l("\u52A0\u8F7D\u8D85\u65F6")},1e4),l=e=>{window.clearTimeout(d),s.abort(),t.remove(),n.textContent="",a.reportError(e)},p=e=>{e.source!==t.contentWindow||e.data?.channel!==u||(e.data.type==="ready"?(c=!0,window.clearTimeout(d),t.style.visibility="visible",n.textContent=""):e.data.type==="height"&&Number.isFinite(e.data.value)?t.style.height=`${Math.max(80,Math.min(1600,e.data.value))}px`:e.data.type==="error"&&typeof e.data.value=="string"&&(c?n.textContent=`\u672C\u6B21\u4EA4\u4E92\u8BA1\u7B97\u9047\u5230\u95EE\u9898\uFF1A${e.data.value.slice(0,300)}\u3002\u53EF\u4EE5\u8C03\u6574\u53C2\u6570\u518D\u8BD5\u3002`:l(e.data.value.slice(0,300))))};window.addEventListener("message",p);let h=!1;t.addEventListener("load",()=>{h&&l("\u6587\u6863\u53D1\u751F\u4E86\u4E0D\u5141\u8BB8\u7684\u5BFC\u822A"),h=!0});let g=new MutationObserver(()=>t.contentWindow?.postMessage({channel:u,type:"theme",value:y(r)},"*"));return g.observe(document.documentElement,{attributes:!0,attributeFilter:["data-theme","style","class"]}),(async()=>{try{if(o.size_bytes>15e5)throw new Error("\u9644\u4EF6\u8D85\u8FC7\u5927\u5C0F\u9650\u5236");let e=await fetch(`/api/chat/artifacts/${encodeURIComponent(o.artifact_id)}`,{signal:s.signal});if(!e.ok)throw new Error(`\u8BFB\u53D6\u5931\u8D25 (${e.status})`);let i=await e.json();if(!i||i.schema!==u||typeof i.title!="string"||typeof i.summary!="string"||typeof i.html!="string"||Array.from(i.html).length>2e5)throw new Error("\u4E0D\u652F\u6301\u7684\u6587\u6863\u683C\u5F0F");if(s.signal.aborted)return;t.title=i.title;let f=a.scripts["frame_runtime.js"];if(!f)throw new Error("\u63D2\u4EF6\u7F3A\u5C11\u56FA\u5B9A\u7ED8\u56FE\u811A\u672C");let m=new URL(f,location.href),v=k=>k.replaceAll("&","&amp;").replaceAll('"',"&quot;");t.srcdoc=b.replace("/*visual-theme*/",()=>y(r)).replace("__VISUAL_RUNTIME_CSP__",()=>v(m.origin+m.pathname)).replace("__VISUAL_RUNTIME_URL__",()=>v(m.href)).replace("<!--visual-content-->",()=>i.html),r.append(t)}catch(e){s.signal.aborted||l(e instanceof Error?e.message:String(e))}})(),()=>{s.abort(),window.clearTimeout(d),g.disconnect(),window.removeEventListener("message",p),t.remove(),n.remove()}}var L={slots:{"message.attachment":{accepts:_,mount:E}}};export{L as default};
