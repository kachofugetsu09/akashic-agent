var p=`<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <meta http-equiv="Content-Security-Policy" content="default-src 'none'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; img-src data:; font-src data:; connect-src 'none'; frame-src 'none'; object-src 'none'; base-uri 'none'; form-action 'none'">
  <style>
    :root {
      color-scheme: light dark;
      font-family: "Source Serif 4", "Noto Serif SC", "Noto Serif CJK SC", serif;
      color: var(--ak-color-text-primary);
      background: var(--ak-color-bg-canvas);
    }
    * { box-sizing: border-box; }
    body { margin: 0; padding: 12px; overflow-wrap: anywhere; }
    h1, h2, h3, p { margin: 0 0 12px; }
    h2 { font-size: 1.25rem; }
    p { line-height: 1.65; }
    svg, canvas { display: block; max-width: 100%; }
    svg text { font-family: inherit; font-size: 12px; fill: currentColor; }
    button, input, select { font: inherit; accent-color: var(--ak-chart-1); }
    button, select, input[type=range] { min-height: 44px; }
    button, select {
      color: inherit;
      background: var(--ak-color-bg-surface);
      border: 1px solid var(--ak-color-border-default);
      border-radius: 4px;
      padding: 8px 12px;
    }
    label { display: block; }
    input[type=range] { width: 100%; }
    output { font-variant-numeric: tabular-nums; }
    button:focus-visible, input:focus-visible, select:focus-visible {
      outline: 2px solid var(--ak-chart-1);
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
`;var u="akashic.visual.v1",g=["color-text-primary","color-text-secondary","color-bg-canvas","color-bg-surface","color-border-default",...[1,2,3,4,5,6].map(n=>`chart-${n}`)];function y(n){if(!n||typeof n!="object")return!1;let a=n;return a.kind==="file"&&a.media_type==="application/json"&&!!a.filename?.endsWith(".akashic-visual.json")}function f(n){let a=getComputedStyle(n);return`:root{${g.map(o=>`--ak-${o}:${a.getPropertyValue(`--ak-${o}`)};`).join("")}
    color-scheme:${a.colorScheme};}`}function b(n,a){let o=a.block,s=new AbortController,r=document.createElement("p");r.setAttribute("role","status"),r.textContent="\u6B63\u5728\u6253\u5F00\u4EA4\u4E92\u56FE\u2026";let t=document.createElement("iframe");t.sandbox.add("allow-scripts"),t.referrerPolicy="no-referrer",t.style.cssText="display:block;width:100%;height:320px;border:0;visibility:hidden",n.append(r,t);let d=!1,c=window.setTimeout(()=>{d||l("\u52A0\u8F7D\u8D85\u65F6")},1e4),l=e=>{window.clearTimeout(c),s.abort(),t.remove(),r.textContent=`\u4EA4\u4E92\u56FE\u4E0D\u53EF\u7528\uFF1A${e}\u3002\u539F\u59CB\u9644\u4EF6\u4ECD\u53EF\u4E0B\u8F7D\u3002`},m=e=>{e.source!==t.contentWindow||e.data?.channel!==u||(e.data.type==="ready"?(d=!0,window.clearTimeout(c),t.style.visibility="visible",r.textContent=""):e.data.type==="height"&&Number.isFinite(e.data.value)?t.style.height=`${Math.max(80,Math.min(1600,e.data.value))}px`:e.data.type==="error"&&typeof e.data.value=="string"&&l(e.data.value.slice(0,300)))};window.addEventListener("message",m);let h=new MutationObserver(()=>t.contentWindow?.postMessage({channel:u,type:"theme",value:f(n)},"*"));return h.observe(document.documentElement,{attributes:!0,attributeFilter:["data-theme","style","class"]}),(async()=>{try{if(o.size_bytes>15e5)throw new Error("\u9644\u4EF6\u8D85\u8FC7\u5927\u5C0F\u9650\u5236");let e=await fetch(`/api/chat/artifacts/${encodeURIComponent(o.artifact_id)}`,{signal:s.signal});if(!e.ok)throw new Error(`\u8BFB\u53D6\u5931\u8D25 (${e.status})`);let i=await e.json();if(!i||i.schema!==u||typeof i.title!="string"||typeof i.summary!="string"||typeof i.html!="string"||Array.from(i.html).length>2e5)throw new Error("\u4E0D\u652F\u6301\u7684\u6587\u6863\u683C\u5F0F");if(s.signal.aborted)return;t.title=i.title,t.srcdoc=p.replace("/*visual-theme*/",()=>f(n)).replace("<!--visual-content-->",()=>i.html)}catch(e){s.signal.aborted||l(e instanceof Error?e.message:String(e))}})(),()=>{s.abort(),window.clearTimeout(c),h.disconnect(),window.removeEventListener("message",m),t.remove(),r.remove()}}var k={slots:{"message.attachment":{accepts:y,mount:b}}};export{k as default};
