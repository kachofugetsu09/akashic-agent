var r=e=>String(e??"").replace(/[&<>"']/g,a=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"})[a]);function u(e){return new Intl.DateTimeFormat("zh-CN",{month:"2-digit",day:"2-digit",hour:"2-digit",minute:"2-digit",hour12:!1}).format(new Date(e))}function h(e){if(e.schema!=="akasha.queries.v1")throw new Error("\u67E5\u8BE2\u683C\u5F0F\u4E0D\u53D7\u652F\u6301\uFF0C\u8BF7\u66F4\u65B0\u9875\u9762");return e}function m(e){return e.kind==="context"?`\u4F1A\u8BDD ${e.session_id} \xB7 ${e.source} \xB7 \u622A\u81F3 #${e.through_seq}`:e.kind==="tool"?`\u4F1A\u8BDD ${e.session_id} \xB7 \u8C03\u7528 ${e.call_ref.message_id}:${e.call_ref.part_index}`:`\u72EC\u7ACB\u67E5\u8BE2 ${e.key}`}function b(e){h(e);let a=[["dense","\u7CBE\u786E\u56DE\u5FC6"],["completion","\u6A21\u5F0F\u8865\u5168"]];return`<section class="akasha-mobile-inspector">
    <button type="button" data-akasha-back>\u8FD4\u56DE\u68C0\u7D22\u5217\u8868</button>
    <header><h2>${r(e.query_text+(e.query_text_truncated?"\u2026":""))}</h2><p>${r(u(e.ts))}</p>
      <p>\u547D\u4E2D ${e.hit_count} \u9879\u8BB0\u5FC6\uFF0C\u5411\u4E0A\u4E0B\u6587\u63D0\u4F9B ${e.presented_count} \u6761\u6D88\u606F\u3002</p>
      <p>${r(m(e.source))}</p>
      <p>\u8FD9\u662F\u67E5\u8BE2\u548C\u6750\u6599\u8BB0\u5F55\uFF0C\u4E0D\u8BC1\u660E\u6A21\u578B\u5DF2\u7ECF\u4F7F\u7528\u8FD9\u4E9B\u5185\u5BB9\u3002</p></header>
    ${a.map(([n,l])=>`<details open class="akasha-mobile-recall akasha-mobile-recall--${n==="completion"?"completion":"precise"}">
      <summary>${l}</summary>
      <ol class="akasha-mobile-memories">${e.hits.filter(t=>t.lane===n).map(t=>`
        <li><div><p>\u5F97\u5206 ${Number(t.score).toFixed(3)} \xB7 ${r(t.sources.join(" \xB7 "))}</p>
        ${t.messages.map(i=>`<article>
          <p>${r(i.preview||"\uFF08\u975E\u6587\u672C\u6D88\u606F\uFF09")}${i.truncated?"\u2026\uFF08\u6B63\u6587\u9884\u89C8\uFF09":""}</p>
          <small>${r(i.message_id)} \xB7 ${i.presented?"\u5DF2\u63D0\u4F9B":"\u672A\u63D0\u4F9B"}</small>
        </article>`).join("")}</div></li>`).join("")||"<li>\u672C\u6B21\u6CA1\u6709\u547D\u4E2D</li>"}</ol>
    </details>`).join("")}
    <p>\u56FE\u7248\u672C ${e.graph_version} \xB7 ${e.pushes} \u6B21\u6269\u6563 \xB7 \u6B8B\u4F59 ${Number(e.residual_l1).toExponential(2)}</p>
  </section>`}function k(e){return h(e),`<section class="akasha-mobile-inspector"><header><h2>Akasha Inspector</h2>
    <p>\u5B9E\u9645\u67E5\u8BE2\u5171 ${e.total} \u6B21\u3002\u9009\u62E9\u4E00\u6761\u67E5\u770B\u547D\u4E2D\u4E0E\u5448\u73B0\u8BB0\u5F55\u3002</p></header>
    ${e.items.length?`<ol class="akasha-mobile-turns">${e.items.map(a=>`
      <li><button type="button" data-akasha-query="${r(a.query_id)}">
        <span>${r(a.query_text+(a.query_text_truncated?"\u2026":""))}</span><small>${r(u(a.ts))} \xB7 \u547D\u4E2D ${a.hit_count} \u9879 \xB7 \u63D0\u4F9B ${a.presented_count} \u6761</small>
      </button></li>`).join("")}</ol>`:"<p>\u8FD8\u6CA1\u6709\u67E5\u8BE2\u8BB0\u5F55\u3002\u4F7F\u7528\u8BB0\u5FC6\u53EC\u56DE\u540E\uFF0C\u53EF\u5728\u8FD9\u91CC\u67E5\u770B\u3002</p>"}
    <nav aria-label="\u68C0\u7D22\u5206\u9875"><button type="button" data-akasha-prev ${e.page===1?"disabled":""}>\u4E0A\u4E00\u9875</button>
      <span>\u7B2C ${e.page} \u9875</span>
      <button type="button" data-akasha-next ${e.page*e.page_size>=e.total?"disabled":""}>\u4E0B\u4E00\u9875</button></nav></section>`}function $(e,a){let n=!0,l,t=0,i=(c,o)=>{n&&o===t&&(e.innerHTML=`<p role="alert">${r(c.message)}\u3002\u8BF7\u91CD\u65B0\u6253\u5F00 Inspector\u3002</p>`)},d=()=>{n&&(e.innerHTML=k(l),e.querySelector("[data-akasha-prev]").addEventListener("click",()=>p(l.page-1)),e.querySelector("[data-akasha-next]").addEventListener("click",()=>p(l.page+1)),e.querySelectorAll("[data-akasha-query]").forEach(c=>{c.addEventListener("click",()=>{let o=++t;e.innerHTML='<p role="status">\u6B63\u5728\u8BFB\u53D6\u68C0\u7D22\u8BB0\u5F55\u2026</p>',a.query("inspector.detail",{query_id:c.getAttribute("data-akasha-query")}).then(s=>{!n||o!==t||(e.innerHTML=b(s),e.querySelector("[data-akasha-back]").addEventListener("click",d))}).catch(s=>i(s,o))})}))},p=c=>{let o=++t;e.innerHTML='<p role="status">\u6B63\u5728\u8BFB\u53D6\u67E5\u8BE2\u5217\u8868\u2026</p>',a.query("inspector.recent",{page:c}).then(s=>{!n||o!==t||(l=s,d())}).catch(s=>i(s,o))};return p(1),()=>{n=!1}}var v={slots:{},dashboard:{mount:$}};export{v as default,$ as mount,b as renderDetail};
