let r=null;async function d(e,t){if(!r)throw new Error("Akasha \u5DE5\u4F5C\u53F0\u9762\u677F\u672A\u6FC0\u6D3B");const a=await r(e,t),s=await a.json();if(!a.ok)throw new Error(String(s.detail??s.message??`HTTP ${a.status}`));return s}function i(e){return String(e).replace(/[&<>"']/g,t=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"})[t]??t)}function v(e){return e.split("/").map(encodeURIComponent).join("/")}function p(e){if(!e)return"\u2014";const t=new Date(String(e));return Number.isNaN(t.getTime())?String(e):new Intl.DateTimeFormat("zh-CN",{month:"2-digit",day:"2-digit",hour:"2-digit",minute:"2-digit",hour12:!1}).format(t)}function g(e,t=3){if(e==null||e==="")return"\u2014";const a=Number(e);return Number.isFinite(a)?a.toFixed(t):"\u2014"}function y(e){return e.kind==="context"?`\u4E0A\u4E0B\u6587 \xB7 ${String(e.source??"")}`:e.kind==="tool"?`\u5DE5\u5177 \xB7 ${String(e.call_ref??"")}`:`\u7A0B\u5E8F \xB7 ${String(e.key??"")}`}function $(e,t){return e.length?`
    <ol class="akasha-evidence-list">
      ${e.map((a,s)=>`
        <li class="akasha-evidence">
          <span class="akasha-evidence-rank" aria-hidden="true">${s+1}</span>
          <div class="akasha-evidence-main">
            <p>${i(a.text||"\uFF08\u7A7A\u6D88\u606F\uFF09")}${a.text_truncated?" \u2026":""}</p>
          </div>
          <div class="akasha-evidence-meta">
            <span class="akasha-chip">${i(a.author)} \xB7 ${i(a.source)}</span>
            <span class="akasha-chip">#${i(a.seq)}</span>
            <time class="akasha-chip akasha-chip--time">${i(p(a.recorded_at))}</time>
            <span class="akasha-chip">${a.presented?"\u5DF2\u5448\u73B0":"\u547D\u4E2D"}</span>
            <code>${i(a.message_id)}</code>
          </div>
        </li>
      `).join("")}
    </ol>
  `:`<p class="akasha-empty">${i(t)}</p>`}function k(e,t,a){const s=a?.messages??[];return`
    <details class="akasha-section akasha-lane akasha-lane--${i(a?.lane??"empty")}">
      <summary>
        <span class="akasha-lane-copy"><strong>${i(e)}</strong><small>${i(t)}</small></span>
        <span class="akasha-lane-count">${s.length}</span>
      </summary>
      ${$(s,"\u8FD9\u4E00\u6761\u901A\u9053\u6CA1\u6709\u547D\u4E2D\u6D88\u606F\u3002")}
    </details>
  `}function c(e,t,a){return`<div class="akasha-metric"><dt>${i(e)}</dt><dd>${i(String(t))}</dd><p>${i(a)}</p></div>`}function f(e,t){const a=t.filters.q??"",s=e.querySelector("[data-akasha-search]");if(s){document.activeElement!==s&&s.value!==a&&(s.value=a);return}e.innerHTML=`<div class="akasha-filter"><label><span>\u641C\u7D22\u68C0\u7D22\u8BB0\u5F55</span><input type="search" value="${i(a)}" placeholder="Query\u3001\u6D88\u606F\u6216 Session" data-akasha-search /></label><md-text-button data-akasha-clear ${a?"":"disabled"}>\u6E05\u7A7A</md-text-button></div>`;const n=e.querySelector("[data-akasha-search]"),l=e.querySelector("[data-akasha-clear]");let o=0;const h=()=>{window.clearTimeout(o),o=window.setTimeout(()=>{const m=n.value.trim();m?t.setFilter("q",m):t.clearFilter("q")},200)},u=()=>{n.value="",t.clearFilter("q")};return n.addEventListener("input",h),l.addEventListener("click",u),()=>{window.clearTimeout(o),n.removeEventListener("input",h),l.removeEventListener("click",u)}}function w(e,t){const a=e.hits.filter(n=>n.lane==="dense"),s=e.hits.filter(n=>n.lane==="completion");return`
    <article class="akasha-inspector">
      <header class="akasha-query"><div><h2>${i(e.query_text)}</h2><p class="akasha-query-meta">${i(p(e.ts))} \xB7 seq ${i(e.seq)} \xB7 ${i(e.session_key||"\u7A0B\u5E8F\u67E5\u8BE2")}</p><p class="akasha-query-meta">${i(y(e.source))}</p></div>${t?'<md-icon-button class="akasha-close" data-akasha-close aria-label="\u5173\u95ED\u8BE6\u60C5"><span aria-hidden="true">\xD7</span></md-icon-button>':""}</header>
      <section class="akasha-overview" aria-labelledby="akasha-overview-title"><div class="akasha-overview-heading"><div><h3 id="akasha-overview-title">${e.presented_count} \u6761\u6D88\u606F\u5B9E\u9645\u5448\u73B0</h3></div><p>\u56FE\u7248\u672C ${e.graph_version} \xB7 \u67E5\u8BE2\u4E0A\u9650 ${e.limit}</p></div><dl class="akasha-metrics">
        ${c("\u547D\u4E2D\u56DE\u5FC6",e.hit_count,"Recall \u8BB0\u5F55\u4E2D\u9009\u4E2D\u7684\u56DE\u5FC6\u6761\u76EE")}
        ${c("\u6D3B\u8DC3\u60C5\u666F\u7C07",e.active_basin_count,"Recall \u8BB0\u5F55\u7684\u771F\u5B9E completion \u6307\u6807")}
        ${c("\u6269\u6563\u6B21\u6570",e.pushes,"\u67E5\u8BE2\u5B8C\u6210\u65F6\u8BB0\u5F55\u7684 pushes")}
        ${c("\u6B8B\u4F59\u8D28\u91CF",g(e.residual_l1),"\u67E5\u8BE2\u5B8C\u6210\u65F6\u8BB0\u5F55\u7684 residual_l1")}
      </dl></section>
      <section class="akasha-evidence-group" aria-labelledby="akasha-evidence-title"><div class="akasha-section-heading"><h3 id="akasha-evidence-title">\u539F\u59CB Message \u8BC1\u636E</h3><small>\u6B63\u6587\u6765\u81EA MessageReader\uFF1B\u5217\u8868\u9875\u53EA\u663E\u793A 240 \u5B57\u9884\u89C8</small></div><div class="akasha-lanes">
        ${a.map(n=>k("Dense \u901A\u9053",n.sources.join(" \xB7 "),n)).join("")}
        ${s.map(n=>k("Completion \u901A\u9053",n.sources.join(" \xB7 "),n)).join("")}
      </div></section>
    </article>
  `}const b={id:"akasha-inspector",label:"Akasha \u68C0\u7D22",viewLabel:"Akasha \u68C0\u7D22",pageSize:25,rowKey:"query_id",countTitle(e){return`${e} \u8F6E\u68C0\u7D22`},columns:[{key:"session_key",label:"\u4F1A\u8BDD",width:120,fmt:"mono-session",cellClass:"mono cell-session",rawTitle:!0},{key:"seq",label:"Seq",width:64,fmt:"metric",cellClass:"mono cell-metric",align:"right"},{key:"query_text",label:"\u67E5\u8BE2",flex:!0,fmt:"text-preview",cellClass:"content-preview"},{key:"dense_count",label:"Dense",width:70,fmt:"metric",cellClass:"mono cell-metric",align:"right"},{key:"completion_count",label:"Completion",width:96,fmt:"metric",cellClass:"mono cell-metric",align:"right"},{key:"presented_count",label:"\u5DF2\u5448\u73B0",width:78,fmt:"metric",cellClass:"mono cell-metric",align:"right"},{key:"active_basin_count",label:"\u60C5\u666F\u7C07",width:78,fmt:"metric",cellClass:"mono cell-metric",align:"right"},{key:"pushes",label:"Pushes",width:78,fmt:"metric",cellClass:"mono cell-metric",align:"right"}],renderFilters:f,async getCount({signal:e}){const t=await d("/api/dashboard/akasha-inspector/overview",{signal:e});return t.available?t.total:null},async fetchPage({page:e,pageSize:t,filters:a,signal:s}){const n=new URLSearchParams({page:String(e),page_size:String(t)});a!=null&&a.session_key&&n.set("session_key",a.session_key),a!=null&&a.q&&n.set("q",a.q);const l=await d(`/api/dashboard/akasha-inspector/turns?${n.toString()}`,{signal:s});return{items:l.items,total:l.total}},async fetchDetail(e,{signal:t}){return d(`/api/dashboard/akasha-inspector/turns/${v(String(e.query_id??""))}`,{signal:t})},renderDetail(e,t,a){var s;if(!e){t.innerHTML='<div class="akasha-detail-empty"><div class="akasha-detail-empty__title">Akasha Inspector</div><div class="akasha-detail-empty__text">\u9009\u62E9\u4E00\u8F6E\u68C0\u7D22\uFF0C\u67E5\u770B\u5B9E\u9645 Recall \u4E0E\u539F\u59CB Message\u3002</div></div>';return}t.innerHTML=w(e,a.closePane),(s=t.querySelector("[data-akasha-close]"))==null||s.addEventListener("click",()=>a.closePane())}};function _(e){r=e.http.request;const t=e.ui.inject("workbench.panels.v2",a=>a.register(b));return()=>{t(),r=null}}export{_ as activate};
