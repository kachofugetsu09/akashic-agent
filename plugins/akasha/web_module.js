let u=null;async function h(a,t){if(!u)throw new Error("Akasha \u5DE5\u4F5C\u53F0\u9762\u677F\u672A\u6FC0\u6D3B");const e=await u(a,t),n=await e.json();if(!e.ok)throw new Error(String(n.detail??n.message??`HTTP ${e.status}`));return n}function s(a){return String(a).replace(/[&<>"']/g,t=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"})[t]??t)}function _(a){return a.split("/").map(encodeURIComponent).join("/")}function p(a){if(!a)return"\u2014";const t=new Date(String(a));return Number.isNaN(t.getTime())?String(a):new Intl.DateTimeFormat("zh-CN",{month:"2-digit",day:"2-digit",hour:"2-digit",minute:"2-digit",hour12:!1}).format(t)}function c(a,t=3){if(a==null||a==="")return"\u2014";const e=Number(a);return Number.isFinite(e)?e.toFixed(t):"\u2014"}function g(a){const t=a.sources??[];return t.length?t.join(" \xB7 "):a.first_relation?a.first_relation:a.graph_only?"\u4EC5\u7531\u5173\u7CFB\u8865\u5168":"\u5916\u90E8\u7EBF\u7D22"}function $(a,t){return a.length?`
    <ol class="akasha-evidence-list">
      ${a.map((e,n)=>{var i;const l=e.score??e.value??e.completion_mass??e.seed_score,r=(i=e.relation_path)!=null&&i.length?`<span class="akasha-path" title="${s(e.relation_path.join(" \u2192 "))}">${s(e.relation_path.join(" \u2192 "))}</span>`:"";return`
          <li class="akasha-evidence">
            <span class="akasha-evidence-rank" aria-hidden="true">${n+1}</span>
            <div class="akasha-evidence-main">
              <p>${s(e.user_text||"\uFF08\u7A7A\u6D88\u606F\uFF09")}</p>
              ${e.assistant_preview?`<p class="akasha-assistant">${s(e.assistant_preview)}</p>`:""}
            </div>
            <div class="akasha-evidence-meta">
              <time class="akasha-chip akasha-chip--time">${s(p(e.ts))}</time>
              <span class="akasha-chip">${s(g(e))}</span>
              ${l==null?"":`<b class="akasha-chip akasha-chip--score">${c(l)}</b>`}
            </div>
            ${r}
          </li>
        `}).join("")}
    </ol>
  `:`<p class="akasha-empty">${s(t)}</p>`}function d(a,t,e,n,i,l,r=!1){return`
    <details class="akasha-section akasha-lane akasha-lane--${s(e)}" ${r?"open":""}>
      <summary>
        <span class="akasha-lane-copy">
          <strong>${s(a)}</strong>
          <small>${s(t)}</small>
        </span>
        <span class="akasha-lane-count">${s(String(i))}</span>
      </summary>
      ${$(n,l)}
    </details>
  `}function o(a,t,e){return`
    <div class="akasha-metric">
      <dt>${s(a)}</dt>
      <dd>${s(String(t))}</dd>
      <p>${s(e)}</p>
    </div>
  `}function y(a,t){const e=t.filters.q??"",n=a.querySelector("[data-akasha-search]");if(n){document.activeElement!==n&&n.value!==e&&(n.value=e);return}a.innerHTML=`
    <div class="akasha-filter">
      <label>
        <span>\u641C\u7D22\u68C0\u7D22\u8BB0\u5F55</span>
        <input
          type="search"
          value="${s(e)}"
          placeholder="Query\u3001\u56DE\u590D\u6216 Session"
          data-akasha-search
        />
      </label>
      <md-text-button data-akasha-clear ${e?"":"disabled"}>\u6E05\u7A7A</md-text-button>
    </div>
  `;const i=a.querySelector("[data-akasha-search]"),l=a.querySelector("[data-akasha-clear]");let r=0;const m=()=>{window.clearTimeout(r),r=window.setTimeout(()=>{const k=i.value.trim();k?t.setFilter("q",k):t.clearFilter("q")},200)},v=()=>{i.value="",t.clearFilter("q")};return i.addEventListener("input",m),l.addEventListener("click",v),()=>{window.clearTimeout(r),i.removeEventListener("input",m),l.removeEventListener("click",v)}}function f(a,t){const e=a.recall_capture_available?a.left_count+a.right_count:a.left_count;return`
    <article class="akasha-inspector">
      <header class="akasha-query">
        <div>
          <h2>${s(a.query_text)}</h2>
          <p class="akasha-query-meta">${s(p(a.ts))} \xB7 seq ${a.seq}<span>${s(a.session_key)}</span></p>
        </div>
        ${t?'<md-icon-button class="akasha-close" data-akasha-close aria-label="\u5173\u95ED\u8BE6\u60C5"><span aria-hidden="true">\xD7</span></md-icon-button>':""}
      </header>

      <section class="akasha-overview" aria-labelledby="akasha-overview-title">
        <div class="akasha-overview-heading">
          <div>
            <h3 id="akasha-overview-title">${e} \u6761\u8BB0\u5FC6\u53C2\u4E0E\u56DE\u7B54</h3>
          </div>
          <p>${a.inject_chars>0?`\u5DF2\u5199\u5165 ${a.inject_chars} \u5B57\u4E0A\u4E0B\u6587`:"\u6CA1\u6709\u5199\u5165 Prompt"}</p>
        </div>
        <dl class="akasha-metrics">
          ${o("\u76F4\u63A5\u7EBF\u7D22",a.seed_count,"Dense\u3001BM25 \u4E0E\u65F6\u5E8F")}
          ${o("\u7CBE\u786E\u56DE\u5FC6",a.left_count,"\u8BED\u4E49\u6700\u63A5\u8FD1\u7684\u5386\u53F2")}
          ${o("\u6A21\u5F0F\u8054\u60F3",a.recall_capture_available?a.right_count:"\u2014",a.recall_capture_available?`${a.basin_count} \u4E2A\u60C5\u666F\u7C07`:"\u672C\u8F6E\u672A\u8BB0\u5F55")}
        </dl>
      </section>

      <details class="akasha-answer">
        <summary>
          <span><strong>\u52A9\u624B\u56DE\u590D</strong><small>\u67E5\u770B\u8FD9\u4E00\u8F6E\u7684\u5B8C\u6574\u56DE\u7B54</small></span>
          <span class="akasha-answer-action">\u5C55\u5F00</span>
        </summary>
        <div class="akasha-answer-body">${s(a.assistant_text||"\uFF08\u52A9\u624B\u6CA1\u6709\u6587\u672C\u56DE\u590D\uFF09")}</div>
      </details>

      <section class="akasha-evidence-group" aria-labelledby="akasha-evidence-title">
        <div class="akasha-section-heading">
          <h3 id="akasha-evidence-title">\u8BB0\u5FC6\u8BC1\u636E</h3>
          <small>\u9009\u62E9\u4E00\u7EC4\u5C55\u5F00\u67E5\u770B</small>
        </div>
        <div class="akasha-lanes">
        ${d("\u76F4\u63A5\u7EBF\u7D22","\u6700\u521D\u547D\u4E2D\u7684\u6D88\u606F","seed",a.seeds,a.seeds.length,"\u8FD9\u4E00\u8F6E\u6CA1\u6709\u5F62\u6210\u53EF\u6301\u4E45\u5316\u7EBF\u7D22\u3002")}
        ${a.activation_capture_available?d("\u56FE\u6269\u6563\u5019\u9009","\u7531\u5173\u7CFB\u7F51\u7EDC\u8865\u5165\u7684\u5019\u9009","activation",a.activation_items,a.activation_items.length,"\u56FE\u6269\u6563\u6CA1\u6709\u589E\u52A0\u5019\u9009\u3002"):""}
        ${d("\u7CBE\u786E\u56DE\u5FC6","\u8BED\u4E49\u6700\u63A5\u8FD1\u7684\u5386\u53F2\u6D88\u606F","precise",a.left,a.left_count,"\u6CA1\u6709\u7CBE\u786E\u547D\u4E2D\u3002")}
        ${d("\u6A21\u5F0F\u8054\u60F3","\u8DE8\u5173\u7CFB\u8865\u5168\u4E14\u5DF2\u4E0E\u7CBE\u786E\u7ED3\u679C\u53BB\u91CD","completion",a.right,a.recall_capture_available?a.right_count:"\u672A\u8BB0\u5F55","\u6CA1\u6709\u4EA7\u751F\u6A21\u5F0F\u8054\u60F3\u3002")}
        ${a.tool_left_count?d("\u5DE5\u5177\u7CBE\u786E\u56DE\u5FC6","recall_memory \u7684\u8BED\u4E49\u547D\u4E2D","precise",a.tool_left,a.tool_left_count,"\u5DE5\u5177\u6CA1\u6709\u4EA7\u751F\u7CBE\u786E\u547D\u4E2D\u3002"):""}
        ${a.tool_right_count?d("\u5DE5\u5177\u6A21\u5F0F\u8054\u60F3","recall_memory \u7684\u56FE\u5173\u7CFB\u7ED3\u679C","completion",a.tool_right,a.tool_right_count,"\u5DE5\u5177\u6CA1\u6709\u4EA7\u751F\u6A21\u5F0F\u8054\u60F3\u3002"):""}
        </div>
      </section>

      <details class="akasha-learning">
        <summary><span><strong>\u5B66\u4E60\u53D8\u5316\u4E0E\u6280\u672F\u6307\u6807</strong><small>${a.activation_count} \u6761\u6269\u6563\u5019\u9009 \xB7 ${a.pushes} \u6B21\u6269\u6563</small></span></summary>
        <dl>
          ${o("\u60CA\u559C\u5EA6",c(a.surprise),"\u5F53\u524D cue \u4E0E\u5DF2\u6709\u6A21\u5F0F\u7684\u5DEE\u5F02")}
          ${o("\u89C2\u5BDF\u8D28\u91CF",c(a.observed_mass),"\u7531\u5916\u90E8\u8BC1\u636E\u652F\u6301\u7684\u5B66\u4E60\u8D28\u91CF")}
          ${o("\u518D\u6FC0\u6D3B",c(a.reactivated_mass),"\u5DF2\u6709\u5173\u7CFB\u91CD\u65B0\u83B7\u5F97\u7684\u6D3B\u6027")}
          ${o("\u589E\u5F3A / \u6291\u5236",`${c(a.potentiated_mass)} / ${c(a.inhibited_mass)}`,"\u8FDE\u63A5\u9884\u7B97\u5185\u7684\u7ADE\u4E89\u7ED3\u679C")}
        </dl>
      </details>

      <details class="akasha-prompt">
        <summary><span><strong>\u5199\u5165 Prompt \u7684\u8BB0\u5FC6</strong><small>${a.inject_chars} \u5B57 \xB7 \u539F\u59CB\u4E0A\u4E0B\u6587\u9884\u89C8</small></span></summary>
        <pre>${s(a.text_block_preview||"\u8FD9\u4E00\u8F6E\u6CA1\u6709\u6CE8\u5165\u8BB0\u5FC6\u3002")}</pre>
      </details>
    </article>
  `}const b={id:"akasha-inspector",label:"Akasha \u68C0\u7D22",viewLabel:"Akasha \u68C0\u7D22",pageSize:25,rowKey:"query_id",countTitle(a){return`${a} \u8F6E\u68C0\u7D22`},columns:[{key:"session_key",label:"\u4F1A\u8BDD",width:120,fmt:"mono-session",cellClass:"mono cell-session",rawTitle:!0},{key:"ts",label:"\u65F6\u95F4",width:110,cellClass:"mono cell-time",rawTitle:!0,renderCell(a){return s(p(a))}},{key:"query_text",label:"\u7528\u6237\u95EE\u9898",flex:!0,fmt:"text-preview",cellClass:"content-preview"},{key:"seed_count",label:"\u7EBF\u7D22",width:64,fmt:"metric",cellClass:"mono cell-metric",align:"right"},{key:"completion_count",label:"\u53EC\u56DE",width:64,fmt:"metric",cellClass:"mono cell-metric",align:"right"}],renderFilters:y,async getCount({signal:a}){try{const t=await h("/api/dashboard/akasha-inspector/overview",{signal:a});return t.available?t.total:null}catch(t){if(a.aborted)throw t;return null}},async fetchPage({page:a,pageSize:t,filters:e,signal:n}){const i=new URLSearchParams({page:String(a),page_size:String(t)});e!=null&&e.session_key&&i.set("session_key",e.session_key),e!=null&&e.q&&i.set("q",e.q);const l=await h(`/api/dashboard/akasha-inspector/turns?${i.toString()}`,{signal:n});return{items:l.items,total:l.total}},async fetchDetail(a,{signal:t}){return h(`/api/dashboard/akasha-inspector/turns/${_(String(a.query_id??""))}`,{signal:t})},renderDetail(a,t,e){var n;if(!a){t.innerHTML=`
        <div class="akasha-detail-empty">
          <div class="akasha-detail-empty__title">Akasha Inspector</div>
          <div class="akasha-detail-empty__text">\u9009\u62E9\u4E00\u8F6E\u68C0\u7D22\uFF0C\u67E5\u770B\u5B83\u4ECE\u54EA\u4E9B\u7EBF\u7D22\u5F00\u59CB\u3001\u6269\u6563\u5230\u54EA\u91CC\uFF0C\u4EE5\u53CA\u6700\u7EC8\u8FDB\u5165 Prompt \u7684\u5185\u5BB9\u3002</div>
        </div>
      `;return}t.innerHTML=f(a,e.closePane),(n=t.querySelector("[data-akasha-close]"))==null||n.addEventListener("click",()=>e.closePane());const i=Array.from(t.querySelectorAll(".akasha-lane"));for(const l of i)l.addEventListener("toggle",()=>{if(l.open)for(const r of i)r!==l&&(r.open=!1)})}};function w(a){u=a.http.request;const t=a.ui.inject("workbench.panels.v2",e=>e.register(b));return()=>{t(),u=null}}export{w as activate};
