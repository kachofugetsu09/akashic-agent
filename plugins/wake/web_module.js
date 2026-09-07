let r=null;async function l(e,t){if(!r)throw new Error("Wake \u5DE5\u4F5C\u53F0\u9762\u677F\u672A\u6FC0\u6D3B");const a=await r(e,t),i=await a.json();if(!a.ok)throw new Error(String(i.detail??i.message??`HTTP ${a.status}`));return i}function d(e){return String(e).replace(/[&<>"']/g,t=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"})[t]??t)}function n(e){if(!e)return"\u8FDB\u884C\u4E2D";const t=new Date(String(e));return Number.isNaN(t.getTime())?String(e):new Intl.DateTimeFormat("zh-CN",{month:"2-digit",day:"2-digit",hour:"2-digit",minute:"2-digit",second:"2-digit",hour12:!1}).format(t)}function o(e){return e?{alert:"Alert",content:"Content",drift:"Drift"}[e]:"\u65E0\u5F85\u529E"}function s(e){return e==null?"\u672A\u8BFB\u53D6":String(e)}function c(e){return{checking:"\u68C0\u67E5\u4E2D",no_due:"\u6CA1\u6709\u5230\u671F\u4FE1\u4EF6",content_insufficient:"Content \u4E0D\u8DB3",admission_rejected:"Admission \u672A\u901A\u8FC7",shared:"\u5DF2\u53D1\u9001",model_skip:"\u6A21\u578B\u8DF3\u8FC7",deferred:"\u5DF2\u5EF6\u671F",cancelled_after_fire:"\u89E6\u53D1\u540E\u5173\u95ED",delivery_unknown:"\u9001\u8FBE\u672A\u77E5",failed:"\u68C0\u67E5\u5931\u8D25"}[e]}function u(e){return`
    <section class="wake-section wake-flow">
      <h3>\u539F\u59CB Wake Flow</h3>
      <dl class="wake-summary">
        <div><dt>\u6062\u590D\u6307\u9488</dt><dd><code>${d(e.flow_id)}</code></dd></div>
        <div><dt>\u8F93\u5165\u4F1A\u8BDD</dt><dd><code>${d(e.pointer.session_id)}</code></dd></div>
        <div><dt>\u539F\u59CB Input</dt><dd><code>${d(e.pointer.input_id)}</code></dd></div>
        <div><dt>\u6307\u9488\u72B6\u6001</dt><dd>${e.pointer.settled?"\u5DF2\u7ED3\u7B97":"\u5F85\u7ED3\u7B97"}</dd></div>
        <div><dt>\u9001\u8FBE\u72B6\u6001</dt><dd>${d(e.delivery.status)}</dd></div>
      </dl>
      <p class="wake-flow-target">${d(e.delivery.channel)} \xB7 ${d(e.delivery.recipient)}</p>
      <ol class="wake-message-list">
        ${e.messages.map(t=>`
          <li>
            <header><strong>#${t.seq} \xB7 ${d(t.author)}</strong><small>${d(t.source)} \xB7 ${d(t.body.kind??"Message")} \xB7 ${d(n(t.recorded_at))}</small></header>
            <p>${d(t.text||"\uFF08\u65E0\u6587\u672C\u5185\u5BB9\uFF09")}</p>
            <code>${d(t.message_id)}</code>
          </li>
        `).join("")}
      </ol>
    </section>
  `}function p(e,t){return`
    <article class="wake-run">
      <header class="wake-run-header">
        <div>
          <p>${d(o(e.owner))} \xB7 ${d(n(e.fired_at))}</p>
          <h2>${d(c(e.outcome))}</h2>
        </div>
        ${t?'<md-icon-button data-wake-close aria-label="\u5173\u95ED\u8BE6\u60C5"><span aria-hidden="true">\xD7</span></md-icon-button>':""}
      </header>

      <dl class="wake-summary">
        <div><dt>\u8BA1\u5212\u65F6\u95F4</dt><dd>${d(n(e.scheduled_for))}</dd></div>
        <div><dt>\u4FE1\u7BB1\u6C34\u4F4D</dt><dd>${d(s(e.mail_watermark))}</dd></div>
        <div><dt>\u68C0\u67E5\u5B8C\u6210</dt><dd>${d(n(e.completed_at))}</dd></div>
      </dl>

      <section class="wake-section">
        <h3>\u8FD9\u6B21\u68C0\u67E5</h3>
        <p>${d(e.detail||"Timer \u5DF2\u89E6\u53D1\uFF0C\u6B63\u5728\u68C0\u67E5 EventMail\u3002")}</p>
        <p><code>${d(e.timer_id)}</code></p>
      </section>
      ${e.flow?u(e.flow):""}
    </article>
  `}const w={id:"wake-attempts",label:"Wake \u68C0\u67E5",viewLabel:"Wake \u68C0\u67E5",pageSize:25,rowKey:"attempt_id",countTitle(e){return`${e} \u6B21\u5B9A\u65F6\u68C0\u67E5`},columns:[{key:"fired_at",label:"\u89E6\u53D1\u65F6\u95F4",width:130,renderCell:n},{key:"owner",label:"\u8F93\u5165",width:90,renderCell:o},{key:"mail_watermark",label:"\u4FE1\u7BB1\u6C34\u4F4D",width:90,renderCell:s},{key:"outcome",label:"\u7ED3\u679C",width:120,renderCell:c},{key:"detail",label:"\u8BF4\u660E",flex:!0,fmt:"text-preview"}],async getCount({signal:e}){return(await l("/api/dashboard/wake/attempts?page=1&page_size=1",{signal:e})).total},async fetchPage({page:e,pageSize:t,signal:a}){return await l(`/api/dashboard/wake/attempts?page=${e}&page_size=${t}`,{signal:a})},async fetchDetail(e,{signal:t}){return l(`/api/dashboard/wake/attempts/${encodeURIComponent(String(e.attempt_id??""))}`,{signal:t})},renderDetail(e,t,a){var i;if(!e){t.innerHTML='<p class="wake-empty">\u9009\u62E9\u4E00\u6B21\u5B9A\u65F6\u68C0\u67E5\uFF0C\u67E5\u770B\u5B83\u5F53\u65F6\u770B\u5230\u7684 EventMail \u6C34\u4F4D\u548C\u7ED3\u679C\u3002</p>';return}t.innerHTML=p(e,a.closePane),(i=t.querySelector("[data-wake-close]"))==null||i.addEventListener("click",()=>a.closePane())}};function m(e){r=e.http.request;const t=e.ui.inject("workbench.panels.v2",a=>a.register(w));return()=>{t(),r=null}}export{m as activate};
