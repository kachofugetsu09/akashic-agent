import { createServer } from "node:http";

let slowPending = 0;
const held = new Set();

// Computer 内的真实页面，业务事件由页面自己的 DOM 处理。
createServer((request, response) => {
  if (request.url === "/status") { response.end(JSON.stringify({slowPending, held: held.size})); return; }
  if (request.url === "/release") {
    for (const waiting of held) waiting.end("<title>Released</title>");
    held.clear(); response.end("released"); return;
  }
  if (request.url.startsWith("/held")) {
    response.setHeader("Content-Type", "text/html"); held.add(response); return;
  }
  response.setHeader("Content-Type", "text/html");
  const send = () => response.end(`<!doctype html><title>${request.url}</title>
    <h1>${request.url}</h1><output id="identity"></output><input id="x"><button id="hit"
    onclick="this.textContent='CLICKED';this.dataset.clicks=String(Number(this.dataset.clicks||0)+1)">CLICK</button>
    <script>if(location.pathname==='/a') localStorage.setItem('identity','A');
    if(location.pathname==='/remember') { localStorage.setItem('identity','PROFILE'); document.cookie='review=PROFILE; max-age=3600; path=/'; }
    document.querySelector('#identity').textContent=String(localStorage.getItem('identity'));</script>`);
  if (request.url === "/slow") { slowPending++; setTimeout(() => { slowPending--; send(); }, 2200); } else send();
}).listen(8989, "127.0.0.1");
