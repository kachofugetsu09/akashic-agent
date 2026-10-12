"""自包含 HTML/CSS/JavaScript 示例；分别以原生 SVG 与 Canvas 计算和绘图。"""

PARAM_STYLE = '''<style>
.param {display:grid;grid-template-columns:1fr auto;align-items:center;gap:0 12px;margin:16px 0;font-size:14px}
.param input {grid-column:1/-1;appearance:none;background:transparent}
.param input::-webkit-slider-runnable-track {height:4px;border-radius:2px;background:linear-gradient(to right,var(--ak-color-action-primary) var(--fill),var(--ak-color-border-default) var(--fill))}
.param input::-webkit-slider-thumb {appearance:none;width:18px;height:18px;margin-top:-7px;border-radius:50%;background:var(--ak-color-bg-canvas);border:2px solid var(--ak-color-action-primary)}
.param input::-moz-range-track {height:4px;background:var(--ak-color-border-default)}
.param input::-moz-range-progress {height:4px;background:var(--ak-color-action-primary)}
.param input::-moz-range-thumb {width:14px;height:14px;border-radius:50%;border:2px solid var(--ak-color-action-primary);background:var(--ak-color-bg-canvas)}
.param output {font-weight:600}.legend {display:flex;justify-content:center;flex-wrap:wrap;gap:8px 24px;font-size:14px}
.legend span {display:inline-flex;align-items:center;gap:8px}.legend i {width:9px;height:9px;border-radius:50%;background:var(--series)}
.chart {width:100%;height:280px;color:var(--ak-color-text-secondary)}
</style>'''

SCALING = PARAM_STYLE + '''<h2>同样的参数量，训练数据越多，Loss 越低</h2>
<p class="caption">横轴每一档参数量放大十倍；拖动滑块，比较数据量改变后的曲线。</p>
<svg id="chart" class="chart" role="img" aria-label="参数量与损失的两条曲线"></svg>
<div class="legend"><span><i style="--series:var(--ak-chart-1)"></i>训练 20B tokens</span><span><i style="--series:var(--ak-chart-2)"></i>训练数据量（可调）</span></div>
<label class="param">训练数据量<output id="tokens">2T tokens</output><input id="data" aria-label="训练数据量" type="range" min="20" max="5000" step="20" value="2000"></label>
<p id="reading" class="caption"></p><p class="caption">这是拟合公式的趋势演示，不代表某个具体模型的实测结果。</p>
<script>
const svg=document.querySelector('#chart'),slider=document.querySelector('#data');
const loss=(n,d)=>1.69+406.4/n**.34+410.7/d**.28,count=n=>n>=1e12?(n/1e12)+'T':(n/1e9)+'B';
function node(tag,attrs,text){const e=document.createElementNS('http://www.w3.org/2000/svg',tag);for(const [k,v] of Object.entries(attrs))e.setAttribute(k,v);if(text!==undefined)e.textContent=text;svg.append(e);return e;}
function draw(){
 const w=svg.clientWidth,h=280,d=slider.valueAsNumber*1e9,x=n=>42+(Math.log10(n)-8)/4*(w-56),y=v=>h-32-(v-1.6)/1.7*(h-52);
 svg.setAttribute('viewBox',`0 0 ${w} ${h}`);svg.replaceChildren();
 for(const v of [1.6,2,2.4,2.8,3.2]){node('line',{x1:42,x2:w-14,y1:y(v),y2:y(v),stroke:'var(--ak-color-border-default)','stroke-dasharray':'2 5'});node('text',{x:32,y:y(v)+4,'text-anchor':'end'},v.toFixed(2));}
 const ns=[1e8,1e9,1e10,1e11,1e12];for(const n of ns)node('text',{x:x(n),y:h-10,'text-anchor':'middle'},count(n));
 node('line',{x1:42,x2:w-14,y1:y(1.69),y2:y(1.69),stroke:'var(--ak-color-text-muted)','stroke-dasharray':'2 4'});
 node('text',{x:w-14,y:y(1.69)-9,'text-anchor':'end',fill:'var(--ak-color-text-muted)'},'不可约损失 1.69');
 for(const [tokens,color,name] of [[20e9,'var(--ak-chart-1)','训练 20B tokens'],[d,'var(--ak-chart-2)','训练数据量（可调）']]){
  const points=Array.from({length:101},(_,i)=>{const n=10**(8+i*.04);return `${x(n)},${y(loss(n,tokens))}`;});
  node('polyline',{points:points.join(' '),fill:'none',stroke:color,'stroke-width':2});
  for(const n of ns){const dot=node('circle',{cx:x(n),cy:y(loss(n,tokens)),r:3.5,fill:color,stroke:'var(--ak-color-bg-canvas)'});const title=document.createElementNS(svg.namespaceURI,'title');title.textContent=`${name} · 参数 ${count(n)} · Loss ${loss(n,tokens).toFixed(2)}`;dot.append(title);}
 }
 document.querySelector('#tokens').textContent=count(d)+' tokens';slider.setAttribute('aria-valuetext',count(d)+' tokens');slider.style.setProperty('--fill',`${100*(slider.valueAsNumber-20)/4980}%`);
 document.querySelector('#reading').textContent=`1T 参数时：20B tokens 的 Loss 为 ${loss(1e12,20e9).toFixed(2)}，${count(d)} tokens 时为 ${loss(1e12,d).toFixed(2)}。`;
}
slider.addEventListener('input',draw);let width=0;new ResizeObserver(()=>{if(svg.clientWidth!==width){width=svg.clientWidth;requestAnimationFrame(draw);}}).observe(svg);draw();
</script>'''

SCATTER = PARAM_STYLE + '''<h2>散点、回归线与置信区间可以叠在一起</h2>
<p class="caption">示意数据：横轴是练习时长，纵轴是表现；阴影为拟合均值的近似 95% 区间。</p>
<svg id="scatter" class="chart" role="img" aria-label="练习时长与表现的散点和回归线"></svg>
<script>
const svg=document.querySelector('#scatter'),data=Array.from({length:24},(_,i)=>({x:i+1,y:14+2.1*i+6*Math.sin(i*1.7)}));
const n=data.length,mx=data.reduce((s,d)=>s+d.x,0)/n,my=data.reduce((s,d)=>s+d.y,0)/n;
const xx=data.reduce((s,d)=>s+(d.x-mx)**2,0),slope=data.reduce((s,d)=>s+(d.x-mx)*(d.y-my),0)/xx,fit=x=>my+slope*(x-mx);
const error=Math.sqrt(data.reduce((s,d)=>s+(d.y-fit(d.x))**2,0)/(n-2)),band=x=>1.96*error*Math.sqrt(1/n+(x-mx)**2/xx);
function node(tag,attrs,text){const e=document.createElementNS(svg.namespaceURI,tag);for(const [k,v] of Object.entries(attrs))e.setAttribute(k,v);if(text!==undefined)e.textContent=text;svg.append(e);return e;}
function draw(){const w=svg.clientWidth,h=280,x=v=>40+(v-1)/23*(w-54),y=v=>h-40-(v-5)/70*(h-60);svg.setAttribute('viewBox',`0 0 ${w} ${h}`);svg.replaceChildren();
 for(const v of [20,40,60]){node('line',{x1:40,x2:w-14,y1:y(v),y2:y(v),stroke:'var(--ak-color-border-default)','stroke-dasharray':'2 5'});node('text',{x:30,y:y(v)+4,'text-anchor':'end'},v);}
 for(const v of [1,6,12,18,24])node('text',{x:x(v),y:h-20,'text-anchor':'middle'},v);
 node('text',{x:40,y:12},'表现');node('text',{x:w-14,y:h-2,'text-anchor':'end'},'练习时长');
 const xs=Array.from({length:24},(_,i)=>i+1),points=[...xs.map(v=>[x(v),y(fit(v)+band(v))]),...xs.reverse().map(v=>[x(v),y(fit(v)-band(v))])];
 node('polygon',{points:points.map(p=>p.join(',')).join(' '),fill:'var(--ak-chart-2)','fill-opacity':.18});
 node('line',{x1:x(1),x2:x(24),y1:y(fit(1)),y2:y(fit(24)),stroke:'var(--ak-chart-2)','stroke-width':2});
 for(const d of data){const dot=node('circle',{cx:x(d.x),cy:y(d.y),r:4,fill:'var(--ak-chart-1)',stroke:'var(--ak-color-bg-canvas)'});const title=document.createElementNS(svg.namespaceURI,'title');title.textContent=`练习 ${d.x} · 表现 ${d.y.toFixed(1)}`;dot.append(title);}
 node('text',{x:w-14,y:y(15),'text-anchor':'end'},'相同趋势，仍有波动');
}
let width=0;new ResizeObserver(()=>{if(svg.clientWidth!==width){width=svg.clientWidth;requestAnimationFrame(draw);}}).observe(svg);draw();
</script>'''

CANVAS = PARAM_STYLE + '''<h2>沿着坡度走，小球逐步接近最低点</h2>
<p class="caption">原生 Canvas 模拟：每一步按 x ← x − η·2x 更新；可以改变学习率。</p>
<canvas id="descent" aria-label="梯度下降抛物线与小球" role="img" style="width:100%;height:240px"></canvas>
<label class="param">学习率 η<output id="rate-value">0.08</output><input id="rate" aria-label="学习率 η" type="range" min=".01" max=".4" step=".01" value=".08"></label>
<div class="actions"><button id="step">走一步</button> <button id="reset">重新开始</button></div>
<p>当前位置 x = <output id="position">3.000</output>，损失 x² = <output id="energy">9.000</output></p>
<script>
const canvas=document.querySelector('#descent'),ctx=canvas.getContext('2d'),slider=document.querySelector('#rate');let x=3,rate=.08;
function draw(){const w=canvas.clientWidth,h=240,dpr=devicePixelRatio;canvas.width=w*dpr;canvas.height=h*dpr;ctx.scale(dpr,dpr);
 const css=getComputedStyle(document.documentElement),px=v=>20+(v+3.2)/6.4*(w-40),py=v=>220-v*18;
 ctx.strokeStyle=css.getPropertyValue('--ak-color-border-default');ctx.beginPath();ctx.moveTo(20,220);ctx.lineTo(w-20,220);ctx.stroke();
 ctx.strokeStyle=css.getPropertyValue('--ak-chart-1');ctx.lineWidth=2;ctx.beginPath();for(let i=0;i<=120;i++){const v=-3.2+i*6.4/120;i?ctx.lineTo(px(v),py(v*v)):ctx.moveTo(px(v),py(v*v));}ctx.stroke();
 ctx.fillStyle=css.getPropertyValue('--ak-chart-2');ctx.beginPath();ctx.arc(px(x),py(x*x),7,0,Math.PI*2);ctx.fill();
 document.querySelector('#position').textContent=x.toFixed(3);document.querySelector('#energy').textContent=(x*x).toFixed(3);}
function setRate(){rate=slider.valueAsNumber;document.querySelector('#rate-value').textContent=rate.toFixed(2);slider.style.setProperty('--fill',`${100*(rate-.01)/.39}%`);slider.setAttribute('aria-valuetext',rate.toFixed(2));}
slider.addEventListener('input',setRate);setRate();document.querySelector('#step').onclick=()=>{x-=rate*2*x;draw();};document.querySelector('#reset').onclick=()=>{x=3;draw();};
new ResizeObserver(()=>requestAnimationFrame(draw)).observe(canvas);addEventListener('akashic:theme',draw);draw();
</script>'''

INTRO = "Scaling Law 描述的是：模型越大、训练数据越多，损失（Loss）越低，但每多一份投入换来的下降会越来越少。\n\n下面用拟合公式画两条曲线。你可以拖动数据量，观察第二条曲线如何变化。"
EXPLANATION = "这幅图有两个特别重要的含义。\n\n**第一，收益递减。** 参数量增加 10 倍，Loss 不会也下降 10 倍。\n\n**第二，瓶颈转移。** 固定数据量时，模型再大也会遇到平台；增加数据量，平台随之下移。"
