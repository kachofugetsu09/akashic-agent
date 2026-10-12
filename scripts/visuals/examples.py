"""以公开 Plot 语法和原生 Canvas 验证自由组合，不定义图表类型清单。"""

SCALING = '''<h2>同样的参数量，训练数据越多，Loss 越低</h2>
<p class="caption">横轴每一档参数量放大十倍；拖动滑块，比较数据量改变后的曲线。</p>
<div id="chart"></div><div id="params"></div><p id="reading" class="caption"></p>
<p class="caption">这是拟合公式的趋势演示，不代表某个具体模型的实测结果。</p>
<script>
const N=[1e8,1e9,1e10,1e11,1e12], loss=(n,d)=>1.69+406.4/n**.34+410.7/d**.28;
let dataTokens=2000;
const count=n=>n>=1e12?(n/1e12)+'T':(n/1e9)+'B';
const draw=ak.plot(document.querySelector('#chart'),width=>{
  const rows=N.flatMap(n=>[{n,y:loss(n,20e9),series:'训练 20B tokens'},{n,y:loss(n,dataTokens*1e9),series:'训练数据量（可调）'}]);
  document.querySelector('#reading').textContent=`1T 参数时：20B tokens 的 Loss 为 ${loss(1e12,20e9).toFixed(2)}，${count(dataTokens*1e9)} tokens 时为 ${loss(1e12,dataTokens*1e9).toFixed(2)}。`;
  return {x:{type:'log',ticks:N,tickFormat:count},y:{domain:[1.6,3.3],ticks:[1.6,2,2.4,2.8,3.2],tickFormat:'.2f'},
    marks:[Plot.ruleY([1.69],{stroke:'var(--ak-color-text-muted)',strokeDasharray:'2,4'}),
      Plot.line(rows,{x:'n',y:'y',stroke:'series',strokeWidth:2,curve:'monotone-x'}),
      Plot.dot(rows,{x:'n',y:'y',fill:'series',stroke:'var(--ak-color-bg-canvas)',r:3.5,tip:true,
        title:d=>`${d.series}\n参数 ${count(d.n)} · Loss ${d.y.toFixed(2)}`}),
      Plot.text([{n:1e12,y:1.69,t:'不可约损失 1.69'}],{x:'n',y:'y',text:'t',textAnchor:'end',dy:-10,fill:'var(--ak-color-text-muted)'})]};
});
ak.param(document.querySelector('#params'),{label:'训练数据量',min:20,max:5000,step:20,value:dataTokens,
  format:d=>count(d*1e9)+' tokens',onInput:d=>{dataTokens=d;draw();}});
</script>'''

SCATTER = '''<h2>散点、回归线与置信区间可以叠在一起</h2>
<p class="caption">示意数据：横轴是练习时长，纵轴是表现；误差带描述拟合的不确定性。</p>
<div id="scatter"></div>
<script>
const data=Array.from({length:24},(_,i)=>({x:i+1,y:14+2.1*i+6*Math.sin(i*1.7)}));
ak.plot(document.querySelector('#scatter'),width=>({x:{label:'练习时长'},y:{label:'表现'},
  marks:[Plot.linearRegressionY(data,{x:'x',y:'y',stroke:'var(--ak-chart-2)',fill:'var(--ak-chart-2)'}),
    Plot.dot(data,{x:'x',y:'y',fill:'var(--ak-chart-1)',stroke:'var(--ak-color-bg-canvas)',r:4,tip:true}),
    Plot.text([{x:18,y:20,text:'相同趋势，仍有波动'}],{x:'x',y:'y',text:'text',fill:'var(--ak-color-text-primary)'})]}));
</script>'''

CANVAS = '''<h2>沿着坡度走，小球逐步接近最低点</h2>
<p class="caption">原生 Canvas 模拟：每一步按 x ← x − η·2x 更新；可以改变学习率。</p>
<canvas id="descent" aria-label="梯度下降抛物线与小球" role="img" style="width:100%;height:240px"></canvas>
<div id="rate"></div><div class="actions"><button id="step">走一步</button> <button id="reset">重新开始</button></div>
<p>当前位置 x = <output id="position">3.000</output>，损失 x² = <output id="energy">9.000</output></p>
<script>
const canvas=document.querySelector('#descent'),ctx=canvas.getContext('2d');let x=3,rate=.08;
function draw(){const w=canvas.clientWidth,h=240,dpr=devicePixelRatio;canvas.width=w*dpr;canvas.height=h*dpr;ctx.scale(dpr,dpr);
 const css=getComputedStyle(document.documentElement),px=v=>20+(v+3.2)/6.4*(w-40),py=v=>220-v*18;
 ctx.strokeStyle=css.getPropertyValue('--ak-color-border-default');ctx.beginPath();ctx.moveTo(20,220);ctx.lineTo(w-20,220);ctx.stroke();
 ctx.strokeStyle=css.getPropertyValue('--ak-chart-1');ctx.lineWidth=2;ctx.beginPath();for(let i=0;i<=120;i++){const v=-3.2+i*6.4/120;i?ctx.lineTo(px(v),py(v*v)):ctx.moveTo(px(v),py(v*v));}ctx.stroke();
 ctx.fillStyle=css.getPropertyValue('--ak-chart-2');ctx.beginPath();ctx.arc(px(x),py(x*x),7,0,Math.PI*2);ctx.fill();
 document.querySelector('#position').textContent=x.toFixed(3);document.querySelector('#energy').textContent=(x*x).toFixed(3);}
ak.param(document.querySelector('#rate'),{label:'学习率 η',min:.01,max:.4,step:.01,value:rate,format:v=>v.toFixed(2),onInput:v=>{rate=v;}});
document.querySelector('#step').onclick=()=>{x-=rate*2*x;draw();};document.querySelector('#reset').onclick=()=>{x=3;draw();};
new ResizeObserver(()=>requestAnimationFrame(draw)).observe(canvas);addEventListener('akashic:theme',draw);draw();
</script>'''

INTRO = "Scaling Law 描述的是：模型越大、训练数据越多，损失（Loss）越低，但每多一份投入换来的下降会越来越少。\n\n下面用拟合公式画两条曲线。你可以拖动数据量，观察第二条曲线如何变化。"
EXPLANATION = "这幅图有两个特别重要的含义。\n\n**第一，收益递减。** 参数量增加 10 倍，Loss 不会也下降 10 倍。\n\n**第二，瓶颈转移。** 固定数据量时，模型再大也会遇到平台；增加数据量，平台随之下移。"
