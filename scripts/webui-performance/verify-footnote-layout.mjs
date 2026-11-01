/** 在真实阅读页对照旧、新选择器的几何，并验证每个脚注引用的往返焦点。 */
import {mkdirSync,writeFileSync,readFileSync} from 'node:fs';
import {parseArgs} from 'node:util';
import {createHash} from 'node:crypto';
import {chromium} from 'playwright-core';
const {values}=parseArgs({options:{url:{type:'string'},output:{type:'string'},browser:{type:'string'},build:{type:'string'}}});
if(!values.url||!values.output||!values.browser||!values.build)throw new Error('需要 --url、--output、--browser、--build');
mkdirSync(values.output);
const state=await(await fetch(new URL('/api/shell/state',values.url))).json();
if(state.build!==values.build)throw new Error('构建不符');
const report={complete:false,build:state.build,harnessSha256:createHash('sha256').update(readFileSync(import.meta.filename)).digest('hex'),samples:[]};
const browser=await chromium.launch({executablePath:values.browser});
report.browser=browser.version();
try{
  for(const width of [1440,390]){
    const context=await browser.newContext({viewport:{width,height:900},reducedMotion:'reduce',serviceWorkers:'block'});
    const page=await context.newPage(),errors=[];
    page.on('pageerror',error=>errors.push(error.message));
    await page.addInitScript(()=>localStorage.setItem('akashic.onboarding.seen','1'));
    await page.goto(new URL('/?session=akashic%3Areading-benchmark',values.url).href);
    const message=page.locator('[data-message-id="reading-sample"]');
    await message.locator('.footnote-reference').first().waitFor();
    await page.waitForFunction(()=>document.querySelector('diffs-container')?.shadowRoot?.querySelector('[data-line] span[style*="--diffs-token-"]'));
    await page.evaluate(()=>document.fonts.ready);
    await page.waitForLoadState('networkidle');
    const refs=message.locator('.footnote-reference a');
    if(await refs.count()!==8)throw new Error('reading-footnotes.md 的八个引用未完整呈现');
    if(await message.locator('.reading-footnote').count()!==2)throw new Error('脚注正文未完整呈现');
    // 1. 在同一份真实 DOM 上切换唯一改动的 CSS 规则，排除内容和滚动状态差异。
    await page.evaluate(()=>{
      const rules=[...document.styleSheets].flatMap(sheet=>[...sheet.cssRules]).filter(rule=>rule.selectorText?.includes('.paragraph-node:has(~ .footnote-reference)'));
      if(rules.length!==1)throw new Error('候选选择器数量不符');
      window.footnoteRule=rules[0];window.footnoteCandidate=rules[0].selectorText;
      rules[0].selectorText=window.footnoteCandidate.slice(0,window.footnoteCandidate.indexOf('.node-content'))+'.node-content:has(> .footnote-reference) > .paragraph-node';
    });
    const baseline=await snapshot(page);
    const before=await page.screenshot({path:`${values.output}/${width}-baseline.png`});
    await page.evaluate(()=>{window.footnoteRule.selectorText=window.footnoteCandidate;});
    const candidate=await snapshot(page);
    const after=await page.screenshot({path:`${values.output}/${width}-candidate.png`});
    const row={width,geometryEqual:JSON.stringify(baseline)===JSON.stringify(candidate),pixelsEqual:before.equals(after),baseline,candidate,roundTrips:0};
    report.samples.push(row);
    if(!row.geometryEqual||!row.pixelsEqual)throw new Error('脚注排版发生变化');
    // 2. 每个引用都经过真实点击与键盘返回；重复引用必须回到自己的位置。
    const originalUrl=page.url();
    for(let index=0;index<8;index++){
      await refs.nth(index).click();
      const backward=page.locator(':focus');
      if(!await backward.evaluate(el=>el.matches('[data-footnote-backref],.footnote-anchor')))throw new Error('未聚焦返回链接');
      await backward.press('Enter');
      if(!await refs.nth(index).evaluate(el=>el===document.activeElement))throw new Error('返回了错误引用');
      if(page.url()!==originalUrl)throw new Error('脚注改变了页面地址');
      row.roundTrips++;
    }
    if(errors.length)throw new Error(errors.join('\n'));
    await context.close();
  }
  report.complete=true;
}catch(error){report.error=String(error);process.exitCode=1;}
finally{writeFileSync(`${values.output}/report.json`,JSON.stringify(report,null,2));await browser.close();}

async function snapshot(page){
  await page.evaluate(()=>new Promise(done=>requestAnimationFrame(()=>requestAnimationFrame(done))));
  return page.locator('[data-message-id="reading-sample"]').evaluate(root=>{
    const origin=root.getBoundingClientRect();
    return [...root.querySelectorAll('.paragraph-node,.node-content,.footnote-reference,.reading-footnote')].map(el=>{
      const style=getComputedStyle(el),rect=el.getBoundingClientRect();
      return {text:el.textContent,display:style.display,margin:style.margin,
        x:rect.x-origin.x,y:rect.y-origin.y,width:rect.width,height:rect.height};
    });
  });
}
