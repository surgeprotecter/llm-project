// Browser QA of actual report HTML. Uses the browser runtime installed with Booking MCP.
const {chromium} = require(process.env.REPORT_BROWSER_MODULE || '/Users/yuming/.npm/_npx/9cffaea4229252f1/node_modules/patchright');
const fs = require('fs');
const path = require('path');
const crypto = require('crypto');
(async () => {
  const reportPath = path.resolve(process.argv[2]);
  const prefix = path.resolve(process.argv[3]);
  const browser = await chromium.launch({headless:true, executablePath:'/Applications/Google Chrome.app/Contents/MacOS/Google Chrome'});
  const page = await browser.newPage({viewport:{width:1440,height:1150}});
  await page.goto('file://'+reportPath);
  await page.addStyleTag({content:'html{scroll-behavior:auto!important}'});
  const checks = {report_file:{path:reportPath,sha256:crypto.createHash('sha256').update(fs.readFileSync(reportPath)).digest('hex'),passed:true}};
  for (const [label, width, height, target] of [['desktop',1440,1150,null],['days',1440,1150,'days'],['hotels',1440,1150,'stay'],['budget',1440,1150,'cost'],['mobile',390,900,null],['mobile-hotels',390,900,'stay']]) {
    await page.setViewportSize({width,height});
    if (target) await page.locator('#'+target).scrollIntoViewIfNeeded();
    else await page.evaluate(()=>window.scrollTo(0,0));
    await page.waitForFunction(()=>[...document.images].filter(i=>{const r=i.getBoundingClientRect();return r.top<innerHeight&&r.bottom>0}).every(i=>i.complete),{timeout:8000}).catch(()=>{});
    await page.screenshot({path:prefix+'-'+label+'.png'});
    const dims=await page.evaluate(()=>({viewport:innerWidth, width:document.documentElement.scrollWidth, tokens:document.body.innerText.includes('{{')||document.body.innerText.includes('{%')}));
    checks[label]={...dims,passed:dims.width<=dims.viewport && !dims.tokens};
  }
  const firstHotel=page.locator('.hotel-image').first();
  if(await firstHotel.count()) {
    await page.setViewportSize({width:1440,height:1150});
    await firstHotel.scrollIntoViewIfNeeded();
    await firstHotel.evaluate(i=>i.decode()).catch(()=>{});
    const images=await page.locator('.hotel-image').evaluateAll(list=>list.map(i=>({loaded:i.complete&&i.naturalWidth>0,hidden:i.hidden})));
    checks.hotel_images={images,passed:images.some(i=>i.loaded)};
    const gallery=page.locator('[data-hotel-gallery]').first();
    if(await gallery.count() && await gallery.locator('[data-gallery-next]').isVisible()) {
      const before=await gallery.locator('.gallery-count').innerText();
      await gallery.locator('[data-gallery-next]').click();
      await gallery.locator('.hotel-slide:not([hidden]) img').first().evaluate(i=>i.decode()).catch(()=>{});
      const after=await gallery.locator('.gallery-count').innerText();
      const loaded=await gallery.locator('.hotel-slide:not([hidden]) img').first().evaluate(i=>i.complete&&i.naturalWidth>0);
      checks.carousel={before,after,loaded,passed:before!==after&&loaded};
    }

    await page.screenshot({path:prefix+'-hotel-cards.png'});
  }
  fs.writeFileSync(prefix+'-layout.json',JSON.stringify(checks,null,2));
  await browser.close();
  console.log(JSON.stringify(checks));
  if(Object.values(checks).some(x=>!x.passed)) process.exitCode=1;
})().catch(e=>{console.error(e);process.exitCode=1;});
