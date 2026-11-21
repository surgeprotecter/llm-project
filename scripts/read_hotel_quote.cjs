// Read visible hotel evidence. Never selects a room or enters a booking flow.
const fs = require('fs');
const path = require('path');
const os = require('os');

function sameProperty(value, requested) {
  try {
    const url = new URL(value);
    // Booking may redirect to the same hotel's language-specific pathname.
    const propertyPath = pathname => pathname.replace(/\.[a-z]{2}(?:-[a-z]{2})?\.html$/i, '.html');
    return url.protocol === 'https:' && ['www.booking.com','booking.com'].includes(url.hostname)
      && propertyPath(url.pathname) === propertyPath(requested.pathname);
  } catch { return false; }
}

// Keep source URLs intact (including signed query strings). Resized copies of
// the same Booking photo share an ID, so they count as one photo.
function uniqueHotelPhotos(values) {
  const seen = new Set();
  return values.filter(value => {
    try {
      const url = new URL(value);
      const photo = url.pathname.match(/^\/xdata\/images\/hotel\/[^/]+\/(\d+)\.(?:jpe?g|png|webp|avif)$/i);
      if (url.protocol !== 'https:' || !/(^|\.)bstatic\.com$/i.test(url.hostname) || !photo || seen.has(photo[1])) return false;
      seen.add(photo[1]);
      return true;
    } catch { return false; }
  }).slice(0, 12);
}

// Read only the requested property's gallery, never all images on the page:
// recommendation cards, reviewer avatars and branding are not hotel evidence.
function readGalleryPhotos() {
  const gallery = document.querySelector('#photo_wrapper, #hotelPhotosInc, [data-testid="property-gallery"], [data-testid="gallery-container"], #hotel_main_content .bh-photo-grid');
  const galleries = gallery ? [gallery] : [];
  const photos = [];
  const add = value => {
    if (!value) return;
    try { photos.push(new URL(value, location.href).href); } catch { /* Ignore malformed lazy-image attributes. */ }
  };
  for (const gallery of galleries) {
    for (const element of gallery.querySelectorAll('img, a, [data-highres], [data-thumb-url], [style*="background-image"]')) {
      for (const attribute of ['data-highres', 'href', 'data-src', 'data-lazy-src', 'data-thumb-url']) {
        const value = element.getAttribute(attribute);
        add(value);
      }
      if (element.tagName === 'IMG') {
        (element.getAttribute('srcset') || '').split(',').reverse().forEach(part => add(part.trim().split(/\s+/)[0]));
        add(element.currentSrc || element.src);
      }
      const background = element.style.backgroundImage.match(/url\(["']?([^"')]+)["']?\)/);
      if (background) add(background[1]);
    }
  }
  return photos;
}

let input='';
process.stdin.on('data', x => input+=x);
process.stdin.on('end', async()=>{
  let browser;
  try {
    const args=JSON.parse(input);
    const root=path.join(os.homedir(),'.npm','_npx');
    const dirs=process.env.REPORT_BROWSER_MODULE ? [process.env.REPORT_BROWSER_MODULE] : fs.readdirSync(root).map(x=>path.join(root,x,'node_modules','patchright'));
    const modulePath=dirs.find(x=>fs.existsSync(x));
    if(!modulePath) throw new Error('Booking browser runtime patchright is not installed');
    const {chromium}=require(modulePath);
    browser=await chromium.launch({channel:'chrome',headless:true});
    const context=await browser.newContext({locale:'en-US',viewport:{width:1366,height:768},
      userAgent:'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
      extraHTTPHeaders:{'Accept-Language':'en-US,en;q=0.9'}});
    const auth=path.join(path.dirname(modulePath),'@striderlabs','mcp-booking','dist','auth.js');
    if(fs.existsSync(auth)) await (await import(require('url').pathToFileURL(auth).href)).loadCookies(context);
    const page=await context.newPage();
    const url=new URL(args.url);
    if(url.protocol!=='https:' || !['www.booking.com','booking.com'].includes(url.hostname) || !url.pathname.startsWith('/hotel/')) throw new Error('Expected a Booking hotel URL');
    for(const [k,v] of Object.entries({checkin:args.checkin,checkout:args.checkout,group_adults:args.adults,no_rooms:args.rooms,selected_currency:args.currency,lang:'en-us'})) url.searchParams.set(k,String(v));
    await page.goto(url.href,{waitUntil:'domcontentloaded',timeout:45000});
    await page.locator('#hprt-table tr[data-block-id], [data-testid="property-card"]').first().waitFor({timeout:15000}).catch(()=>{});
    let data=await page.evaluate(()=>{
      const body=document.body.innerText;
      const currency=document.querySelector('[data-testid="header-currency-picker-trigger"], [data-testid="currency-selector"]');
      return {url:location.href,title:document.title,body:body.slice(0,30000),currency_text:currency?.innerText||currency?.getAttribute('aria-label')||'',
        stay_context:[document.querySelector('#hprt-table thead')?.innerText||'', ...body.matchAll(/.{0,70}(?:\d+ nights?|\d+ adults?|Dec \d+|Nov \d+).{0,90}/g)].map(x=>typeof x==='string'?x:x[0]).slice(0,18),
        rows:[...document.querySelectorAll('#hprt-table tr[data-block-id]')].slice(0,12).map(row=>({text:row.innerText,price:row.getAttribute('data-hotel-rounded-price')||row.getAttribute('data-hotel-price'),block_id:row.getAttribute('data-block-id')})),
        price_context:[...document.querySelectorAll('[data-testid="price-and-discounted-price"], [data-testid="price-for-x-nights"], .prco-wrapper')].slice(0,12).map(e=>e.parentElement.innerText.slice(0,1800)),
        images:[]};
    });
    // A redirect to another property or an interstitial must not supply photos.
    if (sameProperty(page.url(), url)) data.images = uniqueHotelPhotos(await page.evaluate(readGalleryPhotos));
    if(args.name) {
      const search=new URL('https://www.booking.com/searchresults.html');
      for(const [k,v] of Object.entries({ss:args.destination||args.name,checkin:args.checkin,checkout:args.checkout,group_adults:args.adults,no_rooms:args.rooms,group_children:0,selected_currency:args.currency,lang:'en-us'})) search.searchParams.set(k,String(v));
      await page.goto(search.href,{waitUntil:'domcontentloaded',timeout:45000});
      await page.locator('[data-testid="property-card"]').first().waitFor({timeout:15000}).catch(()=>{});
      const cards=await page.evaluate(()=>[...document.querySelectorAll('[data-testid="property-card"]')].slice(0,15).map(e=>({
        name:e.querySelector('[data-testid="title"]')?.textContent,
        url:e.querySelector('a[data-testid="title-link"]')?.href,
        text:e.innerText,
        price_text:e.querySelector('[data-testid="price-and-discounted-price"]')?.textContent,
        images:[...e.querySelectorAll('img')].flatMap(img=>[img.currentSrc, img.src, img.getAttribute('data-src')]).filter(Boolean),
      })));
      data.cards=cards.filter(c=>sameProperty(c.url, url));
      for (const card of data.cards) {
        card.images=uniqueHotelPhotos(card.images);
        card.image=card.images[0] || '';
      }
      data.images=uniqueHotelPhotos([...data.images, ...data.cards.flatMap(card=>card.images)]);
      data.search_url=page.url();
      data.search_context=await page.locator('body').innerText().then(x=>x.slice(0,1500));
    }
    console.log(JSON.stringify(data));
  } catch(e) {console.log(JSON.stringify({error:String(e)}));process.exitCode=1;}
  finally {if(browser) await browser.close();}
});
