const { chromium } = require('playwright');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');

(async () => {
  const root = path.resolve(__dirname, '..');
  const out = path.join(root, 'docs', 'screenshots');
  fs.mkdirSync(out, {recursive:true});
  const browser = await chromium.launch({headless:true, channel:process.env.BROWSER_CHANNEL || 'chrome'});
  const context = await browser.newContext({viewport:{width:1440,height:1000}});
  const page = await context.newPage();
  const results = [];
  const errors = [];
  const sessions = new Set();
  let latest;
  page.on('pageerror', e => errors.push(e.message));
  const origin = process.env.TEST_BASE_URL || 'http://127.0.0.1:8000';
  async function act(route, callback) {
    const waiting = page.waitForResponse(r => r.url().endsWith('/chat/'+route) && r.status()===200);
    await callback();
    const res = await waiting;
    latest = await res.json();
    sessions.add(latest.session_id);
    await page.getByText('Version '+latest.version, {exact:true}).waitFor();
    return latest;
  }
  async function send(text, route) {
    await page.getByRole('textbox').fill(text);
    return act(route, () => page.getByRole('button',{name:'Send',exact:true}).click());
  }
  try {
    await page.goto(origin);
    await page.getByRole('heading',{name:'Symptom Guide',exact:true}).waitFor();
    await send('I am coughing, but no fever.', 'start');
    assert.equal(latest.negative_symptoms[0].kg_id,'DEMO:fever');
    results.push('Initial alias and negation rendered');
    await act('answer', () => page.getByRole('button',{name:'Not sure',exact:true}).click());
    assert.equal(latest.uncertain_symptoms[0].kg_id,'DEMO:breathlessness');
    assert.equal(latest.next_symptom.kg_id,'DEMO:chest_pain');
    results.push('Unknown preserved, no repeated question');
    await act('answer', () => page.getByRole('button',{name:'No',exact:true}).click());
    await act('answer', () => page.getByRole('button',{name:'Yes',exact:true}).click());
    assert.equal(latest.status,'COMPLETE');
    results.push('Complete multi-turn mock integration');
    await act('correct', () => page.getByLabel('Status for fever').selectOption('present'));
    assert(latest.positive_symptoms.some(s=>s.kg_id==='DEMO:fever'));
    results.push('Evidence correction and recalculation');
    await send('Actually no fever. I have a headache.', 'message');
    const id=latest.session_id, version=latest.version;
    await page.reload();
    await page.getByText('Version '+version,{exact:true}).waitFor();
    assert.equal(await page.evaluate(()=>sessionStorage.getItem('symptom-session')), id);
    results.push('Free-text correction and reload recovery');
    let blockedPayload;
    await page.route('**/api/chat/message', async route => {
      blockedPayload=route.request().postDataJSON();
      await route.abort('failed');
      await page.unroute('**/api/chat/message');
    }, {times:1});
    await page.getByRole('textbox').fill('I have nausea.');
    await page.getByRole('button',{name:'Send',exact:true}).click();
    await page.getByRole('button',{name:'Retry same request'}).waitFor();
    assert.equal((await context.request.get(origin+'/api/chat/session/'+id)).status(),200);
    await act('message', () => page.getByRole('button',{name:'Retry same request'}).click());
    assert.equal(latest.version,version+1);
    assert.equal(blockedPayload.expected_version,version);
    results.push('Network failure and same-request retry');
    const downloadPromise=page.waitForEvent('download');
    await page.getByRole('button',{name:'Export session'}).click();
    const download=await downloadPromise;
    const downloaded=await download.path();
    assert.equal(JSON.parse(fs.readFileSync(downloaded,'utf8')).session_id,id);
    results.push('JSON export contains current session');
    await page.screenshot({path:path.join(out,'desktop.png'),fullPage:true});
    await page.setViewportSize({width:390,height:844});
    assert(await page.evaluate(()=>document.documentElement.scrollWidth <= window.innerWidth));
    await page.screenshot({path:path.join(out,'mobile.png'),fullPage:true});
    results.push('Mobile width and screenshot');
    await page.getByRole('button',{name:'Delete & start over'}).click();
    await page.getByText('New session',{exact:true}).waitFor();
    assert.equal((await context.request.get(origin+'/api/chat/session/'+id)).status(),404);
    results.push('Delete removes backend state');
    await send('Hello there', 'start');
    const emptyId=latest.session_id;
    await send('I am dizzy.', 'message');
    assert.equal(latest.session_id,emptyId);
    assert.equal(latest.positive_symptoms[0].kg_id,'DEMO:dizziness');
    results.push('Unmatched input can continue without creating another session');
    assert.deepEqual(errors,[]);
    results.push('No browser JavaScript errors');
    console.log(JSON.stringify({passed:results.length,checks:results},null,2));
    fs.writeFileSync(path.join(root,'evaluation','browser-report.json'),JSON.stringify({
      generated_at:new Date().toISOString(),browser:await browser.version(),passed:results.length,checks:results,errors
    },null,2));
  } catch(e) {
    await page.screenshot({path:path.join(out,'failure.png'),fullPage:true});
    throw e;
  } finally {
    for (const id of sessions) await context.request.delete(origin+'/api/chat/session/'+id);
    await browser.close();
  }
})().catch(e=>{console.error(e);process.exitCode=1});

