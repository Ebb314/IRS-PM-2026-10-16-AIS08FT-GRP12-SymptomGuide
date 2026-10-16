const {chromium}=require('playwright');
const assert=require('node:assert/strict');
const fs=require('node:fs');const path=require('node:path');
(async()=>{
 const root=path.resolve(__dirname,'..'),origin=process.env.TEST_BASE_URL||'http://127.0.0.1:8004';
 const browser=await chromium.launch({headless:true,channel:process.env.BROWSER_CHANNEL||'chrome'});
 const context=await browser.newContext({viewport:{width:1440,height:1100}});
 const page=await context.newPage();page.setDefaultTimeout(180000);
 const checks=[],errors=[],traces=[];let latest,session;
 page.on('pageerror',e=>errors.push(e.message));
 async function action(route,callback){
  const wait=page.waitForResponse(r=>r.url().endsWith('/chat/'+route)&&r.status()===200);
  await callback();latest=await(await wait).json();session=latest.session_id;
  await page.getByText('Version '+latest.version,{exact:true}).waitFor();traces.push(latest);return latest;
 }
 async function send(text,route='message'){
  await page.getByRole('textbox').first().fill(text);
  return action(route,()=>page.getByRole('button',{name:'Send',exact:true}).click());
 }
 try{
  const health=await(await context.request.get(origin+'/api/health',{timeout:180000})).json();
  assert.equal(health.vocabulary_size,15135);assert.equal(health.mode,'random');assert.equal(health.ai_profile,'local');
  await page.goto(origin);
  await send('I have Fever.','start');
  assert(latest.positive_symptoms.some(e=>e.kg_id==='P835'));
  assert(latest.candidate_diseases.every(c=>c.id.startsWith('SIM:')));
  checks.push('Full catalog health and actual P IDs with random disease fixtures');
  await page.getByRole('button',{name:'Yes',exact:true}).waitFor();
  await page.getByRole('button',{name:'Describe in my own words',exact:true}).click();
  assert.equal(await page.getByRole('button',{name:'Yes',exact:true}).count(),0);
  await page.getByRole('button',{name:'Show answer choices',exact:true}).click();
  await page.getByRole('button',{name:'Yes',exact:true}).waitFor();
  await page.getByRole('textbox').first().focus();
  assert.equal(await page.getByRole('button',{name:'Yes',exact:true}).count(),0);
  checks.push('Explicit free-text mode and direct composer focus hide choices; reversible');
  const oldQ=latest.next_symptom.kg_id;
  await send('I have Cough.');
  assert.equal(latest.knowledge_focus.kg_id,'P4018');
  if(latest.next_symptom.kg_id===oldQ)assert.equal(await page.getByRole('button',{name:'Yes',exact:true}).count(),0);
  checks.push('Free-text report advances evidence and old choices remain hidden');
  const evidence=structuredClone(latest.evidence),round=latest.round,q=latest.next_symptom;
  await action('message',()=>page.getByRole('button',{name:'Ask a knowledge question',exact:true}).click());
  assert.equal(latest.ai_trace.topic.kg_id,'P4018');assert(latest.ai_trace.citations.some(c=>c.id==='nhs-cough'));
  assert.deepEqual(latest.evidence,evidence);assert.equal(latest.round,round);assert.deepEqual(latest.next_symptom,q);
  checks.push('Knowledge button resolves B Cough after A Fever without consuming question');
  await page.screenshot({path:path.join(root,'docs/screenshots/fullkg-topic-cough.png'),fullPage:true});
  await send('I have Headache.');
  await action('message',()=>page.getByRole('button',{name:'Ask a knowledge question',exact:true}).click());
  assert.equal(latest.ai_trace.topic.kg_id,'P885');assert.equal(latest.ai_trace.response_mode,'local-llm-general');
  assert(latest.message.includes('Headache'));assert(!latest.message.includes('Fever'));
  checks.push('Knowledge button resolves C Headache with real LLM general explanation');
  await page.reload();await page.getByText('Version '+latest.version,{exact:true}).waitFor();
  await action('message',()=>page.getByRole('button',{name:'Ask a knowledge question',exact:true}).click());
  assert.equal(latest.ai_trace.topic.kg_id,'P885');checks.push('Latest knowledge topic survives reload');
  await send('I have Cough.');
  await action('message',()=>page.getByRole('button',{name:'Ask a knowledge question',exact:true}).click());
  assert.equal(latest.ai_trace.topic.kg_id,'P4018');checks.push('Revisiting an existing symptom updates latest topic by event order');
  await page.getByText('Supported vocabulary (15135)',{exact:true}).click();
  await page.getByLabel('Search vocabulary').fill('Renal cyst');
  assert(await page.getByText('Renal cyst',{exact:false}).count()>0);
  checks.push('Full vocabulary searchable with bounded rendering');
  await page.setViewportSize({width:390,height:844});
  assert(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth+1));
  checks.push('Updated controls fit mobile viewport');
  assert.deepEqual(errors,[]);checks.push('No browser JavaScript errors');
  fs.writeFileSync(path.join(root,'evaluation/browser-fullkg-report.json'),JSON.stringify({created_at:new Date().toISOString(),health,passed:checks.length,checks,traces},null,2));
  console.log(JSON.stringify({passed:checks.length,checks},null,2));
 }finally{if(session)await context.request.delete(origin+'/api/chat/session/'+session);await browser.close()}
})().catch(e=>{console.error(e);process.exit(1)});
