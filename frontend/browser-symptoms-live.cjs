const {chromium}=require('playwright');
const assert=require('node:assert/strict'),fs=require('node:fs'),path=require('node:path');
(async()=>{
 const origin=process.env.TEST_BASE_URL||'http://127.0.0.1:8000';
 const browser=await chromium.launch({headless:true,channel:'chrome'});
 const context=await browser.newContext({viewport:{width:1440,height:1080}});
 const page=await context.newPage();page.setDefaultTimeout(240000);
 let state;const rows=[],errors=[];page.on('pageerror',e=>errors.push(e.message));
 async function send(message){
  const wait=page.waitForResponse(r=>r.url().endsWith('/chat/'+(state?'message':'start'))&&r.status()===200);
  await page.locator('textarea').fill(message);await page.getByRole('button',{name:'Send',exact:true}).click();
  state=await(await wait).json();await page.getByText('Version '+state.version,{exact:true}).waitFor();
  assert.equal(state.ai_trace.reply_pipeline,'direct');
  assert.equal(state.ai_trace.response_mode,'multi-turn-llm');
  rows.push({message,answer:state.message,timings:state.timings_ms});
 }
 try{
  await page.goto(origin);await send('I have Drowsiness.');
  assert(state.evidence.some(e=>e.kg_id==='P986'&&e.status==='present'));
  for(const status of ['absent','uncertain','present']){
   const started=Date.now();const wait=page.waitForResponse(r=>r.url().endsWith('/chat/correct')&&r.status()===200);
   await page.getByLabel('Status for Drowsiness',{exact:true}).selectOption(status);
   state=await(await wait).json();await page.getByText('Version '+state.version,{exact:true}).waitFor();
   assert.equal(state.evidence.find(e=>e.kg_id==='P986').status,status);
   assert.equal(state.ai_trace.response_mode,'explicit-record-update');
   assert.equal(state.symptom_question_active,false);
   rows.push({status,wall_ms:Date.now()-started,server_ms:state.timings_ms.total});
  }
  await page.reload();await page.getByText('Version '+state.version,{exact:true}).waitFor();
  assert.equal(await page.getByLabel('Status for Drowsiness',{exact:true}).inputValue(),'present');
  await send('What have you recorded about me?');assert.equal(state.evidence.length,1);
  assert.match(state.message,/drowsiness/i);
  assert.equal(await page.getByRole('button',{name:'Yes',exact:true}).count(),0);
  assert.deepEqual(errors,[]);
  await page.screenshot({path:path.resolve(__dirname,'../docs/screenshots/symptoms-live.png'),fullPage:true});
  fs.writeFileSync(path.resolve(__dirname,'../evaluation/browser-symptoms-live.json'),JSON.stringify({passed:true,kind:'real-api-local-llm',rows},null,2));
  console.log(JSON.stringify({passed:true,rows}));
 }finally{if(state)await context.request.delete(origin+'/api/chat/session/'+state.session_id);await browser.close()}
})().catch(e=>{console.error(e);process.exit(1)});
