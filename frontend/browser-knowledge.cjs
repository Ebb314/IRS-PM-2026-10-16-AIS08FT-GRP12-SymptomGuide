const {chromium}=require('playwright');const assert=require('node:assert/strict');
const fs=require('node:fs');const path=require('node:path');
(async()=>{
 const root=path.resolve(__dirname,'..'),url=process.env.TEST_BASE_URL||'http://127.0.0.1:8000';
 const browser=await chromium.launch({headless:true,channel:'chrome'});
 const page=await browser.newPage({viewport:{width:1440,height:1100}});page.setDefaultTimeout(240000);
 const rows=[];let state;
 async function action(route,callback){const wait=page.waitForResponse(r=>r.url().endsWith('/chat/'+route)&&r.status()===200);await callback();state=await(await wait).json();await page.getByText('Version '+state.version,{exact:true}).waitFor();return state;}
 try{
  await page.goto(url);
  await page.getByRole('textbox').first().fill('I have Drowsiness.');
  await action('start',()=>page.getByRole('button',{name:'Send',exact:true}).click());
  const evidence=structuredClone(state.evidence),round=state.round,pending=state.next_symptom;
  await action('message',()=>page.getByRole('button',{name:'Ask a knowledge question',exact:true}).click());
  assert.equal(state.ai_trace.topic.kg_id,'P986');assert.equal(state.ai_trace.response_mode,'local-llm-general');
  assert(state.ai_trace.model_call.output_tokens>0);assert.equal(state.ai_trace.model_call.model,'Qwen2.5-1.5B-Instruct');
  assert.deepEqual(state.ai_trace.citations,[]);assert(!state.message.includes('listed as a symptom node'));
  assert.deepEqual(state.evidence,evidence);assert.equal(state.round,round);assert.deepEqual(state.next_symptom,pending);
  rows.push({topic:'Drowsiness',response:state});
  await page.getByText('AI explanation from model knowledge; no retrieved sources.',{exact:true}).waitFor();
  await page.screenshot({path:path.join(root,'docs/screenshots/llm-drowsiness.png'),fullPage:true});
  await page.getByRole('textbox').first().fill('I have Headache.');
  await action('message',()=>page.getByRole('button',{name:'Send',exact:true}).click());
  await action('message',()=>page.getByRole('button',{name:'Ask a knowledge question',exact:true}).click());
  assert.equal(state.ai_trace.topic.kg_id,'P885');assert.equal(state.ai_trace.response_mode,'local-llm-general');
  rows.push({topic:'Headache',response:state});
  fs.writeFileSync(path.join(root,'evaluation/llm-knowledge-browser.json'),JSON.stringify({created_at:new Date().toISOString(),passed:true,rows},null,2));
  console.log(JSON.stringify(rows.map(r=>({topic:r.topic,answer:r.response.message,trace:r.response.ai_trace})),null,2));
 }finally{if(state)await page.request.delete(url+'/api/chat/session/'+state.session_id);await browser.close()}
})().catch(e=>{console.error(e);process.exit(1)});
