const {chromium}=require('playwright');
const assert=require('node:assert/strict'),fs=require('node:fs'),path=require('node:path');
(async()=>{
 const origin=process.env.TEST_BASE_URL||'http://127.0.0.1:8006';
 const browser=await chromium.launch({headless:true,channel:'chrome'});
 const context=await browser.newContext({viewport:{width:1440,height:1080}});
 const page=await context.newPage();page.setDefaultTimeout(240000);
 const errors=[],rows=[];let state;
 page.on('pageerror',e=>errors.push(e.message));
 async function send(text){
  const route=state?'message':'start';
  const wait=page.waitForResponse(r=>r.url().endsWith('/chat/'+route)&&r.status()===200);
  await page.getByRole('textbox').first().fill(text);
  await page.getByRole('button',{name:'Send',exact:true}).click();
  state=await(await wait).json();
  await page.getByText('Version '+state.version,{exact:true}).waitFor();
  rows.push({message:text,answer:state.message,engine:state.ai_trace.engine});
  assert.equal(state.ai_trace.engine,'conversation-agent-v1');
  assert.equal(await page.getByRole('button',{name:'Yes',exact:true}).count(),0);
 }
 try{
  await page.goto(origin);
  await send('I have Drowsiness.');
  assert(state.evidence.some(e=>e.kg_id==='P986'));
  await send('What does that word mean?');
  assert.equal(state.evidence.length,1);
  assert.match(state.message,/sleep|drows|alert/i);
  await page.reload();
  await page.getByText('Version '+state.version,{exact:true}).waitFor();
  await page.getByText(/Supported vocabulary \(15135\)/).waitFor();
  assert.equal(await page.getByRole('button',{name:'Yes',exact:true}).count(),0);
  const folder=path.resolve(__dirname,'../docs/screenshots');fs.mkdirSync(folder,{recursive:true});
  await page.screenshot({path:path.join(folder,'agent-dialogue.png'),fullPage:true});
  assert.deepEqual(errors,[]);
  fs.writeFileSync(path.resolve(__dirname,'../evaluation/browser-agent.json'),JSON.stringify({passed:true,rows,checks:['real model response','reference to last symptom','no unrelated Yes buttons','session restored after reload','no browser errors']},null,2));
  console.log(JSON.stringify({passed:true,rows}));
 }finally{
  if(state)await context.request.delete(origin+'/api/chat/session/'+state.session_id);
  await browser.close();
 }
})().catch(e=>{console.error(e);process.exit(1)});
