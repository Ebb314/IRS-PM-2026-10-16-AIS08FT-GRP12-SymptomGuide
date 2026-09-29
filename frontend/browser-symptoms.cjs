// Isolated UI contract tests; fixtures are explicit, not LLM quality evidence.
const {chromium}=require('playwright');
const assert=require('node:assert/strict'),fs=require('node:fs'),path=require('node:path');
const origin=process.env.TEST_BASE_URL||'http://127.0.0.1:8007';
const evidence=(id,name,status)=>({kg_id:id,kg_name:name,status,original_text:'fictional test input',turn:1,source:'test-fixture',duration_text:'',severity_text:''});
(async()=>{
 const browser=await chromium.launch({headless:true,channel:'chrome'});
 const page=await browser.newPage({viewport:{width:1440,height:1080}});
 let state={session_id:'sidebar-fixture',version:1,status:'RUNNING',evidence:[evidence('P986','Drowsiness','present'),evidence('T2','Cough','absent'),evidence('T3','Fatigue','uncertain')],pending_evidence:[evidence('T4','Headache','present'),evidence('T5','Dizziness','present')],messages:[],warnings:[],excluded_evidence:[],candidate_diseases:[],drug_recommendations:[],ai_trace:{},timings_ms:{},evidence_history:[],mode:'random',round:0,symptom_question_active:false};
 const requests=[],errors=[];let fail=false;
 page.on('pageerror',e=>errors.push(e.message));
 await page.route('**/api/**',async route=>{
  const url=route.request().url();
  if(url.endsWith('/vocabulary'))return route.fulfill({json:[]});
  if(url.endsWith('/correct')){
   const body=route.request().postDataJSON();requests.push(body);
   if(fail)return route.fulfill({status:503,json:{detail:'Fixture adapter unavailable'}});
   assert.equal(body.expected_version,state.version);
   let item=state.evidence.find(e=>e.kg_id===body.kg_id)||state.pending_evidence.find(e=>e.kg_id===body.kg_id);
   state.pending_evidence=state.pending_evidence.filter(e=>e.kg_id!==body.kg_id);
   if(body.status!=='dismiss'){
    state.evidence=state.evidence.filter(e=>e.kg_id!==body.kg_id);
    state.evidence.push({...item,status:body.status});
   }
   state.version++;
  }
  return route.fulfill({json:state});
 });
 try{
  await page.addInitScript(()=>sessionStorage.setItem('symptom-session','sidebar-fixture'));
  await page.goto(origin);
  const panel=page.getByRole('region',{name:'Symptom log',exact:true});
  await panel.getByText('Drowsiness',{exact:true}).waitFor();
  await page.getByLabel('Status for Drowsiness',{exact:true}).selectOption('absent');
  await page.getByText('Saved · session version 2',{exact:true}).waitFor();
  assert.equal(await panel.getByRole('region',{name:'Absent symptoms'}).getByText('Drowsiness',{exact:true}).count(),1);
  await page.getByLabel('Status for Drowsiness',{exact:true}).selectOption('uncertain');
  await page.getByText('Saved · session version 3',{exact:true}).waitFor();
  await page.getByRole('button',{name:'Confirm Headache as Present',exact:true}).click();
  await page.getByText('Saved · session version 4',{exact:true}).waitFor();
  await page.getByRole('button',{name:'Ignore suggestion',exact:true}).click();
  await page.getByText('Saved · session version 5',{exact:true}).waitFor();
  assert.equal(await page.getByRole('region',{name:'Suggested matches',exact:true}).count(),0);
  await page.reload();await page.getByText('Saved · session version 5',{exact:true}).waitFor();
  assert.equal(await page.getByLabel('Status for Drowsiness',{exact:true}).inputValue(),'uncertain');
  fail=true;
  await page.getByLabel('Status for Drowsiness',{exact:true}).selectOption('present');
  await page.getByRole('alert').getByText('Fixture adapter unavailable',{exact:true}).waitFor();
  assert.equal(await page.getByLabel('Status for Drowsiness',{exact:true}).inputValue(),'uncertain');
  fail=false;await page.getByRole('button',{name:'Retry same request',exact:true}).click();
  await page.getByText('Saved · session version 6',{exact:true}).waitFor();
  assert.equal(requests.at(-1).request_id,requests.at(-2).request_id);
  await page.screenshot({path:path.resolve(__dirname,'../docs/screenshots/symptoms-desktop.png'),fullPage:true});
  await page.setViewportSize({width:390,height:844});
  assert(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth));
  await panel.scrollIntoViewIfNeeded();
  await page.screenshot({path:path.resolve(__dirname,'../docs/screenshots/symptoms-mobile.png'),fullPage:true});
  assert.deepEqual(errors,[]);
  fs.writeFileSync(path.resolve(__dirname,'../evaluation/browser-symptoms.json'),JSON.stringify({passed:true,kind:'mocked-api-ui-contract',checks:['three status groups','status changes move groups','confirm suggestion','ignore suggestion','restore after reload','failed update retains saved status','retry preserves request ID','390px no horizontal overflow','no JS errors'],requests},null,2));
  console.log('Sidebar UI contract checks passed');
 } finally{await browser.close()}
})().catch(e=>{console.error(e);process.exit(1)});

