// Targeted behavior checks, not a general language-understanding benchmark.
const fs=require('node:fs'),path=require('node:path'),assert=require('node:assert/strict');
const file=path.resolve(__dirname,'../evaluation/conversation-agent.json');
const report=JSON.parse(fs.readFileSync(file,'utf8'));
const find=text=>report.rows.find(r=>r.message===text);
const checks=[];
function check(name,ok){checks.push({name,passed:!!ok})}
check('Mechanical checks',report.mechanical_checks_passed);
check('Description repair addresses the user',!/^(I feel|My (?:head|chest|body))/i.test(find('I feel bad but i dont know how to say').answer));
check('Short What uses conversation rather than narration',!/the question|what do you mean by what/i.test(find('What?').answer));
const cold=find('Whats cold').answer;
check('Cold ambiguity acknowledged',/chill|temperature|feeling cold/i.test(cold)&&/illness|infection|virus|common cold/i.test(cold));
const yes=report.rows.filter(r=>r.scenario==='repeated-acknowledgment'&&r.message==='yes');
check('Repeated yes has no question-mark prompts (syntax only)',yes.length===2&&yes.every(r=>!r.answer.includes('?')));
check('Knowledge and other-person dialogue does not create evidence',report.rows.filter(r=>r.scenario!=='report-correct-explain').every(r=>r.evidence.length===0));
check('Correction retained in saved evidence',find('What have you recorded about me?').evidence.some(e=>e.kg_id==='P986'&&e.status==='absent'));
const trip=find('Can you plan a trip to Tokyo?').answer;
check('Unrelated travel redirected',/symptom|health/i.test(trip)&&!/tailored itinerary|travel dates/i.test(trip));
check('Real app correction control',/dropdown|drop-down/i.test(find('Never mind the trip. How do I change an answer in this app?').answer));
const calls=report.rows.map(r=>[r.trace.understanding?.model_call,r.trace.understanding?.repair_call,...(r.trace.model_calls||[]),...(r.trace.review_calls||[])].filter(Boolean));
const durations=calls.map(list=>list.reduce((sum,c)=>sum+c.latency_ms,0)).sort((a,b)=>a-b);
const summary={created_at:new Date().toISOString(),scope:'16 targeted turns; pattern checks plus separately documented qualitative inspection, not a clinical or general NLU benchmark',checks,
passed:checks.every(c=>c.passed),model:report.rows[0].trace.model_call.model,quantization:report.rows[0].trace.model_call.quantization,
generation_seconds:{min:durations[0]/1000,median:durations[Math.floor(durations.length/2)]/1000,max:durations.at(-1)/1000},
latency_note:'Sum of generation calls; excludes cold model/index loading, retrieval, queueing and HTTP overhead.',
quality_note:'passed refers only to these automatic checks. See conversation-agent-human-review.md for remaining semantic issues; this is not a general understanding accuracy score.',
remaining_limitations:['Small targeted sample','Same-model review can retain errors','Bounded recent-message context','Conservative evidence validation','No clinical factuality evaluation']};
fs.writeFileSync(path.resolve(__dirname,'../evaluation/conversation-agent-review.json'),JSON.stringify(summary,null,2));
console.log(JSON.stringify(summary,null,2));assert(summary.passed,'One or more behavior checks failed');
