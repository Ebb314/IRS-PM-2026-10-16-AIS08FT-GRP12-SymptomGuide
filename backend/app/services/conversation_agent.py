"""Open-ended multi-turn dialogue, separate from validated evidence writes.

There is deliberately no intent enum, phrase router or intent example bank here.
"""
import html
import json
import os
import re
from pathlib import Path
from typing import Literal
from pydantic import Field,field_validator
from app.models.chat_models import Model,ExtractedSymptom
from app.services.model_runtime import parse_json
from app.services.retrieval_service import HybridIndex

class Observation(Model):
    concept_id: str
    quote: str=Field(min_length=1,max_length=400)
    status: Literal['present','absent','uncertain']

class Understanding(Model):
    user_goal: str=Field(default='',max_length=500)
    reference_resolution: str=Field(default='',max_length=500)
    unresolved: str=Field(default='',max_length=500)
    possible_meanings: list[str]=Field(default_factory=list,max_length=3)
    observations: list[Observation]=Field(default_factory=list,max_length=8)

    @field_validator('user_goal','reference_resolution','unresolved',mode='before')
    @classmethod
    def normalize_optional_description(cls,value):
        if value is None: return ''
        if isinstance(value,list) and all(isinstance(item,str) for item in value): return '; '.join(value)
        return value

class ConversationAgent:
    def __init__(self,generator,encoder=None):
        self.generator=generator
        documents=json.loads((Path(__file__).parents[1]/'data/knowledge.json').read_text(encoding='utf-8-sig'))
        self.knowledge=HybridIndex(documents,encoder)

    def understand(self,text,state,nlp):
        hits=nlp.index.search(text,k=6) if hasattr(nlp,'index') else []
        allowed={h['id'] for h in hits}
        allowed.update(e.kg_id for e in state.pending_evidence)
        allowed.update(e.kg_id for e in state.evidence)
        # Exact matches supplement candidate retrieval; they never write evidence.
        rules=getattr(nlp,'rules',nlp)
        baseline=rules.analyze(text,state.version+1)
        allowed.update(e.kg_id for e in baseline.evidence+baseline.excluded)
        previous=next((m.text for m in reversed(state.messages[:-1]) if m.sender=='assistant'),'')
        if state.last_question: allowed.add(state.last_question.kg_id)
        trace={'engine':'contextual-evidence-proposals','candidates':hits,'accepted':[],'rejected':[]}
        context=[{'role':m.sender,'text':m.text[:500]} for m in state.messages[-7:]]
        try:
            raw,meta=self.generator.generate(
                'Analyze the latest USER message in its conversation. Describe their goal and any reference in plain language, not category labels. '
                'For an ambiguous word or request, list its distinct plausible meanings in possible_meanings; otherwise use an empty array. '
                'Candidate labels are ONLY a lookup for record IDs, not the definition of what the person means. '
                'Propose observations when the latest message asserts the speaker currently has, does NOT have, or is unsure about a symptom, or clearly answers an actual symptom question. '
                'An explicit correction such as no longer having a previously recorded symptom must propose status absent. '
                'Keep negative and uncertain symptoms in observations too: absent means the speaker says they do not have it '
                '(including informal spelling such as dont and negated belief such as I do not think I have it). '
                'Uncertain means they cannot tell whether they have it, such as not sure, maybe, or unsure. '
                'Determine each symptom status separately; negation of one symptom does not negate the others. '
                'Questions ABOUT a symptom, examples, another person, past symptoms, confusion and acknowledgments are NOT current symptoms. '
                'Never infer a symptom from a request to explain it. Use only candidate IDs and copy an exact quote from the latest message. '
                'Return one valid JSON object. Fields user_goal, reference_resolution, unresolved are short strings in your own words; use an empty string when none. '
                'Field observations is an array of objects with exactly concept_id (candidate ID), quote (exact user text), and status (present, absent, uncertain). '
                'If no assertion about current presence, absence or uncertainty exists, return an empty array for observations. No Markdown or extra text.',
                json.dumps({'conversation':context,'latest_message':text,
                    'candidates':[{'id':k,'name':nlp.nodes[k].kg_name} for k in sorted(allowed)]},ensure_ascii=False),280)
            trace['model_call']=meta
            try:
                plan=Understanding.model_validate(parse_json(raw))
            except (ValueError,RuntimeError):
                repaired,repair_meta=self.generator.generate(
                    'Repair this output into valid JSON matching the provided schema. Preserve its meaning. Do not add symptom observations. Return JSON only.',
                    json.dumps({'schema':Understanding.model_json_schema(),'output':raw}),320)
                trace['repair_call']=repair_meta
                plan=Understanding.model_validate(parse_json(repaired))
            trace['interpretation']=plan.model_dump(exclude={'observations'})
            trace['proposed']=[p.model_dump() for p in plan.observations]
            state.conversation_memory={'user_goal':plan.user_goal,'unresolved':plan.unresolved}
            accepted=[];seen=set()
            pending=[]
            excluded={e.kg_id for e in baseline.excluded}
            exact={e.kg_id:e for e in baseline.evidence}
            for proposal in plan.observations:
                reason=None
                if proposal.concept_id not in allowed or proposal.concept_id in seen: reason='unknown-or-duplicate-concept'
                elif proposal.concept_id in excluded: reason='non-current-or-other-person'
                elif text.count(proposal.quote)!=1: reason='quote-not-unique-in-current-message'
                # A write requires a personal assertion, not a term inside a question.
                elif self._question_blocks_assertion(text,proposal,nlp): reason='question-not-an-assertion'
                elif proposal.concept_id not in exact:
                    # Implicit replies must refer to a symptom explicitly asked in the last reply.
                    name=nlp.nodes[proposal.concept_id].kg_name
                    if not (name.casefold() in previous.casefold() and '?' in previous):
                        reason='unconfirmed-semantic-link'
                        if text.lower().strip(' .!?') not in {'yes','no','ok','okay','not sure','thanks'}:
                            pending.append(ExtractedSymptom(**nlp.nodes[proposal.concept_id].model_dump(),
                                status=proposal.status,original_text=proposal.quote,turn=state.version+1,source='unconfirmed-agent-proposal'))
                if reason:
                    trace['rejected'].append({'concept_id':proposal.concept_id,'reason':reason});continue
                node=nlp.nodes[proposal.concept_id]
                # An exact vocabulary hit validates identity, not assertion status.
                # The rule baseline defaults unmatched cues to present; applying it
                # here used to overwrite valid LLM negation and uncertainty.
                status=proposal.status
                start=text.index(proposal.quote)
                accepted.append(ExtractedSymptom(**node.model_dump(),status=status,original_text=proposal.quote,
                    start=start,end=start+len(proposal.quote),turn=state.version+1,source='agent-validated-proposal'))
                seen.add(proposal.concept_id)
            trace['accepted']=[e.model_dump() for e in accepted]
            if pending: state.pending_evidence=pending
            elif accepted: state.pending_evidence=[e for e in state.pending_evidence if e.kg_id not in seen]
            trace['validation']='proposal-validation-complete'
            return accepted,trace
        except Exception as exc:
            trace.update(validation='no-write-on-analysis-failure',error_type=type(exc).__name__)
            if hasattr(exc,'errors'):
                trace['schema_errors']=[{'field':list(e['loc']),'type':e['type']} for e in exc.errors()]
            return [],trace

    @staticmethod
    def _question_blocks_assertion(text,proposal,nlp):
        if re.match(r'\s*(?:what|whats|why|how|define|explain|tell me|can you|do i|am i|could i)\b',text,re.I):
            return True
        if '?' not in text: return False
        if proposal.status!='uncertain': return True
        # Accept explicit personal uncertainty despite conversational punctuation,
        # not general questions about a term's meaning. This is a write guard,
        # not a dispatcher for conversational intent.
        normalized=text.replace('’',"'").strip()
        personal=re.match(r"^I(?:'m| am)?\s+(?:not sure|unsure|uncertain|don'?t know)\s+(?:if|whether)\s+I\s+(?:have|feel|am experiencing)\b",normalized,re.I)
        name=nlp.nodes[proposal.concept_id].kg_name
        topic=re.fullmatch(re.escape(name)+r"\s*\?\s*(?:I(?:'m| am)\s+)?(?:not sure|unsure|uncertain)[.!?\s]*",normalized,re.I)
        return not (personal or topic)

    def reply(self,text,state,analysis):
        # Model receives genuine user/assistant turns, not a transcript to classify.
        hits=self.knowledge.search(text,k=2)
        sources=[{'id':h['id'],'text':h['text']} for h in hits if h['cosine']>=0.45 or h['bm25']>=2]
        records=[{'symptom':e.kg_name,'status':e.status} for e in state.evidence]
        facts={'recorded_symptoms':records,'reasoning_mode':state.mode,
            'understanding':analysis.get('interpretation',{}),
            'possible_meanings':analysis.get('interpretation',{}).get('possible_meanings',[]),
            'unconfirmed_meanings':[{'phrase':e.original_text,'possible_symptom':e.kg_name} for e in state.pending_evidence],
            'adapter_suggestion':state.last_question.kg_name if state.last_question else None,
            'evidence_write':analysis.get('validation'),'sources':sources}
        system=(
            'You assist ONLY with symptom descriptions, general health knowledge and this symptom-recording app. '
            'For unrelated requests, politely say you cannot help with that task in this app and offer relevant help. '
            'Talk directly to the person. '
            'Read the conversation and respond to what they mean now, including spelling mistakes, references, corrections and changes of topic. '
            'Do not narrate what the user said or label their intent. Do not offer a generic three-option menu. '
            'When confused, simplify your own last explanation. When they cannot describe a feeling, help with ordinary words and one small question. '
            'A yes to an open question does not tell you a symptom: explain gently and help them give a specific detail without repeating yourself. '
            'If a health term has two plausible meanings, ask which meaning. Answer knowledge questions without recording symptoms. '
            'When APP FACTS lists distinct possible meanings, mention those alternatives briefly and clarify before selecting one. '
            'You may explain general health terms, but do not diagnose or prescribe. Briefly redirect unrelated tasks to what this demo can help with. '
            'Write a short conversational reply of at most 65 words and ONE question. No bullet lists. '
            'When a person acknowledges an open question without supplying the detail, demonstrate a possible answer and STOP without asking that question again. '
            'Repeated short acknowledgments mean your current approach is not helping: give a fill-in sentence instead of another request to describe symptoms. '
            'Example of this conversational repair: You can write "My ___ feels ___ since ___." Leaving blanks is okay. '
            'When they ask a definition or clarify the meaning of a word, explain the word and stop. Do not turn a vocabulary question into a symptom interview. '
            'Facts below are authoritative for what was saved. Do not claim a symptom was recorded unless it appears there. '
            'If the user reported a sensation with an unconfirmed possible meaning, ask them to confirm that meaning before treating it as a symptom. '
            'An adapter suggestion is optional context, NOT a question already asked. In random mode it is a random test fixture, not clinical reasoning. '
            'App controls: in Symptom log, records are grouped as Present/Absent/Not sure; each has a Change status dropdown. '
            'Suggested matches need confirmation: use Present, Absent, Not sure, or Ignore suggestion. Ignoring a match does not mean absent. '
            'To correct a record use that dropdown or tell me the correction in chat. There is no Edit button or pencil icon. '
            'Other controls: Export session, Delete & start over. An example is hypothetical and has not been recorded. '
            'Use supplied sources when relevant; otherwise general model knowledge has no verified citation. Never invent a source.\n'
            'APP FACTS: '+json.dumps(facts,ensure_ascii=False)+
            '\nBefore replying, check that your answer follows the app scope, answers the latest request in context, '
            'does not invent saved records or controls, and does not repeat an unanswered request. Output only your short final reply.')
        history=[{'role':'assistant' if m.sender=='assistant' else 'user','content':html.unescape(m.text)} for m in state.messages[-16:]]
        if history and history[0]['role']=='assistant': history=history[1:]
        trace={'engine':'conversation-agent-v1','response_mode':'multi-turn-llm','understanding':analysis,
            'context_policy':'recent-16-messages-plus-validated-records','source_context':sources,'citations':[]}
        previous=next((m.text for m in reversed(state.messages[:-1]) if m.sender=='assistant'),'')
        # Keep the full reviewed pipeline as an explicit comparison / rollback mode.
        # Direct mode retains contextual understanding and all evidence validators.
        pipeline=os.getenv('AGENT_REPLY_PIPELINE','direct')
        trace['reply_pipeline']=pipeline
        try:
            for attempt in range(2):
                instruction=system if not attempt else system+'\nYour previous draft repeated or narrated the conversation. Give a direct, useful answer to the latest user instead.'
                answer=''
                if pipeline!='direct':
                    answer,meta=self.generator.chat([{'role':'system','content':instruction}]+history,240)
                    trace.setdefault('model_calls',[]).append(meta)
                review_system=(
                    'You edit a draft reply for a health vocabulary and symptom-recording assistant. '
                    'Write the improved reply to the LAST user message, using the conversation. Output only the reply, under 65 words. '
                    'Fix these errors if present: speaking as the patient; invented symptoms or saved records; repeating a question the user cannot answer; '
                    'answering a different topic; giving unrelated travel, coding or other non-health assistance. '
                    'Clearly label hypothetical examples as examples. When the user asks what a word means, explain it without assuming they have it. '
                    'When a health term has multiple plausible meanings, distinguish them briefly before assuming one. If they acknowledge an open question, offer a sample sentence instead of asking again. '
                    'For unrelated tasks say this app helps with symptoms and health terms. For app changes use the recorded symptom status dropdown or a correction in chat. '
                    'No diagnosis or prescribing. The supplied draft is fallible; the actual conversation and saved records take precedence. '
                    'Answer latest_user_message; older messages only supply context. Use supplied source passages when relevant. '
                    'If understanding.possible_meanings contains different meanings, the reply should briefly distinguish them and ask which is intended. '
                    'SAVED RECORDS: '+json.dumps(records))
                if pipeline=='direct':
                    review_system=review_system.replace('You edit a draft reply','You write the final reply').replace('Write the improved reply','Write a helpful reply')
                    review_system += (' No draft is needed. Reply as the assistant, never as the patient. '
                        'Do not invent first-person symptom descriptions unless explicitly marked as hypothetical examples. '
                        'When the user is confused, simplify the previous answer with a short concrete example. '
                        'Repeated acknowledgments to an open question need a fill-in example, not another request for the same detail. '
                        'Discuss app controls ONLY when asked about the app or a record change. '
                        'When saved_records is empty there is no status dropdown yet. Do not suggest selecting Not sure to record a vague feeling. '
                        'Help describe feelings with ordinary words instead. Knowledge definitions should not end with recording instructions. '
                        'If asked: Symptom log groups saved records as Present, Absent, Not sure, each with a Change status dropdown. '
                        'Only existing unconfirmed suggestions can be confirmed or ignored; ignoring does not mean absent.')
                # Reviewing is an editing task, not another continuation of the user's turn.
                # A native "yes" as the last review message otherwise re-triggers the same draft.
                review_input=json.dumps({'task':('Write a helpful final reply to the latest user. Output ONLY that reply.' if pipeline=='direct' else 'Rewrite the draft as a helpful reply to the latest user. Output ONLY the replacement reply.'),
                    'latest_user_message':text,'understanding':analysis.get('interpretation',{}),
                    'previous_assistant_message':previous,'conversation':history,'saved_records':records,
                    'pending_matches':facts['unconfirmed_meanings'],'reasoning_mode':state.mode,
                    'evidence_validation':analysis.get('validation'),
                    'sources':sources,'draft':answer,
                    'revision_feedback':('The previous attempt repeated an earlier reply or narrated instead of helping. '
                        'Write a different concrete response. If the person is stuck, demonstrate one example or fill-in sentence, '
                        'rather than requesting the same detail again.') if attempt else ''},ensure_ascii=False)
                answer,review_meta=self.generator.chat([{'role':'system','content':review_system},
                    {'role':'user','content':review_input}],180)
                trace.setdefault('model_calls' if pipeline=='direct' else 'review_calls',[]).append(review_meta)
                answer=answer.strip()
                invalid=(not answer or len(answer)>1800 or answer==previous or
                    re.search(r'\bthe user (?:feels|wants|asks)|\bthe question.{0,90}(?:was|has been) (?:asked|answered)|\[insert',answer,re.I) or
                    re.match(r'^(?:my (?:chest|head|body) feels|i (?:feel|have been feeling|am experiencing))\b',answer,re.I))
                if not invalid:
                    trace['model_call']=review_meta
                    return answer,trace
                trace['review_reason']='output-validation'
            raise ValueError('Unhelpful model output after retry')
        except Exception as exc:
            trace.update(response_mode='agent-generation-failed',error_type=type(exc).__name__)
            return "I couldn't produce a useful reply just now. Please try rephrasing your last message.",trace

    def phrase_question(self,state,fallback):
        return fallback,{'response_mode':'explicit-adapter-question'}


