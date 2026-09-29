"""Intent-aware, source-grounded dialogue. Does not replace teammate reasoning."""
import json,re
from pathlib import Path
from pydantic import Field
from app.models.chat_models import Model
from app.services.model_runtime import parse_json
from app.services.retrieval_service import HybridIndex

class Draft(Model):
    answer:str=Field(min_length=1,max_length=1200)
    citations:list[str]=Field(max_length=5)

class DialogueAI:
    def __init__(self,generator=None,encoder=None,knowledge=None,vocabulary=None,mention_finder=None):
        self.generator=generator
        self.vocabulary=vocabulary or {}
        self.mention_finder=mention_finder
        documents=knowledge or json.loads((Path(__file__).parents[1]/"data/knowledge.json").read_text(encoding="utf-8-sig"))
        self.index=HybridIndex(documents,encoder)
        self.documents=documents
        from app.services.intent_service import IntentService
        self.intent_service=IntentService(generator,encoder) if generator else None

    def knowledge_target(self,text,state):
        from app.services.conversation_context import normalize
        text=normalize(text)
        generic=bool(re.search(r'\b(?:latest|last|current) symptom\b|\b(?:it|that|this symptom)\b|question a knowledge',text,re.I))
        if self.mention_finder:
            matches=self.mention_finder(text)
            if matches and len(matches[-1][2])==1:
                return self.vocabulary[matches[-1][2][0]],False
        return (state.knowledge_focus,True) if generic else (None,False)

    def route(self,text):
        lower=text.lower().strip()
        if re.search(r"\b(ignore (?:all |previous )?instructions|system prompt|reveal.*prompt)\b",lower): return "out_of_scope"
        if re.search(r"\b(summar(?:y|ize|ise)|what (?:have i|did i) (?:said|told|reported)|what do you know about me)\b",lower): return "summary"
        if re.search(r"\b(why.*(?:ask\w*|question)|why this|explain.*(?:question|result|score))\b",lower): return "explain"
        if re.search(r"^(what (?:is|are|does)|how (?:does|do|can)|can you explain|tell me about|define|why (?:is|are|does)|explain (?:it|that|this|my latest symptom)|question a knowledge)\b",lower): return "knowledge"
        return "report"

    def classify(self,text,state):
        if self.generator is None:
            return self.route(text),{'mode':'offline-rules'}
        from app.services.intent_service import IntentService
        from app.services.conversation_context import normalize,conversation_act
        normalized=normalize(text)
        intent,trace=self.intent_service.classify(normalized,state)
        act=conversation_act(normalized,state)
        if act:
            trace.update(conversation_act=act,selected_intent=act,selection_policy='explicit-conversation-cue')
            intent=act
        return intent,trace

    def general_knowledge(self,question,target,topic_trace):
        trace={'intent':'knowledge','response_mode':'llm-unavailable','citations':[],
            'answer_basis':'model-general-knowledge','prompt_version':'general-knowledge-v1',**topic_trace}
        if self.generator is None:
            return 'The local language model is not enabled. Enable the local AI profile to explain '+target.kg_name+'.',trace
        try:
            raw,meta=self.generator.generate(
                'Explain medical terms for general education in plain English. Answer the question about the supplied TERM in 2 to 4 concise sentences. '
                'Start with what the term means and describe how it feels in everyday language. '
                'Only discuss causes if the question explicitly asks about causes. Do not infer that the user has a condition. '
                'Do not prescribe medication, doses or treatment. If the term is not a symptom or you do not know, say so. '
                'No documents were retrieved: do not invent sources, citations, links, or claim verification. '
                'Return only the explanation as plain text, including the supplied term. No headings or formatting.',
                json.dumps({'TERM':target.kg_name,'QUESTION':question}),260)
            trace['model_call']=meta
            # Free-form education does not require the NLU extractor's JSON contract.
            draft=(Draft.model_validate(parse_json(raw)) if raw.lstrip().startswith('{')
                   else Draft(answer=raw.strip(),citations=[]))
            if draft.citations or re.search(r'https?://|www\.|\[\d+\]',draft.answer,re.I):
                raise ValueError('Invented source')
            if target.kg_name.casefold() not in draft.answer.casefold():
                raise ValueError('Topic not preserved')
            if re.search(r'\b(?:you (?:have|suffer from)|take \d|dosage|\d+\s*mg|diagnosed with)\b',draft.answer,re.I):
                raise ValueError('Individual diagnosis or medication statement')
            trace.update(response_mode='local-llm-general',validation='format-and-scope-checked')
            return draft.answer,trace
        except Exception as exc:
            trace.update(response_mode='llm-generation-failed',fallback_reason=type(exc).__name__)
            return 'I could not generate a reliable explanation of '+target.kg_name+' this time. Please try asking again.',trace

    def conversational(self,text,state,intent):
        from app.services.conversation_context import normalize
        text=normalize(text)
        if self.generator and intent in {'guide','repair','clarify_term','clarification_retry','clarification_stop','ask_term'}:
            return self.intent_service.support(text,state,intent)
        if self.generator and intent!='knowledge':
            from app.services.intent_service import IntentService
            return self.intent_service.respond(text,state,intent)
        context=[]
        if intent=="summary":
            summary="; ".join(f"{e.kg_name}: {e.status}"+(f" ({e.duration_text})" if e.duration_text else "") for e in state.evidence)
            return "Your recorded evidence: "+(summary or "none yet")+". You can correct any item in the evidence panel.",{
                "intent":intent,"response_mode":"structured-summary","citations":[],"validation":"rendered-from-state"}
        if intent=="out_of_scope":
            return "I can explain this prototype and its recorded evidence, but cannot reveal system instructions or provide a diagnosis.",{
                "intent":intent,"response_mode":"scope-boundary","citations":[]}
        query=text
        target,generic=self.knowledge_target(text,state) if intent=='knowledge' else (None,False)
        if self.generator and intent=='knowledge' and target is None and self.mention_finder:
            from app.services.intent_service import IntentService
            return self.intent_service.respond(text,state,'clarify' if generic else 'knowledge')
        if generic and target is None:
            return 'Please name the symptom you want explained. Your latest description did not identify a supported symptom.',{
                'intent':intent,'response_mode':'topic-clarification','citations':[]}
        topic_trace={}
        if target:
            query='What is '+target.kg_name+'?' if generic else text
            topic_trace={'topic':target.model_dump(),'topic_resolution':'latest-reported' if generic else 'explicit-name'}
        if intent=="explain": query+=" why follow-up question mock scores"
        hits=self.index.search(query,k=3)
        if target:
            # Exact source-topic binding prevents a related A passage answering B.
            key=target.kg_name.casefold().replace(' ','_')
            supported=[d for d in self.documents if any(c.split(':')[-1].casefold()==key for c in d.get('concepts',[]))]
            if not supported:
                return self.general_knowledge(query,target,topic_trace)
            supported_ids={d['id'] for d in supported}
            hits=[h for h in hits if h['id'] in supported_ids] or supported
        if not hits:
            return "I do not have a supporting passage for that question in this demonstration knowledge base.",{
                "intent":intent,"response_mode":"abstention","citations":[]}
        if intent=="explain" and state.last_question:
            context.append("Pending question: "+state.last_question.kg_name+". Rationale: "+state.last_question.rationale)
        trace={"intent":intent,"response_mode":"retrieved-extract","retrieval":hits,"citations":hits[:1],
            "prompt_version":"grounded-dialogue-v3","validation":"source-extract",**topic_trace}
        fallback=hits[0]["text"]
        if self.generator is None: return fallback,trace
        system=("You explain an academic symptom chatbot. Answer only from SOURCES. "
            "Never diagnose, prescribe or claim mock scores are probabilities. "
            "If the sources do not answer the question say that you do not know. "
            "Return JSON only with keys answer (one short paragraph) and citations (source IDs). "
            'Use at least one source ID. No additional keys. '
            'Example format: {"answer":"The prototype stores confirmed evidence.","citations":["project-scope"]}. '
            'Use the supplied sources instead of the example when answering.')
        try:
            raw,meta=self.generator.generate(system,json.dumps({"QUESTION":query if target else text,"CONTEXT":context,
                "SOURCES":[{k:h[k] for k in ["id","title","text"]} for h in hits]})+'\nOutput JSON:',220)
            trace["model_call"]=meta
            draft=Draft.model_validate(parse_json(raw))
            allowed={h["id"] for h in hits}
            if not draft.citations or not set(draft.citations).issubset(allowed): raise ValueError("Unbacked citation")
            if re.search(r"\b(you (?:have|suffer from)|take \d|dosage|mg\b|diagnosed with)\b",draft.answer,re.I):
                raise ValueError("Individual diagnosis or medication statement")
            if re.search(r"\b(?:for|towards|toward) (?:further )?(?:diagnosis|diagnostic|assessment)\b",draft.answer,re.I):
                raise ValueError("Unsupported clinical purpose")
            # Numbers may not be invented; citation validity alone is not semantic entailment.
            source_numbers=set(re.findall(r"\d+", " ".join(h["text"] for h in hits if h['id'] in draft.citations)))
            if not set(re.findall(r"\d+",draft.answer)).issubset(source_numbers): raise ValueError("Unbacked number")
            trace["response_mode"]="local-llm-grounded"; trace["validation"]="schema-and-citations-checked"
            trace["citations"]=[h for h in hits if h["id"] in draft.citations]
            return draft.answer,trace
        except Exception as exc:
            trace["response_mode"]="retrieved-extract-fallback"; trace["fallback_reason"]=type(exc).__name__
            return fallback,trace

    def phrase_question(self,state,fallback):
        # Algorithm chooses the concept; generator only rephrases the approved question.
        if not self.generator or not state.last_question: return fallback,{}
        name=state.last_question.kg_name
        trace={"prompt_version":"question-v2","response_mode":"template"}
        try:
            raw,meta=self.generator.generate(
                "Rewrite the supplied question in friendly English. Keep the symptom phrase exactly. "
                "Ask only one question. Do not add symptoms, causes or advice. Return the question only.",
                "Are you experiencing "+name+"?",55)
            trace["model_call"]=meta
            if name.lower() not in raw.lower() or raw.count("?")!=1 or len(raw)>220 or "\n" in raw.strip():
                raise ValueError("Question changed its target")
            # Restrict generation to a single question ending at the question mark.
            if not raw.strip().endswith("?"): raise ValueError("Trailing advice")
            # Only allow neutral wrappers around the exact algorithm-selected concept.
            # This deliberately rejects creative wording that changes assertion meaning.
            wrapper=r"(?:(?:are you|have you been) (?:experiencing|having|feeling)|do you (?:have|experience)|have you (?:noticed|experienced))(?: any)? "
            if not re.fullmatch(wrapper+re.escape(name)+r"(?: (?:currently|right now|today|recently))?\?",raw.strip(),re.I):
                raise ValueError("Unsupported question wording")
            trace["response_mode"]="local-llm-question"
            return raw.strip()+" You can answer Yes, No or Not sure.",trace
        except Exception as exc:
            trace["fallback_reason"]=type(exc).__name__
            return fallback,trace


