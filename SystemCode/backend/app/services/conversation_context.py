"""Normalize informal text and identify explicit conversation-repair cues."""
import html
import re

def normalize(text):
    text=html.unescape(text).replace('\u2019',"'").strip()
    text=re.sub(r"\bwhats\b|\bwhat's\b",'what is',text,flags=re.I)
    text=re.sub(r"\bdont\b","don't",text,flags=re.I)
    text=re.sub(r'\bwaht\b','what',text,flags=re.I)
    return re.sub(r'\s+',' ',text)

def conversation_act(text,state):
    text=normalize(text).lower().strip(' .!?')
    prompt=state.dialogue_prompt
    last=next((m.text.lower() for m in reversed(state.messages) if m.sender=='assistant'),'')
    # Older stored sessions predate the explicit pending-dialogue field.
    if prompt=='none' and 'symptoms' in last and ('health term' in last or 'health knowledge' in last) and ('app' in last):
        prompt='menu'
    if prompt=='menu' and text in {'1','2','3'}:
        return {'1':'guide','2':'ask_term','3':'help'}[text]
    if text in {'yes','yeah','yep','ok','okay','no','nope','not sure'}:
        if prompt!='none' or state.last_question is None:
            return 'clarification_stop' if state.clarification_attempts>=1 else 'clarification_retry'
        return 'report'
    if re.search(r"\bi (?:don.t|do not|can.t|cannot) (?:get|understand|follow)\b",text):
        return 'repair' if last else 'guide'
    if re.fullmatch(r'what|huh|pardon|sorry|what do you mean|i (?:do not|don.t) understand',text):
        return 'repair' if any(m.sender=='assistant' for m in state.messages) else 'guide'
    if (re.search(r'\bhow (?:to|do i|can i|should i) (?:describe|explain|say)\b',text)
        or re.search(r'\b(?:don.t know how|cannot|can.t|struggl\w* to) (?:to )?(?:say|describe|explain|express)\b',text)):
        return 'guide'
    if re.fullmatch(r'(?:what is|define|explain|tell me about) (?:a |the )?cold',text):
        return 'clarify_term'
    return None

def recent_exchange(state):
    # Current user message has already been appended by ChatbotService.
    return [{'role':m.sender,'text':m.text[:700]} for m in state.messages[-5:-1]]

def is_meta_narration(answer):
    return bool(re.search(r'\bthe user (?:feels|asks|wants|is|has|struggles)|\bthe question .{0,100}(?:has been|was) (?:asked|answered)|\[insert',answer,re.I))
