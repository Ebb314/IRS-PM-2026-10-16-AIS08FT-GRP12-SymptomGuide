"""Strict positive/negative/unknown encoding, independent of training-label policy."""
import numpy as np


ALIASES = {'发热':'fever','发烧':'fever','咳嗽':'cough','呼吸困难':'shortness of breath'}
STATES = {'positive':'positive','negative':'negative','unknown':'unknown',
          '阳性':'positive','阴性':'negative','未知':'unknown'}


def encode_evidence(vocabulary, present_symptoms=(), absent_symptoms=(),
                    unknown_symptoms=(), symptom_states=None):
    """Return [positive bits, negative bits]; missing evidence is unknown.

    A valid representation is not a claim that suitable negative training labels
    exist. Callers must separately establish the model's training-data policy.
    """
    lookup = {name.casefold():i for i,name in enumerate(vocabulary)}
    if len(lookup)!=len(vocabulary):
        raise ValueError('Vocabulary contains duplicate names after case folding')
    for items in [present_symptoms,absent_symptoms,unknown_symptoms]:
        if not isinstance(items,(list,tuple,set)) or any(not isinstance(s,str) for s in items):
            raise ValueError('Symptom lists must contain names, not free text or null')
    if symptom_states is not None:
        if not isinstance(symptom_states,dict):
            raise ValueError('symptom_states must be a name-to-state object')
        if any([present_symptoms,absent_symptoms,unknown_symptoms]):
            raise ValueError('Use symptom_states OR the three lists, not both')
        entries = []
        for name,state in symptom_states.items():
            if not isinstance(name,str) or not isinstance(state,str) or state.casefold() not in STATES:
                raise ValueError('States must be positive, negative, or unknown')
            entries.append((name,STATES[state.casefold()]))
    else:
        entries = [(name,state) for items,state in [(present_symptoms,'positive'),
                   (absent_symptoms,'negative'),(unknown_symptoms,'unknown')] for name in items]
    resolved = {}
    for name,state in entries:
        key = name.strip().casefold()
        key = ALIASES.get(key,key)
        if key not in lookup:
            raise ValueError(f'Unsupported symptom: {name}')
        idx = lookup[key]
        if idx in resolved and resolved[idx]!=state:
            raise ValueError(f'Contradictory symptom states: {vocabulary[idx]}')
        resolved[idx] = state
    vector = np.zeros((1,len(vocabulary)*2),dtype=np.float32)
    for idx,state in resolved.items():
        if state=='positive':
            vector[0,idx] = 1
        elif state=='negative':
            vector[0,len(vocabulary)+idx] = 1
    explicit = {vocabulary[idx]:state for idx,state in sorted(resolved.items())}
    return vector, {'explicit_states':explicit,
        'positive_symptoms':[name for name,state in explicit.items() if state=='positive'],
        'negative_symptoms':[name for name,state in explicit.items() if state=='negative'],
        'explicit_unknown_symptoms':[name for name,state in explicit.items() if state=='unknown'],
        'unknown_count':len(vocabulary)-sum(state!='unknown' for state in resolved.values())}
