"""Conservative exact-text slot projection for a single symptom in a message.
Multi-symptom attachment remains the model's job to avoid assigning slots across entities.
"""
import re
DURATION=re.compile(r'\bsince (?:yesterday|this morning|last night|Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)\b|\b(?:a|an|one|two|three|four|five|six|seven|\d+) (?:hours?|days?|weeks?|months?)\b|\blast night\b',re.I)
SEVERITY=re.compile(r'\b(?:mild|moderate|severe|slight)\b',re.I)
def project_slots(text,evidence):
    if len(evidence)!=1 or evidence[0].status!='present': return
    e=evidence[0]
    for field,pattern in [('duration_text',DURATION),('severity_text',SEVERITY)]:
        matches=list(pattern.finditer(text))
        if len(matches)==1:
            setattr(e,field,matches[0].group())
