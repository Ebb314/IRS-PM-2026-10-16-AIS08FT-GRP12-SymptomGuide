"""Seeded random test doubles. They do not implement disease inference."""
import hashlib,json,random
from app.models.chat_models import DiseaseCandidate,NextSymptom
from app.services.interfaces import RecallService,InformedSearchService

def rng(seed,values):
    digest=hashlib.sha256(json.dumps(values,sort_keys=True).encode()).hexdigest()
    return random.Random(str(seed)+digest)

class RandomRecallService(RecallService):
    def __init__(self,seed=20260915,pool_size=50):
        self.seed=seed
        self.pool=[(f'SIM:D{i:04d}',f'Simulated disease {i:04d}') for i in range(1,pool_size+1)]
    def recall(self,positive,negative):
        known=sorted(s.kg_id for s in positive+negative)
        if not known:return []
        randomizer=rng(self.seed,known)
        sampled=randomizer.sample(self.pool,randomizer.randint(3,6))
        return sorted([DiseaseCandidate(id=k,name=n,score=round(randomizer.random(),4),evidence_ids=known,
            source='seeded-random-fixture-v1') for k,n in sampled],key=lambda c:c.score,reverse=True)

class RandomSearchService(InformedSearchService):
    def __init__(self,nodes,seed=20260915):
        self.nodes=nodes;self.seed=seed
    def select_next(self,candidates,positive,negative,asked_symptom_ids):
        excluded={s.kg_id for s in positive+negative}|set(asked_symptom_ids)
        available=sorted(set(self.nodes)-excluded)
        if not available or not candidates:return None
        randomizer=rng(self.seed,{'excluded':sorted(excluded),'candidates':[c.id for c in candidates]})
        node=self.nodes[randomizer.choice(available)]
        return NextSymptom(**node.model_dump(),heuristic_score=round(randomizer.random(),4),
            rationale='Random test question from the full symptom catalog; no clinical ranking.')
