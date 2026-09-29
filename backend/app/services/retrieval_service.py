"""BM25 plus dense embeddings and reciprocal rank fusion.
Retrieves concept descriptions/educational passages, not disease candidates.
"""
import hashlib,json,math,re
from collections import Counter
from threading import Lock

STOPWORDS=set('a an the is are was were do does did what how why can could would should i you my me it its this that these those of to for from in on at and or with have has be been am about'.split())
def tokens(text): return [w for w in re.findall(r"[a-z0-9]+",text.lower()) if w not in STOPWORDS]

class HybridIndex:
    def __init__(self,documents,encoder=None):
        self.documents=documents; self.encoder=encoder
        self.rows=[tokens(d["title"]+" "+d["text"]) for d in documents]
        self.counts=[Counter(row) for row in self.rows]
        self.avg=sum(map(len,self.rows))/max(1,len(self.rows))
        self.df=Counter(w for row in self.rows for w in set(row))
        self.vectors=None; self.lock=Lock()
        self.fingerprint=hashlib.sha256(json.dumps(documents,sort_keys=True).encode()).hexdigest()[:12]
        if encoder is not None:
            self.vectors=encoder.encode([d["title"]+". "+d["text"] for d in documents],normalize_embeddings=True)

    def search(self,query,k=3,mode="hybrid"):
        words=tokens(query); n=len(self.rows)
        if not words or not n: return []
        lexical=[]
        for row,counts in zip(self.rows,self.counts):
            score=0.
            for word in words:
                freq=counts[word]; idf=math.log(1+(n-self.df[word]+.5)/(self.df[word]+.5))
                score+=idf*freq*2.5/(freq+1.5*(.25+.75*len(row)/max(self.avg,1)))
            lexical.append(score)
        dense=[0.]*n
        if self.encoder is not None and mode!="bm25":
            import numpy as np
            with self.lock: vector=self.encoder.encode([query],normalize_embeddings=True)[0]
            dense=np.dot(self.vectors,vector).tolist()
        rankings=[]
        if mode not in {"bm25","dense","hybrid"}: raise ValueError("Unknown retrieval mode")
        if mode!="dense": rankings.append(sorted((i for i in range(n) if lexical[i]>0),key=lambda i:lexical[i],reverse=True))
        if mode!="bm25" and self.encoder is not None: rankings.append(sorted((i for i in range(n) if dense[i]>=.30),key=lambda i:dense[i],reverse=True))
        votes=[{i:1/(60+position) for position,i in enumerate(r,1)} for r in rankings]
        fused={i:sum(v.get(i,0) for v in votes) for i in range(n)}
        ranked=sorted(range(n),key=lambda i:fused[i],reverse=True)
        return [{**self.documents[i],"bm25":round(lexical[i],4),"cosine":round(dense[i],4),
            "rrf":round(fused[i],6),"index_version":self.fingerprint}
            for i in ranked if fused[i]>0][:k]

def load_encoder(name="sentence-transformers/all-MiniLM-L6-v2"):
    import os
    from pathlib import Path
    from sentence_transformers import SentenceTransformer
    bundled=Path(__file__).resolve().parents[3]/"models"/"all-MiniLM-L6-v2"
    name=os.getenv("EMBEDDING_MODEL_PATH",str(bundled) if bundled.is_dir() else name)
    return SentenceTransformer(name,local_files_only=True)

