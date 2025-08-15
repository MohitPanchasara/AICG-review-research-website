# backend/intuition.py
import re
from typing import List, Dict, Tuple
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

# --- text utils ---
STOPWORDS = {
    "the","a","an","and","or","but","if","while","with","of","to","in","on","for","at","by","from","as",
    "is","am","are","was","were","be","been","being","this","that","these","those","it","its","into",
    "we","you","he","she","they","them","his","her","their","our","us","my","your"
}
TOKEN_RE = re.compile(r"[a-z]{3,}")

def _content_tokens(text: str):
    toks = TOKEN_RE.findall((text or "").lower())
    return [w for w in toks if w not in STOPWORDS]

class IntuitionScorer:
    """
    Computes per-segment 'randomness' from captions with:
      - TF-IDF cosine (max over last N)
      - Jaccard on content words (max over last N)
      randomness = 1 - (w_tfidf * cos + w_jaccard * jaccard)
    Returns (per_segment_list, final_score_in_0_100)
    """
    def __init__(self, history=1, w_tfidf=0.7, w_jaccard=0.3, flag_threshold=0.60):
        self.history = int(history)
        self.w_tfidf = float(w_tfidf)
        self.w_jaccard = float(w_jaccard)
        self.flag_threshold = float(flag_threshold)

    def compute(self, items: List[List]) -> Tuple[List[Dict], float]:
        # items: [[start, end, text], ...]
        segments = [{"start": float(s), "end": float(e), "text": str(t or "").strip()} for s, e, t in items]
        n = len(segments)
        if n == 0:
            return [], 0.0

        texts = [s["text"] for s in segments]

        vec = TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            ngram_range=(1, 2),
            sublinear_tf=True,
            norm="l2",  # row L2-normalized => cosine = dot
        )
        X = vec.fit_transform(texts)  # csr_matrix (n, d)
        toks = [_content_tokens(t) for t in texts]

        def cos(i: int, j: int) -> float:
            # robust sparse cosine (since rows are L2-normalized, dot == cosine)
            return float(X[i].multiply(X[j]).sum())

        def jac(i: int, j: int) -> float:
            a, b = set(toks[i]), set(toks[j])
            if not a and not b:
                return 1.0
            if not a or not b:
                return 0.0
            return len(a & b) / max(1, len(a | b))

        per = [{
            "idx": 0,
            "start": segments[0]["start"],
            "end": segments[0]["end"],
            "text": segments[0]["text"],
            "cosine": None,
            "jaccard": None,
            "similarity": None,
            "randomness": 0.0,
            "flag": False,
        }]

        for i in range(1, n):
            sims_cos, sims_jac = [], []
            for h in range(1, min(self.history, i) + 1):
                j = i - h
                sims_cos.append(cos(i, j))
                sims_jac.append(jac(i, j))

            cos_sim = max(sims_cos) if sims_cos else 0.0
            jac_sim = max(sims_jac) if sims_jac else 0.0

            combined = self.w_tfidf * cos_sim + self.w_jaccard * jac_sim
            combined = max(0.0, min(1.0, combined))
            randomness = 1.0 - combined
            flag = randomness >= self.flag_threshold

            per.append({
                "idx": i,
                "start": segments[i]["start"],
                "end": segments[i]["end"],
                "text": segments[i]["text"],
                "cosine": round(cos_sim, 4),
                "jaccard": round(jac_sim, 4),
                "similarity": round(combined, 4),
                "randomness": round(randomness, 4),
                "flag": flag,
            })

        final = float(np.mean([p["randomness"] for p in per]) * 100.0)
        return per, final
