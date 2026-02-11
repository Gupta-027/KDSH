import re
from typing import List

_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")

def extract_claims(backstory: str, caption: str | None = None, max_claims: int = 10) -> List[str]:
    """
    Turn backstory into short 'claims' to improve retrieval + verification.
    Heuristic: split into sentences, clean, keep informative ones.
    """
    text = (caption.strip() + ". " if caption and str(caption).strip() else "") + (backstory or "").strip()
    text = re.sub(r"\s+", " ", text)

    sents = [s.strip() for s in _SENT_SPLIT.split(text) if s.strip()]
    claims = []
    for s in sents:
        if len(s) < 25:
            continue
        # drop super generic sentences
        if s.lower().startswith(("in conclusion", "overall", "this shows", "therefore")):
            continue
        claims.append(s)

    # de-dup (simple)
    seen = set()
    uniq = []
    for c in claims:
        key = c.lower()
        if key not in seen:
            seen.add(key)
            uniq.append(c)

    return uniq[:max_claims]
