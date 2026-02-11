from __future__ import annotations
from typing import Dict, Tuple
import re

from transformers import pipeline
from config import NLI_MODEL, DEVICE

_label_norm = re.compile(r"[^a-z]+")

def _norm_label(label: str) -> str:
    return _label_norm.sub("", label.lower())

class NLIVerifier:
    """
    Score (evidence -> claim) as contradiction / entailment / neutral.
    """
    def __init__(self):
        self.pipe = pipeline(
            task="text-classification",
            model=NLI_MODEL,
            device=DEVICE,
            return_all_scores=True,
        )

    def score(self, evidence: str, claim: str) -> Dict[str, float]:
        # NLI expects premise-hypothesis; many models accept a pair as dict
        # transformers pipeline supports {"text": premise, "text_pair": hypothesis}
        out = self.pipe({"text": evidence, "text_pair": claim})[0]

        scores = {"entailment": 0.0, "neutral": 0.0, "contradiction": 0.0}

        for item in out:
            lab = _norm_label(item["label"])
            val = float(item["score"])
            if "contrad" in lab or lab.endswith("0") and "contradiction" in NLI_MODEL.lower():
                scores["contradiction"] = max(scores["contradiction"], val)
            elif "entail" in lab or lab.endswith("2") and "mnli" in NLI_MODEL.lower():
                scores["entailment"] = max(scores["entailment"], val)
            elif "neutral" in lab or lab.endswith("1"):
                scores["neutral"] = max(scores["neutral"], val)

        # If labels are LABEL_0/LABEL_1/LABEL_2, map by relative ordering:
        # (Fallback) If all zeros, do a simple heuristic:
        if scores["entailment"] == 0 and scores["neutral"] == 0 and scores["contradiction"] == 0:
            # try to infer from max label
            best = max(out, key=lambda x: x["score"])
            lab = _norm_label(best["label"])
            if lab.endswith("0"):
                scores["contradiction"] = best["score"]
            elif lab.endswith("1"):
                scores["neutral"] = best["score"]
            else:
                scores["entailment"] = best["score"]

        return scores

    def best_contradiction(self, evidence_chunks: list[str], claim: str) -> Tuple[float, str]:
        best = 0.0
        best_txt = ""
        for ev in evidence_chunks:
            s = self.score(ev, claim)
            if s["contradiction"] > best:
                best = s["contradiction"]
                best_txt = ev
        return best, best_txt
