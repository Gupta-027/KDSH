from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Dict, Any

from claim_extractor import extract_claims
from rag_client import retrieve_chunks
from nli_verifier import NLIVerifier


@dataclass
class Prediction:
    label: str                    # "consistent" | "contradict"
    confidence: float
    max_contradiction: float
    key_evidence: Optional[str]


class ConsistencyPredictor:
    def __init__(
        self,
        k_per_claim: int = 8,
        max_claims: int = 10,
        contradiction_threshold: float = 0.65,
    ):
        self.k_per_claim = k_per_claim
        self.max_claims = max_claims
        self.contradiction_threshold = contradiction_threshold
        self.nli = NLIVerifier()

    def predict(
        self,
        book_name: str,
        char_name: str,
        backstory: str,
        caption: Optional[str] = None,
    ) -> Prediction:
        claims = extract_claims(backstory, caption=caption, max_claims=self.max_claims)
        if not claims:
            # conservative
            return Prediction(label="contradict", confidence=0.51, max_contradiction=0.0, key_evidence=None)

        global_best_contra = 0.0
        global_best_evidence = None

        for claim in claims:
            query = f"Character: {char_name}\nClaim: {claim}"
            chunks = retrieve_chunks(query=query, book_name=book_name, k=self.k_per_claim)
            if not chunks:
                continue

            best_contra, best_ev = self.nli.best_contradiction(chunks, claim)
            if best_contra > global_best_contra:
                global_best_contra = best_contra
                global_best_evidence = best_ev

        # Decision rule:
        if global_best_contra >= self.contradiction_threshold:
            # confidence grows with contradiction score
            conf = min(0.99, 0.50 + 0.50 * global_best_contra)
            return Prediction("contradict", conf, global_best_contra, global_best_evidence)

        # If no strong contradiction found, mark consistent
        conf = min(0.95, 0.55 + 0.40 * (1.0 - global_best_contra))
        return Prediction("consistent", conf, global_best_contra, global_best_evidence)
