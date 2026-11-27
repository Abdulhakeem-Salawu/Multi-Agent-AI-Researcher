from __future__ import annotations

import re
from collections import Counter

from .retrieval import tokenize
from .schemas import ClaimAssessment

_CITATION_RE = re.compile(r"\[(S\d+)\]")


def extract_candidate_claims(answer: str) -> list[str]:
    claims: list[str] = []
    for raw_line in answer.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        line = re.sub(r"^[-*•]\s*", "", line)
        line = re.sub(r"^\d+[.)]\s*", "", line)
        if len(tokenize(line)) >= 5:
            claims.append(line)
    return claims


def lexical_support_score(claim: str, evidence: str) -> float:
    claim_tokens = [token for token in tokenize(claim) if len(token) > 2]
    evidence_tokens = set(token for token in tokenize(evidence) if len(token) > 2)
    if not claim_tokens or not evidence_tokens:
        return 0.0
    counts = Counter(claim_tokens)
    matched = sum(count for token, count in counts.items() if token in evidence_tokens)
    return matched / sum(counts.values())


def audit_claims(
    answer: str,
    source_map: dict[str, str],
    threshold: float = 0.12,
) -> list[ClaimAssessment]:
    assessments: list[ClaimAssessment] = []
    for claim in extract_candidate_claims(answer):
        cited_ids = list(dict.fromkeys(_CITATION_RE.findall(claim)))
        valid_ids = [source_id for source_id in cited_ids if source_id in source_map]
        evidence = " ".join(source_map[source_id] for source_id in valid_ids)
        score = lexical_support_score(_CITATION_RE.sub("", claim), evidence)

        if not cited_ids:
            status = "uncited"
            reason = "No source citation was attached to this candidate claim."
        elif len(valid_ids) != len(cited_ids):
            status = "invalid_citation"
            reason = "At least one cited source ID does not exist in this report."
        elif score < threshold:
            status = "needs_review"
            reason = "The cited text has weak lexical overlap; inspect it manually."
        else:
            status = "candidate_supported"
            reason = (
                "Lexical overlap passed the screening threshold. This does not establish entailment or truth."
            )

        assessments.append(
            ClaimAssessment(
                claim=claim,
                cited_sources=valid_ids,
                lexical_support_score=round(score, 4),
                status=status,
                reason=reason,
            )
        )
    return assessments
