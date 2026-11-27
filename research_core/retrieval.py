from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Sequence

from .schemas import RetrievedHit, SearchHit

_TOKEN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_'-]*")


def tokenize(text: str) -> list[str]:
    return [token.lower() for token in _TOKEN_RE.findall(text)]


def _bm25_scores(query: str, hits: Sequence[SearchHit]) -> list[float]:
    query_terms = set(tokenize(query))
    tokenized = [tokenize(f"{hit.title} {hit.raw_content or hit.content}") for hit in hits]
    lengths = [len(tokens) for tokens in tokenized]
    average_length = sum(lengths) / max(1, len(lengths))
    document_frequencies: Counter[str] = Counter()
    term_frequencies: list[Counter[str]] = []

    for tokens in tokenized:
        frequencies = Counter(tokens)
        term_frequencies.append(frequencies)
        document_frequencies.update(frequencies.keys())

    scores: list[float] = []
    total_docs = len(hits)
    for frequencies, length in zip(term_frequencies, lengths):
        score = 0.0
        for term in query_terms:
            frequency = frequencies.get(term, 0)
            if not frequency:
                continue
            document_frequency = document_frequencies[term]
            inverse_frequency = math.log(
                1.0 + (total_docs - document_frequency + 0.5) / (document_frequency + 0.5)
            )
            denominator = frequency + 1.5 * (1.0 - 0.75 + 0.75 * length / max(average_length, 1.0))
            score += inverse_frequency * (frequency * 2.5 / denominator)
        scores.append(score)
    return scores


def rank_hits(
    question: str,
    hits: list[RetrievedHit],
    top_k: int = 8,
    mode: str = "hybrid",
) -> list[tuple[RetrievedHit, float]]:
    """Deduplicate by URL and combine lexical relevance with the search provider score."""
    if mode not in {"hybrid", "bm25"}:
        raise ValueError(f"Unsupported retrieval mode: {mode}")
    unique: dict[str, RetrievedHit] = {}
    for hit in sorted(hits, key=lambda item: (item.query_index, item.result_index, str(item.url))):
        unique.setdefault(str(hit.url), hit)
    candidates = list(unique.values())
    if not candidates:
        return []

    lexical_scores = _bm25_scores(question, candidates)
    maximum_lexical = max(lexical_scores, default=0.0)
    normalized_lexical = [
        score / maximum_lexical if maximum_lexical else 0.0 for score in lexical_scores
    ]

    scored: list[tuple[RetrievedHit, float]] = []
    for hit, lexical in zip(candidates, normalized_lexical):
        if mode == "bm25":
            score = lexical if maximum_lexical else hit.score
        else:
            score = 0.7 * lexical + 0.3 * hit.score if maximum_lexical else hit.score
        scored.append((hit, float(score)))

    scored.sort(key=lambda item: (-item[1], str(item[0].url)))
    return scored[:top_k]
