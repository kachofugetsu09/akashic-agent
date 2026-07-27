"""Online lexical encoder for full, causally weighted term features."""

from __future__ import annotations

import json
import hashlib
import math
import re
import warnings
from collections import Counter
from dataclasses import dataclass, field as dataclass_field
from importlib.metadata import version

with warnings.catch_warnings():
    warnings.filterwarnings(
        "ignore",
        message="pkg_resources is deprecated as an API.*",
        category=UserWarning,
    )
    import jieba

from .model import SparseFeature

TOKEN_CHUNKS = re.compile(r"[A-Za-z0-9_]+|[\u3400-\u9fff]+")
NORMALIZER_VERSION = "lower-search-v1"
_TOKENIZER = jieba.Tokenizer()


def tokenize(text: str) -> Counter[str]:
    """Tokenize Chinese and ASCII text into term frequencies."""

    terms: list[str] = []
    for chunk in TOKEN_CHUNKS.findall(text.lower()):
        pieces = (
            _TOKENIZER.lcut_for_search(chunk)
            if any("\u3400" <= c <= "\u9fff" for c in chunk)
            else [chunk]
        )
        terms.extend(piece.strip() for piece in pieces if len(piece.strip()) >= 2)
    return Counter(terms)


def lexical_identity() -> dict[str, str]:
    """Return the isolated tokenizer and dictionary identity."""

    digest = hashlib.sha256()
    dictionary = _TOKENIZER.get_dict_file()
    try:
        for block in iter(lambda: dictionary.read(1024 * 1024), b""):
            digest.update(block)
    finally:
        dictionary.close()
    return {
        "jieba_version": version("jieba"),
        "jieba_dictionary_sha256": digest.hexdigest(),
        "lexical_normalizer_version": NORMALIZER_VERSION,
    }


@dataclass
class LexicalState:
    """Maintain causal BM25 document frequencies for one text field."""

    field: str
    doc_count: int = 0
    total_length: int = 0
    document_frequency: dict[str, int] = dataclass_field(
        default_factory=dict
    )

    def encode(self, terms: Counter[str], k1: float, b: float) -> list[SparseFeature]:
        """Score a document against prior-only statistics without mutating state."""

        length = sum(terms.values())
        average_length = self.total_length / self.doc_count if self.doc_count else max(length, 1)
        scored = [
            (self._bm25_weight(term, tf, length, average_length, k1, b), term, tf)
            for term, tf in terms.items()
        ]
        scored.sort(key=lambda item: (-item[0], item[1]))
        return [
            SparseFeature(
                family=f"lex_{self.field}",
                feature_id=term,
                value=score,
                rank=rank,
                evidence_json=json.dumps(
                    {"tf": tf, "prior_df": self.document_frequency.get(term, 0), "prior_docs": self.doc_count},
                    ensure_ascii=False,
                    sort_keys=True,
                ),
            )
            for rank, (score, term, tf) in enumerate(scored, start=1)
        ]

    def update(self, terms: Counter[str]) -> None:
        """Commit one document after its causal scores have been produced."""

        self.doc_count += 1
        self.total_length += sum(terms.values())
        for term in terms:
            self.document_frequency[term] = self.document_frequency.get(term, 0) + 1

    def _bm25_weight(
        self,
        term: str,
        tf: int,
        length: int,
        average_length: float,
        k1: float,
        b: float,
    ) -> float:
        prior_df = self.document_frequency.get(term, 0)
        idf = math.log1p((self.doc_count - prior_df + 0.5) / (prior_df + 0.5))
        denominator = tf + k1 * (1.0 - b + b * length / average_length)
        return idf * tf * (k1 + 1.0) / denominator
