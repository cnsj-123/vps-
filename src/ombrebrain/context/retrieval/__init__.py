from __future__ import annotations

# Context-layer Retrieval v2 shadow pipeline.
#
# Package layout:
#   candidate.py - RetrievalCandidate + normalize_candidate()
#   scorer.py    - score_candidate() (fixed weight formula)
#   reranker.py  - rerank_candidates()
#   quality.py   - privacy-safe [gateway.retrieval_quality_v2]
#   legacy.py    - the original ContextRetrievalAdapter
#
# The original ombrebrain.context.retrieval module moved to
# legacy.py when this package was introduced (a package and a
# same-named module cannot coexist). Its public API is
# re-exported below, so the historical import
#   from ombrebrain.context.retrieval import (
#       ContextRetrievalAdapter,
#   )
# keeps working unchanged.
#
# Nothing in this package changes the real retrieval result:
# it is a shadow-only pipeline observed at the service layer.

from ombrebrain.context.retrieval.candidate import (
    RetrievalCandidate,
    normalize_candidate,
)
from ombrebrain.context.retrieval.quality import (
    RetrievalQualityShadow,
)
from ombrebrain.context.retrieval.reranker import (
    RetrievalReranker,
    rerank_candidates,
)
from ombrebrain.context.retrieval.scorer import (
    RetrievalScorer,
    RetrievalScorerWeights,
    RetrievalScoringContext,
    score_candidate,
)

# Backward-compatible re-export. Import after the new modules
# so legacy.py can import its shadow hook from this package
# without a cycle.
from ombrebrain.context.retrieval.legacy import (
    ContextRetrievalAdapter,
)

__all__ = [
    "ContextRetrievalAdapter",
    "RetrievalCandidate",
    "RetrievalQualityShadow",
    "RetrievalReranker",
    "RetrievalScorer",
    "RetrievalScorerWeights",
    "RetrievalScoringContext",
    "normalize_candidate",
    "rerank_candidates",
    "score_candidate",
]
