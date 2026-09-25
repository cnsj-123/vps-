from __future__ import annotations

# Context-layer Retrieval v2 *shadow* package.
#
# Canonical retrieval domain (candidate / features / scoring /
# policy / context compilers) lives in ``ombrebrain.retrieval``.
# This package must never grow a second retrieval domain: it is a
# thin adapter + observability layer for the shadow pipeline.
#
# Package layout:
#   candidate.py - ShadowCandidate (adapter view) + normalize_candidate()
#   scorer.py    - score_candidate() (fixed shadow formula)
#   reranker.py  - rank_candidates() (rank only, never selects)
#   quality.py   - privacy-safe [gateway.retrieval_quality_v2]
#   legacy.py    - the original live ContextRetrievalAdapter
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
# legacy retrieval is the live source and this is shadow-only.

from ombrebrain.context.retrieval.candidate import (
    ShadowCandidate,
    normalize_candidate,
)
from ombrebrain.context.retrieval.quality import (
    RetrievalQualityShadow,
)
from ombrebrain.context.retrieval.reranker import (
    ShadowRanker,
    rank_candidates,
)
from ombrebrain.context.retrieval.scorer import (
    ShadowScorer,
    ShadowScoringContext,
    ShadowScoringWeights,
    score_candidate,
)

# Backward-compatible re-export. Import after the shadow modules
# so legacy.py can be imported without a cycle.
from ombrebrain.context.retrieval.legacy import (
    ContextRetrievalAdapter,
)

__all__ = [
    "ContextRetrievalAdapter",
    "RetrievalQualityShadow",
    "ShadowCandidate",
    "ShadowRanker",
    "ShadowScorer",
    "ShadowScoringContext",
    "ShadowScoringWeights",
    "normalize_candidate",
    "rank_candidates",
    "score_candidate",
]
