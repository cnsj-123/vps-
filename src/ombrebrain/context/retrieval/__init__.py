from __future__ import annotations

# Context-layer Retrieval v2 shadow pipeline.
#
# Package layout:
#   candidate.py - normalized RetrievalCandidate
#   scorer.py    - configurable shadow scoring
#   reranker.py  - shadow reorder (top_k / threshold)
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

from ombrebrain.context.retrieval.candidate import (
    RetrievalCandidate,
)
from ombrebrain.context.retrieval.quality import (
    RetrievalQualityShadow,
)
from ombrebrain.context.retrieval.reranker import (
    RetrievalReranker,
)
from ombrebrain.context.retrieval.scorer import (
    RetrievalScorer,
    RetrievalScorerWeights,
    RetrievalScoringContext,
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
]
