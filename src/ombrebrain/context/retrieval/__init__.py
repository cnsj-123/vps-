from __future__ import annotations

# Context-layer Retrieval v2 *shadow* package.
#
# Canonical retrieval domain (candidate / features / scoring /
# policy / context compilers) lives in ``ombrebrain.retrieval``.
# This package must never grow a second retrieval domain, and it must
# not re-export one either: it is a thin adapter + observability layer
# for the shadow pipeline.
#
# Package layout:
#   candidate.py - project_candidate() -> canonical RetrievalCandidate
#                  (+ the real OB bucket metadata readers)
#   scorer.py    - pure shadow scoring functions (fixed weights)
#   reranker.py  - rank_candidates() (rank only, never selects)
#   quality.py   - privacy-safe [gateway.retrieval_quality_v2]
#   legacy.py    - the original live ContextRetrievalAdapter
#
# Deliberately NOT exported here: any shadow candidate / scorer /
# weights / context / ranker type. The only public surface is the
# live adapter plus the observability observer.
#
# Nothing in this package changes the real retrieval result:
# legacy retrieval is the live source and this is shadow-only.

from ombrebrain.context.retrieval.quality import (
    RetrievalQualityShadow,
)

# Import after the shadow modules so legacy.py can be imported
# without a cycle. The historical import
#   from ombrebrain.context.retrieval import (
#       ContextRetrievalAdapter,
#   )
# keeps working unchanged.
from ombrebrain.context.retrieval.legacy import (
    ContextRetrievalAdapter,
)

__all__ = [
    "ContextRetrievalAdapter",
    "RetrievalQualityShadow",
]
