"""Retriever backends behind one protocol."""

from lsr.retrievers.base import Retriever, ScoringRetriever, score_dense

__all__ = ["Retriever", "ScoringRetriever", "score_dense"]
