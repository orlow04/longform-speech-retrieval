"""The metric path, tested against a hand-built task with no model and no audio.

Every expected number below is worked out by hand, so the metric is trustworthy
before there is a backend to blame.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pytest

from lsr.eval import bootstrap_ci, compute_metrics, ranks_from_scores
from lsr.retrievers.base import score_dense
from lsr.store import current_manifest, read_embeddings, write_embeddings
from lsr.tasks import Document, Query, check_task

QUERY_IDS = ["q1", "q2", "q3", "q4", "q5"]
DOC_IDS = ["d1", "d2", "d3", "d4", "d5", "d6"]
# q5 has two relevant documents; the rest have one.
QRELS = {"q1": {"d1"}, "q2": {"d2"}, "q3": {"d3"}, "q4": {"d4"}, "q5": {"d5", "d6"}}

# Rows are queries, columns d1..d6. Expected best-relevant rank in the comment.
SCORES = np.array(
    [
        [0.9, 0.1, 0.2, 0.0, 0.0, 0.0],  # q1: d1 top                        -> 1
        [0.8, 0.5, 0.1, 0.0, 0.0, 0.0],  # q2: d1 above d2                   -> 2
        [0.7, 0.6, 0.5, 0.4, 0.0, 0.0],  # q3: d1, d2 above d3               -> 3
        [0.5, 0.0, 0.0, 0.5, 0.0, 0.0],  # q4: d1 ties d4, pessimistic       -> 2
        [0.9, 0.0, 0.0, 0.0, 0.1, 0.8],  # q5: d6 is the better relevant one -> 2
    ]
)
EXPECTED_RANKS = [1, 2, 3, 2, 2]


@dataclass
class ToyTask:
    name: str = "toy_text_to_text"
    query_modality: str = "text"
    document_modality: str = "text"
    query_list: list = field(default_factory=lambda: [Query(id=q, text=q) for q in QUERY_IDS])
    doc_list: list = field(default_factory=lambda: [Document(id=d, text=d) for d in DOC_IDS])
    qrel_map: dict = field(default_factory=lambda: {k: set(v) for k, v in QRELS.items()})

    def queries(self):
        yield from self.query_list

    def documents(self):
        yield from self.doc_list

    def qrels(self):
        return self.qrel_map


def unit(x):
    x = np.asarray(x, dtype=np.float32)
    return x / np.linalg.norm(x, axis=1, keepdims=True)


# --- ranks ------------------------------------------------------------------


def test_ranks_by_hand():
    ranks = ranks_from_scores(SCORES, QRELS, QUERY_IDS, DOC_IDS)
    assert ranks.tolist() == EXPECTED_RANKS


def test_ranks_ignore_pool_order():
    """Ground truth comes from ids, not from the diagonal."""
    perm = np.random.default_rng(0).permutation(len(DOC_IDS))
    ranks = ranks_from_scores(SCORES[:, perm], QRELS, QUERY_IDS, [DOC_IDS[i] for i in perm])
    assert ranks.tolist() == EXPECTED_RANKS


def test_collapsed_retriever_scores_worst_not_best():
    """All scores tied: optimistic ties would report Hits@1 = 1.0 here."""
    ranks = ranks_from_scores(np.zeros_like(SCORES), QRELS, QUERY_IDS, DOC_IDS)
    # Single-relevant queries sit below the 5 other docs; q5 below the 4 non-relevant.
    assert ranks.tolist() == [6, 6, 6, 6, 5]
    assert compute_metrics(ranks)["Hits@1"] == 0.0


def test_missing_relevant_raises_by_default():
    qrels = {**QRELS, "q1": {"not_in_pool"}}
    with pytest.raises(ValueError, match="q1"):
        ranks_from_scores(SCORES, qrels, QUERY_IDS, DOC_IDS)


def test_missing_relevant_ranked_last_when_asked():
    qrels = {**QRELS, "q1": {"not_in_pool"}}
    ranks = ranks_from_scores(SCORES, qrels, QUERY_IDS, DOC_IDS, on_missing="last")
    assert ranks.tolist() == [7, *EXPECTED_RANKS[1:]]


@pytest.mark.parametrize(
    "scores",
    [SCORES[:, :5], np.where(SCORES == 0.9, np.nan, SCORES)],
    ids=["wrong_shape", "nan"],
)
def test_ranks_reject_bad_scores(scores):
    with pytest.raises(ValueError):
        ranks_from_scores(scores, QRELS, QUERY_IDS, DOC_IDS)


# --- metrics ----------------------------------------------------------------


def test_metrics_by_hand():
    m = compute_metrics(np.array(EXPECTED_RANKS), ks=(1, 2, 3))
    assert m["Hits@1"] == pytest.approx(1 / 5)
    assert m["Hits@2"] == pytest.approx(4 / 5)
    assert m["Hits@3"] == pytest.approx(1.0)
    assert m["MRR"] == pytest.approx((1 + 1 / 2 + 1 / 3 + 1 / 2 + 1 / 2) / 5)
    assert m["mean_rank"] == pytest.approx(2.0)
    assert m["median_rank"] == 2.0
    assert "MAP" not in m


@pytest.mark.parametrize("ranks", [np.array([]), np.array([0, 1]), np.array([[1, 2]])])
def test_metrics_reject_bad_ranks(ranks):
    with pytest.raises(ValueError):
        compute_metrics(ranks)


def test_bootstrap_is_seeded_and_brackets_the_estimate():
    ranks = np.random.default_rng(1).integers(1, 20, size=500)
    point = compute_metrics(ranks)["Hits@5"]
    lo, hi = bootstrap_ci(ranks, "Hits@5", seed=3)
    assert lo < point < hi
    assert (lo, hi) == bootstrap_ci(ranks, "Hits@5", seed=3)
    lo_mrr, hi_mrr = bootstrap_ci(ranks, "MRR")
    assert lo_mrr < compute_metrics(ranks)["MRR"] < hi_mrr


def test_bootstrap_degenerate_pool():
    assert bootstrap_ci(np.ones(10, dtype=int)) == (1.0, 1.0)


def test_bootstrap_unknown_metric():
    with pytest.raises(ValueError):
        bootstrap_ci(np.ones(3), "MAP")


# --- check_task -------------------------------------------------------------


def test_check_task_accepts_consistent_task():
    check_task(ToyTask())


@pytest.mark.parametrize(
    ("task", "match"),
    [
        (ToyTask(name=""), "no name"),
        (ToyTask(query_list=[Query(id="q1", text="a")] * 2), "duplicate query"),
        (ToyTask(doc_list=[]), "empty pool"),
        (ToyTask(query_modality="audio"), "'audio' field"),
        (ToyTask(doc_list=[Document(id=d, text=d, audio_path="x.wav") for d in DOC_IDS]),
         "'text' field"),
        (ToyTask(qrel_map={**QRELS, "q9": {"d1"}}), "unknown query"),
        (ToyTask(qrel_map={**QRELS, "q1": {"d9"}}), "outside the pool"),
        (ToyTask(qrel_map={k: v for k, v in QRELS.items() if k != "q3"}), "no relevant"),
    ],
)
def test_check_task_catches(task, match):
    with pytest.raises(ValueError, match=match):
        check_task(task)


def test_check_task_reports_every_problem_at_once():
    task = ToyTask(doc_list=[], qrel_map={"q9": {"d1"}})
    with pytest.raises(ValueError) as exc:
        check_task(task)
    msg = str(exc.value)
    assert "empty pool" in msg and "unknown query" in msg and "no relevant" in msg


# --- scoring and the cached-embedding boundary --------------------------------


def test_score_dense_is_cosine():
    q, d = unit([[1, 0], [1, 1]]), unit([[1, 0], [0, 1], [-1, 0]])
    np.testing.assert_allclose(score_dense(q, d), [[1, 0, -1], [0.7071, 0.7071, -0.7071]],
                               atol=1e-4)
    with pytest.raises(ValueError):
        score_dense(q, np.zeros((3, 4)))


def test_embeddings_round_trip(tmp_path):
    rng = np.random.default_rng(0)
    q, d = unit(rng.normal(size=(5, 8))), unit(rng.normal(size=(6, 8)))
    write_embeddings(tmp_path / "run", q, d, QUERY_IDS, DOC_IDS)
    q2, d2, qids, dids = read_embeddings(tmp_path / "run")
    np.testing.assert_array_equal(q, q2)
    np.testing.assert_array_equal(d, d2)
    assert (qids, dids) == (QUERY_IDS, DOC_IDS)


@pytest.mark.parametrize(
    ("q", "match"),
    [
        (np.ones((5, 8), dtype=np.float32), "unit norm"),
        (unit(np.ones((5, 8))).astype(np.float64), "float32"),
        (unit(np.ones((4, 8))), "5 query ids but 4"),
        (unit(np.ones((5, 4))), "dimension mismatch"),
    ],
    ids=["unnormalised", "float64", "row_count", "dim"],
)
def test_write_embeddings_rejects(tmp_path, q, match):
    d = unit(np.ones((6, 8)))
    with pytest.raises(ValueError, match=match):
        write_embeddings(tmp_path, q, d, QUERY_IDS, DOC_IDS)
    assert not (tmp_path / "embeddings.npz").exists()


def test_positive_control_end_to_end(tmp_path):
    """Queries embedded identically to their gold doc must score Hits@1 = 1.

    Goes through the boundary with a shuffled pool, so a diagonal assumption or
    an id/row misalignment anywhere on the path would show up here.
    """
    rng = np.random.default_rng(0)
    doc_emb = unit(rng.normal(size=(6, 16)))
    gold = {q: sorted(QRELS[q])[0] for q in QUERY_IDS}
    query_emb = doc_emb[[DOC_IDS.index(gold[q]) for q in QUERY_IDS]]
    perm = rng.permutation(6)
    write_embeddings(tmp_path, query_emb, doc_emb[perm], QUERY_IDS, [DOC_IDS[i] for i in perm])

    q, d, qids, dids = read_embeddings(tmp_path)
    ranks = ranks_from_scores(score_dense(q, d), QRELS, qids, dids)
    assert compute_metrics(ranks)["Hits@1"] == 1.0


def test_manifest_fills_git_state_and_respects_overrides():
    base = dict(run_id="r", task="t", retriever="x", config_hash="h", seed=0,
                n_queries=5, n_documents=6)
    m = current_manifest(**base)
    assert len(m.git_sha) == 40 and isinstance(m.git_dirty, bool) and m.hardware
    m2 = current_manifest(**base, git_sha="abc", git_dirty=False)
    assert (m2.git_sha, m2.git_dirty) == ("abc", False)
