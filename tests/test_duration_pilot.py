import numpy as np
import pytest

from lsr.eval import compute_metrics, sampled_pool_values
from lsr.eval.geometry import embedding_stats, score_stats
from lsr.longaudio.grouping import group_containing, group_units


def _units(lengths):
    t, out = 0.0, []
    for d in lengths:
        out.append({"start_s": t, "end_s": t + d})
        t += d
    return out


def test_groups_are_contiguous_unsplit_and_near_target():
    units = _units([7, 9, 12, 6, 8, 11, 10, 9, 7, 13, 6, 8])
    groups = group_units("r", units, 30)
    assert [i for g in groups for i in g.units] == list(range(len(units)))
    assert groups[0].start_s == 0 and groups[-1].end_s == units[-1]["end_s"]
    for a, b in zip(groups, groups[1:]):
        assert a.end_s == b.start_s
    # within a unit of the target, except a merged tail (< target / 2) on the last one
    assert all(20 <= g.duration_s <= 40 for g in groups[:-1])
    assert groups[-1].duration_s <= 30 * 1.5 + 13


def test_short_tail_is_merged_not_left_as_a_runt():
    groups = group_units("r", _units([30, 30, 5]), 30)
    assert [g.duration_s for g in groups] == [30, 35]


def test_target_longer_than_recording_gives_one_group():
    groups = group_units("r", _units([5, 5, 5]), 300)
    assert len(groups) == 1 and groups[0].id == "r@0"


def test_group_containing_rejects_straddling_spans():
    groups = group_units("r", _units([10, 10, 10, 10]), 20)
    assert group_containing(groups, 12, 18).id == "r@0"
    assert group_containing(groups, 15, 25) is None


def test_sampled_pool_full_pool_is_identity():
    ranks = np.array([1, 2, 5, 9])
    v = sampled_pool_values(ranks, n_documents=10, pool_size=10, ks=(1, 5))
    m = compute_metrics(ranks, ks=(1, 5))
    assert v["Hits@1"].mean() == pytest.approx(m["Hits@1"])
    assert v["Hits@5"].mean() == pytest.approx(m["Hits@5"])
    assert v["MRR"].mean() == pytest.approx(m["MRR"])


def test_sampled_pool_matches_monte_carlo():
    rng = np.random.default_rng(0)
    n_docs, k = 200, 25
    ranks = np.array([1, 3, 20, 100, 200])
    v = sampled_pool_values(ranks, n_docs, k, ks=(1, 5))
    for qi, r in enumerate(ranks):
        # negatives 0..r-2 score at or above gold; draw k-1 of the n_docs-1 negatives
        draws = np.array([rng.choice(n_docs - 1, k - 1, replace=False) for _ in range(20000)])
        sub_rank = 1 + (draws < r - 1).sum(axis=1)
        assert v["Hits@1"][qi] == pytest.approx((sub_rank == 1).mean(), abs=0.01)
        assert v["Hits@5"][qi] == pytest.approx((sub_rank <= 5).mean(), abs=0.01)
        assert v["MRR"][qi] == pytest.approx((1 / sub_rank).mean(), abs=0.01)


def test_sampled_pool_chance_level():
    # a gold ranked uniformly at random has expected Hits@1 = 1/pool
    n = 50
    v = sampled_pool_values(np.arange(1, n + 1), n, 10, ks=(1,))
    assert v["Hits@1"].mean() == pytest.approx(0.1)


def test_sampled_pool_validates():
    with pytest.raises(ValueError):
        sampled_pool_values(np.array([1]), 10, 11)
    with pytest.raises(ValueError):
        sampled_pool_values(np.array([11]), 10, 5)


def test_embedding_stats_detect_collapse():
    rng = np.random.default_rng(0)
    spread = rng.normal(size=(100, 32))
    collapsed = np.ones((100, 32)) + 0.01 * rng.normal(size=(100, 32))
    a, b = embedding_stats(spread), embedding_stats(collapsed)
    assert abs(a["pairwise_cos"]) < 0.05 and b["pairwise_cos"] > 0.99
    assert a["effective_rank"] > 20


def test_score_stats_margin():
    s = np.array([[0.9, 0.5, 0.1], [0.2, 0.3, 0.4]])
    st = score_stats(s, np.array([0, 0]))
    assert st["gold_cos"] == pytest.approx(0.55)
    assert st["best_negative_cos"] == pytest.approx(0.45)
    assert st["margin"] == pytest.approx(0.1)


def test_effective_rank_subsample_is_bounded_by_its_size():
    st = embedding_stats(np.random.default_rng(1).normal(size=(200, 64)), n_sub=20)
    assert st["effective_rank_sub"] < 20 < st["effective_rank"]
