"""Checks the skeleton holds together, and pins two invariants worth keeping.

These pass today against stubs. As each part gets written, the
``NotImplementedError`` expectations below should be replaced by real tests of
behaviour — they are a to-do list as much as a test file.
"""

import importlib

import pytest

CORE_MODULES = [
    "lsr",
    "lsr.tasks",
    "lsr.retrievers",
    "lsr.eval",
    "lsr.store",
    "lsr.perturb",
    "lsr.longaudio",
]


@pytest.mark.parametrize("module", CORE_MODULES)
def test_core_imports(module):
    importlib.import_module(module)


def test_core_does_not_import_torch():
    """The cached-embedding boundary only works if the core stays light.

    If this fails, a backend's dependency has leaked into the core and the
    metric path can no longer run in an environment that cannot install torch.
    """
    import sys

    for module in CORE_MODULES:
        importlib.import_module(module)
    assert "torch" not in sys.modules


def test_query_and_document_are_hashable():
    """Both are frozen dataclasses, so they can key a dict or join a set."""
    from lsr.tasks import Document, Query

    assert len({Query(id="q1", text="a"), Query(id="q1", text="a")}) == 1
    assert len({Document(id="d1", text="b")}) == 1


def test_segment_duration():
    from lsr.longaudio import Segment

    seg = Segment(id="s0", parent_id="rec0", audio_path="/dev/null", start_s=10.0, end_s=50.0)
    assert seg.duration_s == 40.0


@pytest.mark.parametrize(
    "call",
    [
        pytest.param(lambda: __import__("lsr.eval", fromlist=["compute_metrics"]).compute_metrics(
            __import__("numpy").array([1, 2, 3])
        ), id="compute_metrics"),
        pytest.param(lambda: __import__("lsr.retrievers", fromlist=["score_dense"]).score_dense(
            __import__("numpy").zeros((2, 4)), __import__("numpy").zeros((3, 4))
        ), id="score_dense"),
        pytest.param(
            lambda: __import__("lsr.store", fromlist=["current_manifest"]).current_manifest(),
            id="current_manifest",
        ),
    ],
)
def test_stubs_are_honest(call):
    """A stub raises rather than returning something plausible and wrong."""
    with pytest.raises(NotImplementedError):
        call()
