"""Checks the skeleton holds together, and pins two invariants worth keeping.

Behaviour of the implemented core is tested in test_core.py.
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
    import subprocess
    import sys

    # A fresh interpreter: other tests in this process may have imported torch.
    code = (f"import sys; [__import__(m) for m in {CORE_MODULES!r}]; "
            "assert 'torch' not in sys.modules, 'torch leaked into the core'")
    subprocess.run([sys.executable, "-c", code], check=True)


def test_query_and_document_are_hashable():
    """Both are frozen dataclasses, so they can key a dict or join a set."""
    from lsr.tasks import Document, Query

    assert len({Query(id="q1", text="a"), Query(id="q1", text="a")}) == 1
    assert len({Document(id="d1", text="b")}) == 1


def test_segment_duration():
    from lsr.longaudio import Segment

    seg = Segment(id="s0", parent_id="rec0", audio_path="/dev/null", start_s=10.0, end_s=50.0)
    assert seg.duration_s == 40.0
