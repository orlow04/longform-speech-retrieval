import sys
import types

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from lsr.backends.clasp import _fusion_class, load_fusion  # noqa: E402


def _model():
    torch.manual_seed(0)
    return _fusion_class()(1024, 1000).eval()


def _out(model):
    torch.manual_seed(1)
    h, i = torch.randn(3, 1024), torch.randn(3, 1000)
    with torch.no_grad():
        return model(h, i).numpy()


def test_loads_module_pickled_from_the_clasp_package(tmp_path, monkeypatch):
    # scripts/train.py pickles clasp.models.fusion.HubertLabseConcat
    model = _model()
    cls = type(model)
    for name in ("clasp", "clasp.models", "clasp.models.fusion"):
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    sys.modules["clasp.models.fusion"].HubertLabseConcat = cls
    cls.__module__, cls.__qualname__ = "clasp.models.fusion", "HubertLabseConcat"  # fresh class
    torch.save(model, tmp_path / "ours.pt")
    for name in ("clasp", "clasp.models", "clasp.models.fusion"):
        monkeypatch.delitem(sys.modules, name)  # not importable at load time
    np.testing.assert_allclose(_out(load_fusion(tmp_path / "ours.pt", "cpu")), _out(model),
                               atol=1e-6)


def test_loads_a_state_dict(tmp_path):
    model = _model()
    torch.save(model.state_dict(), tmp_path / "sd.pt")
    loaded = load_fusion(tmp_path / "sd.pt", "cpu")
    assert loaded.mode == "joint"
    np.testing.assert_allclose(_out(loaded), _out(model), atol=1e-6)
