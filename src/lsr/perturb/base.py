"""Acoustic perturbations, applicable to any task and any retriever.

A perturbation changes the acoustic surface of a signal while leaving its
lexical content intact, so that retrieval degradation can be attributed to
non-semantic variation. The axes worth having, roughly in order of how much
damage they did in earlier work: additive noise (white, ambient) > speaking rate
> reverberation > speaker identity > emotion/prosody > codec compression.

Conventions that earlier work got wrong and this module should not:

- **Perturb one side only.** The query, or the document, never both silently.
  Which side it was applied to is part of the result.
- **SNR needs a definition.** Full-signal RMS and P.56 active speech level give
  different dB for the same file. Pick one, name it in the config, keep it.
- **A severity knob must be the thing it is named after.** A "reverb SNR" axis
  that actually drives a decay time in milliseconds produces a curve that reads
  as a finding and is an artifact. If the knob is decay time, call it that.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np


class Perturbation(Protocol):
    """A deterministic, parameterised transform of a waveform.

    Deterministic given its seed: two runs of the same condition must produce
    the same audio, or no robustness number is comparable to any other.
    """

    name: str
    severity: float

    @property
    def params(self) -> dict[str, object]:
        """Everything needed to reproduce this condition exactly, for provenance."""
        ...

    def apply(self, waveform: np.ndarray, sample_rate: int, *, seed: int) -> np.ndarray:
        """Return the perturbed waveform at the same sample rate and length.

        Length preservation is not universal — time-scale modification changes
        duration by design — so an implementation that changes length says so in
        its docstring and in ``params``.
        """
        ...
