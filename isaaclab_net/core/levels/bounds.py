"""Value-of-information bounds: ORACLE (instant, lossless delivery) and NOCOMM (nothing is ever delivered).

They are not network models. They bracket what any network level can give a task: ORACLE is the best any
network could do (every accepted message arrives in the step it was captured, with delay 0), and NOCOMM is a
task in which communication does not exist (every accepted message sits in its FIFO until the application
timeout drops it, so the queue fills and later messages overflow, exactly as behind a dead link). Train or
evaluate a task under both before comparing fidelity levels: if the ORACLE return is not clearly above the
NOCOMM return, the network carries little information the policy uses, and no fidelity level can matter for
that task. Same semantics as the ORACLE / NOCOMM modes of the kill-test NetDelay.
"""
from __future__ import annotations

import torch

from .base import INF, LevelNet


class NetOracle(LevelNet):
    """Every accepted message is delivered at its capture step t, delay 0, never lost."""

    level = "ORACLE"

    def _arrival(self, new, count, nact, dr):
        return self._t[:, None].expand_as(self._send).float()


class NetNoComm(LevelNet):
    """No accepted message is ever delivered; each one times out after the application deadline."""

    level = "NOCOMM"

    def _arrival(self, new, count, nact, dr):
        return torch.full_like(self._snr_add, INF)
