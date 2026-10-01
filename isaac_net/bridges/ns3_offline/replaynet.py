"""ReplayNet: NetBase that replays recorded per-frame outcomes, keyed by (env, robot, frame id).

Frame id = capture step (one frame per robot per step). When the running policy sends a frame the
recording has no outcome for (a different step, robot or size class), a fallback is used and the
lookup is counted as a miss:
    exact      key and size class match the recording
    nearest    nearest recorded frame of the same robot and size class (in time)
    env        nearest recorded frame of the same size class from any robot of the env
    lost       nothing usable: the frame is treated as lost
The known flaw: outcomes do not react to the running policy's own load or positions.
"""
from __future__ import annotations

import bisect
from collections import Counter, defaultdict

import torch

from ...core.proto.netsim import NetBase


class ReplayNet(NetBase):
    """NetBase whose frame outcomes come from a recorded rollout, with the fallbacks of the module docstring.
    Valid only for the policy that produced the recording: the outcomes do not react to the running policy."""

    def __init__(self, E, R, device, sizes, outcomes):
        """outcomes: {(e, r, t): (cls, delay_steps or inf)}."""
        self.table = outcomes
        self.by_robot = defaultdict(list)     # (e, r, cls) -> sorted [(t, delay)]
        self.by_env = defaultdict(list)       # (e, cls) -> sorted [(t, delay)]
        for (e, r, t), (c, d) in outcomes.items():
            self.by_robot[(e, r, c)].append((t, d))
            self.by_env[(e, c)].append((t, d))
        for v in list(self.by_robot.values()) + list(self.by_env.values()):
            v.sort()
        self.hits = Counter()
        super().__init__(E, R, device, sizes)

    @staticmethod
    def _nearest(lst, t):
        if not lst:
            return None
        ts = [x[0] for x in lst]
        k = bisect.bisect_left(ts, t)
        cands = [lst[j] for j in (k - 1, k) if 0 <= j < len(lst)]
        return min(cands, key=lambda x: abs(x[0] - t))[1]

    def lookup(self, e, r, t, c):
        rec = self.table.get((e, r, t))
        if rec is not None and rec[0] == c:
            self.hits["exact"] += 1
            return rec[1]
        d = self._nearest(self.by_robot.get((e, r, c), []), t)
        if d is not None:
            self.hits["nearest"] += 1
            return d
        d = self._nearest(self.by_env.get((e, c), []), t)
        if d is not None:
            self.hits["env"] += 1
            return d
        self.hits["lost"] += 1
        return float("inf")

    def _on_arrival(self, t, e, r, i, draws):
        c = self.cls[e, r, i]
        dl = [tt + self.lookup(ee, rr, tt, cc) for ee, rr, cc, tt in zip(e.tolist(), r.tolist(), c.tolist(), t[e].tolist())]
        self.dlv[e, r, i] = torch.tensor(dl, dtype=self.dlv.dtype, device=self.dev)

    def _transmit(self, t, snr_db):
        ok = (self.cap >= 0) & (self.dlv < (t + 1)[:, None, None])
        return torch.where(ok, self.dlv, torch.full_like(self.dlv, float("inf")))
