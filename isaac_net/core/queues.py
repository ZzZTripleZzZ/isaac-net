"""Fixed-shape per-robot frame FIFO with byte-stream offsets (NR engine).

Every accepted frame occupies a contiguous byte range [start, end) of a per-robot stream;
`enq` is the stream length. MAC layers hand out stream bytes to transport blocks and compute an
in-order pointer; a frame completes when end <= pointer (RLC in-order delivery).
"""
from __future__ import annotations

import torch


def onehot(idx, n):
    """Sync-free one-hot (F.one_hot validates indices on the host)."""
    return idx[..., None] == torch.arange(n, device=idx.device)


def env_mask(E, env_ids, device):
    """Bool [E] mask from env_ids (None = all envs; bool mask [E], index tensor or list of indices)."""
    if env_ids is None:
        return torch.ones(E, dtype=torch.bool, device=device)
    if torch.is_tensor(env_ids) and env_ids.dtype == torch.bool:
        return env_ids.to(device)
    m = torch.zeros(E, dtype=torch.bool, device=device)
    m[torch.as_tensor(env_ids, device=device, dtype=torch.long)] = True
    return m


def reset_where(t, mask_e, value):
    """Set t[e, ...] = value for envs where mask_e [E] is True (fixed shape, no host sync)."""
    m = mask_e.view(-1, *([1] * (t.dim() - 1)))
    if torch.is_tensor(value):
        return torch.where(m, value.expand_as(t), t)
    return torch.where(m, torch.full_like(t, value), t)


class FrameQueue:
    """FIFO of F frame slots per robot, compacted each control step (index 0 = oldest).

    enable_extras() adds per-frame fields that the traffic models (traffic.py) and submit(..., tag=, priority=,
    deadline_ms=) fill and that compaction and reset carry along: `off` (arrival offset inside the capture step,
    in control steps), `tag`, `prio` and `dline` (deadline in ms, inf = none). They are off by default, so an
    engine without traffic models runs exactly the ops it ran before.
    """

    INIT = {"cap": -1, "start": 0, "end": 0, "lost": False, "fin": float("inf")}
    EXTRAS = {"off": (torch.float64, 0.0), "tag": (torch.long, 0), "prio": (torch.long, 0),
              "dline": (torch.float64, float("inf"))}

    def __init__(self, E, R, F, device, meta=()):
        self.E, self.R, self.F, self.dev = E, R, F, device
        self.meta = tuple(meta)
        z = lambda dt, v: torch.full((E, R, F), v, dtype=dt, device=device)
        self.cap = z(torch.long, -1)
        self.start = z(torch.long, 0)
        self.end = z(torch.long, 0)
        self.lost = z(torch.bool, False)
        self.fin = z(torch.float64, float("inf"))
        self.enq = torch.zeros((E, R), dtype=torch.long, device=device)
        for n, dt in self.meta:
            setattr(self, n, z(dt, 0))
        self.extras = {}

    def enable_extras(self):
        """Add the EXTRAS fields (idempotent); frames already queued get the defaults."""
        for n, (dt, v) in self.EXTRAS.items():
            if n not in self.extras:
                self.extras[n] = v
                setattr(self, n, torch.full((self.E, self.R, self.F), v, dtype=dt, device=self.dev))

    def reset(self, env_ids=None):
        m = env_mask(self.E, env_ids, self.dev)
        for n, v in self.INIT.items():
            setattr(self, n, reset_where(getattr(self, n), m, v))
        self.enq = reset_where(self.enq, m, 0)
        for n, _ in self.meta:
            setattr(self, n, reset_where(getattr(self, n), m, 0))
        for n, v in self.extras.items():
            setattr(self, n, reset_where(getattr(self, n), m, v))

    @property
    def fields(self):
        return list(self.INIT) + [n for n, _ in self.meta] + list(self.extras)

    def count(self):
        return (self.cap >= 0).sum(-1)

    def add(self, t, mask, nbytes):
        """Append one frame of nbytes [E,R] where mask; returns (accepted, slot index, one-hot)."""
        cnt = self.count()
        acc = mask & (cnt < self.F) & (nbytes > 0)
        i = cnt.clamp(max=self.F - 1)
        oh = onehot(i, self.F) & acc[..., None]
        nb = nbytes.round().long()
        self.cap = torch.where(oh, torch.full_like(self.cap, t), self.cap)
        self.start = torch.where(oh, self.enq[..., None], self.start)
        self.end = torch.where(oh, (self.enq + nb)[..., None], self.end)
        self.lost = self.lost & ~oh
        self.fin = torch.where(oh, torch.full_like(self.fin, float("inf")), self.fin)
        self.enq = self.enq + nb * acc
        for n, v in self.extras.items():         # a new frame starts with the defaults; callers overwrite
            t_ = getattr(self, n)
            setattr(self, n, torch.where(oh, torch.full_like(t_, v), t_))
        return acc, i, oh

    def remove(self, gone):
        self.cap = torch.where(gone, torch.full_like(self.cap, -1), self.cap)
        key = (self.cap < 0).long() * self.F + torch.arange(self.F, device=self.dev)
        order = key.argsort(-1)
        for n in self.fields:
            setattr(self, n, getattr(self, n).gather(-1, order))

    def reorder(self, sent, key, nkeys):
        """Class-ordered byte assignment (NRConfig.scheduler="qos"): reorder the frames whose bytes are all unsent
        and visible (start >= sent, end <= enq) by key [E,R,F] (long in [0, nkeys), smaller first), stably, and lay
        their bytes out again contiguously so that the last one ends at enq. Frames with sent bytes (start < sent)
        keep their place and range, and so do frames beyond enq (not arrived yet behind the traffic-model gate), so
        no HARQ process and no stream offset outside the reordered block changes; slot order stays byte order.
        Bytes of the block that belong to no frame (frames purged out of the middle of the stream) end up in front
        of it; when no frame has unsent bytes below the block they are skipped. Returns the new send pointer [E,R]
        (fixed shape, no host sync)."""
        F = self.F
        idx = torch.arange(F, device=self.dev)
        s_ = sent[..., None]
        valid = self.cap >= 0
        mov = valid & (self.start >= s_) & (self.end <= self.enq[..., None])
        pre = valid & (self.start < s_)
        k = torch.where(mov, (1 + key) * F, torch.where(pre, torch.zeros_like(key), torch.full_like(key, (nkeys + 1) * F)))
        order = (k + idx).argsort(-1)
        for n in self.fields:
            setattr(self, n, getattr(self, n).gather(-1, order))
        mov = mov.gather(-1, order)
        ln = (self.end - self.start) * mov
        b0 = self.enq - ln.sum(-1)
        end = b0[..., None] + ln.cumsum(-1)
        self.start = torch.where(mov, end - ln, self.start)
        self.end = torch.where(mov, end, self.end)
        tail = torch.where(pre.gather(-1, order), self.end, torch.zeros_like(self.end)).max(-1).values
        return torch.where(tail <= sent, torch.maximum(sent, b0), sent)

    def floor(self):
        """Stream offset below which every byte belongs to a frame that has left the queue."""
        head = self.cap[..., 0] >= 0
        return torch.where(head, self.start[..., 0], self.enq)
