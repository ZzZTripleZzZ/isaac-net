"""Split a batch of envs across several GPUs: ShardedEngine, with the API of one engine.

    from isaaclab_net.core.sharded import ShardedEngine
    net = ShardedEngine("L2-legacy", E=8192, R=16, devices=["cuda:0", "cuda:1"], config=cfg, backend="graph", seed=0)
    net.submit(None, send)               # send [E, R] on devices[0]; scattered to the shards
    out = net.step(None, poses)          # every output gathered back to devices[0], env order unchanged
    net.reset(done_ids)                  # global env ids; each shard resets its own

Shard i is an independent engine from make_engine(level, E_i, R, devices[i], config, backend, ...) that holds the
contiguous global envs offset_i .. offset_i + E_i - 1 (split near-equally, or by `split`). Global env e maps to
(shard_of[e], local_of[e]) (locate(e)). Inputs whose first dim is E are sliced per shard and moved to its device;
outputs are concatenated on devices[0] (the primary device). reset(env_ids) builds a mask [E] and gives every shard
its slice, so a shard resets exactly its envs of the list (reset(None) is a full reset of every shard).

Shard invariance. With engine-owned randomness (NRConfig.rng = "engine", the default) every draw of an env is keyed by
its env id (proto/rng.py). Each shard keys its rows by global id (CounterRNG.set_env_offset, also in the Triton
kernels), and all shards share the seed, so an env draws the same numbers in shard 0 or shard 1 and the sharded
engine is bitwise equal to the unsharded one on the same device type. This holds for the prototype levels, L2-legacy
with one cell (reference and fast backends), the surrogates and bounds, and the background and energy wrappers.
It does not hold where draws come from a sequential generator whose position depends on the batch: the NR engine L2
(stepping draws from the global RNG, reset and traffic draws per engine), the multi-cell legacy radio (RadioMC), and
rng="global". Those shards get distinct derived seeds, so shards are independent but the results depend on the
split (`shard_invariant` is False).

Multi-GPU: the shards run one after another from one Python thread, and since the engines' steps have no host syncs,
kernels on different GPUs overlap. Resets and bool-mask conversions sync. scripts/hazel/sharded_gpu.sbatch runs the
equality check and the throughput benchmark (benchmarks/bench_sharded.py) on a two-GPU node.
"""
from __future__ import annotations

import torch

from .config import NRConfig
from .levels import BOUND_LEVELS, SURROGATE_LEVELS
from .proto.rng import CounterRNG
from .traffic import Requests

INVARIANT_LEVELS = ("L0", "L0DR", "L05", "L05Q", "L1", "L2-legacy") + SURROGATE_LEVELS + BOUND_LEVELS


def set_env_offset(engine, offset):
    """Key every engine-owned stream in a chain of wrappers by global env id offset + local id, and redo the
    engine's initial reset with those keys (its construction drew the initial state under local ids). The episode
    counters are rewound first, so the envs stay in their first episode, as in an unsharded engine."""
    rngs, obj, n = {}, engine, 0
    while obj is not None and n < 16:
        for r in (obj.__dict__.get("rng"), getattr(obj.__dict__.get("radio"), "rng", None)):
            if isinstance(r, CounterRNG):
                rngs[id(r)] = r
        obj = obj.__dict__.get("engine")
        n += 1
    if not rngs or all(r.env_offset == offset for r in rngs.values()):
        return
    for r in rngs.values():
        r.set_env_offset(offset)
        r.episode.sub_(1)
    engine.reset(None)


class ShardedEngine:
    """E envs split across `devices` (a device may repeat: two shards on one GPU). See the module docstring."""

    def __init__(self, level, E, R, devices, config: NRConfig | None = None, backend="reference", *, seed=None,
                 split=None, **kw):
        from .engine import make_engine
        devices = [torch.device(d) for d in devices]
        assert len(devices) >= 1
        n = len(devices)
        if split is None:
            split = [E // n + (1 if i < E % n else 0) for i in range(n)]
        split = [int(s) for s in split]
        assert len(split) == n and sum(split) == E and min(split) >= 1, "split: one positive size per device, sum E"
        cfg = config if config is not None else NRConfig()
        if seed is None:
            seed = cfg.seed if cfg.seed is not None else int(torch.randint(0, 2 ** 62, ()).item())
        self.level, self.E, self.R, self.config = level, E, R, cfg
        self.devices, self.split = devices, split
        self.dev = devices[0]
        self.offsets = [sum(split[:i]) for i in range(n)]
        self.shard_invariant = (level in INVARIANT_LEVELS and cfg.rng == "engine"
                                and (level != "L2-legacy" or cfg.is_legacy_cell()))
        self.shards = []
        for i, (d, e) in enumerate(zip(devices, split)):
            s = seed if self.shard_invariant else (int(seed) + i * 0x9E3779B97F4A7C15) % 2 ** 62
            eng = make_engine(level, e, R, d, cfg, backend, seed=s, **kw)
            set_env_offset(eng, self.offsets[i])
            self.shards.append(eng)
        self.shard_of = torch.cat([torch.full((e,), i, dtype=torch.long) for i, e in enumerate(split)]).to(self.dev)
        self.local_of = torch.cat([torch.arange(e) for e in split]).to(self.dev)

    # ------------------------------------------------------------------ mapping
    def locate(self, env_id):
        """(shard index, local env id) of global env env_id."""
        e = int(env_id)
        for i, (o, s) in enumerate(zip(self.offsets, self.split)):
            if o <= e < o + s:
                return i, e - o
        raise IndexError(env_id)

    def _slice(self, x, i):
        """Shard i's part of x: tensors whose first dim is E are sliced and moved; the rest pass unchanged."""
        if torch.is_tensor(x) and x.dim() >= 1 and x.shape[0] == self.E:
            o = self.offsets[i]
            return x[o:o + self.split[i]].to(self.devices[i], non_blocking=True)
        if isinstance(x, Requests):
            return Requests(*(self._slice(v, i) for v in (x.send, x.det, x.hid)))
        if isinstance(x, dict):
            return {k: self._slice(v, i) for k, v in x.items()}
        return x

    def _gather(self, parts):
        p0 = parts[0]
        if torch.is_tensor(p0):
            if p0.dim() == 0:
                return p0.to(self.dev)
            return torch.cat([p.to(self.dev, non_blocking=True) for p in parts], 0)
        if isinstance(p0, dict):
            return {k: self._gather([p[k] for p in parts]) for k in p0}
        if isinstance(p0, tuple):
            return tuple(self._gather(list(z)) for z in zip(*parts))
        return p0

    def _each(self, name, *args, **kw):
        return [getattr(s, name)(*(self._slice(a, i) for a in args), **{k: self._slice(v, i) for k, v in kw.items()})
                for i, s in enumerate(self.shards)]

    # ------------------------------------------------------------------ contract API
    @property
    def clock(self):
        return self._gather([s.clock for s in self.shards])

    def reset(self, env_ids=None):
        if env_ids is None or (isinstance(env_ids, slice) and env_ids == slice(None)):
            for s in self.shards:
                s.reset(None)
            return
        if torch.is_tensor(env_ids) and env_ids.dtype == torch.bool:
            m = env_ids.to(self.dev)
        else:
            m = torch.zeros(self.E, dtype=torch.bool, device=self.dev)
            m[torch.as_tensor(env_ids, dtype=torch.long).to(self.dev)] = True
        for i, s in enumerate(self.shards):
            s.reset(self._slice(m, i))

    def submit(self, t, requests, snr_db=None, **kw):
        return self._gather(self._each("submit", t, requests, snr_db, **kw))

    def add_frames(self, t, send, det, hid, snr_db):
        self._each("add_frames", t, send, det, hid, snr_db)

    def add_dl_frames(self, t, nbytes, cls=None):
        self._each("add_dl_frames", t, nbytes, cls)

    def step(self, t, x=None, cur_hid=None, **kw):
        if cur_hid is None:
            return self._gather(self._each("step", t, x, **kw))
        return self._gather(self._each("step", t, x, cur_hid, **kw))

    def queued(self):
        return self._gather([s.queued() for s in self.shards])

    def energy_obs(self):
        return self._gather([s.energy_obs() for s in self.shards])

    def __getattr__(self, name):
        if name == "shards":
            raise AttributeError(name)
        raise AttributeError(f"ShardedEngine has no attribute {name!r}; per-shard state is in net.shards[i]")
