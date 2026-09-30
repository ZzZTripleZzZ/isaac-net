"""LevelNet: the shared base of the surrogate levels (TR, GE, QA, NN) and the bound levels (ORACLE, NOCOMM).

Each level is written once, with graph-safe ops only (no nonzero, no boolean-mask writes, no host syncs inside
submit / step). The backends run that one implementation:
  "reference" (alias "eager")  the ops run eagerly, on any device
  "graph"                      the same ops captured in two CUDA graphs (submit, step), replayed every call

The FIFO, the enqueue and the end-of-step bookkeeping reuse the prototype's graph-safe bodies
(proto/netsim_fast.add_body / finish_body), which are bitwise equal to the prototype reference, so every level has
the prototype's message semantics: F = 16 frames per robot, the 2 s (20-step) application timeout, in-order
compaction, and the same dict outputs.

A level implements some of these hooks:
  _alloc()                          allocate its state buffers (listed in STATE, so graph capture restores them)
  _reset_rows(ids, n, gen)          re-initialize the rows ids (None = all) of its state; draws from gen only
  _arrival(new, count, nact, dr)    delivery time [E,R] of the frame each robot enqueues now (delay levels)
  _serve(t, dr)                     finish times [E,R,F] this step (default: frames whose delivery time < t + 1)
  _observe(fin, t)                  see this step's outcomes before the FIFO drops them (NN's history)
  _after(dr)                        end-of-step update (GE transition, QA's previous queue)
SUBMIT_DRAWS / STEP_DRAWS name the random draws a level takes per call, as (name, shape, kind) with shape "ER"
([E,R]) or "E" and kind "rand" or "randn". They come from the global torch RNG of the device, one value per robot
or env whether it is used or not, so a partial reset never shifts the stream other envs consume. With
inject=True they are read instead from static buffers filled by set_noise(submit=..., step=...) (equivalence
tests feed the reference and the graph backend the same draws).
"""
from __future__ import annotations

import torch

from ..proto import netsim as _ns
from ..proto.netsim_fast import add_body, finish_body, put_slot

F, TIMEOUT = _ns.F, _ns.TIMEOUT
Requests, env_index, fill_rows = _ns.Requests, _ns.env_index, _ns.fill_rows
INF = float("inf")


class LevelNet:
    """Graph-safe engine base with the contract API (reset / submit / step dict / clock / legacy calls)."""

    level = None
    FIELDS = _ns.NetBase.FIELDS
    FEATS = _ns.NetBase.FEATS
    INIT, DTYPE = _ns.NetBase.INIT, _ns.NetBase.DTYPE
    OUTS = ["newest", "det_env", "delivered", "timed", "delay", "cap_out", "cls_out"]
    STATE: tuple = ()
    SUBMIT_DRAWS: tuple = ()
    STEP_DRAWS: tuple = ()
    DELAY_LEVEL = True            # frames get a delivery time at arrival (no byte service)

    def __init__(self, E, R, device, sizes, params=None, backend="reference", inject=False, seed=None):
        if backend == "eager":
            backend = "reference"
        if backend not in ("reference", "graph"):
            raise ValueError(f"backend {backend!r}: the {self.level} level has 'reference' and 'graph'")
        self.E, self.R, self.dev = E, R, torch.device(device)
        if backend == "graph" and self.dev.type != "cuda":
            raise ValueError("the graph backend needs a CUDA device")
        self.backend, self.inject, self.params = backend, inject, params
        self.sizes = torch.tensor(sizes, device=self.dev, dtype=torch.float32)
        self.log_stats = False
        self.log_cap_max = 10 ** 9
        if seed is None:
            seed = int(torch.randint(0, 2 ** 62, ()).item())
        self.gen = torch.Generator(device=self.dev)
        self.gen.manual_seed(seed)
        self.radio = None
        d = self.dev
        z = lambda shape, dt, v: torch.full(shape, v, dtype=dt, device=d)
        for n in self.FIELDS:
            setattr(self, n, z((E, R, F), self.DTYPE[n], self.INIT[n]))
        self.clock = z((E,), torch.long, 0)
        self._last_snr = z((E, R), torch.float32, 0.0)
        self._last_hid = z((E,), torch.long, 0)
        # static inputs / outputs of the captured graphs
        self._t = z((E,), torch.long, 0)
        self._send = z((E, R), torch.long, 0)
        self._det_in = z((E, R), torch.bool, False)
        self._hid_in = z((E,), torch.long, 0)
        self._snr_add = z((E, R), torch.float32, 0.0)
        self._snr = z((E, R), torch.float32, 0.0)
        self._cur_hid = z((E,), torch.long, 0)
        self._newest = z((E, R), torch.long, -1)
        self._det_env = z((E,), torch.bool, False)
        self._delivered = z((E, R, F), torch.bool, False)
        self._timed = z((E, R, F), torch.bool, False)
        self._delay = z((E, R, F), torch.float32, float("nan"))
        self._cap_out = z((E, R, F), torch.long, -1)
        self._cls_out = z((E, R, F), torch.long, 0)
        self._qlen = z((E, R), torch.long, 0)
        self._qbytes = z((E, R), torch.float32, 0.0)
        self._accepted = z((E, R), torch.bool, False)
        self._fin = z((E, R, F), torch.float32, INF)
        self._overflow = torch.zeros((), dtype=torch.long, device=d)
        self._arF = torch.arange(F, device=d)
        self._noise = {}
        if inject:
            for name, shape, _ in self.SUBMIT_DRAWS + self.STEP_DRAWS:
                self._noise[name] = torch.zeros((E, R) if shape == "ER" else (E,), device=d)
        self._alloc()
        self._graphs = {}
        self._pool = None
        self.clear_stats()
        self.reset()

    # ---------------------------------------------------------------- level hooks
    def _alloc(self):
        pass

    def _reset_rows(self, ids, n, gen):
        pass

    def _arrival(self, new, count, nact, dr):
        raise NotImplementedError

    def _serve(self, t, dr):
        ok = (self.cap >= 0) & (self.dlv < (t + 1)[:, None, None])
        return torch.where(ok, self.dlv, torch.full_like(self.dlv, INF))

    def _observe(self, fin, t):
        pass

    def _after(self, dr):
        pass

    # ---------------------------------------------------------------- reset
    def reset(self, env_ids=None):
        """Re-initialize env_ids (None = all) in place, outside any captured graph: frame buffers, level state,
        clock and radio. Draws come from the engine generator only."""
        ids = env_index(env_ids, self.E, self.dev)
        if ids is not None and ids.numel() == 0:
            return
        for n in self.FIELDS:
            fill_rows(getattr(self, n), ids, self.INIT[n])
        fill_rows(self.clock, ids, 0)
        fill_rows(self._last_snr, ids, 0.0)
        fill_rows(self._last_hid, ids, 0)
        self._reset_rows(ids, self.E if ids is None else ids.numel(), self.gen)
        if self.radio is not None:
            self.radio.reset(ids)

    def clear_stats(self):
        self.stats = {"delay": [], "overflow": 0}
        for f in self.FEATS:
            self.stats["d_" + f] = []
            self.stats["x_" + f] = []

    def queued(self):
        return (self.cap >= 0).sum(-1)

    def collect(self):
        st = self.stats
        cat = lambda k: torch.cat(st[k]) if st[k] else torch.zeros(0)
        return {k: cat(k) for k in st if k != "overflow"} | {"overflow": st["overflow"]}

    def set_noise(self, submit=None, step=None):
        """Inject the draws of the next submit and / or step (inject=True only): tuples in the order of
        SUBMIT_DRAWS / STEP_DRAWS."""
        for spec, vals in ((self.SUBMIT_DRAWS, submit), (self.STEP_DRAWS, step)):
            if vals is not None:
                for (name, _, _), v in zip(spec, vals):
                    self._noise[name].copy_(v)

    def _draws(self, spec):
        if self.inject:
            return {name: self._noise[name] for name, _, _ in spec}
        E, R, d = self.E, self.R, self.dev
        out = {}
        for name, shape, kind in spec:
            fn = torch.rand if kind == "rand" else torch.randn
            out[name] = fn((E, R) if shape == "ER" else (E,), device=d)
        return out

    # ---------------------------------------------------------------- API
    def _tvec(self, t):
        return _ns.NetBase._tvec(self, t)

    def attach_radio(self, radio):
        """Use an external Radio for step(t, poses); reset(env_ids) then resets its rows too."""
        self.radio = radio

    def _snr_from(self, x):
        return _ns.NetBase._snr_from(self, x)

    def submit(self, t, requests, snr_db=None):
        """Enqueue new messages at capture time t (None = engine clock). requests: Requests or send [E,R].
        snr_db [E,R] is the SNR recorded as a frame feature (default: the SNR of the previous step).
        Returns accepted [E,R] bool."""
        t = self._tvec(t)
        req = requests if isinstance(requests, Requests) else Requests(send=requests)
        self._t.copy_(t)
        self._send.copy_(req.send)
        if req.det is None:
            self._det_in.zero_()
        else:
            self._det_in.copy_(req.det)
        if req.hid is None:
            self._hid_in.zero_()
        else:
            self._hid_in.copy_(req.hid)
        self._last_hid.copy_(self._hid_in)
        self._snr_add.copy_(self._last_snr if snr_db is None else snr_db)
        self._run("add")
        if self.log_stats:
            self.stats["overflow"] += int(self._overflow)
        return self._accepted.clone()

    def add_frames(self, t, send, det, hid, snr_db):
        """Legacy wrapper (NetSlot API)."""
        self.submit(t, Requests(send, det, hid), snr_db)

    def step(self, t, x, cur_hid=None):
        """Advance [t, t+1). x: SNR [E,R] dB or positions [E,R,2|3]. Same outputs as netsim.NetBase.step: the
        legacy form step(t, snr_db, cur_hid) returns (newest, det_env), otherwise a dict."""
        if cur_hid is not None:
            out = self._advance(t, x, cur_hid, full=False)
            return out["newest"], out["det_env"]
        return self._advance(t, x, None, full=True)

    def _advance(self, t, x, cur_hid, full):
        t = self._tvec(t)
        snr = self._snr_from(x)
        pre = None
        if self.log_stats:
            pre = {f: getattr(self, f).clone() for f in ["cap"] + self.FEATS}
        self._t.copy_(t)
        self._snr.copy_(snr)
        self._last_snr.copy_(snr)
        self._cur_hid.copy_(self._last_hid if cur_hid is None else cur_hid)
        self._run("step")
        if self.log_stats:
            st, cap = self.stats, pre["cap"]
            keep = cap <= self.log_cap_max
            dk, tk = self._delivered & keep, self._timed & keep
            st["delay"].append((self._fin - cap.float())[dk].cpu())
            for f in self.FEATS:
                st["d_" + f].append(pre[f][dk].cpu())
                st["x_" + f].append(pre[f][tk].cpu())
        out = {"newest": self._newest.clone(), "det_env": self._det_env.clone()}
        if full:
            out.update(delivered=self._delivered.clone(), timed_out=self._timed.clone(), cap=self._cap_out.clone(),
                       cls=self._cls_out.clone(), delay=self._delay.clone(), queue_len=self._qlen.clone(),
                       queue_bytes=self._qbytes.clone(), sinr_db=self._snr.clone(), t=t.clone())
        self.clock.copy_(t + 1)
        return out

    # ---------------------------------------------------------------- regions (the captured code)
    @torch.no_grad()
    def _add_region(self):
        overflow, new, idx, count, nact = add_body(
            self.cap, self.cls, self.det, self.hid, self.rem, self.f_nact, self.f_snr, self.f_own,
            self._send, self._det_in, self._hid_in, self._snr_add, self._t, self.sizes)
        self._overflow.copy_(overflow)
        self._accepted.copy_(new)
        if self.DELAY_LEVEL:
            dl = self._arrival(new, count, nact, self._draws(self.SUBMIT_DRAWS))
            put_slot(self.dlv, idx, new, dl)

    @torch.no_grad()
    def _step_region(self):
        t = self._t
        dr = self._draws(self.STEP_DRAWS)
        fin = self._serve(t, dr)
        self._observe(fin, t)
        fields, outs = finish_body(self.cap, self.cls, self.det, self.hid, self.rem, self.dlv, self.f_nact,
                                   self.f_snr, self.f_own, fin, t, self._cur_hid)
        for name, v in zip(self.OUTS, outs):          # outputs first: cap_out / cls_out may alias self.cap / cls
            getattr(self, "_" + name).copy_(v)
        for name, v in zip(self.FIELDS, fields):
            getattr(self, name).copy_(v)
        self._after(dr)
        self._fin.copy_(fin)
        self._qlen.copy_((self.cap >= 0).sum(-1))
        self._qbytes.copy_(self.rem.sum(-1))

    def _run(self, which):
        region = self._add_region if which == "add" else self._step_region
        if self.backend == "reference":
            region()
            return
        gr = self._graphs.get(which)
        if gr is None:
            gr = self._capture(region)
            self._graphs[which] = gr
        gr.replay()

    def _io(self):
        return [self._overflow, self._accepted, self._fin, self._qlen, self._qbytes] + \
               [getattr(self, "_" + n) for n in self.OUTS]

    def _capture(self, region):
        """Warm up and capture region into a CUDA graph. State, outputs and both RNG states are snapshotted and
        restored, so capture has no side effect on the simulation or on the random stream."""
        names = list(self.FIELDS) + list(self.STATE)
        snap = {n: getattr(self, n).clone() for n in names}
        snap_io = [x.clone() for x in self._io()]
        cpu_rng, cuda_rng = torch.get_rng_state(), torch.cuda.get_rng_state(self.dev)
        s = torch.cuda.Stream(self.dev)
        s.wait_stream(torch.cuda.current_stream(self.dev))
        with torch.cuda.stream(s):
            for _ in range(2):
                region()
                for n, v in snap.items():
                    getattr(self, n).copy_(v)
        torch.cuda.current_stream(self.dev).wait_stream(s)
        g = torch.cuda.CUDAGraph()
        if self._pool is None:
            self._pool = torch.cuda.graph_pool_handle()
        with torch.cuda.graph(g, pool=self._pool):
            region()
        torch.cuda.synchronize(self.dev)
        for n, v in snap.items():
            getattr(self, n).copy_(v)
        for dst, src in zip(self._io(), snap_io):
            dst.copy_(src)
        torch.set_rng_state(cpu_rng)
        torch.cuda.set_rng_state(cuda_rng, self.dev)
        return g
