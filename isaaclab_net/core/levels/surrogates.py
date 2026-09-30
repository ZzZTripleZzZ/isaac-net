"""Cheap surrogate levels fitted from L2 / L2-legacy rollouts: TR, GE, QA, NN.

  TR  trace replay: each env replays one recorded (episode, env) trace of the source level, open loop
  GE  3-state Markov-modulated delay and loss, one chain per env, one transition per control step
  QA  analytic queue: per-step processor sharing among the backlogged robots of an env, FIFO service
  NN  learned stateful surrogate: an MLP maps send-time features to a drop probability and delay quantiles

Their parameters come from `python -m isaaclab_net.tools.fit_levels` (see that module for the fits and the exact
TR matching rule). The parameter formats are those of the earlier prototype baselines, so their fit files load too.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn

from ..proto import netsim as _ns
from .base import INF, TIMEOUT, LevelNet

UL_PER_STEP, S, BYTES_PER_SE, SE_MAX = _ns.UL_PER_STEP, _ns.S, _ns.BYTES_PER_SE, _ns.SE_MAX
PHR_MIN_DB, SR_DELAY = _ns.PHR_MIN_DB, _ns.SR_DELAY
H = 4                                                    # NN: env delay-history length in control steps
RAYLEIGH_DB = 10 * math.log10(math.exp(-0.5772156649))   # ergodic Rayleigh loss, -2.51 dB


def _interp_quantiles(qf, u):
    """Linear interpolation in a 101-point quantile function qf [..., 101] at u [...] in [0, 1)."""
    x = u * 100
    lo = x.floor().long().clamp(max=99)
    w = x - lo
    return qf.gather(-1, lo[..., None]).squeeze(-1) * (1 - w) + qf.gather(-1, (lo + 1)[..., None]).squeeze(-1) * w


# ----------------------------------------------------------------------------- TR
class NetTR(LevelNet):
    """Trace replay. At every reset, env e picks one recorded trace j uniformly (engine generator).

    A frame of class c captured at env step t gets an outcome from trace j: the outcomes of the class-c frames
    captured at t* are indexed by start[j, c, t] and cnt[j, c, t] (the fit resolves t* = the nearest step with
    at least one class-c frame, ties to the earlier step), and one of them is drawn uniformly. If the trace has
    no class-c frame at all (cnt = 0), the draw comes from the pooled class-c outcomes. The frame is delivered
    at t + delay, or lost (delay = inf). Steps beyond the recorded horizon use its last step. Position, SNR,
    queue state and the policy's sends are ignored.
    """

    level = "TR"
    STATE = ("trace",)
    SUBMIT_DRAWS = (("u_pick", "ER", "rand"), ("u_pool", "ER", "rand"))

    def _alloc(self):
        p, d = self.params, self.dev
        self.out = p["delay"].to(d, torch.float32)
        self.start, self.cnt = p["start"].to(d, torch.long), p["cnt"].to(d, torch.long)
        self.pool = p["pool"].to(d, torch.float32)
        self.pool_start, self.pool_cnt = p["pool_start"].to(d, torch.long), p["pool_cnt"].to(d, torch.long)
        self.J, _, self.Tmax = self.start.shape
        self.trace = torch.zeros(self.E, dtype=torch.long, device=d)

    def _reset_rows(self, ids, n, gen):
        _ns.fill_rows(self.trace, ids, torch.randint(0, self.J, (n,), device=self.dev, generator=gen))

    def _arrival(self, new, count, nact, dr):
        t = self._t
        c = (self._send - 1).clamp(0, self.start.shape[1] - 1)
        tt = t.clamp(0, self.Tmax - 1)[:, None].expand_as(c)
        j = self.trace[:, None].expand_as(c)
        st, cn = self.start[j, c, tt], self.cnt[j, c, tt]
        pick = st + (dr["u_pick"] * cn).long().clamp(max=(cn - 1).clamp(min=0))
        delay = self.out[pick.clamp(0, self.out.numel() - 1)]
        ps, pc = self.pool_start[c], self.pool_cnt[c]
        pp = ps + (dr["u_pool"] * pc).long().clamp(max=(pc - 1).clamp(min=0))
        pooled = torch.where(pc > 0, self.pool[pp.clamp(0, max(self.pool.numel() - 1, 0))],
                             torch.full_like(delay, INF))
        delay = torch.where(cn > 0, delay, pooled)
        return t[:, None] + delay                     # inf stays inf (lost)


# ----------------------------------------------------------------------------- GE
class NetGE(LevelNet):
    """K-state Markov chain per env (Gilbert-Elliott style), one transition per control step. The state s_t
    applies to the frames captured at step t. Each (state, class) has a loss probability p [K,C] and a
    delivered-delay distribution: the 101-point empirical quantile function q [K,C,101] or, if the fit has
    no q, a lognormal (mu, sig [K,C]). The initial state of a reset env is drawn from pi0 [K] (engine
    generator); the chain is exogenous, so the policy's sends never change it."""

    level = "GE"
    STATE = ("s",)
    SUBMIT_DRAWS = (("z", "ER", "randn"), ("u_delay", "ER", "rand"), ("u_loss", "ER", "rand"))
    STEP_DRAWS = (("u_trans", "E", "rand"),)

    def _alloc(self):
        p, d = self.params, self.dev
        self.P = p["P"].to(d, torch.float32)
        self.cumP = self.P.cumsum(-1)
        self.cum0 = p["pi0"].to(d, torch.float32).cumsum(-1)
        self.mu, self.sig, self.p = (p[k].to(d, torch.float32) for k in ("mu", "sig", "p"))
        self.q = p["q"].to(d, torch.float32) if "q" in p else None
        self.K = self.P.shape[0]
        self.s = torch.zeros(self.E, dtype=torch.long, device=d)

    def _pick(self, cum, u):
        """Inverse CDF: the first state whose cumulative probability exceeds u."""
        return (u[..., None] >= cum).sum(-1).clamp(max=self.K - 1)

    def _reset_rows(self, ids, n, gen):
        u = torch.rand(n, device=self.dev, generator=gen)
        _ns.fill_rows(self.s, ids, self._pick(self.cum0, u))

    def _arrival(self, new, count, nact, dr):
        c = (self._send - 1).clamp(0, self.p.shape[1] - 1)
        k = self.s[:, None].expand_as(c)
        if self.q is None:
            delay = torch.exp(self.mu[k, c] + self.sig[k, c] * dr["z"])
        else:
            delay = _interp_quantiles(self.q[k, c], dr["u_delay"])
        lost = dr["u_loss"] < self.p[k, c]
        return torch.where(lost, torch.full_like(delay, INF), self._t[:, None] + delay)

    def _after(self, dr):
        self.s.copy_(self._pick(self.cumP[self.s], dr["u_trans"]))


# ----------------------------------------------------------------------------- QA
class NetQA(LevelNet):
    """Analytic per-step processor-sharing queue, once per control step (not per slot).

    n = backlogged robots in the env after arrivals. Each gets a time share of min(S/n, n_max) subbands (n_max =
    the L2-legacy power-headroom cap), with its power split over more than one subband as in L1. With pf, the
    per-subband SNR gains the analytic PF multi-user-diversity term 10 log10(H_n) + RAYLEIGH_DB (H_n = the n-th
    harmonic number). The byte rate per step is b = share * eta * 0.75 log2(1 + SNR) (SE capped) * BYTES_PER_SE
    * 40. A robot whose queue was empty at the end of the previous step starts SR_DELAY slots late. Frames are
    served FIFO in continuous time: a frame finishes at t + off + (bytes through it) / b if that is within the
    step. Params: eta (default 0.9) and pf (default True), calibrated by the fit tool.
    """

    level = "QA"
    STATE = ("prev_q",)
    DELAY_LEVEL = False

    def _alloc(self):
        p = self.params or {}
        self.eta = float(p.get("eta", 0.9))
        self.pf = bool(p.get("pf", True))
        self.harm = torch.cumsum(1.0 / torch.arange(1, self.R + 1, device=self.dev, dtype=torch.float32), 0)
        self.prev_q = torch.zeros(self.E, self.R, device=self.dev)

    def _reset_rows(self, ids, n, gen):
        _ns.fill_rows(self.prev_q, ids, 0.0)

    def _serve(self, t, dr):
        snr_db = self._snr
        q = self.rem.sum(-1)
        back = q > 0
        nb = back.sum(-1, keepdim=True).clamp(min=1)
        n_max = torch.floor(10 ** ((snr_db - PHR_MIN_DB) / 10)).clamp(1, S)
        share = torch.minimum(S / nb.float(), n_max)
        snr_sb = snr_db - 10 * torch.log10(share.clamp(min=1.0))
        if self.pf:
            snr_sb = snr_sb + 10 * torch.log10(self.harm[nb - 1]) + RAYLEIGH_DB
        se = (0.75 * torch.log2(1 + 10 ** (snr_sb / 10))).clamp(max=SE_MAX) * self.eta
        b = share * se * BYTES_PER_SE * UL_PER_STEP * back                 # bytes per control step
        off = torch.where(back & (self.prev_q <= 0), torch.full_like(q, SR_DELAY / UL_PER_STEP),
                          torch.zeros_like(q))
        cum = self.rem.cumsum(-1)
        rem, fin = _ns.serve_fifo(self.rem, b * (1 - off))
        self.rem.copy_(rem)
        tt = t[:, None, None]
        tfin = tt + off[..., None] + cum / b.clamp(min=1e-6)[..., None]
        tfin = torch.minimum(tfin, (tt + 1) - 1e-6)
        return torch.where(fin, tfin, torch.full_like(tfin, INF))

    def _after(self, dr):
        self.prev_q.copy_(self.rem.sum(-1))


# ----------------------------------------------------------------------------- NN
class DelayNet(nn.Module):
    """MLP: drop logit + Q monotone quantiles of log(delay in control steps) conditional on delivery."""

    def __init__(self, din, Q=32, h=128):
        super().__init__()
        self.Q = Q
        self.body = nn.Sequential(nn.Linear(din, h), nn.SiLU(), nn.Linear(h, h), nn.SiLU(), nn.Linear(h, h), nn.SiLU())
        self.head = nn.Linear(h, 1 + Q)

    def forward(self, x):
        o = self.head(self.body(x))
        q = torch.cat([o[:, 1:2], o[:, 1:2] + torch.cumsum(nn.functional.softplus(o[:, 2:]), -1)], -1)
        return o[:, 0], q

    @staticmethod
    def sample_quantiles(q, u):
        """Piecewise-linear inverse CDF through tau_k = (k + 0.5) / Q; flat beyond the end knots."""
        Q = q.shape[1]
        pos = u * Q - 0.5
        lo = pos.floor().long().clamp(0, Q - 2)
        w = (pos - lo).clamp(0, 1)
        return q.gather(1, lo[:, None]).squeeze(1) * (1 - w) + q.gather(1, (lo + 1)[:, None]).squeeze(1) * w


def nn_features(cls, snr, own, ownb, nact, totb, to, hist, R):
    """Raw NN input [n, 9 + H] from send-time features (the fit stores a mean / std normalizer on top):
    cls traffic class, snr own SNR (dB), own frames queued ahead, ownb their nominal bytes, nact backlogged
    robots in the env after this submit, totb nominal bytes queued in the env including this step's arrivals,
    to EWMA of timeouts per robot per step, hist [n, H] mean delivered delay of the env in the last H steps."""
    cols = [(cls == 1).float(), (cls == 2).float(), snr / 40.0, own.float() / _ns.F,
            torch.log1p(ownb / 1000.0), nact.float() / R, torch.log1p(totb / 1000.0), to]
    cols += [torch.log1p(10.0 * hist[:, k]) for k in range(hist.shape[1])]
    cols += [(hist[:, 0] > 0).float()]
    return torch.stack(cols, -1)


class NetNN(LevelNet):
    """Learned stateful surrogate (MimicNet / DeepQueueNet style). At submit every enqueued frame is dropped with
    probability sigmoid(logit), otherwise its delay is sampled by inverse-CDF interpolation through the Q
    quantiles. The history features (hist, to) come from this engine's own sampled outcomes, so the model is
    autoregressive. With fifo (default), a delivered frame cannot arrive before the frames queued ahead of it,
    and a lost frame ahead blocks until its own timeout.

    Params (the fit file's "NN" entry): state (DelayNet state dict), xm / xs (input normalizer), din, Q, h, and
    optionally fifo. The MLP runs on every robot at every submit (fixed shape); only enqueued frames use it.
    """

    level = "NN"
    STATE = ("hist", "to_ewma")
    SUBMIT_DRAWS = (("u_drop", "ER", "rand"), ("u_delay", "ER", "rand"))

    def _alloc(self):
        p, d = self.params, self.dev
        self.model = DelayNet(int(p["din"]), int(p["Q"]), int(p["h"])).to(d)
        self.model.load_state_dict(p["state"])
        self.model.eval()
        self.xm, self.xs = p["xm"].to(d, torch.float32), p["xs"].to(d, torch.float32)
        self.fifo = bool(p.get("fifo", True))
        self.hist = torch.zeros(self.E, H, device=d)
        self.to_ewma = torch.zeros(self.E, device=d)

    def _reset_rows(self, ids, n, gen):
        _ns.fill_rows(self.hist, ids, 0.0)
        _ns.fill_rows(self.to_ewma, ids, 0.0)

    def _arrival(self, new, count, nact, dr):
        E, R = self.E, self.R
        send = self._send
        nom = torch.where(self.cap >= 0, self.sizes[(self.cls - 1).clamp(min=0)], torch.zeros_like(self.rem))
        own_nom = torch.where(new, self.sizes[(send - 1).clamp(min=0)], torch.zeros_like(self._snr_add))
        ownb = nom.sum(-1) - own_nom                       # nominal bytes of the robot's frames ahead of this one
        totb = nom.sum((1, 2))[:, None].expand(E, R)
        cls = torch.where(new, send, torch.ones_like(send))
        ER = lambda x: x.reshape(E * R)
        x = nn_features(ER(cls), ER(self._snr_add), ER(count), ER(ownb), ER(nact[:, None].expand(E, R)),
                        ER(totb), ER(self.to_ewma[:, None].expand(E, R)),
                        self.hist[:, None, :].expand(E, R, H).reshape(E * R, H), R)
        logit, q = self.model((x - self.xm) / self.xs)
        lost = ER(dr["u_drop"]) < torch.sigmoid(logit)
        delay = torch.exp(DelayNet.sample_quantiles(q, ER(dr["u_delay"])))
        dlv = self._t[:, None] + delay.view(E, R)
        if self.fifo:
            val = torch.where(torch.isfinite(self.dlv), self.dlv, self.cap.float() + TIMEOUT)
            ahead = self._arF < count[..., None]
            bound = torch.where(ahead, val, torch.full_like(val, -INF)).max(-1).values
            dlv = torch.maximum(dlv, bound)
        return torch.where(lost.view(E, R), torch.full_like(dlv, INF), dlv)

    def _observe(self, fin, t):
        live = self.cap >= 0
        dlv = live & torch.isfinite(fin)
        timed = live & ~dlv & ((t[:, None, None] + 1 - self.cap) >= TIMEOUT)
        n = dlv.sum((1, 2))
        dsum = torch.where(dlv, fin - self.cap.float(), torch.zeros_like(fin)).sum((1, 2))
        newest = torch.where(n > 0, dsum / n.clamp(min=1), self.hist[:, 0])
        self.hist.copy_(torch.cat([newest[:, None], self.hist[:, :-1]], 1))
        self.to_ewma.copy_(0.7 * self.to_ewma + 0.3 * timed.sum((1, 2)).float() / self.R)
