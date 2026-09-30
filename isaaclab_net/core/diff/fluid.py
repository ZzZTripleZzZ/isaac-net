"""Differentiable fluid network models: L1D (relaxed L1 NetFluid) and QAD (relaxed QA analytic queue).

Both keep the uplink of E envs x R robots as fixed-shape tensors and advance one control step per call. Every
input that the discrete level treats as a decision is a real-valued tensor here, and the KPIs are smooth
functions of it, so torch.autograd gives d(KPI) / d(decision):

    send       [E,R]   send weight in [0, 1]: the probability (mean-field) or a relaxed Bernoulli sample
                       (relaxed_bernoulli) of enqueuing one message this step
    msg_bytes  [E,R]   message size in bytes (continuous)
    tx_dbm     [E,R]   transmit power; shifts the SNR by tx_dbm - 23 dBm (the legacy reference power), which
                       moves the spectral efficiency, and sets the transmit energy
    snr_db     [E,R]   single-subband SNR at 23 dBm, or pos [E,R,2|3] through proto.netsim.Radio (log-distance
                       path loss plus plane-wave shadowing), which is differentiable in the positions
    deadline   [E,R]   application deadline in control steps (default: timeout), continuous

State layout. Messages live in an age-indexed buffer [E,R,J], J = timeout (index a = captured a steps ago, 0 =
youngest). A robot submits at most one message per step and a message leaves after at most `timeout` steps, so
slot a holds at most one message and FIFO order is age order. This replaces the discrete level's compacted FIFO
(a permutation, not differentiable) by a fixed shift. Each slot carries
    x  expected remaining bytes (send weight x size while unserved)
    w  mass: the probability-like weight of the message still being queued (1 for a sent message at tau = 0)

Relaxations (see relax.py; tau = 0 is the discrete model):
    FIFO completion  frame a is done in slot k when the bytes served to its robot since the step began reach the
                     bytes queued up to and including it: ge(S_k, cum_a), a sigmoid on the log ratio. Its
                     expected finish time is sum_k (done_a(k) - done_a(k-1)) (k+1)/K.
    FIFO remainder   bytes left = cum - smin(cum, S), a log-domain soft minimum, exact at S = 0 and S >> cum.
    backlog          a robot is backlogged while its youngest queued byte is unserved (1 - done of the queue).
    share            S / max(n_backlogged, 1) and the power split max(share, 1) use smax; the SE cap uses smin.
    buffer overflow  a message is accepted when fewer than F messages are queued: sigmoid((F - 0.5 - n) / tau_n).
    deadline         a message still queued at the end of the step at age a survives with weight
                     sigmoid((deadline - 0.5 - (a + 1)) / tau_n) relative to age a - 1 (exact drop at tau_n = 0).
    QA only          the power-headroom cap floor(.) is a sum of sigmoid steps; the PF diversity term uses the
                     harmonic number H(n) = digamma(n + 1) + gamma at a real n; the SR offset uses a soft
                     "was empty" gate.

Not differentiated: the scheduler's discrete choices of L2 (which robot wins which subband), HARQ, BLER draws and
OLLA are not in these models at all; L1D / QAD are fluid models of the uplink with the L1 / QA physics. See
docs/differentiable.md.

KPIs per step (all [E,R] unless noted, weights are masses): accepted, overflow, delivered (per age [E,R,J]),
delivered_mass, delay_sum (sum of mass x delay in control steps), timed_out, aoi (age of information at the end
of the step, in control steps), energy_j (transmit energy in joules), queue_bytes, queue_len, served_bytes.
"""
from __future__ import annotations

import math

import torch

from ..proto import netsim as _ns
from . import relax as rx

S_SB, BYTES_PER_SE, SE_MAX = _ns.S, _ns.BYTES_PER_SE, _ns.SE_MAX
PHR_MIN_DB, SR_DELAY, P_REF_DBM = _ns.PHR_MIN_DB, _ns.SR_DELAY, _ns.P_TX_DBM
RAYLEIGH_DB = 10 * math.log10(math.exp(-0.5772156649))
QMIN = 1e-3                 # bytes: a queue below this is empty (the discrete FIFO zeroes frames within 1e-3 B)


def relaxed_bernoulli(p, u, lam):
    """Relaxed Bernoulli(p) sample from uniforms u (binary-concrete). lam = 0 gives the hard sample (u < p)."""
    if lam <= 0:
        return (u < p).to(p.dtype)
    lg = lambda z: torch.log(z.clamp(1e-7, 1 - 1e-7)) - torch.log1p(-z.clamp(1e-7, 1 - 1e-7))   # noqa: E731
    return torch.sigmoid((lg(p) - lg(u)) / lam)


def radio_snr_db(radio, pos):
    """proto.netsim.Radio.snr_db computed in the dtype of pos (float64 for gradcheck), same formula."""
    k, phi = radio.k.to(pos.dtype), radio.phi.to(pos.dtype)
    d = pos.norm(dim=-1).clamp(min=1.0)
    pl = 40 + 35 * torch.log10(d)
    sh = radio.amp * torch.cos(torch.einsum("erc,ekc->erk", pos[..., :2], k) + phi[:, None, :]).sum(-1)
    return _ns.P_TX_DBM - pl - sh - _ns.NI_DBM


def _l1_slot(S, energy, q0, present, snr, p_w, t_slot, eta, tau):
    """One UL slot of L1D: backlog from the youngest byte's completion, equal shares, SE, bytes served."""
    back = present * (1 - rx.ge(S, q0, tau, atol=QMIN))
    nb = back.sum(-1, keepdim=True)
    share = S_SB / rx.smax(nb, 1.0, tau)
    split = rx.smax(share, 1.0, tau)
    snr_sb = snr - 10 * torch.log10(split)
    se = rx.smin(0.75 * torch.log2(1 + 10 ** (snr_sb / 10)), SE_MAX, tau) * eta
    return S + share * se * BYTES_PER_SE * back, energy + p_w * t_slot * back * rx.smin(share, 1.0, tau)


class DiffFluid:
    """Differentiable fluid uplink. mode "L1": relaxed NetFluid (K UL slots per step, equal subband shares).
    mode "QA": relaxed NetQA (one analytic processor-sharing interval per step, PF gain, headroom cap, SR offset).

    tau: relative temperature of the byte and share relaxations; tau_n: additive temperature of the count and
    age gates (default = tau). tau = tau_n = 0 runs the exact hard operations (the discrete level). Both can be
    changed between steps (net.tau = ...). compile=True runs the L1D slot body through torch.compile (fewer
    kernel launches per UL slot; the first call compiles). All state tensors are replaced, never written in place,
    so a rollout is one autograd graph; call detach() to truncate it.
    """

    def __init__(self, E, R, device="cpu", mode="L1", tau=0.05, tau_n=None, timeout=_ns.TIMEOUT, fb=_ns.F,
                 ul_per_step=_ns.UL_PER_STEP, eta=None, pf=True, step_s=0.1, dtype=torch.float32, compile=False):
        if mode not in ("L1", "QA"):
            raise ValueError(f"mode {mode!r}: 'L1' or 'QA'")
        self.E, self.R, self.dev, self.dtype, self.mode = E, R, torch.device(device), dtype, mode
        self.J, self.F, self.K = int(timeout), int(fb), int(ul_per_step)
        self.tau = float(tau)
        self.tau_n = float(tau if tau_n is None else tau_n)
        self.eta = float(_ns.L1_ETA if eta is None else eta)
        self.pf, self.step_s = bool(pf), float(step_s)
        self.compile, self._compiled = bool(compile), None
        kw = dict(device=self.dev, dtype=dtype)
        self.age = torch.arange(self.J, **kw)
        self.slot_end = torch.arange(1, self.K + 1, **kw) / self.K        # finish offset of UL slot k
        self.radio = None
        self.reset()

    # ------------------------------------------------------------------ state
    def reset(self, env_ids=None):
        """Re-initialize env_ids (None = all; index tensor, list or bool mask [E]). Other rows keep their values
        and their autograd history."""
        E, R, J = self.E, self.R, self.J
        kw = dict(device=self.dev, dtype=self.dtype)
        if env_ids is None or not hasattr(self, "x"):
            self.x = torch.zeros(E, R, J, **kw)
            self.w = torch.zeros(E, R, J, **kw)
            self.aoi = torch.zeros(E, R, **kw)
            self.prev_q = torch.zeros(E, R, **kw)
            self.clock = torch.zeros(E, dtype=torch.long, device=self.dev)
            if self.radio is not None:
                self.radio.reset()
            return
        m = torch.zeros(E, dtype=torch.bool, device=self.dev)
        ids = _ns.env_index(env_ids, E, self.dev)
        m[ids] = True
        z = lambda v: torch.where(m.view(-1, *([1] * (v.dim() - 1))), torch.zeros_like(v), v)   # noqa: E731
        self.x, self.w, self.aoi, self.prev_q = z(self.x), z(self.w), z(self.aoi), z(self.prev_q)
        self.clock = torch.where(m, torch.zeros_like(self.clock), self.clock)
        if self.radio is not None:
            self.radio.reset(ids)

    def detach(self):
        """Cut the autograd history of the state (truncated backpropagation)."""
        self.x, self.w, self.aoi, self.prev_q = (v.detach() for v in (self.x, self.w, self.aoi, self.prev_q))

    def attach_radio(self, radio):
        """Use this proto.netsim.Radio for pos inputs (reset(env_ids) resets its rows too)."""
        self.radio = radio

    def snr_from(self, snr_db=None, pos=None, tx_dbm=None):
        """SNR [E,R] in dB at the robot's transmit power: snr_db (at 23 dBm) or Radio(pos), plus tx_dbm - 23."""
        if pos is not None:
            if self.radio is None:
                self.radio = _ns.Radio(self.E, self.dev)
            snr_db = radio_snr_db(self.radio, pos)
        if snr_db is None:
            raise ValueError("give snr_db or pos")
        if tx_dbm is not None:
            snr_db = snr_db + (tx_dbm - P_REF_DBM)
        return snr_db

    # ------------------------------------------------------------------ one control step
    def step(self, send, msg_bytes, snr_db=None, pos=None, tx_dbm=None, deadline=None):
        """Enqueue (send, msg_bytes) at age 0, serve one control step, drop messages past their deadline, age the
        buffer. Returns the KPI dict of this step (see the module docstring)."""
        tau, tau_n, J = self.tau, self.tau_n, self.J
        dt = self.dtype
        send = torch.as_tensor(send, device=self.dev).to(dt).expand(self.E, self.R)
        msg_bytes = torch.as_tensor(msg_bytes, device=self.dev).to(dt).expand(self.E, self.R)
        snr = self.snr_from(snr_db, pos, tx_dbm).to(dt)
        p_w = 10 ** (((P_REF_DBM if tx_dbm is None else tx_dbm) - 30.0) / 10.0)
        p_w = torch.as_tensor(p_w, device=self.dev, dtype=dt)

        # ---- enqueue: soft buffer-overflow gate on the queued message count
        gate = rx.below(self.w.sum(-1), self.F, tau_n)
        acc = send * gate
        x = torch.cat([(acc * msg_bytes)[..., None], self.x[..., 1:]], -1)
        w = torch.cat([acc[..., None], self.w[..., 1:]], -1)

        # ---- serve: cum[a] = bytes queued up to and including the message of age a (older first)
        cum = x.flip(-1).cumsum(-1).flip(-1)
        q0 = cum[..., 0]
        if self.mode == "L1":
            served, done_k, energy = self._serve_l1(cum, q0, snr, p_w)
            dd = torch.diff(done_k, dim=-2, prepend=torch.zeros_like(done_k[..., :1, :]))   # [E,R,K,J]
            fin_off = (dd * self.slot_end[:, None]).sum(-2)                                   # E[finish] | done
            done = done_k[..., -1, :]
        else:
            served, done, fin_off, energy = self._serve_qa(cum, q0, snr, p_w)
        dmass = w * done                                           # delivered mass by age
        delay_sum = (w * (done * self.age + fin_off)).sum(-1)      # mass x (age + finish offset)
        newcum = cum - rx.smin(cum, served[..., None], tau)
        older = torch.cat([newcum[..., 1:], torch.zeros_like(newcum[..., :1])], -1)
        # x = mass x bytes per unit mass: the surviving mass w (1 - done) keeps the unserved bytes of the message.
        # At tau = 0 this is the discrete FIFO zeroing a finished frame (and its sub-1e-3 B residue).
        x = (newcum - older).clamp(min=0) * (1 - done)
        w = w * (1 - done)

        # ---- deadline: survival weight relative to the previous age (exact drop at tau_n = 0)
        dl = torch.full((self.E, self.R), float(J), device=self.dev, dtype=dt) if deadline is None else \
            torch.as_tensor(deadline, device=self.dev).to(dt).expand(self.E, self.R)
        surv = rx.below(self.age + 1, dl[..., None], tau_n)                     # [E,R,J]
        prev = torch.cat([torch.ones_like(surv[..., :1]), surv[..., :-1]], -1)
        keep = (surv / prev.clamp(min=1e-12)).clamp(max=1.0)
        timed = (w * (1 - keep)).sum(-1)
        w, x = w * keep, x * keep
        timed = timed + w[..., -1]                                  # the buffer ends at age J - 1
        # ---- age of information: newest delivered message this step (youngest first)
        none_before = torch.cumprod(torch.cat([torch.ones_like(dmass[..., :1]), 1 - dmass[..., :-1]], -1), -1)
        p_newest = dmass * none_before
        aoi = (1 - dmass).prod(-1) * (self.aoi + 1) + (p_newest * (self.age + 1)).sum(-1)
        # ---- shift ages
        self.x = torch.cat([torch.zeros_like(x[..., :1]), x[..., :-1]], -1)
        self.w = torch.cat([torch.zeros_like(w[..., :1]), w[..., :-1]], -1)
        self.prev_q = self.x.sum(-1)
        self.aoi = aoi
        self.clock = self.clock + 1
        return {"accepted": acc, "overflow": send - acc, "delivered": dmass, "delivered_mass": dmass.sum(-1),
                "delay_sum": delay_sum, "timed_out": timed, "aoi": aoi, "energy_j": energy,
                "queue_bytes": self.x.sum(-1), "queue_len": self.w.sum(-1), "served_bytes": rx.smin(served, q0, tau),
                "sinr_db": snr}

    # ------------------------------------------------------------------ service models
    def _serve_l1(self, cum, q0, snr, p_w):
        """K UL slots; equal subband shares among backlogged robots. Returns served bytes [E,R], done [E,R,K,J]
        (soft indicator that the message of age a finished by the end of slot k) and energy [E,R]."""
        tau = self.tau
        present = rx.positive(q0, QMIN, tau)
        S = torch.zeros_like(q0)
        Ss = []
        energy = torch.zeros_like(q0)
        slot = self._slot_fn()
        for _ in range(self.K):
            S, energy = slot(S, energy, q0, present, snr, p_w, self.step_s / self.K, self.eta, tau)
            Ss.append(S)
        Sk = torch.stack(Ss, -1)                                     # [E,R,K]
        done_k = rx.ge(Sk[..., None], cum[..., None, :], tau, atol=QMIN)
        return S, done_k, energy

    def _slot_fn(self):
        if not self.compile:
            return _l1_slot
        if self._compiled is None:
            self._compiled = torch.compile(_l1_slot, dynamic=False)
        return self._compiled

    def _serve_qa(self, cum, q0, snr, p_w):
        """One processor-sharing interval per step (NetQA). Returns served bytes, done [E,R,J], finish offset
        mass-free [E,R,J] (finish offset within the step, times done) and energy."""
        tau, tau_n = self.tau, self.tau_n
        back = rx.positive(q0, QMIN, tau)
        nb = rx.smax(back.sum(-1, keepdim=True), 1.0, tau)
        n_max = rx.floor_int(10 ** ((snr - PHR_MIN_DB) / 10), 1, S_SB, tau_n)
        share = rx.smin(S_SB / nb, n_max, tau)
        snr_sb = snr - 10 * torch.log10(rx.smax(share, 1.0, tau))
        if self.pf:
            snr_sb = snr_sb + 10 * torch.log10(rx.harmonic(nb)) + RAYLEIGH_DB
        se = rx.smin(0.75 * torch.log2(1 + 10 ** (snr_sb / 10)), SE_MAX, tau) * self.eta
        b = share * se * BYTES_PER_SE * self.K * back                              # bytes per control step
        off = (SR_DELAY / self.K) * back * (1 - rx.positive(self.prev_q, QMIN, tau))
        cap = b * (1 - off)
        done = rx.ge(cap[..., None], cum, tau, atol=QMIN)
        tfin = rx.smin(off[..., None] + cum / b.clamp(min=1e-6)[..., None], 1 - 1e-6, tau)
        busy = rx.smin(q0 / b.clamp(min=1e-6), 1 - off, tau)
        energy = p_w * self.step_s * back * rx.smin(share, 1.0, tau) * busy
        return cap, done, done * tfin, energy


# ---------------------------------------------------------------------- rollouts and KPIs
def rollout(net, T, send, msg_bytes, snr_db=None, pos=None, tx_dbm=None, deadline=None, warmup=0, time_axis=None):
    """Run T steps and aggregate the KPIs over steps warmup..T-1. Inputs are constant ([E,R], pos [E,R,2|3]) or
    per step with a leading axis T; time_axis (a set of argument names) says which ones are per step, e.g.
    {"send"} for a [T,E,R] relaxed-Bernoulli send sequence. Returns per-robot [E,R] tensors and batch means:
      offered, delivered, timed_out, overflow (masses), delivery (delivered / offered), delay (mean delay of the
      delivered mass, control steps), aoi (mean end-of-step AoI, control steps), energy_j (per step), and
      mean_delay / mean_delivery / mean_aoi / mean_energy_j (scalars; delay weighted by delivered mass)."""
    ta = set(time_axis or ())
    args = dict(send=send, msg_bytes=msg_bytes, snr_db=snr_db, pos=pos, tx_dbm=tx_dbm, deadline=deadline)
    acc = {}
    for t in range(T):
        kw = {k: (v[t] if k in ta and v is not None else v) for k, v in args.items()}
        out = net.step(**kw)
        if t < warmup:
            continue
        s = kw["send"] if torch.is_tensor(kw["send"]) else torch.as_tensor(kw["send"])
        for k, v in (("offered", s.to(net.dtype).expand(net.E, net.R)), ("delivered", out["delivered_mass"]),
                     ("delay_sum", out["delay_sum"]), ("timed_out", out["timed_out"]), ("overflow", out["overflow"]),
                     ("aoi_sum", out["aoi"]), ("energy_sum", out["energy_j"])):
            acc[k] = acc.get(k, 0) + v
    n = T - warmup
    eps = 1e-9
    res = {k: acc[k] for k in ("offered", "delivered", "timed_out", "overflow")}
    res["delivery"] = acc["delivered"] / (acc["offered"] + eps)
    res["delay"] = acc["delay_sum"] / (acc["delivered"] + eps)
    res["aoi"] = acc["aoi_sum"] / n
    res["energy_j"] = acc["energy_sum"] / n
    res["mean_delay"] = acc["delay_sum"].sum() / (acc["delivered"].sum() + eps)
    res["mean_delivery"] = acc["delivered"].sum() / (acc["offered"].sum() + eps)
    res["mean_aoi"] = res["aoi"].mean()
    res["mean_energy_j"] = res["energy_j"].mean()
    return res
