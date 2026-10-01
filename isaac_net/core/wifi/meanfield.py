"""Mean-field (Bianchi-style) contention model of 802.11 DCF / EDCA on tensors.

Every contending station i is described by its attempt probability tau_i per generic slot. Given the conditional
failure probability p_i of an attempt, the backoff chain of a station with contention windows W_k = min(2^k W0, Wmax)
(W0 = CWmin + 1, Wmax = CWmax + 1) and at most L = max_tx attempts per frame gives (renewal reward over one frame)

    tau(p) = sum_{k<L} p^k / sum_{k<L} p^k (W_k + 1) / 2                                      (1)

which is Bianchi's b_{0,0} expression when L -> infinity and his (W, m) are W0 and log2(Wmax / W0). The failure
probability couples the stations:

    p_i = 1 - (1 - p_sense_i) (1 - p_hidden_i) (1 - FER)
    p_sense_i  = 1 - prod_{j in V(i), j != i} (1 - tau_j)                                     (2)
    p_hidden_i = 1 - exp(-sum_{j in H(i)} tau_j / Tslot_j * (Tv_i + Tv_j))                  (3)

V(i) is the set of stations i senses (its collision domain; without hidden nodes, every station on its channel), H(i)
the stations it does not sense but that reach its AP: they start transmitting at their own attempt rate during the
vulnerable window of i's frame (the PPDU, or only the RTS with RTS/CTS). FER is a residual frame error rate.

EDCA AIFS differentiation: a station whose AIFSN exceeds the smallest one it senses by d_i counts down only after
d_i idle slots following each busy period. Slots are sorted into zones z = 0, 1, ..., Z - 1 (idle slots since the
last busy period; the last zone collects the rest); in zone z only stations with d_j <= z count down. With
b_z = 1 - prod_{d_j <= z} (1 - tau_j) the busy probability in zone z, the zone of a generic slot is distributed as
pi_z ~ prod_{y<z} (1 - b_y) (z < Z - 1) and pi_{Z-1} ~ prod_{y<Z-1} (1 - b_y) / b_{Z-1} (a busy slot restarts the
zone at 0). tau_i is then the attempt probability per slot in which station i counts down, (2) is evaluated per
zone and p_i averages it over the zones i can attempt in (weights pi_z, z >= d_i). With one AIFS (Z = 1) this is
the plain model. The expected duration of a generic slot seen by station i is, per zone and then averaged,

    Tslot_i = sum_z pi_z [sigma P_idle,z + P_idle,z sum_j tau_j / (1 - tau_j) Ts_j + P_coll,z Tc_avg,z]      (4)

over j in V(i) with d_j <= z (i included), with Tc_avg the attempt-weighted mean collision time, and station i
completes mu_i = tau_i sum_{z >= d_i} pi_z (1 - p_i,z) / Tslot_i successful accesses per microsecond. A station may stand for several identical
stations (weight w_i, e.g. saturated background clients); w_i = 0 means not contending.

The fixed point (1)-(3) is solved by damped iteration with a fixed number of iterations, so the solve has no
data-dependent control flow (CUDA-graph safe). bianchi_* below is the scalar reference (exact root by bisection)
used by the tests and the validation.
"""
from __future__ import annotations

import math

import torch

BIG = 1e9


def stage_factors(W0, Wmax, max_tx):
    """(W_k + 1) / 2 for the backoff stages k < max_tx, shape [..., max_tx] (W0, Wmax: tensors of window sizes)."""
    k = torch.arange(int(max_tx), device=W0.device, dtype=W0.dtype)
    Wk = torch.minimum(W0[..., None] * torch.exp2(k.clamp(max=60)), Wmax[..., None])
    return (Wk + 1.0) * 0.5


def tau_from_p(p, W0, Wmax, max_tx, Wfac=None):
    """Attempt probability (1) for failure probability p; W0, Wmax: contention window sizes (CW + 1). Wfac: the
    stage factors of stage_factors(W0, Wmax, max_tx), if already computed (vectorized over the stages)."""
    if Wfac is None:
        Wfac = stage_factors(W0.expand_as(p), Wmax.expand_as(p), max_tx)
    k = torch.arange(Wfac.shape[-1], device=p.device, dtype=p.dtype)
    pk = p[..., None] ** k
    return pk.sum(-1) / (pk * Wfac).sum(-1)


class SingleView:
    """One collision domain (one channel, no hidden nodes): every station senses every station of its env."""

    def sum(self, x):
        return x.sum(-1, keepdim=True).expand_as(x)

    def min(self, x, mask):
        v = torch.where(mask, x, torch.full_like(x, BIG))
        return v.amin(-1, keepdim=True).expand_as(x)


class DomainView:
    """Collision domains by channel: onehot [..., N, D] float; every station senses every station of its domain."""

    def __init__(self, onehot):
        self.M = onehot
        self.MT = onehot.transpose(-1, -2)

    def sum(self, x):
        return (self.M @ (self.MT @ x[..., None]))[..., 0]

    def min(self, x, mask):
        v = torch.where(mask, x, torch.full_like(x, BIG))
        agg = (v[..., :, None] + (1.0 - self.M) * BIG).amin(-2)              # [..., D]
        return (self.M * agg[..., None, :]).sum(-1)


class MatrixView:
    """Pairwise sensing: V [..., N, N] float, V[i, j] = 1 if station i senses station j (V[i, i] = 1)."""

    def __init__(self, V):
        self.V = V

    def sum(self, x):
        return (self.V @ x[..., None])[..., 0]

    def min(self, x, mask):
        v = torch.where(mask, x, torch.full_like(x, BIG))
        return (v[..., None, :] + (1.0 - self.V) * BIG).amin(-1)


def _zones(view, w, tau, d, Z, active):
    """Per zone z < Z (idle slots since the last busy period, beyond the smallest AIFS; the last zone collects the
    rest): the view sums S_z = sum_{j in V(i), d_j <= z} w_j log(1 - tau_j), and the stationary zone distribution
    pi_z seen by each station (after a busy slot the zone restarts at 0; in zone z only stations with d_j <= z count
    down, so the channel stays idle with probability exp(S_z))."""
    lt = torch.log1p(-tau)
    wl = w * lt
    S, own = [], []
    for z in range(Z):
        if Z == 1:
            S.append(view.sum(wl))
            own.append(torch.where(active, lt, torch.zeros_like(lt)))
            break
        m = d <= z
        S.append(view.sum(torch.where(m, wl, torch.zeros_like(wl))))
        own.append(torch.where(active & m, lt, torch.zeros_like(lt)))
    if Z == 1:
        return S, own, [torch.ones_like(tau)]
    g = torch.ones_like(tau)
    raw = []
    for z in range(Z):
        idle = torch.exp(S[z])
        raw.append(g if z < Z - 1 else g / (1.0 - idle).clamp(min=1e-12))
        g = g * idle
    tot = sum(raw)
    return S, own, [r / tot for r in raw]


def slot_time(view, w, tau, Ts, Tc, sigma, d=None, Z=1, zones=None, active=None):
    """(Tslot [...,N], P_idle [...,N]) of (4), averaged over the AIFS zones: in zone z only the stations with
    d_j <= z take part. zones = _zones(...) if already computed."""
    active = w > 0 if active is None else active
    S, _, pi = zones if zones is not None else _zones(view, w, tau, d, Z, active)
    r = w * tau / (1.0 - tau)
    wt = w * tau
    tslot = torch.zeros_like(tau)
    p_idle = torch.zeros_like(tau)
    for z in range(Z):
        if Z == 1:
            rz, wtz = r, wt
        else:
            m = d <= z
            rz, wtz = torch.where(m, r, torch.zeros_like(r)), torch.where(m, wt, torch.zeros_like(wt))
        idle = torch.exp(S[z])
        a0 = view.sum(rz)
        a1 = view.sum(rz * Ts)
        gd = view.sum(wtz)
        g = view.sum(wtz * Tc)
        p_coll = (1.0 - idle - idle * a0).clamp(min=0.0)
        tc = g / gd.clamp(min=1e-12)
        tslot = tslot + pi[z] * (sigma * idle + idle * a1 + p_coll * tc)
        p_idle = p_idle + pi[z] * idle
    return tslot, p_idle


def _fail(view, w, tau, d, Z, active, fer, hidden, Tv, Ts, Tc, sigma):
    """Attempt-weighted failure probability p_i, the share O_i of generic slots in which i counts down, the
    per-zone success probabilities and the zones."""
    zones = _zones(view, w, tau, d, Z, active)
    S, own, pi = zones
    hid = 1.0
    if hidden is not None:
        Tslot, _ = slot_time(view, w, tau, Ts, Tc, sigma, d, Z, zones, active)
        rate = w * tau * _open(pi, d, Z) / Tslot
        hid = torch.exp(-(Tv * (hidden @ rate[..., None])[..., 0] + (hidden @ (rate * Tv)[..., None])[..., 0]))
    num = torch.zeros_like(tau)
    O = torch.zeros_like(tau)
    for z in range(Z):
        ok = torch.exp(S[z] - own[z]) * hid * (1.0 - fer)
        wz = pi[z] if Z == 1 else torch.where(d <= z, pi[z], torch.zeros_like(pi[z]))
        num = num + wz * ok
        O = O + wz
    return 1.0 - num / O.clamp(min=1e-12), O, num, zones


def _open(pi, d, Z):
    if Z == 1:
        return pi[0]
    return sum(torch.where(d <= z, pi[z], torch.zeros_like(pi[z])) for z in range(Z))


def solve(w, W0, Wmax, max_tx, Ts, Tc, sigma, view, *, fer=0.0, d=None, Z=1, hidden=None, Tv=None, tau=None,
          iters=20, damp=0.6, Wfac=None):
    """Damped fixed-point iteration of (1)-(3). All per-station inputs are [..., N] tensors (or scalars that
    broadcast). d [..., N]: AIFSN_i - the smallest AIFSN among the stations i senses (EDCA), with Z = 1 + the
    largest possible d (Z = 1: plain DCF / one AIFS, d ignored). hidden: H [..., N, N] float (H[i, j] = 1: j is
    hidden from i and reaches i's AP) with Tv [..., N]. tau: warm start (default 2 / (W0 + 1)); tau is the attempt
    probability per slot in which the station counts down. Returns a dict with tau, p (failure probability of an
    attempt), mu (successful accesses per us), tslot (us) and p_idle. Wfac: stage_factors(W0, Wmax, max_tx)."""
    tau = (2.0 / (W0 + 1.0)).expand_as(w).clone() if tau is None else tau
    if Wfac is None:
        Wfac = stage_factors(W0.expand_as(w), Wmax.expand_as(w), max_tx)
    active = w > 0
    if Z > 1 and d is None:
        raise ValueError("Z > 1 needs the AIFSN offsets d")
    for _ in range(int(iters)):
        tau = tau.clamp(1e-7, 0.999)
        p, _, _, _ = _fail(view, w, tau, d, Z, active, fer, hidden, Tv, Ts, Tc, sigma)
        tau = tau + damp * (tau_from_p(p, W0, Wmax, max_tx, Wfac) - tau)
    tau = tau.clamp(1e-7, 0.999)
    p, O, succ, zones = _fail(view, w, tau, d, Z, active, fer, hidden, Tv, Ts, Tc, sigma)
    Tslot, p_idle = slot_time(view, w, tau, Ts, Tc, sigma, d, Z, zones, active)
    mu = torch.where(active, tau * succ / Tslot, torch.zeros_like(tau))
    return {"tau": tau, "p": p, "mu": mu, "tslot": Tslot, "p_idle": p_idle, "open": O}


# ----------------------------------------------------------------------------------------------- scalar reference
def tau_scalar(p, W0, Wmax, max_tx):
    num = den = 0.0
    pk = 1.0
    for k in range(int(max_tx)):
        Wk = min(W0 * 2 ** k, Wmax)
        num += pk
        den += pk * (Wk + 1) * 0.5
        pk *= p
    return num / den


def fixed_point_scalar(n, W0, Wmax, max_tx, fer=0.0, tol=1e-13):
    """(tau, p) of n identical saturated stations: exact root of p = 1 - (1 - tau(p))^(n-1) (1 - fer)."""
    lo, hi = 0.0, 1.0 - 1e-12
    for _ in range(200):
        p = 0.5 * (lo + hi)
        tau = tau_scalar(p, W0, Wmax, max_tx)
        f = p - (1 - (1 - tau) ** (n - 1) * (1 - fer))
        if f > 0:
            hi = p
        else:
            lo = p
        if hi - lo < tol:
            break
    p = 0.5 * (lo + hi)
    return tau_scalar(p, W0, Wmax, max_tx), p


def saturation_scalar(n, W0, Wmax, max_tx, Ts, Tc, sigma, payload_us=None, fer=0.0):
    """Saturation of n identical stations (Bianchi): dict with tau, p, per-station successes per us, and the
    normalized throughput S = P_s P_tr E[P] / E[slot] when payload_us (the payload's air time) is given."""
    tau, p = fixed_point_scalar(n, W0, Wmax, max_tx, fer)
    ptr = 1 - (1 - tau) ** n
    ps_any = n * tau * (1 - tau) ** (n - 1)                  # one transmitter (before FER)
    slot = (1 - ptr) * sigma + ps_any * Ts + (ptr - ps_any) * Tc
    succ = ps_any * (1 - fer)
    out = {"tau": tau, "p": p, "slot_us": slot, "mu_per_us": succ / n / slot, "access_us": n * slot / succ}
    if payload_us is not None:
        out["S"] = succ * payload_us / slot
    return out


# Bianchi (IEEE JSAC 18(3), 2000), Table I: FHSS PHY, 1 Mbit/s, times in us (1 bit = 1 us)
BIANCHI_FHSS = {"payload": 8184.0, "mac_hdr": 272.0, "phy_hdr": 128.0, "ack": 112.0 + 128.0, "rts": 160.0 + 128.0,
                "cts": 112.0 + 128.0, "delta": 1.0, "slot": 50.0, "sifs": 28.0, "difs": 128.0}


def bianchi_times(access="basic", prm=BIANCHI_FHSS):
    """(Ts, Tc, payload_us) of Bianchi's model: basic access or RTS/CTS, as in his Section IV."""
    H = prm["phy_hdr"] + prm["mac_hdr"]
    P, d = prm["payload"], prm["delta"]
    if access == "basic":
        ts = H + P + prm["sifs"] + d + prm["ack"] + prm["difs"] + d
        tc = H + P + prm["difs"] + d
    else:
        ts = prm["rts"] + prm["sifs"] + d + prm["cts"] + prm["sifs"] + d + H + P + prm["sifs"] + d + prm["ack"] + \
            prm["difs"] + d
        tc = prm["rts"] + prm["difs"] + d
    return ts, tc, P


def bianchi_throughput(n, W=32, m=3, access="basic", prm=BIANCHI_FHSS):
    """Bianchi's saturation throughput (normalized to the 1 Mbit/s channel) for n stations, window W, m stages,
    no retry limit."""
    ts, tc, P = bianchi_times(access, prm)
    return saturation_scalar(n, W, W * 2 ** m, 400, ts, tc, prm["slot"], P)["S"]


def db(x):
    return 10 * math.log10(x)
