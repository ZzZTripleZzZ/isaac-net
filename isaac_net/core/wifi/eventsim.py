"""Event-driven CSMA/CA reference simulator (CPU, numpy): one BSS, every station senses every other one.

It keeps the exact slot-level semantics of 802.11 DCF / EDCA that the mean-field model abstracts away:
  * after a busy period ends, station i waits SIFS + AIFSN_i slots and then decrements its backoff counter once per
    idle slot; it transmits in the slot where the counter reaches zero. A busy period freezes every counter;
  * two or more stations that transmit in the same slot collide (no capture); a single transmitter succeeds unless
    the residual frame error draw fails;
  * after a failure the station doubles its contention window (W_k = min(2^k W0, Wmax), counter uniform in
    [0, W_k - 1]); after max_tx attempts the access's bytes are dropped; after a success it returns to W0;
  * post-backoff: after every transmission the station draws a new counter even with an empty queue and counts it
    down while the medium is idle. A frame that arrives at an empty station whose counter is zero while the medium
    has been idle for its AIFS is sent at the next slot boundary (immediate access); if the medium is busy or not
    idle long enough, it draws a counter first;
  * one access carries B = min(queued bytes, cap_i) application bytes (A-MPDU aggregation), as in the mean-field
    model; the busy time of a single transmission is busy_succ(i, B), whether it is received or lost to the residual
    frame error (the PPDU still occupies the medium for its full duration, and the ACK timeout / EIFS that follows
    is about the SIFS + ACK of a success), and of a collision the longest busy_coll(i, B) of the colliding
    stations, as in meanfield.slot_time. Channel times come from phy.AccessTiming (or Bianchi's parameters).
Not modeled (as in the mean-field model): propagation delay, capture, hidden nodes, EIFS after a collision seen
by a third station, rate adaptation dynamics, and several queues per station.

run() takes per-station arrivals (times in us, sizes in bytes) or saturated stations and returns throughput,
per-access delays (head of line to the end of the successful exchange) and per-message delays (arrival to the end
of the exchange that carried its last byte) with their arrival times.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


class Const:
    """Picklable constant busy time (us), whatever the bytes."""

    def __init__(self, us):
        self.us = float(us)

    def __call__(self, nbytes):
        return self.us


@dataclass
class Station:
    W0: int                       # CWmin + 1
    Wmax: int                     # CWmax + 1
    aifsn: int
    cap: float                    # application bytes per access
    busy_succ: object             # f(B) -> us: busy time of a lone transmission, received or errored (after AIFS)
    busy_coll: object             # f(B) -> us: busy time of a collision
    max_tx: int = 7
    fer: float = 0.0


def run(stations, sim_us, sigma, sifs, arrivals=None, seed=0, timeout_us=math.inf, warmup_us=0.0):
    """Simulate `sim_us` microseconds. arrivals: None (every station saturated) or a list, per station, of
    (times_us, sizes_bytes) arrays sorted by time. Messages older than timeout_us are purged from the queue before
    each access. Statistics count events after warmup_us."""
    rng = np.random.default_rng(seed)
    n = len(stations)
    W0 = np.array([s.W0 for s in stations])
    Wmax = np.array([s.Wmax for s in stations])
    aifsn = np.array([s.aifsn for s in stations])
    sat = arrivals is None
    b = np.array([rng.integers(0, W0[i]) for i in range(n)])
    stage = np.zeros(n, dtype=int)
    # queues: per station, list of [arrival_us, remaining_bytes, lost, size, index]
    queues = [[] for _ in range(n)]
    ptr = np.zeros(n, dtype=int)
    hol_since = np.zeros(n)
    t_idle0 = 0.0
    delivered = np.zeros(n)
    dropped = np.zeros(n)
    access_delays = [[] for _ in range(n)]
    msg_delays = [[] for _ in range(n)]
    msg_lost = np.zeros(n, dtype=int)
    msg_done = np.zeros(n, dtype=int)
    n_succ = np.zeros(n, dtype=int)
    n_coll = np.zeros(n, dtype=int)
    n_att = np.zeros(n, dtype=int)
    busy_total = 0.0
    INF = math.inf

    def has_data(i):
        return sat or len(queues[i]) > 0

    def next_arrival():
        if sat:
            return INF, -1
        best, who = INF, -1
        for i in range(n):
            ts = arrivals[i][0]
            if ptr[i] < len(ts) and ts[ptr[i]] < best:
                best, who = ts[ptr[i]], i
        return best, who

    def qbytes(i):
        return sum(m[1] for m in queues[i])

    while True:
        ta, j = next_arrival()
        cand = [i for i in range(n) if has_data(i)]
        if cand:
            k = np.array([aifsn[i] + b[i] for i in cand])
            K = int(k.min())
            t_tx = t_idle0 + sifs + K * sigma
        else:
            K, t_tx = None, INF
        if min(ta, t_tx) >= sim_us:
            break
        if ta < t_tx:
            # ---- arrival at station j
            elapsed = (ta - t_idle0 - sifs) / sigma
            ka = math.floor(elapsed) if elapsed >= 0 else -1
            if not queues[j]:
                b_eff = max(0, b[j] - max(0, ka - aifsn[j]))
                if b_eff == 0:
                    if ka >= aifsn[j]:
                        b[j] = ka + 1 - aifsn[j]              # immediate access at the next slot boundary
                    else:
                        b[j] = rng.integers(0, W0[j])         # medium busy or not idle for AIFS: back off
                hol_since[j] = ta
            ts, sz = arrivals[j]
            queues[j].append([ta, float(sz[ptr[j]]), False, float(sz[ptr[j]])])
            ptr[j] += 1
            continue
        # ---- transmission slot K
        tx = [i for i in cand if aifsn[i] + b[i] == K]
        if not sat and timeout_us < INF:
            for i in tx:
                while queues[i] and t_tx - queues[i][0][0] >= timeout_us:
                    queues[i].pop(0)
                    msg_lost[i] += t_tx > warmup_us
            tx = [i for i in tx if queues[i]]
            if not tx:
                continue                  # the purged stations stop with a zero counter; the medium stays idle
        for i in range(n):
            b[i] = max(0, b[i] - max(0, K - aifsn[i]))
        Bs = {i: (stations[i].cap if sat else min(qbytes(i), stations[i].cap)) for i in tx}
        count = t_tx > warmup_us
        if len(tx) == 1:
            i = tx[0]
            ok = rng.random() >= stations[i].fer
            busy = stations[i].busy_succ(Bs[i])          # an errored frame occupies the medium as a success
            fails = [] if ok else [i]
        else:
            busy = max(stations[i].busy_coll(Bs[i]) for i in tx)
            fails = list(tx)
            ok = False
        t_end = t_tx + busy
        if count:
            busy_total += busy
        for i in tx:
            n_att[i] += count
        if ok:
            i = tx[0]
            if count:
                n_succ[i] += 1
                delivered[i] += Bs[i]
                access_delays[i].append(t_end - hol_since[i])
            done, nl = _consume(queues[i], Bs[i], sat, t_end, msg_delays[i], lost=False, count=count)
            msg_done[i] += done
            msg_lost[i] += nl
            stage[i] = 0
            b[i] = rng.integers(0, W0[i])
            hol_since[i] = t_end
        for i in fails:
            if count and len(tx) > 1:
                n_coll[i] += 1
            stage[i] += 1
            if stage[i] >= stations[i].max_tx:
                if count:
                    dropped[i] += Bs[i]
                msg_lost[i] += _consume(queues[i], Bs[i], sat, t_end, None, lost=True, count=count)[1]
                stage[i] = 0
                hol_since[i] = t_end
            W = min(W0[i] * 2 ** stage[i], Wmax[i])
            b[i] = rng.integers(0, W)
        t_idle0 = t_end
    T = sim_us - warmup_us
    return {"throughput_bps": delivered * 8 / (T * 1e-6), "delivered_bytes": delivered, "dropped_bytes": dropped,
            "access_us": [np.asarray(a) for a in access_delays], "msg_delay_us": [np.asarray([x[1] for x in d]) for d in msg_delays],
            "msg_arrival_us": [np.asarray([x[0] for x in d]) for d in msg_delays],
            "msg_done": msg_done, "msg_lost": msg_lost, "n_succ": n_succ, "n_coll": n_coll, "n_att": n_att,
            "busy_frac": busy_total / T, "sim_us": T}


def _consume(q, B, sat, t_end, delays, lost, count):
    """Remove B bytes from the head of queue q (lost=True: they were dropped). Returns (messages completed,
    messages lost) among those whose last byte left now; a message with any dropped byte counts as lost."""
    if sat:
        return 0, 0
    done = nlost = 0
    while B > 1e-9 and q:
        m = q[0]
        take = min(B, m[1])
        m[1] -= take
        B -= take
        if lost:
            m[2] = True
        if m[1] <= 1e-9:
            q.pop(0)
            if m[2]:
                nlost += count
            elif count:
                delays.append((m[0], t_end - m[0]))
                done += 1
    return done, nlost
