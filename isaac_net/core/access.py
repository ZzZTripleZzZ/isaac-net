"""Per-robot access state machine of the NR engine (level "L2"): RACH / connection setup and connected-mode DRX.

    from isaac_net.core import NRConfig, make_engine
    cfg = NRConfig(rach=True, rach_initial="idle", rach_release_after_ms=2000, drx=True, drx_cycle_ms=160)
    net = make_engine("L2", E, R, dev, cfg)
    out = net.step(None, poses)          # + access_state, access_sleep_frac, rach_attempts [E, R]
    net.counters()["access"]             # preamble transmissions, collisions, connections, failed procedures

States per robot (out["access_state"], long [E, R], the state at the last slot of the step):

    IDLE (0) --UL or DL data arrives--> RACH (1) --preamble alone at its RO--> RAR + Msg3/Msg4 --> CONNECTED (2)
                                          ^   \\--same preamble as another robot of its cell--> backoff --/
                                          |                                (after rach_max_attempts: restart)
    CONNECTED (2) <--> DORMANT (3): DRX sleep after drx_inactivity_ms without scheduling activity; awake again in
                                    the next on-duration, at once on UL data with drx_ul_wake="sr"
    CONNECTED (2) --> IDLE (0): rach_release_after_ms without activity (RRC release), if set

A robot is schedulable (both directions) only while CONNECTED and awake. The gate is MacLink.sched_ok, the mask the
handover interruption already uses, so the MAC code is unchanged: AccessStage wraps the slot() of the UL and DL links
of this engine instance and ANDs its mask into sched_ok for that slot (with several cells, into the handover mask the
engine sets). Messages of unschedulable robots wait in their queues (the frame buffer, timeouts and AoI apply as
usual); the SR state machine keeps running, so the grant is ready when the robot becomes schedulable (Msg3 carries
the buffer status).

RACH (TS 38.321 Sec. 5.1, contention based). ROs are the first UL-capable slot of every rach_occasion_slots window.
A robot that has to attempt picks one of rach_preambles preambles uniformly at the next RO; two or more robots of the
same env and cell that pick the same preamble at the same RO collide, and all of them fail (no capture). The count per
(env, cell, preamble) is one scatter_add over [E, C * preambles], so the step has no loop over robots. Success: the
robot is served from RO + rach_rar_window_slots + rach_msg3_slots on. Collision: the failure is known at contention
resolution (the same RO + RAR + Msg3 delay), the robot backs off uniformly in [0, rach_backoff_ms] and retries at the
next RO after that; rach_max_attempts failed attempts end the procedure (counted) and a new one starts at once.
Triggers: UL data (submit() or a traffic model, at its arrival slot) or DL data (paging is not modelled: the robot
starts RACH at the arrival) for an IDLE robot. With rach_initial="idle" every robot starts IDLE after a reset, so a
fleet that powers on together contends at the same ROs. With several cells a robot contends in the cell it was
associated with at the end of the previous step.

DRX (TS 38.321 Sec. 5.7). Active Time = drx-InactivityTimer running (drx_inactivity_ms after the last scheduling
activity, restarted by each), or the on-duration of the current cycle (slot - drx_start_offset mod cycle <
drx_on_ms; the short cycle for drx_short_cycles cycles after the inactivity timer expires, if drx_short_cycle_ms is
set, then the long cycle), or, with drx_ul_wake="sr", UL data in the buffer (a pending SR is Active Time; the SR then
goes through the engine's own SR / grant delay). drx_ul_wake="on_duration" makes UL data wait for the next on-duration
like DL data. Scheduling activity is approximated per slot as "awake and with data to send or a HARQ process waiting"
in that direction: the inactivity timer starts when the buffers have drained, which is when the gNB stops issuing new
grants. HARQ retransmission timers (drx-HARQ-RTT, drx-RetransmissionTimer) are not separate: a pending HARQ process
keeps an awake robot awake. Clock: global slots (the SFN), so an env's on-durations do not move with its resets.

RRC release (rach_release_after_ms): a CONNECTED robot with empty buffers whose last activity is that long ago goes
IDLE. The release is decided per control step from the step's first data arrival (exact when the release time falls
before it; a robot with no arrival in the step is released if the time falls inside the step).

Energy: out["access_sleep_frac"] [E, R] is the share of the step's engine slots in which the robot was DRX-dormant
or IDLE (sampled at the slots the engine runs, i.e. every slot with data symbols of an active direction).
core/energy.py charges EnergyConfig.drx_sleep_power_w for that share of the step instead of idle_power_w.

Randomness: the preamble and backoff draws come from the engine's counter RNG (nr_rng.NRRng, sites ACCESS_PRE and
ACCESS_BO, stream = RO index inside the step), keyed by (seed, env, episode, step), so an env's access process is
independent of E and of other envs' resets, and the graph backend draws the same numbers. With rng="global" the
stage keeps its own counter RNG seeded from the engine seed.

Graph safety: fixed shapes, state updated by reassignment of [E, R] tensors (the graph backend re-binds them as it
does the MAC's state; core/nr_fast.state_owners registers this stage), no host sync, time as a device tensor when
captured. The fused Triton kernel of the triton backend has no schedulable-mask input, so rach / drx are refused
there (use "graph", bitwise equal to "reference").
"""
from __future__ import annotations

import torch

from .mac import BIG
from .nr_rng import NRRng, stream_id
from .proto.rng import STEP, mix32
from .queues import env_mask

IDLE, RACH, CONNECTED, DORMANT = 0, 1, 2, 3
STATE_NAMES = ("idle", "rach", "connected", "dormant")
ACCESS_PRE, ACCESS_BO = 16, 17           # engine RNG sites (nr_rng.py uses 1..4)
CTR_NAMES = ("rach_attempts", "rach_collisions", "rach_successes", "rach_failures", "rrc_releases")


def _ms_to_slots(ms, slot_ms):
    return int(round(ms / slot_ms))


class AccessStage:
    """The access state machine of one NREngine (any backend but triton). Built by NREngine when cfg.rach or
    cfg.drx is set; see the module docstring."""

    def __init__(self, eng):
        cfg = eng.config
        if getattr(eng, "backend", None) == "triton":
            raise ValueError("rach / drx block scheduling through MacLink.sched_ok, which the fused triton kernel does "
                             "not read; use backend='graph' (bitwise equal to the reference) or 'reference'")
        self.eng = eng
        self.net = net = eng.net
        self.cfg = cfg
        self.E, self.R, self.dev = E, R, d = eng.E, eng.R, eng.dev
        self.N = cfg.slots_per_step
        sm = cfg.slot_ms
        P = len(cfg.tdd_pattern)
        # ---- RACH constants ----
        self.rach = bool(cfg.rach)
        self.ro = int(cfg.rach_occasion_slots)
        if self.rach and self.ro % P:
            raise ValueError(f"rach_occasion_slots={self.ro} must be a multiple of the TDD period ({P} slots of "
                             f"{cfg.tdd_pattern!r}) so that every RO window has the same first UL-capable slot")
        self.ro_off = next((x for x in range(self.ro) if cfg.ul_capable(x)), None) if self.rach else 0
        if self.rach and self.ro_off is None:
            raise ValueError("no UL-capable slot in a RACH occasion window")
        self.n_pre = int(cfg.rach_preambles)
        self.cres = int(cfg.rach_rar_window_slots) + int(cfg.rach_msg3_slots)   # preamble -> served / failure known
        self.bo = _ms_to_slots(cfg.rach_backoff_ms, sm)
        self.max_att = int(cfg.rach_max_attempts)
        self.release = (None if (not self.rach or cfg.rach_release_after_ms is None)
                        else max(1, _ms_to_slots(cfg.rach_release_after_ms, sm)))
        self.init_state = IDLE if (self.rach and cfg.rach_initial == "idle") else CONNECTED
        # ---- DRX constants ----
        self.drx = bool(cfg.drx)
        self.inact = _ms_to_slots(cfg.drx_inactivity_ms, sm)
        self.cyc = max(1, _ms_to_slots(cfg.drx_cycle_ms, sm))
        self.on = max(1, _ms_to_slots(cfg.drx_on_ms, sm))
        self.scyc = None if cfg.drx_short_cycle_ms is None else max(1, _ms_to_slots(cfg.drx_short_cycle_ms, sm))
        self.n_short = int(cfg.drx_short_cycles)
        self.off = _ms_to_slots(cfg.drx_start_offset_ms, sm)
        self.ul_wake = cfg.drx_ul_wake == "sr"
        # ---- randomness ----
        if net.rng is not None:
            self.rng, self._own_rng = net.rng, False
        else:                   # rng="global": a private counter RNG from the engine seed (reference backend only)
            self.rng, self._own_rng = NRRng(mix32(int(eng.seed) & 0xFFFFFFFF) ^ 0x6A09E667, E, d), True
        # ---- state ([E, R], reassigned each step; the graph backend re-binds it) ----
        z = lambda v: torch.full((E, R), v, dtype=torch.long, device=d)      # noqa: E731
        self.st = z(self.init_state)
        self.ra_next = z(BIG)         # slot of the next preamble (an RO), BIG = none planned
        self.ra_att = z(0)            # failed attempts of the current procedure
        self.conn_at = z(BIG)         # slot from which a robot in RACH is served
        self.last_act = z(0)          # slot of the last scheduling activity (DRX inactivity, RRC release)
        self.ul_seen = z(0)           # queue stream ends seen at the end of the previous step (arrival detection)
        self.dl_seen = z(0)
        self.ra_step = z(0)           # preamble transmissions this step
        self.sleep_cnt = torch.zeros(E, R, device=d)
        self.vis_cnt = torch.zeros((), device=d)
        self.ctr = {k: torch.zeros((), device=d) for k in CTR_NAMES}
        self._install()

    # ------------------------------------------------------------------ hooks on this engine instance
    def _install(self):
        net = self.net
        for link in (net.ul, net.dl):
            if link is not None:
                link.slot = self._gated(link, link.slot)
        step0 = net.step

        def step(t, *a, **k):
            self.pre(t)
            return self.post(t, step0(t, *a, **k))
        net.step = step
        if net.C > 1:
            cells0 = net.step_cells

            def step_cells(t, *a, **k):
                self.pre(t)
                return self.post(t, cells0(t, *a, **k))
            net.step_cells = step_cells

    def _gated(self, link, slot0):
        def slot(g, *a, **k):
            busy = self._busy(link)
            ok, conn, awake = self._ok(g)
            prev = link.sched_ok
            link.sched_ok = ok if prev is None else prev & ok
            try:
                r = slot0(g, *a, **k)
            finally:
                link.sched_ok = prev
            self.last_act = torch.where(ok & busy, g, self.last_act)
            self.sleep_cnt = self.sleep_cnt + ((self.st == IDLE) | (conn & ~awake)).float()
            self.vis_cnt = self.vis_cnt + 1.0
            return r
        return slot

    # ------------------------------------------------------------------ per-slot masks
    @staticmethod
    def _busy(link):
        """[E,R] the link has data the robot has not sent (or a HARQ process waiting for a retransmission)."""
        return (link.unsent() > 0) | (link.h_state == 1).any(-1)

    def _awake(self, g):
        """DRX Active Time at slot g [E,R] (all True without DRX)."""
        la = self.last_act
        gg = torch.zeros_like(la) + g                 # g: host int (reference) or 0-dim device tensor (graph)
        awake = (gg - la) < self.inact
        if self.scyc is None:
            on = ((gg - self.off) % self.cyc) < self.on
        else:                                         # short cycle for n_short cycles after the timer expires
            short = (gg - (la + self.inact)) < self.n_short * self.scyc
            phase = torch.where(short, (gg - self.off) % self.scyc, (gg - self.off) % self.cyc)
            on = phase < self.on
        awake = awake | on
        if self.ul_wake:
            awake = awake | self._busy(self.net.ul)
        return awake

    def _ok(self, g):
        """(schedulable, connected, awake) [E,R] at slot g."""
        conn = (self.st == CONNECTED) | ((self.st == RACH) & (self.conn_at <= g))
        if not self.drx:
            return conn, conn, conn
        awake = self._awake(g)
        return conn & awake, conn, awake

    def _next_ro(self, x):
        """First RO slot >= x (x an int or a long tensor)."""
        return (x - self.ro_off + self.ro - 1) // self.ro * self.ro + self.ro_off

    # ------------------------------------------------------------------ step
    def _arrivals(self, g0):
        """Slot of the first UL / DL data arrival in the step [E,R] (BIG = none)."""
        net, N = self.net, self.N
        A = torch.full_like(self.st, BIG)
        A = torch.where(net.ul.q.enq > self.ul_seen, g0, A)
        gate = getattr(self.eng, "_gate", None)            # traffic models: arrivals inside the step by slot
        if gate is not None and gate[1].shape[-1] > 0:
            first = gate[1].min(-1).values
            A = torch.where(first < N, torch.minimum(A, g0 + first), A)
        if net.dl is not None:
            A = torch.where(net.dl.q.enq > self.dl_seen, torch.minimum(A, torch.zeros_like(A) + g0), A)
        return A

    def pre(self, t):
        """Before the slots of step t: releases, triggers and every RO of the step."""
        net, N = self.net, self.N
        tv, _ = net._times(t)
        g0 = tv * N
        self.sleep_cnt = torch.zeros_like(self.sleep_cnt)
        self.vis_cnt = torch.zeros_like(self.vis_cnt)
        self.ra_step = torch.zeros_like(self.ra_step)
        if not self.rach:
            return
        st = self.st
        A = self._arrivals(g0)
        if self.release is not None:
            busy0 = self._busy(net.ul) | (self._busy(net.dl) if net.dl is not None else False)
            # released at last_act + release if that comes before the first arrival (or inside the step without one)
            lim = torch.minimum(A, torch.zeros_like(A) + (g0 + N - 1))
            rel = (st == CONNECTED) & ~busy0 & (self.last_act + self.release <= lim)
            self.ctr["rrc_releases"] += rel.sum()
            st = torch.where(rel, IDLE, st)
        trig = (st == IDLE) & (A < BIG)
        st = torch.where(trig, RACH, st)
        self.ra_att = torch.where(trig, 0, self.ra_att)
        self.ra_next = torch.where(trig, self._next_ro(A), self.ra_next)
        self.conn_at = torch.where(trig, BIG, self.conn_at)
        self.st = st
        self._occasions(g0)

    def _occasions(self, g0):
        """Every RO in [g0, g0 + N): preamble draws, collisions per (env, cell, preamble), RAR / backoff."""
        net, N, R, P = self.net, self.N, self.R, self.n_pre
        C = net.C
        cell = net.assoc.serv.clamp(min=0) if C > 1 else None
        first = self._next_ro(g0)
        for k in range(N // self.ro + 1):
            r = first + k * self.ro
            att = (self.st == RACH) & (self.ra_next == r) & (r < g0 + N)
            u = self.rng.uniform(STEP, stream_id(ACCESS_PRE, k), R)
            key = (u * P).long().clamp(max=P - 1)
            if cell is not None:
                key = key + cell * P
            cnt = torch.zeros(self.E, C * P, dtype=torch.long, device=self.dev).scatter_add(1, key, att.long())
            mine = cnt.gather(1, key)
            ok = att & (mine == 1)
            col = att & (mine > 1)
            c = self.ctr
            c["rach_attempts"] += att.sum()
            c["rach_successes"] += ok.sum()
            c["rach_collisions"] += col.sum()
            self.ra_step = self.ra_step + att.long()
            done = r + self.cres
            self.conn_at = torch.where(ok, done, self.conn_at)
            self.last_act = torch.where(ok, done, self.last_act)      # connection = activity (DRX, release)
            self.ra_next = torch.where(ok, BIG, self.ra_next)
            n = self.ra_att + col.long()
            fail = col & (n >= self.max_att)
            c["rach_failures"] += fail.sum()
            self.ra_att = torch.where(fail, 0, n)
            ub = self.rng.uniform(STEP, stream_id(ACCESS_BO, k), R)
            back = (ub * (self.bo + 1)).long().clamp(max=self.bo)
            self.ra_next = torch.where(col, self._next_ro(done + back), self.ra_next)

    def post(self, t, out):
        """After the slots of step t: robots whose connection completed become CONNECTED; step outputs."""
        net, N = self.net, self.N
        tv, _ = net._times(t)
        g_last = tv * N + N - 1
        newly = (self.st == RACH) & (self.conn_at <= g_last)
        self.st = torch.where(newly, CONNECTED, self.st)
        self.conn_at = torch.where(newly, BIG, self.conn_at)
        self.ul_seen = net.ul.q.enq.clone()
        if net.dl is not None:
            self.dl_seen = net.dl.q.enq.clone()
        if self._own_rng:
            self.rng.tick_step()
        if not isinstance(out, dict):
            return out
        state = self.st
        if self.drx:
            state = torch.where((state == CONNECTED) & ~self._awake(g_last), DORMANT, state)
        out["access_state"] = state.clone()
        out["access_sleep_frac"] = self.sleep_cnt / self.vis_cnt.clamp(min=1.0)
        out["rach_attempts"] = self.ra_step.clone()
        return out

    # ------------------------------------------------------------------ reset and counters
    def reset(self, ids):
        """Envs ids (None = all) back to rach_initial with no procedure running; counters only on a full reset."""
        E, d = self.E, self.dev
        m = env_mask(E, ids, d)[:, None]
        g = self.eng.T * self.N                      # first slot of the next step (global clock)
        put = lambda x, v: torch.where(m, v, x)      # noqa: E731
        self.st = put(self.st, self.init_state)
        self.ra_next = put(self.ra_next, BIG)
        self.ra_att = put(self.ra_att, 0)
        self.conn_at = put(self.conn_at, BIG)
        self.last_act = put(self.last_act, g)
        self.ul_seen = put(self.ul_seen, self.net.ul.q.enq)
        if self.net.dl is not None:
            self.dl_seen = put(self.dl_seen, self.net.dl.q.enq)
        self.ra_step = put(self.ra_step, 0)
        self.sleep_cnt = put(self.sleep_cnt, 0.0)
        if self._own_rng:
            self.rng.reset_mask(None if ids is None else env_mask(E, ids, d))
        if ids is None:
            for v in self.ctr.values():
                v.zero_()

    def counters(self):
        return {k: float(v) for k, v in self.ctr.items()}

    def state_names(self):
        return STATE_NAMES


def access_on(cfg):
    """True when the config asks for the access state machine (rach or drx)."""
    return bool(getattr(cfg, "rach", False) or getattr(cfg, "drx", False))


__all__ = ["AccessStage", "IDLE", "RACH", "CONNECTED", "DORMANT", "STATE_NAMES", "access_on"]
