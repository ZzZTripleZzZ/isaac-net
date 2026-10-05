"""Configurable batched NR cell engine: the "L2" fidelity level (merged from nrconfig/).

NRNet runs the slots of one control step for every [E, R] robot at once: fading evolution, DL CQI,
DL data slots (mac_dl.DlMac), SR opportunities and UL data slots (mac_ul.UlMac), then frame
delivery / timeouts / RLC-UM losses per control step. It keeps the legacy netsim.NetSlot API
(add_frames / step / queued / stats / collect) and adds add_dl_frames, step_rx (link budget in)
and partial reset(env_ids). See mac.py for the byte-stream / HARQ model.

Time is one global control-step clock t (a Python int) shared by all envs. engine.NREngine wraps NRNet
in the contract API of ARCHITECTURE.md (submit / step -> dict, per-env episode clocks) and is what
`make_engine("L2", ...)` returns.

Multi-cell (cfg.n_cells = C > 1; the design of proto/netsim_mc.NetSlotMC on this MAC). Input: the large-scale
path gain of every robot-cell link [E,R,C] (step_cells, or poses through the engine's RadioMC). Every cell runs
its own PF scheduler (per RBG) and HARQ over the same carrier (MacLink.member), with a TDD pattern that is
synchronous across cells. Same-slot interference goes through MacLink.sinr_hook: in every UL data slot gNB c
sees the robots the other cells scheduled on the same RBG, at their transmit PSD (power split and fractional
power control) and with the fading of that robot-gNB link; in every DL data slot a robot sees every other gNB
that transmits on the RBG, at gnb_tx_dbm spread over the carrier. Link adaptation (scheduler estimate, MCS, PHR
cap, DL CQI) uses the N+I measured in the previous data slot of that direction (EWMA li_alpha); decoding uses the
actual same-slot N+I; OLLA absorbs the mismatch. The PHR cap on RBGs per UE uses the noise-only SNR (power headroom
does not depend on interference; NetSlotMC uses the SINR there, which starves loaded cells of RBGs). Interference needs noise_model="thermal" (the fixed floor
already contains it) and ul_interference / dl_interference. UL fractional power control (P0 + alpha PL) is on by
default when C > 1. CellAssociation attaches robots at max RSRP after every reset and runs A3 with
hysteresis and time-to-trigger (in slots); a handover fires at its exact slot, moves the robot's MAC state
(MacLink.handover) and blocks scheduling for the interruption. Fading is per link [E,R,C,S,2] and shared by UL
and DL (TDD reciprocity). Shapes are fixed; the only Python loops are the constant slot and RBG loops. At
C = 1 none of this code runs and NRNet is bitwise the single-cell engine.

Randomness (cfg.rng): "engine" draws the fading innovations, the TB decodes and the reset fading from the engine's
counter-based streams (nr_rng.NRRng, keyed by seed, env, episode, step and slot); "global" draws the step noise from
the global torch RNG and the reset fading from `generator` (the behavior before the engine RNG).

Rician fading (cfg.fading_rician, docs/channels.md): the gain of every subband is |sqrt(K/(K+1)) e^{j phi} +
sqrt(1/(K+1)) h|^2 with h the AR(1) Rayleigh state above (unchanged), phi a fixed per-link phase and K per link
[E,R,(C)]: rician_k_db for every link, or (rician_k_from_los) a log-normal draw per link with the mu_K / sigma_K of
TR 38.901 Table 7.5-6 where the radio reports LOS and no blockage and 0 elsewhere. A new LOS state enters through
set_los() (NREngine calls it after the radio's path gain), and step() / step_cells() turn a changed target into a
linear ramp over rician_k_ramp_slots slots that starts at the first slot of that control step; the K of slot g is a
pure function of g and the ramp state (_k_at), which the fused Triton kernel evaluates the same way. With
fading_rician off none of this code runs.

Frequency-correlated fading (cfg.fading_freq_corr, docs/channels.md "Frequency-selective fading"): the AR(1) innovation
z [E,R,(C),S,2] (and the initial state) becomes L z, with L [S,S] a square root of the subband correlation matrix of
an exponential power-delay profile with rms delay spread tau (subband_corr). With fading_delay_spread_ns, L is one
constant; otherwise every link draws a log-normal tau (TR 38.901 Table 7.5-6, LOS and NLOS values, the state from
set_los) at reset and indexes a precomputed table of L on a log grid of tau (fading_ds_grid), so the step stays a
fixed-shape gather and a batched [S,S] matmul. With fading_freq_corr off none of this code runs.

Capture (nr_fast.py): with self._tdev set to a 0-dim long device tensor holding t, every time-dependent value of the
step is computed on the device from it, and host decisions depend only on t through the TDD / SR / CQI schedule key
and the fading step of the first slot, so one captured graph per (key, first fading step) serves every control step.
"""
from __future__ import annotations

import math

import torch

from .config import NRConfig, occasion_symbols
from .mac_dl import DlMac
from .mac_ul import UlMac
from .nr_rng import DSPR, FADING, H0, KFAC, KPHI, make_rng
from .queues import env_mask, onehot, reset_where
from .radio import CellAssociation, pick

# Rician K-factor of LOS links, (mu_K, sigma_K) in dB: TR 38.901 V17.0.0 Table 7.5-6, checked against ETSI TR 138 901
# V17.0.0 (2022-04): Part-1 UMi-Street Canyon 9 / 5, UMa 9 / 3.5; Part-2 RMa 7 / 4, Indoor-Office 7 / 4; Part-3 InF
# 7 / 8. K is defined for LOS only (N/A for NLOS and O2I). The four InF sub-scenarios share one column.
RICIAN_K_DB = {"UMi": (9.0, 5.0), "UMa": (9.0, 3.5), "RMa": (7.0, 4.0), "InH": (7.0, 4.0),
               "InF-SL": (7.0, 8.0), "InF-DL": (7.0, 8.0), "InF-SH": (7.0, 8.0), "InF-DH": (7.0, 8.0)}


def rician_k_params(scenario):
    """(mu_K, sigma_K) dB of a TR 38.901 scenario name (aliases as config.tr38901_scenario accepts)."""
    from .channels.tr38901 import scenario_name
    return RICIAN_K_DB[scenario_name(scenario)]


# rms delay spread lgDS = log10(DS / 1 s), (mu, sigma) per scenario and LOS state, fc in GHz: TR 38.901 V17.0.0 Table
# 7.5-6 Parts 1-3. Checked on 2026-10-05 against Sionna's parameter files of V16.1 (src/sionna/phy/channel/tr38901/
# models/v16_1/*.json; the V17.0.0 rows are those of V16.1) and, for RMa, InH and InF, against the itecspec.com mirror
# of clause 7.5 (Rel-19; those rows are unchanged). Rel-19 (note 8, the 7-24 GHz study) updated UMi and UMa to
# UMi LOS -0.18 lg(1 + fc) - 7.28 / 0.39, NLOS -0.22 lg(1 + fc) - 6.87 / 0.19 lg(1 + fc) + 0.22, UMa LOS
# -7.067 - 0.0794 lg(fc) / 0.57 + 0.026 lg(fc), NLOS -6.47 - 0.134 lg(fc) / 0.39; the V17.0.0 values are kept here,
# as in the rest of the package. Frequency floors (notes 6, 7): UMa and InH use fc = 6 GHz below 6 GHz, UMi fc = 2 GHz
# below 2 GHz. InF: V the hall volume (m^3), S its total surface (walls + floor + ceiling, m^2).
def _lg(x):
    return math.log10(x)


LG_DS = {
    "UMi": lambda fc, los: (-0.24 * _lg(1 + max(fc, 2.0)) - 7.14, 0.38) if los else
    (-0.24 * _lg(1 + max(fc, 2.0)) - 6.83, 0.16 * _lg(1 + max(fc, 2.0)) + 0.28),
    "UMa": lambda fc, los: (-6.955 - 0.0963 * _lg(max(fc, 6.0)), 0.66) if los else
    (-6.28 - 0.204 * _lg(max(fc, 6.0)), 0.39),
    "RMa": lambda fc, los: (-7.49, 0.55) if los else (-7.43, 0.48),
    "InH": lambda fc, los: (-0.01 * _lg(1 + max(fc, 6.0)) - 7.692, 0.18) if los else
    (-0.28 * _lg(1 + max(fc, 6.0)) - 7.173, 0.10 * _lg(1 + max(fc, 6.0)) + 0.055),
}
# InF default hall (length x width x height, m) of TR 38.901 Table 7.8-7 as Sionna implements it (inf_scenario.py,
# _DEFAULT_HALL_DIMENSIONS; not checked against the spec text): SL and DH 120 x 60 x 10 (V/S = 4.0), DL and SH
# 300 x 150 x 10 (V/S = 4.545)
INF_HALL_M = {"InF-SL": (120.0, 60.0, 10.0), "InF-DL": (300.0, 150.0, 10.0), "InF-SH": (300.0, 150.0, 10.0),
              "InF-DH": (120.0, 60.0, 10.0)}


def lg_ds_params(cfg, los):
    """(mu_lgDS, sigma_lgDS) of cfg.tr38901_scenario for the LOS (los=True) or NLOS state at cfg.carrier_ghz."""
    from .channels.tr38901 import scenario_name
    sc = scenario_name(cfg.tr38901_scenario)
    if not sc.startswith("InF"):
        return LG_DS[sc](cfg.carrier_ghz, los)
    sigma = 0.15 if los else 0.19
    if cfg.inf_lg_ds is not None:
        return cfg.inf_lg_ds, sigma
    if cfg.inf_hall_volume_m3 is not None:
        vs = cfg.inf_hall_volume_m3 / cfg.inf_hall_surface_m2
    else:
        a, b, h = INF_HALL_M[sc]
        vs = a * b * h / (2 * (a * b + a * h + b * h))
    return (_lg(26 * vs + 14) - 9.35 if los else _lg(30 * vs + 32) - 9.44), sigma


def subband_centers_hz(cfg):
    """Center frequency of every subband (RBG) relative to the carrier edge, Hz [S] (float64): PRBs per RBG x 12 x
    SCS, the last RBG possibly narrower (cfg.subband_prbs)."""
    w = torch.tensor([float(x) for x in cfg.subband_prbs], dtype=torch.float64) * (12 * cfg.scs_khz * 1e3)
    return torch.cumsum(w, 0) - w / 2


def subband_corr(tau_s, centers_hz):
    """Real correlation matrix [..., S, S] (float64) of the subband fading for rms delay spreads tau_s [...] (s).

    Exponential power-delay profile P(t) = exp(-t / tau) / tau, t >= 0. The complex channel H(f) = sum of the taps
    a_k exp(-j 2 pi f t_k), uncorrelated taps of mean power P(t_k), has the frequency correlation
        rho(df) = E[H(f + df) H*(f)] = int_0^inf P(t) exp(-j 2 pi df t) dt = 1 / (1 + j 2 pi df tau).
    The exact circular model would give the innovation w = x + j y the complex covariance R_ij = rho(f_i - f_j), that
    is, the real 2S x 2S covariance [[Re R, -Im R], [Im R, Re R]] / 2 for (x, y). The engine keeps the real and
    imaginary parts independent and identically correlated, x = L z_x / sqrt 2, y = L z_y / sqrt 2 with one real
    L L^T = Cm, so E[w w^H] = Cm and E[w w^T] = 0 (still circular). We take the magnitude-consistent choice
        Cm_ij = |rho(f_i - f_j)| = 1 / sqrt(1 + (2 pi (f_i - f_j) tau)^2),
    because
      * Cm_ii = 1: every subband keeps unit power, so the AR(1) recursion h <- rho_t h + sqrt(1 - rho_t^2) w stays
        stationary with E|h_s|^2 = 1, and its temporal correlation rho_t^k per subband is unchanged;
      * for circular Gaussian h the power correlation between subbands is corr(|h_i|^2, |h_j|^2) = |E h_i h_j^*|^2,
        here Cm_ij^2 = |rho|^2 = 1 / (1 + (2 pi df tau)^2), the same as the exact complex model; the engine's gain
        uses only |h|^2 (Rayleigh) or |a e^{j phi} + b h|^2 (Rician), so the Rayleigh gains have the exact model's
        joint statistics up to the (unobservable) phase rotation arg rho = -atan(2 pi df tau);
      * Cm is positive definite: 1 / sqrt(1 + (2 pi tau f)^2) is the Fourier transform of a positive function (a
        modified Bessel function K_0, up to a scale), so by Bochner's theorem every Cm is a valid covariance.
    With Rician fading the cross term Re(e^{-j phi} h) correlates across subbands as Cm_ij / 2 instead of the exact
    Re(rho) / 2, slightly more correlated (Re rho = 1 / (1 + (2 pi df tau)^2) <= Cm_ij)."""
    df = centers_hz[:, None] - centers_hz[None, :]
    x = 2 * math.pi * df * torch.as_tensor(tau_s, dtype=torch.float64)[..., None, None]
    return 1.0 / torch.sqrt(1.0 + x * x)


def corr_sqrt(c):
    """Lower-triangular L [..., S, S] with L L^T = c and unit row norms (unit diagonal of L L^T exactly, in float64):
    the Cholesky factor, with a diagonal jitter for nearly singular c (delay spreads far below 1 / bandwidth, where
    all subbands are almost fully correlated) and rows renormalized after it."""
    eye = torch.eye(c.shape[-1], dtype=c.dtype)
    for eps in (0.0, 1e-12, 1e-10, 1e-8, 1e-6):
        L, info = torch.linalg.cholesky_ex(c + eps * eye)
        if not bool((info != 0).any()):
            return L / L.norm(dim=-1, keepdim=True)
    raise ValueError("subband correlation matrix is not positive definite")


class NRNet:
    """Drop-in replacement for netsim.NetSlot (same add_frames / step / queued / stats API) with an
    optional downlink (add_dl_frames, dl_newest) and a link-budget entry point (step_rx)."""

    FEATS = ["cls", "f_nact", "f_snr", "f_own"]

    def __init__(self, E, R, device, sizes, cfg: NRConfig | None = None, generator=None, seed=None):
        self.cfg = cfg or NRConfig()
        self.C = self.cfg.n_cells
        self.gen = generator       # rng="global": draws of reset() (fading state); stepping uses the global RNG
        self.rng = make_rng(self.cfg, E, device, seed)     # rng="engine": every draw (nr_rng.py)
        self._tdev = None          # graph capture: 0-dim long device tensor holding t (see the module docstring)
        self.fading_rho_ms = None  # optional per-robot AR(1) fading correlation per ms [E,R] (per-robot Doppler,
                                   # channels.doppler); None = the global cfg.fading_rho_per_ms
        self.E, self.R, self.dev = E, R, device
        self.rician = self.cfg.rician_mode                # None | "fixed" | "los" (config.NRConfig.rician_mode)
        self.k_ramp = self.cfg.rician_k_ramp_slots if self.rician == "los" else 0
        self.fc_mode = self.cfg.freq_corr_mode           # None | "fixed" | "los" (config.NRConfig.freq_corr_mode)
        self.sizes = torch.tensor(sizes, device=device, dtype=torch.float32)
        self.S = self.cfg.n_subbands
        meta = [("cls", torch.long), ("det", torch.bool), ("hid", torch.long), ("f_nact", torch.long),
                ("f_snr", torch.float32), ("f_own", torch.long)]
        self.ul = UlMac(self.cfg, E, R, device, meta)
        self.dl = DlMac(self.cfg, E, R, device, [("cls", torch.long)]) if self.cfg.dl else None
        for link in (self.ul, self.dl):
            if link is not None:
                link.rng = self.rng
                if link.mimo:          # rank decision input: the serving link's Rician K (mac.MacLink._rank)
                    link.rank_k = self._rank_k
        self.log_stats = False
        self.log_cap_max = 10 ** 9
        self._sched_cache = {}
        self._occ_cache = {}
        self.trace_frames = None
        self.trace_frames_dl = None
        self.log_sinr = False      # multi-cell: per-TB SINR samples in self.sinr_log (host copies; sweeps only)
        if self.fc_mode:
            self._init_fcorr()
        if self.C > 1:
            self._init_cells()
        elif self.cfg.rlf:
            raise ValueError("rlf=True needs n_cells > 1: radio link failure and re-establishment live in the "
                             "multi-cell association (CellAssociation), which the single-cell engine does not run")
        self.reset()

    def _init_cells(self):
        cfg, C, d = self.cfg, self.C, self.dev
        self.assoc = CellAssociation(cfg, self.E, self.R, C, d, cfg.slots_per_step, slot_ms=cfg.slot_ms, rlf=cfg.rlf)
        thermal = cfg.noise_model == "thermal"
        self.use_int = {"ul": thermal and cfg.ul_interference, "dl": thermal and cfg.dl_interference}
        self.noise_mw = {"ul": 10 ** (cfg.noise_dbm_per_prb("gnb") / 10), "dl": 10 ** (cfg.noise_dbm_per_prb("ue") / 10)}
        self.ref_db = 10 * math.log10(cfg.snr_ref_prbs)
        self.dl_psd_db = cfg.gnb_tx_dbm - 10 * math.log10(cfg.nprb)      # gNB transmit power per PRB
        self._ici = {"ul": self._ul_ici, "dl": self._dl_ici}
        self._user_hook = {"ul": None, "dl": None}
        for link in (self.ul, self.dl):
            if link is not None:
                link.n_cells = C
                link.sinr_hook = self._ici[link.dir] if self.use_int[link.dir] else None
        self.sinr_log = []

    def set_sinr_hook(self, fn, direction="ul"):
        """Install fn(g, dir, won, n_prb, sinr) -> sinr on that direction. With several cells it runs after the
        engine's own inter-cell interference (fn=None removes it)."""
        link = self.ul if direction == "ul" else self.dl
        if self.C == 1:
            link.sinr_hook = fn
            return
        self._user_hook[direction] = fn
        own = self._ici[direction] if self.use_int[direction] else None
        if own is None or fn is None:
            link.sinr_hook = own or fn
        else:
            link.sinr_hook = lambda g, dr, won, n_prb, act: fn(g, dr, won, n_prb, own(g, dr, won, n_prb, act))

    @property
    def cap(self):
        return self.ul.q.cap

    def reset(self, env_ids=None):
        """Full reset (env_ids None) or partial reset of the given envs: queues, HARQ, SR/BSR, OLLA,
        PF averages, CSI and fading of those envs return to their initial state. The slot clock is
        global (all envs share t); statistics and counters are cleared only by a full reset."""
        E, R, S, d = self.E, self.R, self.S, self.dev
        self.ul.reset(env_ids)
        if self.dl is not None:
            self.dl.reset(env_ids)
        cdim = () if self.C == 1 else (self.C,)
        if self.rng is None:
            h0 = torch.randn(E, R, *cdim, S, 2, device=d, generator=self.gen) / math.sqrt(2)
        else:              # new episode for the reset envs, then their draws (rows of other envs are discarded)
            self.rng.reset_mask(None if env_ids is None else env_mask(E, env_ids, d))
            h0 = self.rng.reset_normal_all(H0, R * math.prod(cdim) * S * 2).view(E, R, *cdim, S, 2) / math.sqrt(2)
        if self.fc_mode:                    # stationary start: the initial state has the subband correlation too
            self._reset_fcorr(env_ids, cdim)
            h0 = self._fcorr(h0)
        if env_ids is None:
            self.h = h0
            self.last_g = None
            self.dl_newest = torch.full((E, R), -1, dtype=torch.long, device=d)
        else:
            m = env_mask(E, env_ids, d)
            self.h = reset_where(self.h, m, h0)
            self.dl_newest = reset_where(self.dl_newest, m, -1)
        if self.rician:
            self._reset_rician(env_ids, cdim)
        if self.C > 1:
            self._reset_cells(env_ids)
        if not hasattr(self, "stats"):
            self.clear_stats()

    def _reset_rician(self, env_ids, cdim):
        """Per-link Rician state of the reset envs: unit phasor e^{j phi} of the specular term [E,R,(C),2] with
        phi ~ U(0, 2 pi), the LOS K draw k_los (rician_k_db, or 10^(N(mu_K, sigma_K) / 10)), and the ramp state
        k_in (input), k_lin (target), k_from (ramp start) and k_g0 (slot of the ramp start). Engine RNG: reset draws
        keyed by (seed, env, episode) (sites KPHI, KFAC); rng="global": the engine's generator."""
        cfg, E, R, d = self.cfg, self.E, self.R, self.dev
        n = R * math.prod(cdim)
        if self.rng is None:
            u = torch.rand(E, n, device=d, generator=self.gen)
            z = torch.randn(E, n, device=d, generator=self.gen)
        else:
            u = self.rng.reset_uniform_all(KPHI, n)
            z = self.rng.reset_normal_all(KFAC, n)
        phi = (2 * math.pi) * u.view(E, R, *cdim)
        spec = torch.stack((torch.cos(phi), torch.sin(phi)), -1)
        if self.rician == "fixed":
            k_los = torch.full((E, R, *cdim), 10 ** (cfg.rician_k_db / 10), device=d)
            k0 = k_los.clone()
        else:
            mu, sigma = rician_k_params(cfg.tr38901_scenario)
            k_los = 10 ** ((mu + sigma * z.view(E, R, *cdim)) / 10)
            k0 = torch.zeros_like(k_los)              # no LOS state yet: K = 0 until set_los()
        g0 = torch.zeros((E, R, *cdim), dtype=torch.long, device=d)
        fresh = torch.ones(E, dtype=torch.bool, device=d)
        if env_ids is None:
            # distinct tensors: the graph backend makes each state attribute's tensor its persistent buffer
            self.spec, self.k_los, self.k_g0 = spec, k_los, g0
            self.k_in, self.k_lin, self.k_from = k0.clone(), k0.clone(), k0.clone()
            if self.rician == "los":
                self.k_fresh = fresh
            return
        m = env_mask(E, env_ids, d)
        self.spec = reset_where(self.spec, m, spec)
        self.k_los = reset_where(self.k_los, m, k_los)
        for name in ("k_in", "k_lin", "k_from"):
            setattr(self, name, reset_where(getattr(self, name), m, k0))
        self.k_g0 = reset_where(self.k_g0, m, g0)
        if self.rician == "los":
            self.k_fresh = self.k_fresh | m

    def _init_fcorr(self):
        """Square roots of the subband correlation (subband_corr): fixed delay spread -> fc_L [S,S]; per-link
        delay spread -> fc_tab [G,S,S] on the log grid fc_grid_ns of fading_ds_grid (G points)."""
        cfg = self.cfg
        f = subband_centers_hz(cfg)
        if self.fc_mode == "fixed":
            self.fc_L = corr_sqrt(subband_corr(cfg.fading_delay_spread_ns * 1e-9, f)).float().to(self.dev).contiguous()
            return
        lo, hi, G = cfg.fading_ds_grid
        self.fc_grid_ns = torch.logspace(math.log10(lo), math.log10(hi), int(G), dtype=torch.float64)
        self.fc_tab = corr_sqrt(subband_corr(self.fc_grid_ns * 1e-9, f)).float().to(self.dev).contiguous()
        self.fc_lgds = (lg_ds_params(cfg, False), lg_ds_params(cfg, True))       # ((mu, sigma) NLOS, LOS)

    def _ds_index(self, lg):
        """Nearest grid point (in log tau) of lgDS = log10(tau / 1 s) values, clamped to the grid ends."""
        lo, hi, G = self.cfg.fading_ds_grid
        u = (lg + 9.0 - math.log10(lo)) * ((int(G) - 1) / (math.log10(hi) - math.log10(lo)))
        return torch.round(u).clamp(0, int(G) - 1).long()

    def _reset_fcorr(self, env_ids, cdim):
        """Per-link delay spread of the reset envs (fc_mode "los"): two independent normals per link (site DSPR, keyed
        by (seed, env, episode); rng="global": the engine's generator), lgDS = mu + sigma z for the NLOS and LOS
        state, kept as grid indices fc_idx_nlos / fc_idx_los [E,R,(C)]; the live index fc_idx starts at the NLOS
        value until set_los delivers a LOS state."""
        if self.fc_mode != "los":
            return
        E, R, d = self.E, self.R, self.dev
        n = R * math.prod(cdim) * 2
        if self.rng is None:
            z = torch.randn(E, n, device=d, generator=self.gen)
        else:
            z = self.rng.reset_normal_all(DSPR, n)
        z = z.view(E, R, *cdim, 2)
        (mn, sn), (ml, sl) = self.fc_lgds
        i_n, i_l = self._ds_index(mn + sn * z[..., 0]), self._ds_index(ml + sl * z[..., 1])
        if env_ids is None:
            self.fc_idx_nlos, self.fc_idx_los, self.fc_idx = i_n, i_l, i_n.clone()
            return
        m = env_mask(E, env_ids, d)
        self.fc_idx_nlos = reset_where(self.fc_idx_nlos, m, i_n)
        self.fc_idx_los = reset_where(self.fc_idx_los, m, i_l)
        self.fc_idx = reset_where(self.fc_idx, m, i_n)

    def _fcorr(self, z):
        """Correlate z [E,R,(C),S,2] across the subbands: L z per link (real and imaginary parts alike)."""
        L = self.fc_L if self.fc_mode == "fixed" else self.fc_tab[self.fc_idx]
        return L @ z

    def set_los(self, los, blocked=None):
        """LOS state of every link (bool [E,R,C], or [E,R] with one cell) and optionally its blockage: the Rician K
        target becomes the link's LOS draw k_los where LOS and not blocked, else 0. Eager (before the step); the
        next step() / step_cells() ramps K to it. Right after a reset an env's first LOS state applies at once.
        Per-link delay spread (freq_corr_mode "los"): the link's LOS or NLOS draw, NLOS where blocked; applies from
        the next slot (no ramp: the AR(1) state carries the change over the coherence time).
        No-op unless rician_mode == "los" or freq_corr_mode == "los"."""
        if self.fc_mode == "los":
            fon = los if blocked is None else los & ~blocked
            if self.C == 1 and fon.dim() == 3:
                fon = fon[..., 0]
            self.fc_idx = torch.where(fon, self.fc_idx_los, self.fc_idx_nlos)
        if self.rician != "los":
            return
        on = los if blocked is None else los & ~blocked
        if self.C == 1 and on.dim() == 3:
            on = on[..., 0]
        k = torch.where(on, self.k_los, torch.zeros_like(self.k_los))
        fr = self.k_fresh.view(-1, *([1] * (k.dim() - 1)))
        self.k_in = k
        self.k_lin = torch.where(fr, k, self.k_lin)
        self.k_from = torch.where(fr, k, self.k_from)
        self.k_fresh = torch.zeros_like(self.k_fresh)

    def _rank_k(self):
        """Rician K target (linear) of every robot's serving link [E,R] for the rank rule, None without Rician fading
        (K = 0 there: the Rayleigh channel is rich scattering)."""
        k = getattr(self, "k_lin", None)
        if k is None or self.C == 1:
            return k
        return pick(k, self.assoc.serv)

    def _k_at(self, gv=None):
        """Rician K of every link [E,R,(C)] in slot gv (host int or 0-dim device long; None = the target)."""
        if self.k_ramp == 0 or gv is None:
            return self.k_lin
        n = gv - self.k_g0 + 1
        return torch.where(n >= self.k_ramp, self.k_lin,
                           self.k_from + (self.k_lin - self.k_from) * (n * (1.0 / self.k_ramp)))

    def _rician_update(self, g0v):
        """Start of a control step (first slot g0v): links whose input K differs from the target start a ramp from
        their K of the previous slot."""
        if self.k_ramp == 0:
            self.k_lin = self.k_in
            return
        ch = self.k_in != self.k_lin
        self.k_from = torch.where(ch, self._k_at(g0v - 1), self.k_from)
        self.k_g0 = torch.where(ch, g0v, self.k_g0)
        self.k_lin = self.k_in

    # per-env multi-cell state (besides h and the association) and its reset value
    CELL_INIT = {"ni_ul": "ul", "ni_dl": "dl"}

    def _reset_cells(self, env_ids):
        """Association (re-attach at the next step), measured N+I back to the noise floor; the interference
        statistics are global and cleared only by a full reset."""
        E, R, C, S, d = self.E, self.R, self.C, self.S, self.dev
        self.assoc.reset(env_ids)
        if env_ids is None:
            self.ni_ul = torch.full((E, C, S), self.noise_mw["ul"], device=d)     # per PRB (mW) at each gNB
            self.ni_dl = torch.full((E, R, S), self.noise_mw["dl"], device=d)     # per PRB (mW) at each robot
            self.ioN = {k: torch.zeros(E, device=d) for k in ("ul", "dl")}        # sum of I/N over RBG-slots
            self.ioN_n = {"ul": 0, "dl": 0}
            self._pg = torch.zeros(E, R, C, device=d)
        else:
            m = env_mask(E, env_ids, d)
            self.ni_ul = reset_where(self.ni_ul, m, self.noise_mw["ul"])
            self.ni_dl = reset_where(self.ni_dl, m, self.noise_mw["dl"])

    def clear_stats(self):
        self.stats = {"delay": [], "overflow": 0, "dl_delay": [], "dropped": 0, "late": 0, "discarded": 0,
                      "d_env": [], "x_env": []}
        self.refused_env = torch.zeros(self.E, dtype=torch.long, device=self.dev)   # overflow + PDCP discard
        for f in self.FEATS:
            self.stats["d_" + f] = []
            self.stats["x_" + f] = []

    def queued(self):
        return self.ul.q.count()

    def air_bytes(self, size):
        """Application bytes -> bytes on the air (per-packet overhead)."""
        c = self.cfg
        if c.pkt_overhead_bytes == 0:
            return size
        return size + torch.ceil(size / c.pkt_payload_bytes) * c.pkt_overhead_bytes

    def _admit(self, link, t, want):
        """5G-LENA-style PDCP discard: refuse arriving frames while the head-of-line frame is stale."""
        if self.cfg.discard != "pdcp_arrival":
            return want
        q = link.q
        stale = (q.cap[..., 0] >= 0) & ((t - q.cap[..., 0]) >= self.cfg.timeout_steps)
        return want & ~stale

    def add_frames(self, t, send, det, hid, snr_db):
        """Enqueue one UL frame of class send [E,R] (0 = none) at capture step t; returns accepted [E,R]."""
        q = self.ul.q
        count = q.count()
        want = send > 0
        adm = self._admit(self.ul, t, want)
        if self.log_stats:
            self.stats["overflow"] += int((adm & (count >= q.F)).sum())
            self.stats["discarded"] += int((want & ~adm).sum())
        size = self.air_bytes(self.sizes[(send - 1).clamp(min=0)])
        nact = (q.cap >= 0).any(-1).sum(-1)
        acc, i, oh = q.add(t, adm, size)
        if self.log_stats:
            self.refused_env += (want & ~acc).sum(-1)
        put = lambda name, v: setattr(q, name, torch.where(oh, v[..., None].to(getattr(q, name).dtype), getattr(q, name)))
        put("cls", send)
        put("det", det)
        put("hid", hid[:, None].expand(-1, self.R))
        put("f_nact", nact[:, None].expand(-1, self.R))
        put("f_snr", snr_db if snr_db.dim() == 2 else snr_db.mean(-1))
        put("f_own", i)
        return acc

    def add_dl_frames(self, t, nbytes, cls=None):
        """nbytes [E,R] application bytes, 0 = no frame."""
        assert self.dl is not None, "cfg.dl is False"
        want = self._admit(self.dl, t, nbytes > 0)
        acc, i, oh = self.dl.q.add(t, want, self.air_bytes(nbytes))
        if cls is not None:
            self.dl.q.cls = torch.where(oh, cls[..., None], self.dl.q.cls)
        if self.log_stats:
            self.stats["overflow_dl"] = self.stats.get("overflow_dl", 0) + int(((nbytes > 0) & ~acc).sum())

    def _qos_prepare(self, tf):
        """scheduler="qos": class-ordered queues and class weights of both links for this step (MacLink.qos_prepare)."""
        for lk in (self.ul, self.dl):
            if lk is not None and getattr(lk, "qos", False):
                lk.qos_prepare(tf)

    # ---- slot schedule ----
    def _schedule(self, g0):
        cfg = self.cfg
        P = len(cfg.tdd_pattern)
        key = g0 % math.lcm(P, cfg.sr_period_slots, cfg.cqi_period_slots)
        if key not in self._sched_cache:
            out = []
            for rel in range(cfg.slots_per_step):
                g = g0 + rel
                dls, uls = cfg.slot_symbols(g)
                sr = cfg.first_ul_in_window(g, cfg.sr_period_slots)
                cqi = cfg.first_ul_in_window(g, cfg.cqi_period_slots)
                ack = cfg.next_ul_capable(g + cfg.k1) - g0
                dls = dls if cfg.dl else 0
                uls = uls if cfg.ul else 0
                if dls or uls or (cqi and cfg.dl) or (sr and cfg.ul):
                    out.append((rel, dls, uls, sr and cfg.ul, cqi and cfg.dl, ack))
            self._sched_cache[key] = out
        return self._sched_cache[key]

    def _occasions(self, d, nsym):
        """Scheduling occasions of a data slot of direction d ("ul" / "dl") with nsym data symbols: a tuple of
        (symbols, end, j), end = completion time of the occasion as a fraction of the slot (None = the slot end, the
        whole-slot path) and j = occasion index. Occasions fill the data symbols back to back and the last one ends
        at the slot end (as a whole-slot TB), so occasion j ends sum(symbols of the later occasions) / 14 slots
        earlier. Without mini-slots: ((nsym, None, 0),)."""
        key = (d, nsym)
        occ = self._occ_cache.get(key)
        if occ is None:
            cfg = self.cfg
            m = cfg.ul_mini_slot_symbols if d == "ul" or cfg.mini_slot_dl else None
            sym = occasion_symbols(nsym, m)
            if len(sym) == 1:
                occ = ((nsym, None, 0),)
            else:
                occ = tuple((L, 1.0 - sum(sym[j + 1:]) / 14.0, j) for j, L in enumerate(sym))
            self._occ_cache[key] = occ
        return occ

    def _occ_slot(self, link, d, nsym, gv, tf, rel, N, *args, gh):
        """MacLink.slot of every scheduling occasion of this data slot (one call without mini-slots, bitwise the
        whole-slot call). Occasion j completes at slot rel + end_j and draws its TB decodes from the engine RNG slot
        key rel + j N (j = 0 keeps the whole-slot key; the keys stay distinct over the step). The occasions share
        the slot's timing: link adaptation uses the OLLA offset of the slot start in every occasion (the HARQ
        feedback of an occasion moves it from the next slot on), and the UL side freezes its BSR and CSI the same
        way (UlMac.occ_begin)."""
        occ = self._occasions(d, nsym)
        if len(occ) == 1:
            link.slot(gv, self._frac(tf, rel, N), nsym, *args, gh=gh, rel=rel)
            return
        ul = d == "ul"
        if ul:
            link.occ_begin()
        olla0 = link.olla
        for ns, end, j in occ:
            if j:
                seq, link.olla = link.olla, olla0
            link.occ_share = ns / nsym
            link.slot(gv, tf + (rel + end) / N, ns, *args, gh=gh, rel=rel + j * N)
            if j:                                  # this occasion's OLLA steps, applied after the earlier ones
                link.olla = (seq + (link.olla - olla0)).clamp(-10, 10)
        link.occ_share = 1.0
        if ul:
            link.occ_end()

    def _evolve(self, g, rel=0):
        """AR(1) fading step to slot g (host int); rel = slot index inside the control step (engine RNG stream).
        With self.fading_rho_ms [E,R] set (per-robot Doppler) every robot uses its own correlation per ms."""
        if not self.cfg.fading:
            return
        dt = 1 if self.last_g is None else g - self.last_g
        if dt > 0:
            if self.rng is None:
                z = torch.randn_like(self.h)
            else:
                z = self.rng.step_normal(FADING, rel, self.h[0].numel()).view(self.h.shape)
            if self.fc_mode:
                z = self._fcorr(z)
            if self.fading_rho_ms is None:
                rho = self.cfg.fading_rho_per_ms ** (dt * self.cfg.slot_ms)
                self.h = rho * self.h + math.sqrt(1 - rho ** 2) * z / math.sqrt(2)
            else:
                r = self.fading_rho_ms
                rho = (r ** (dt * self.cfg.slot_ms)).reshape(*r.shape, *([1] * (self.h.dim() - 2)))
                self.h = rho * self.h + torch.sqrt(1 - rho ** 2) * z / math.sqrt(2)
        self.last_g = g

    def _times(self, t):
        """(t as used by device ops, t as float64 or float) for the step at control step t (host int)."""
        tv = t if self._tdev is None else self._tdev
        return tv, (tv if self._tdev is None else tv.double())

    @staticmethod
    def _frac(tf, rel, N):
        """Completion time of slot rel of the step, in control steps (float64 on the device when captured)."""
        return tf + (rel + 1) / N

    def _gain(self, gv=None):
        """Fading gain (dB) per subband [E,R,(C),S] in slot gv (gv: Rician ramp only; None = its target)."""
        if not self.cfg.fading:
            return torch.zeros(self.h.shape[:-1], device=self.dev)
        if not self.rician:
            return 10 * torch.log10((self.h ** 2).sum(-1).clamp(min=1e-6))
        k = self._k_at(gv)[..., None, None]
        inv = 1 / (k + 1)
        x = torch.sqrt(k * inv) * self.spec[..., None, :] + torch.sqrt(inv) * self.h
        return 10 * torch.log10((x ** 2).sum(-1).clamp(min=1e-6))

    def step_rx(self, t, pathgain_db, cur_hid=None, ul_interf_dbm_prb=None, dl_interf_dbm_prb=None,
                dl_pathgain_db=None, full=False):
        """Link-budget entry point (for multicell/): pathgain_db [E,R] or [E,R,S] (negative, incl.
        shadowing), optional interference PSDs in dBm per PRB [E,R,S] at the gNB (UL) / UE (DL).
        Noise from cfg.noise_model. Builds the SINR inputs of step()."""
        c = self.cfg
        pg = pathgain_db if pathgain_db.dim() == 3 else pathgain_db[..., None].expand(-1, -1, self.S)
        n_ul = torch.full_like(pg, c.noise_dbm_per_prb("gnb"))
        if ul_interf_dbm_prb is not None:
            n_ul = 10 * torch.log10(10 ** (n_ul / 10) + 10 ** (ul_interf_dbm_prb / 10))
        ul_ref = c.ue_tx_dbm - 10 * math.log10(c.snr_ref_prbs) + pg - n_ul
        dl = None
        if self.dl is not None:
            dpg = pg if dl_pathgain_db is None else (dl_pathgain_db if dl_pathgain_db.dim() == 3
                                                     else dl_pathgain_db[..., None].expand(-1, -1, self.S))
            n_dl = torch.full_like(dpg, c.noise_dbm_per_prb("ue"))
            if dl_interf_dbm_prb is not None:
                n_dl = 10 * torch.log10(10 ** (n_dl / 10) + 10 ** (dl_interf_dbm_prb / 10))
            dl = c.gnb_tx_dbm - 10 * math.log10(c.nprb) + dpg - n_dl
        return self.step(t, ul_ref, cur_hid, dl, full=full)

    def step(self, t, snr_db, cur_hid=None, dl_snr_db=None, full=False):
        """Advance [t, t+1). snr_db: [E,R] or per-subband [E,R,S] UL SINR if the full UE power were
        spread over snr_ref_prbs PRBs (legacy env: full-power SNR over one 10-PRB subband).
        dl_snr_db: per-PRB DL SINR, same shapes; default snr_db + dl_snr_offset_db.
        Returns (newest delivered UL capture [E,R], detection delivered [E]) like NetSlot.step;
        DL results in self.dl_newest. full=True returns a dict instead, which adds the per-frame masks and
        times of the frames as queued before the step (see _finish)."""
        cfg = self.cfg
        N = cfg.slots_per_step
        g0 = t * N
        assert self.C == 1, "several cells: use step_cells(t, pathgain [E,R,C]) (or poses through NREngine)"
        tv, tf = self._times(t)
        g0v = tv * N
        self._qos_prepare(tf)
        ul_ref = snr_db if snr_db.dim() == 3 else snr_db[..., None].expand(-1, -1, self.S)
        if cfg.ul_pc_on:           # one cell with ul_pc=True: path loss from the input SNR
            self.ul.pc_backoff = self._pc_backoff(ul_ref.mean(-1) + cfg.subband_noise_dbm)
        if self.dl is not None:
            dref = dl_snr_db if dl_snr_db is not None else snr_db + cfg.dl_snr_offset_db
            dl_ref = dref if dref.dim() == 3 else dref[..., None].expand(-1, -1, self.S)
        if self.rician:
            self._rician_update(g0v)
        for rel, dls, uls, sr, cqi, ack in self._schedule(g0):
            g, gv = g0 + rel, g0v + rel
            self._evolve(g, rel)
            gain = self._gain(gv) if self.rician else self._gain()
            if cqi:
                self.dl.cqi_report(dl_ref, gain)
            if dls:
                self._occ_slot(self.dl, "dl", dls, gv, tf, rel, N, dl_ref, gain, g0v + ack, gh=g)
            if sr:
                self.ul.sr_step(gv)
            if uls:
                self._occ_slot(self.ul, "ul", uls, gv, tf, rel, N, ul_ref, gain, 0, gh=g)
        return self._finish(tv, cur_hid, full)

    # ---- several cells ----
    def _pc_backoff(self, rx_serv):
        """Fractional UL power control: backoff (dB) from the full power over snr_ref_prbs PRBs so that this PSD
        is P0 + alpha PL (P0 per snr_ref_prbs PRBs); rx_serv = serving RSRP at full power [E,R]."""
        c = self.cfg
        return c.ue_tx_dbm - (c.ul_pc_p0_dbm + c.ul_pc_alpha * (c.ue_tx_dbm - rx_serv))

    def _handover(self, ho, target, g):
        self.assoc.switch(ho, target, g)
        flush = self.cfg.ho_rlc == "flush"
        self.ul.handover(ho, flush)
        if self.dl is not None:
            self.dl.handover(ho, flush)

    def _rlf_links(self, m, flush):
        """RLF declaration (flush per rlf_rlc), re-establishment (flush=False: per-cell MAC state starts afresh at the
        new cell) or idle (flush=True) of the robots m [E,R]: the MAC side of a handover."""
        self.ul.handover(m, flush)
        if self.dl is not None:
            self.dl.handover(m, flush)

    def _rlf_events(self, g, k_ho, fired, g_ho, rx, sinr_c):
        """RLF declarations and re-establishment completions due by slot g, before the handovers of that slot. A T310
        expiry that a handover precedes (or ties) is cancelled by that handover (switch() stops T310). Returns (the
        handover candidates up to slot g without the robots in RLF, the robots declared now): a handover planned in
        this step for a robot that failed first never fires, even if its re-establishment completes in the step."""
        asc = self.assoc
        ho = (k_ho >= 0) & (g_ho <= g) & ~fired
        exp = asc.expired(g) & ~(ho & (g_ho <= asc.t310_end))
        asc.declare(exp, rx, sinr_c)
        self._rlf_links(exp, self.cfg.rlf_flush)
        self._rlf_links(asc.complete(g), False)
        return ho & ~asc.rlf_active & ~exp, exp

    def _cells_sinr_db(self):
        """serving_sinr_db toward every cell [E,R,C] (each gNB's latest wideband N+I estimate): the RLF cell search."""
        ni = 10 * torch.log10(self.ni_ul.mean(-1))[:, None, :]
        return self._pg + self.cfg.ue_tx_dbm - self.ref_db - ni

    def serving_sinr_db(self):
        """Full-power SINR over snr_ref_prbs PRBs on the serving link against the gNB's latest wideband N+I
        estimate [E,R] (the multi-cell counterpart of the SNR input; observation and frame feature)."""
        ni = self.ni_ul.gather(1, self.assoc.serv[..., None].expand(-1, -1, self.S)).mean(-1)
        return pick(self._pg, self.assoc.serv) + self.cfg.ue_tx_dbm - self.ref_db - 10 * torch.log10(ni)

    def geometry_db(self):
        return self.assoc.geometry_db(self._pg)

    def step_cells(self, t, pathgain_db, cur_hid=None, full=False):
        """Advance [t, t+1) with several cells. pathgain_db [E,R,C]: large-scale gain of every robot-gNB link
        (negative dB, incl. shadowing; reciprocal, so it serves UL and DL). Returns what step() returns; the
        full dict adds serving_cell [E,R]."""
        cfg, C, S = self.cfg, self.C, self.S
        N = cfg.slots_per_step
        g0 = t * N
        tv, tf = self._times(t)
        g0v = tv * N
        self._qos_prepare(tf)
        self._pg = pathgain_db
        rx = pathgain_db + cfg.ue_tx_dbm                      # RSRP up to a constant: full UE power, no fading
        asc = self.assoc
        asc.associate(rx)
        rlf = asc.rlf
        if rlf:                                               # radio link monitoring once per step (CellAssociation)
            sinr_c = self._cells_sinr_db()
            self._rlf_links(asc.rlm(self.serving_sinr_db(), rx, sinr_c, g0v), True)
        k_ho, tgt = asc.plan(rx)
        g_ho = g0v + k_ho
        fired = torch.zeros(self.E, self.R, dtype=torch.bool, device=self.dev)
        links = [x for x in (self.ul, self.dl) if x is not None]
        if self.rician:
            self._rician_update(g0v)
        for rel, dls, uls, sr, cqi, ack in self._schedule(g0):
            g, gv = g0 + rel, g0v + rel
            if rlf:
                ho, failed = self._rlf_events(gv, k_ho, fired, g_ho, rx, sinr_c)
                fired = fired | failed
            else:
                ho = (k_ho >= 0) & (k_ho <= rel) & ~fired      # A3 triggers up to this slot switch now
            self._handover(ho, tgt, g_ho)
            fired = fired | ho
            self._evolve(g, rel)
            gain_c = self._gain(gv) if self.rician else self._gain()      # [E,R,C,S]
            self._gain_c = gain_c
            serv = asc.serv
            gain = pick(gain_c, serv)
            member = onehot(serv, C).permute(0, 2, 1)         # [E,C,R]
            ok = asc.schedulable(gv)
            for link in links:
                link.member, link.sched_ok = member, ok
            pg_s = pick(pathgain_db, serv)[..., None]
            if cqi or dls:
                self._ni_la_dl = 10 * torch.log10(self.ni_dl)
                dl_ref = self.dl_psd_db + pg_s - self._ni_la_dl
            if cqi:
                self.dl.cqi_report(dl_ref, gain)
            if dls:
                self._occ_slot(self.dl, "dl", dls, gv, tf, rel, N, dl_ref, gain, g0v + ack, gh=g)
            if sr:
                self.ul.sr_step(gv)
            if uls:
                self._ni_la_ul = 10 * torch.log10(self.ni_ul.gather(1, serv[..., None].expand(-1, -1, S)))
                ul_ref = cfg.ue_tx_dbm - self.ref_db + pg_s - self._ni_la_ul
                rx_s = pg_s[..., 0] + cfg.ue_tx_dbm
                self.ul.phr_snr = rx_s - cfg.subband_noise_dbm
                if cfg.ul_pc_on:
                    self.ul.pc_backoff = self._pc_backoff(rx_s)
                self._occ_slot(self.ul, "ul", uls, gv, tf, rel, N, ul_ref, gain, 0, gh=g)
        if rlf:
            late = self._rlf_events(g0v + N - 1, k_ho, fired, g_ho, rx, sinr_c)[0]
        else:
            late = (k_ho >= 0) & ~fired                       # triggers after the last active slot of the step
        self._handover(late, tgt, g_ho)
        out = self._finish(tv, cur_hid, full)
        if full:
            out["serving_cell"] = asc.serv.clone()
            if rlf:
                out["rlf"] = asc.rlf_active.clone()
        return out

    def _log_sinr(self, d, won, act, i_db):
        """Per-TB mean SINR with and without the inter-cell interference, and the RBG count (host copy)."""
        n = won.sum(-1)
        tx = n > 0
        nf = n.clamp(min=1).float()
        sinr = (act * won).sum(-1) / nf
        snr = ((act + i_db) * won).sum(-1) / nf
        self.sinr_log.append((d, torch.stack([sinr, snr, n.float(), self.geometry_db().clamp(max=99.0)], -1)[tx].cpu()))

    def _ul_ici(self, g, d, won, n_prb, act):
        """UL sinr_hook: interference at every gNB from the robots the other cells scheduled on the same RBG in
        this slot (transmit PSD after power split and power control, fading of the robot-gNB link)."""
        serv = self.assoc.serv
        psd = (self._pg + self.cfg.ue_tx_dbm - self.ref_db)[..., None] - self.ul._split(n_prb)[..., None, None]
        p_rx = 10 ** ((psd + self._gain_c) / 10) * won[:, :, None, :]                    # [E,R,C,S] mW per PRB
        other = ~onehot(serv, self.C)                                                    # [E,R,C]
        interf = (p_rx * other[..., None]).sum(1)                                        # [E,C,S]
        ni_now = self.noise_mw["ul"] + interf
        ni_act = 10 * torch.log10(ni_now.gather(1, serv[..., None].expand(-1, -1, self.S)))
        a = self.cfg.li_alpha
        self.ni_ul = ni_now if a == 1.0 else a * ni_now + (1 - a) * self.ni_ul
        self.ioN["ul"] += (interf / self.noise_mw["ul"]).sum((1, 2))
        self.ioN_n["ul"] += self.C * self.S
        out = act - (ni_act - self._ni_la_ul)
        if self.log_sinr:
            self._log_sinr("ul", won, out, ni_act - 10 * math.log10(self.noise_mw["ul"]))
        return out

    def _dl_ici(self, g, d, won, n_prb, act):
        """DL sinr_hook: interference at every robot from the other gNBs that transmit on the same RBG in this
        slot (gnb_tx_dbm over the carrier, fading of the gNB-robot link)."""
        serv = self.assoc.serv
        member = onehot(serv, self.C)                                                    # [E,R,C]
        active = (member[..., None] & won[:, :, None, :]).any(1)                         # [E,C,S]
        p_rx = 10 ** ((self._pg[..., None] + self.dl_psd_db + self._gain_c) / 10)        # [E,R,C,S]
        interf = (p_rx * (active[:, None] & ~member[..., None])).sum(2)                  # [E,R,S]
        ni_now = self.noise_mw["dl"] + interf
        a = self.cfg.li_alpha
        self.ni_dl = ni_now if a == 1.0 else a * ni_now + (1 - a) * self.ni_dl
        self.ioN["dl"] += (interf / self.noise_mw["dl"]).sum((1, 2)) / self.R
        self.ioN_n["dl"] += self.S
        ni_act = 10 * torch.log10(ni_now)
        out = act - (ni_act - self._ni_la_dl)
        if self.log_sinr:
            self._log_sinr("dl", won, out, ni_act - 10 * math.log10(self.noise_mw["dl"]))
        return out

    def iot_db(self, direction="ul"):
        """Mean interference over thermal (dB): UL per gNB-RBG-slot, DL per robot-RBG-slot, since the last full
        reset [E]."""
        n = max(self.ioN_n[direction], 1)
        return 10 * torch.log10(1 + self.ioN[direction] / n)

    def _finish(self, t, cur_hid, full=False):
        cfg = self.cfg
        u = self.ul
        q = u.q
        if self.rng is not None:
            self.rng.tick_step()
        delivered, timed, dropped = u.end_step(t, cfg.timeout_steps)
        capd = torch.where(delivered, q.cap, torch.full_like(q.cap, -1))
        newest = capd.max(-1).values
        if cur_hid is None:
            det_env = torch.zeros(self.E, dtype=torch.bool, device=self.dev)
        else:
            det_env = (delivered & q.det & (q.hid == cur_hid[:, None, None])).flatten(1).any(-1)
        if self.log_stats:
            self._log_ul(q, delivered, timed, dropped)
        if self.trace_frames is not None:
            self.trace_frames.append(tuple(x.cpu() for x in (q.cap, q.start, q.end, q.fin, delivered, timed, dropped)))
        if full:
            valid = q.cap >= 0
            out = {"newest": newest, "det_env": det_env, "delivered": delivered, "timed_out": timed,
                   "dropped": dropped, "cap": q.cap.clone(), "cls": torch.where(valid, q.cls, torch.zeros_like(q.cls)),
                   "delay": torch.where(delivered, (q.fin - q.cap.double()).float(),
                                        torch.full(q.cap.shape, float("nan"), device=self.dev))}
        u.compact(delivered | timed | dropped)
        if self.dl is not None:
            dq = self.dl.q
            dd, dt_, dr = self.dl.end_step(t, cfg.timeout_steps)
            if self.trace_frames_dl is not None:
                self.trace_frames_dl.append(tuple(x.cpu() for x in (dq.cap, dq.start, dq.end, dq.fin, dd, dt_, dr)))
            self.dl_newest = torch.where(dd, dq.cap, torch.full_like(dq.cap, -1)).max(-1).values
            if self.log_stats:
                self._log_dl(dq, dd)
            self.dl.compact(dd | dt_ | dr)
        if not full:
            return newest, det_env
        out["queue_len"] = u.q.count()
        out["queue_bytes"] = (u.q.enq - u.ack_ptr()).clamp(min=0).float()     # accepted, not yet resolved in order
        if self.dl is not None:
            out["dl_newest"] = self.dl_newest.clone()
            out["dl_queue_len"] = self.dl.q.count()
        if cfg.n_layers_max > 1:       # rank of each robot's last new TB [E,R] (1 in a direction without rank 2)
            for key, lk in (("rank", u), ("dl_rank", self.dl)):
                if lk is not None:
                    out[key] = lk.last_rank.clone() if lk.mimo else torch.ones(self.E, self.R, dtype=torch.long,
                                                                                device=self.dev)
        return out

    def _log_ul(self, q, delivered, timed, dropped):
        """Statistics of the UL frames resolved this step (host copies; q = the queue before compaction)."""
        st, cfg = self.stats, self.cfg
        keep = q.cap <= self.log_cap_max
        dk, tk = delivered & keep, (timed | dropped) & keep
        delay = (q.fin - q.cap.double()).float()
        st["delay"].append(delay[dk].cpu())
        eidx = torch.arange(self.E, device=self.dev)[:, None, None].expand_as(q.cap)
        st["d_env"].append(eidx[dk].cpu())
        st["x_env"].append(eidx[tk].cpu())
        st["dropped"] += int(dropped.sum())
        st["late"] += int((dk & (delay >= cfg.timeout_steps)).sum())
        for f in self.FEATS:
            st["d_" + f].append(getattr(q, f)[dk].cpu())
            st["x_" + f].append(getattr(q, f)[tk].cpu())

    def _log_dl(self, dq, dd):
        self.stats["dl_delay"].append((dq.fin - dq.cap.double()).float()[dd].cpu())

    def collect(self):
        st = self.stats
        cat = lambda k: torch.cat(st[k]) if st[k] else torch.zeros(0)
        out = {k: cat(k) for k in st if isinstance(st[k], list)}
        out.update({k: v for k, v in st.items() if not isinstance(v, list)})
        return out

    def counters(self):
        f = lambda link: {k: float(v) for k, v in link.ctr.items()} | {
            "ntx_hist": link.ntx_hist.tolist(), "rv_tx": link.rv_tx.sum(0).tolist(),
            "rv_fail": link.rv_fail.sum(0).tolist()}
        res = {"ul": f(self.ul)}
        if self.dl is not None:
            res["dl"] = f(self.dl)
        if self.C > 1 and self.cfg.rlf:
            res["rlf"] = {k: float(v) for k, v in self.assoc.ctr.items()}
        return res
