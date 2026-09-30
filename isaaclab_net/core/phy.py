"""3GPP PHY abstraction, batched in torch: MCS tables, exact TBS, EESM and SINR-to-BLER lookup.

- MCS: TS 38.214 Table 5.1.3.1-1 (qam64) and 5.1.3.1-2 (qam256), (Qm, R x 1024).
- TBS: TS 38.214 Sec. 5.1.3.2 steps 1-4 incl. Table 5.1.3.2-1, exact and elementwise.
- LDPC segmentation for the code-block size used by the BLER lookup: TS 38.212 Sec. 5.2.2 / 7.2.
- BLER vs SINR per (MCS, code-block size): Sionna SYS 2.2.0 tables (Apache-2.0, shipped in core/data/),
  exported by isaaclab_net/tools/export_sionna_tables.py, AWGN LDPC link-level curves. TB error = 1 - (1 - BLER_cb)^C.
- bler_source="lena": 5G-LENA v5.1 EESM tables. They are GPL-derived data and are never shipped: build them from
  your own 5G-LENA checkout with `python -m isaaclab_net.tools.extract_lena_tables` (see lena_tables_path()).
- EESM betas per MCS: Sionna SYS 2.2.0 esm_params/eesm_beta_table.json (Apache-2.0).
"""
import math
import os

import numpy as np
import torch

# (Qm, R*1024). Table 5.1.3.1-1, MCS 0..28 (29-31 reserved)
MCS_T1 = [(2, 120), (2, 157), (2, 193), (2, 251), (2, 308), (2, 379), (2, 449), (2, 526), (2, 602),
          (2, 679), (4, 340), (4, 378), (4, 434), (4, 490), (4, 553), (4, 616), (4, 658), (6, 438),
          (6, 466), (6, 517), (6, 567), (6, 616), (6, 666), (6, 719), (6, 772), (6, 822), (6, 873),
          (6, 910), (6, 948)]
# Table 5.1.3.1-2, MCS 0..27 (28-31 reserved)
MCS_T2 = [(2, 120), (2, 193), (2, 308), (2, 449), (2, 602), (4, 378), (4, 434), (4, 490), (4, 553),
          (4, 616), (4, 658), (6, 466), (6, 517), (6, 567), (6, 616), (6, 666), (6, 719), (6, 772),
          (6, 822), (6, 873), (8, 682.5), (8, 711), (8, 754), (8, 797), (8, 841), (8, 885),
          (8, 916.5), (8, 948)]
MCS_TABLES = {1: MCS_T1, 2: MCS_T2}

# TS 38.214 Table 5.1.3.2-1, TBS for N_info <= 3824
TBS_TABLE = [24, 32, 40, 48, 56, 64, 72, 80, 88, 96, 104, 112, 120, 128, 136, 144, 152, 160, 168, 176,
             184, 192, 208, 224, 240, 256, 272, 288, 304, 320, 336, 352, 368, 384, 408, 432, 456, 480,
             504, 528, 552, 576, 608, 640, 672, 704, 736, 768, 808, 848, 888, 928, 984, 1032, 1064,
             1128, 1160, 1192, 1224, 1256, 1288, 1320, 1352, 1416, 1480, 1544, 1608, 1672, 1736, 1800,
             1864, 1928, 2024, 2088, 2152, 2216, 2280, 2408, 2472, 2536, 2600, 2664, 2728, 2792, 2856,
             2976, 3104, 3240, 3368, 3496, 3624, 3752, 3824]

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "sionna_phy_tables.npz")


_CONST = {}


def _const(name, values, dtype, device):
    """Device copy of a constant table, made once per device (no host-to-device copy inside a captured step)."""
    key = (name, str(device), dtype)
    if key not in _CONST:
        _CONST[key] = torch.tensor(values, dtype=dtype, device=device)
    return _CONST[key]


def _f64(x, dev):
    """float64 tensor of a tensor or a Python number (a fill kernel, not a host copy, so it can be captured)."""
    if torch.is_tensor(x):
        return x.to(dtype=torch.float64, device=dev)
    return torch.full((), float(x), dtype=torch.float64, device=dev)


def n_re_per_prb(nsym, dmrs, oh):
    """N'_RE (38.214 5.1.3.2 step 1), capped at 156."""
    return torch.clamp(12 * nsym - dmrs - oh, max=156)


def tbs_38214(qm, r, n_prb, nsym, dmrs=12, oh=0, layers=1):
    """Exact TBS in bits, 38.214 Sec. 5.1.3.2. All tensor args broadcast; returns int64.
    qm: modulation order, r: target code rate (float, R/1024 already divided), n_prb: allocated PRBs,
    nsym: allocated OFDM symbols (incl. DMRS symbols). Zero PRBs give TBS 0."""
    dev = n_prb.device if torch.is_tensor(n_prb) else (qm.device if torch.is_tensor(qm) else None)
    f64 = lambda x: _f64(x, dev)
    qm, r, n_prb, nsym = f64(qm), f64(r), f64(n_prb), f64(nsym)
    n_re = n_re_per_prb(nsym, f64(dmrs), f64(oh)) * n_prb
    n_info = n_re * r * qm * layers
    # step 3: N_info <= 3824
    n = torch.clamp(torch.floor(torch.log2(n_info.clamp(min=1.0))) - 6, min=3)
    ninfo_q = torch.clamp(2 ** n * torch.floor(n_info / 2 ** n), min=24)
    tab = _const("tbs", TBS_TABLE, torch.float64, dev)
    idx = torch.searchsorted(tab, ninfo_q.contiguous(), right=False).clamp(max=len(TBS_TABLE) - 1)
    tbs_small = tab[idx]
    # step 4: N_info > 3824
    n4 = torch.floor(torch.log2((n_info - 24).clamp(min=1.0))) - 5
    x = (n_info - 24) / 2 ** n4
    ninfo_q4 = torch.clamp(2 ** n4 * torch.floor(x + 0.5), min=3840)      # round half up
    c_low = torch.ceil((ninfo_q4 + 24) / 3816)
    tbs_low = 8 * c_low * torch.ceil((ninfo_q4 + 24) / (8 * c_low)) - 24
    c_hi = torch.ceil((ninfo_q4 + 24) / 8424)
    tbs_hi = 8 * c_hi * torch.ceil((ninfo_q4 + 24) / (8 * c_hi)) - 24
    tbs_mid = 8 * torch.ceil((ninfo_q4 + 24) / 8) - 24
    tbs_large = torch.where(r <= 0.25, tbs_low, torch.where(ninfo_q4 > 8424, tbs_hi, tbs_mid))
    tbs = torch.where(n_info <= 3824, tbs_small, tbs_large)
    return torch.where(n_re > 0, tbs, torch.zeros_like(tbs)).long()


def segment(tbs, r):
    """LDPC base graph + code-block segmentation (38.212 7.2.2 / 5.2.2).
    Returns (cb_size_bits incl. CRC, n_cb). Matches Sionna calculate_tb_size's cb_size."""
    tbs = tbs.double()
    r = _f64(r, tbs.device)
    bg2 = (tbs <= 292) | ((tbs <= 3824) & (r <= 0.67)) | (r <= 0.25)
    kcb = torch.where(bg2, torch.full_like(tbs, 3840.0), torch.full_like(tbs, 8448.0))
    b = tbs + torch.where(tbs > 3824, torch.full_like(tbs, 24.0), torch.full_like(tbs, 16.0))
    c = torch.where(b <= kcb, torch.ones_like(b), torch.ceil(b / (kcb - 24)))
    bp = torch.where(c > 1, b + 24 * c, b)
    return (bp / c), c


def lena_tables_path():
    """Where the locally generated 5G-LENA tables live: $ISAACLAB_NET_LENA_TABLES, else
    ~/.cache/isaaclab_net/lena_eesm_tables.npz (outside the source tree, so they cannot be committed)."""
    return os.environ.get("ISAACLAB_NET_LENA_TABLES",
                          os.path.join(os.path.expanduser("~"), ".cache", "isaaclab_net", "lena_eesm_tables.npz"))


LENA_DATA = lena_tables_path()
# TS 38.212 Table 5.3.2-1: all LDPC lifting sizes Zc, sorted
LIFTING = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 18, 20, 22, 24, 26, 28, 30, 32, 36, 40, 44, 48,
           52, 56, 60, 64, 72, 80, 88, 96, 104, 112, 120, 128, 144, 160, 176, 192, 208, 224, 240, 256, 288, 320,
           352, 384]
_TAB_CACHE = {}


def _db(lin):
    """Linear -> dB, with empty allocations (NaN) mapped to -30 dB so table indices stay valid."""
    return (10 * torch.log10(lin.clamp(min=1e-6))).nan_to_num(nan=-30.0, posinf=60.0, neginf=-30.0)


def tbs_lena(qm, r, n_prb, nsym, ref_sc=1):
    """5G-LENA v5.1 TB size in bits (NrAmc::CalculateTbSize behaviour, reimplemented): payload bytes
    P = floor((12 - ref_sc) * n_prb * nsym * Qm * R / 8), minus a 3-byte CRC, minus 3 bytes per code
    block when the TB exceeds the largest code block (1056 B for BG1, 480 B for BG2)."""
    f64 = lambda x: _f64(x, n_prb.device)
    qm, r, n_prb = f64(qm), f64(r), f64(n_prb)
    p = torch.floor((12 - ref_sc) * n_prb * nsym * qm * r / 8)
    tb = torch.where(p >= 3, p - 3, p)
    bits = p * 8
    bg2 = (bits <= 292) | (r <= 0.25) | ((bits <= 3824) & (r <= 0.67))
    cbmax = torch.where(bg2, torch.full_like(p, 480.0), torch.full_like(p, 1056.0))
    c = torch.ceil(tb / cbmax)
    tb = torch.where(tb > cbmax, p - 3 * c, tb)
    return (tb.clamp(min=0) * 8).long()


def segment_ldpc_k(tbs, r):
    """Code-block size K = Zc * Kb (38.212 5.2.2 / 5.3.2) as 5G-LENA's EESM model uses it for its
    table lookup, with B = TBS + 24. Returns (K, C, bg) with bg 0 = BG1, 1 = BG2."""
    tbs = tbs.double()
    r = _f64(r, tbs.device)
    bg2 = (tbs <= 292) | (r <= 0.25) | ((tbs <= 3824) & (r <= 0.67))
    b = tbs + 24
    kcb = torch.where(bg2, torch.full_like(b, 3840.0), torch.full_like(b, 8448.0))
    c = torch.where(b <= kcb, torch.ones_like(b), torch.ceil(b / (kcb - 24)))
    b1 = torch.where(c > 1, b + 24 * c, b)
    k1 = torch.floor(b1 / c)
    kb2 = torch.where(b > 640, 10.0, torch.where(b > 560, 9.0, torch.where(b > 192, 8.0, 6.0)))
    kb = torch.where(bg2, kb2, torch.full_like(b, 22.0))
    z = _const("lifting", LIFTING, torch.float64, tbs.device)
    zi = torch.searchsorted(z, (k1 / kb).contiguous(), right=False).clamp(max=len(LIFTING) - 1)
    zc = z[zi]
    k = zc * torch.where(bg2, torch.full_like(b, 10.0), torch.full_like(b, 22.0))
    return k, c, bg2.long()


class PHY:
    """MCS / TBS / BLER / EESM machinery of one direction on one device.

    source: "pdsch" (Sionna SYS PDSCH curves for both directions, default), "sionna_label"
    (Sionna curves as labelled, audit only) or "lena" (5G-LENA v5.1 EESM tables from a local
    extraction, lookup semantics of NrEesmErrorModel). tbs_mode: "38214" (exact 5.1.3.2) or "lena"."""

    def __init__(self, direction, mcs_table, device, bler_target=0.1, source="pdsch", tbs_mode="38214",
                 lena_ref_sc=1, max_tx=16):
        mcs = MCS_TABLES[mcs_table]
        self.M = M = len(mcs)
        self.dev, self.source, self.tbs_mode, self.lena_ref_sc = device, source, tbs_mode, lena_ref_sc
        self.qm = torch.tensor([q for q, _ in mcs], dtype=torch.float32, device=device)
        self.r = torch.tensor([c / 1024 for _, c in mcs], dtype=torch.float32, device=device)
        self.se = self.qm * self.r
        key = (source, direction if source == "sionna_label" else "-", mcs_table, str(device))
        if key not in _TAB_CACHE:
            if source == "lena":
                path = lena_tables_path()
                if not os.path.exists(path):
                    raise FileNotFoundError(
                        f"{path} missing: bler_source='lena' needs the 5G-LENA EESM tables, which are GPL-2.0 data "
                        "and not shipped. Build them from your own 5G-LENA checkout with "
                        "`python -m isaaclab_net.tools.extract_lena_tables /path/to/nr` (see README).")
                d = np.load(path)
                tab = torch.tensor(d[f"bler_t{mcs_table}"][:, :M], device=device)
                grid = torch.tensor(d["sinr_grid"], device=device)
                cbs = torch.tensor(d["cbs_axis"], device=device)
                beta = torch.tensor(d[f"beta_t{mcs_table}"][:M], device=device)
                notes = ["5G-LENA NrEesm tables (step lookup)"]
            else:
                d = np.load(DATA)
                t, g, notes = build_bler_table(d, direction, mcs_table, source)
                tab = torch.tensor(t[:M], device=device)[None]
                grid = torch.tensor(g, device=device)
                cbs = torch.tensor(d["cbs_grid"], device=device)
                beta = torch.tensor(d[f"eesm_beta_t{mcs_table}"][:M], device=device)
            _TAB_CACHE[key] = (tab.float().contiguous(), grid.float(), cbs.float(), beta.float(), notes)
        self.bler, self.snr_grid, self.cbs_axis, self.beta, self.fill_log = _TAB_CACHE[key]
        self.step = source == "lena"
        self.NB, _, self.C, self.G = self.bler.shape
        self.s0 = float(self.snr_grid[0]); self.ds = float(self.snr_grid[1] - self.snr_grid[0])
        self.c0 = float(self.cbs_axis[0]); self.dc = float(self.cbs_axis[1] - self.cbs_axis[0])
        # HARQ-IR equivalent MCS (effective code rate R / ntx, same modulation), [M, max_tx + 1]
        eq = torch.arange(M).repeat(max_tx + 1, 1).T.clone()
        for m in range(1, M):
            for n in range(2, max_tx + 1):
                reff = mcs[m][1] / 1024 / n
                eq[m, n] = m - sum(1 for k in range(m) if mcs[k][0] == mcs[m][0] and mcs[k][1] / 1024 > reff)
        self.mcs_eq = eq.to(device)
        self.mcs_max = M - 1                                   # link-adaptation cap (NRConfig.*_mcs_max)
        self.set_target(bler_target)

    def set_target(self, target):
        """thr[bg, m, c]: smallest SINR (dB) with code-block BLER <= target."""
        ok = self.bler <= target
        first = torch.where(ok.any(-1), ok.float().argmax(-1), torch.full(ok.shape[:-1], self.G - 1, device=self.dev))
        self.thr = self.snr_grid[first]
        self.target = target
        ref = torch.full((self.M,), 1524.0, device=self.dev)
        tb = self.tbs_all(torch.tensor(10.0, device=self.dev), 12, 12)            # one 10-PRB RBG
        k, _, bg = self.cb(tb, self.r)
        self.thr_ref = self.thr_lookup(k.float(), bg)

    # -------- TBS and code blocks --------
    def tbs_all(self, n_prb, nsym, dmrs=12, oh=0):
        """TBS (bits) of every MCS for n_prb [...] -> [..., M]."""
        n = n_prb[..., None] if n_prb.dim() else n_prb
        if self.tbs_mode == "lena":
            return tbs_lena(self.qm, self.r, n, nsym, self.lena_ref_sc)
        return tbs_38214(self.qm, self.r, n, nsym, dmrs, oh)

    def cb(self, tbs, r):
        """(code-block size used for the BLER lookup, number of CBs, base graph index)."""
        if self.step:
            return segment_ldpc_k(tbs, r)
        cbs, c = segment(tbs, r)
        return cbs, c, torch.zeros_like(c, dtype=torch.long)

    # -------- lookups --------
    def _cidx(self, cbs):
        if self.step:          # LENA: curve of the largest simulated CBS <= K (smallest if below all)
            i = (torch.searchsorted(self.cbs_axis, cbs.contiguous(), right=True) - 1).clamp(min=0)
            return i, torch.zeros_like(cbs)
        x = ((cbs - self.c0) / self.dc).clamp(0, self.C - 1)
        i0 = x.floor().clamp(max=self.C - 2).long()
        return i0, x - i0

    def bler_lookup(self, mcs, sinr_db, cbs, bg=None):
        """Code-block BLER for integer mcs, effective SINR (dB), code-block size (bits)."""
        bg = torch.zeros_like(mcs, dtype=torch.long) if bg is None else bg.long()
        i0, wi = self._cidx(cbs.float())
        flat = self.bler.reshape(-1)
        base0 = ((bg * self.M + mcs.long()) * self.C + i0) * self.G
        if self.step:          # LENA: BLER of the largest simulated SINR point <= x
            j = torch.floor((sinr_db.nan_to_num(-30.0) - self.s0) / self.ds + 1e-6).clamp(0, self.G - 1).long()
            return flat[base0 + j]
        x = ((sinr_db.nan_to_num(-30.0) - self.s0) / self.ds).clamp(0, self.G - 1)
        j0 = x.floor().clamp(max=self.G - 2).long()
        wj = x - j0
        lo = flat[base0 + j0] * (1 - wj) + flat[base0 + j0 + 1] * wj
        hi = flat[base0 + self.G + j0] * (1 - wj) + flat[base0 + self.G + j0 + 1] * wj
        return lo * (1 - wi) + hi * wi

    def thr_lookup(self, cbs, bg=None):
        """Target-BLER SINR threshold for every MCS at code-block sizes cbs [..., M]."""
        bg = torch.zeros_like(cbs, dtype=torch.long) if bg is None else bg.long()
        i0, wi = self._cidx(cbs)
        m = torch.arange(self.M, device=self.dev).expand_as(i0)
        t0 = self.thr[bg, m, i0]
        if self.step:
            return t0
        return t0 * (1 - wi) + self.thr[bg, m, i0 + 1] * wi

    def se_at(self, sinr_db):
        """Spectral efficiency (Qm*R) of the highest MCS meeting the BLER target at sinr_db for a
        10-PRB allocation; MCS 0 when nothing qualifies. Used for PF rate estimates and CQI."""
        return self.se[self.mcs_at(sinr_db)]

    def mcs_at(self, sinr_db):
        ok = (sinr_db[..., None] >= self.thr_ref) & (torch.arange(self.M, device=self.dev) <= self.mcs_max)
        return (ok.long() * torch.arange(1, self.M + 1, device=self.dev)).max(-1).values.clamp(min=1) - 1

    # -------- effective SINR --------
    def _lw(self, mask, w):
        lw = torch.log(w.float()) if w is not None else torch.zeros(mask.shape[-1], device=self.dev)
        return torch.where(mask, lw.expand_as(mask), torch.full(mask.shape, -float("inf"), device=self.dev))

    def eff_sinr_all(self, sinr_db, mask, mode="eesm", w=None):
        """Effective SINR (dB) over masked subbands (weights w = PRBs per subband) for every MCS."""
        lw = self._lw(mask, w)
        lnw = torch.logsumexp(lw, -1, keepdim=True)
        if mode == "mean_db":
            v = (sinr_db * torch.exp(lw - lnw)).nan_to_num(0).sum(-1, keepdim=True)
            return v.expand(*v.shape[:-1], self.M)
        lin = 10 ** (sinr_db.clamp(-30, 60) / 10)
        x = -lin[..., None, :] / self.beta[:, None] + lw[..., None, :]           # [..., M, S]
        eff = -self.beta * (torch.logsumexp(x, -1) - lnw)
        return _db(eff)

    def eesm_lse(self, sinr_db, mask, mcs, w=None):
        """log(sum_s w_s exp(-SINR_s / beta_mcs)) over masked subbands, and log(sum w)."""
        lw = self._lw(mask, w)
        beta = self.beta[mcs.long()][..., None]
        lin = 10 ** (sinr_db.clamp(-30, 60) / 10)
        return torch.logsumexp(-lin / beta + lw, -1), torch.logsumexp(lw, -1)

    def eff_sinr(self, sinr_db, mask, mcs, mode="eesm", w=None):
        """Effective SINR (dB) for a given MCS per UE: sinr_db, mask [..., S], mcs [...] -> [...]."""
        if mode == "mean_db":
            lw = self._lw(mask, w)
            return (sinr_db * torch.exp(lw - torch.logsumexp(lw, -1, keepdim=True))).nan_to_num(0).sum(-1)
        lse, lnw = self.eesm_lse(sinr_db, mask, mcs, w)
        return _db(-self.beta[mcs.long()] * (lse - lnw))

    # -------- link adaptation --------
    def select_mcs(self, est_db, mask, offset_db, n_prb, nsym, dmrs, oh=0, mode="eesm", w=None):
        """Highest MCS whose TB error probability at the (offset) effective SINR of the allocated
        subbands is <= the BLER target (5G-LENA ErrorModel AMC rule). Returns (mcs, tbs)."""
        eff = self.eff_sinr_all(est_db, mask, mode, w) + offset_db[..., None]      # [..., M]
        tbs = self.tbs_all(n_prb, nsym, dmrs, oh)                                  # [..., M]
        cbs, c, bg = self.cb(tbs, self.r)
        m_all = torch.arange(self.M, device=self.dev).expand_as(tbs)
        tbler = 1 - (1 - self.bler_lookup(m_all, eff, cbs.float(), bg)) ** c.float()
        ok = (tbler <= self.target) & (tbs > 0) & (m_all <= self.mcs_max)
        m = (ok.long() * torch.arange(1, self.M + 1, device=self.dev)).max(-1).values.clamp(min=1) - 1
        return m, tbs.gather(-1, m[..., None]).squeeze(-1)

    def tb_error_prob(self, mcs, sinr_eff_db, tbs, mcs_eq=None):
        cbs, c, bg = self.cb(tbs, self.r[mcs.long()])
        b = self.bler_lookup(mcs if mcs_eq is None else mcs_eq, sinr_eff_db, cbs.float(), bg)
        return 1 - (1 - b) ** c.float()


_RAW = {}
SNR_MIN = -15.0


def _raw(d):
    if "raw" not in _RAW:
        import json
        _RAW["raw"] = json.loads(bytes(d["raw_json"]).decode())
    return _RAW["raw"]


def build_bler_table(d, direction, mcs_table, source="pdsch"):
    """Build the [M, C, G] BLER table used by the engine from the exported Sionna data.

    source="pdsch" (default): both directions use Sionna's PDSCH (category 1) curves. Sionna 2.2.0's
    PUSCH_table1.json is the same simulation as PDSCH_table1.json for MCS 3..16 but its keys
    17..27 carry the PDSCH MCS 18..28 curves (off by one), and PUSCH_table2.json does not match
    38.214 Table 5.1.3.1-2 (10% points about 8 dB below Shannon); see README. source="sionna_label"
    uses the tables exactly as Sionna labels them (for audit only).
    Post-processing: (a) rows that are missing, or never reach BLER 0.1 inside the simulated SNR
    range, are filled by copying a curve with identical (Qm, R) from the other table or by
    shifting the nearest valid MCS curve by the Shannon-gap SNR difference; (b) above the last
    simulated SNR point, where Sionna's interpolator clamps to the edge value (an artificial error
    floor), the waterfall continues log-linearly with the slope of the last simulated segment
    (at least 1 decade/dB); (c) BLER is forced non-increasing in SNR.
    The grid is extended below Sionna's -5 dB edge to SNR_MIN with BLER 1 (Sionna's lookup would
    clamp to the edge value there), so the shifted low-MCS curves keep their waterfall.
    Returns (table [M, C, G] float32, SNR grid [G], list of notes)."""
    cat = "dl" if source == "pdsch" else direction
    fname = ("PDSCH" if cat == "dl" else "PUSCH") + "_table{}.json"
    snr0 = d["snr_grid"].astype(np.float64)
    ds = snr0[1] - snr0[0]
    pad = int(round((snr0[0] - SNR_MIN) / ds))       # extend the grid down to SNR_MIN with BLER 1
    snr = np.concatenate([snr0[0] + ds * np.arange(-pad, 0), snr0])
    ci_ref = int(np.argmin(np.abs(d["cbs_grid"] - 1924)))
    notes = []
    gap = lambda q, c: 10 * math.log10(2 ** (q * c / 1024) - 1)

    def load(tab):
        a = d[f"bler_{cat}_t{tab}"].astype(np.float64)
        a = np.concatenate([np.ones(a.shape[:-1] + (pad,)), a], -1)
        av = d[f"avail_{cat}_t{tab}"].copy()
        raw = _raw(d)[fname.format(tab)]["category"]
        raw = raw[list(raw.keys())[0]]["index"][str(tab)]["MCS"]
        s_last = max(max(v["SNR_db"]) for v in raw.values())
        s_step = min(np.diff(v["SNR_db"]).min() for v in raw.values())
        inr = snr <= s_last + 1e-6
        for m in range(a.shape[0]):
            if av[m] and not (a[m, ci_ref, inr] <= 0.1).any():
                av[m] = False
                notes.append(f"{cat} t{tab} MCS {m}: never reaches BLER 0.1 by {s_last:.0f} dB, treated as missing")
        # (b) continuation above the last simulated SNR
        last = int(np.searchsorted(snr, s_last - 1e-6))
        j1 = int(np.searchsorted(snr, s_last - s_step - 1e-6))
        b1 = np.clip(a[..., j1], 1e-9, 1)
        b2 = np.clip(a[..., last], 1e-9, 1)
        slope = np.maximum((np.log10(b1) - np.log10(b2)) / (snr[last] - snr[j1]), 1.0)
        ext = np.log10(b2)[..., None] - slope[..., None] * (snr[None, None, last:] - snr[last])
        a[..., last:] = np.where(a[..., last:last + 1] > 0, 10 ** ext, 0.0)
        return a, av

    tabs = {t: load(t) for t in (1, 2)}
    order = [mcs_table, 3 - mcs_table]
    for t in (1, 2):                                   # fill both tables (table 2 copies from filled table 1)
        a, av = tabs[t]
        mcs = MCS_TABLES[t]
        for m in range(len(mcs)):
            if av[m]:
                continue
            o = 3 - t
            oa, oav = tabs[o]
            same = [k for k in range(len(MCS_TABLES[o])) if MCS_TABLES[o][k] == mcs[m] and oav[k]]
            if same:
                a[m] = oa[same[0]]
                notes.append(f"{cat} t{t} MCS {m}: copied from table {o} MCS {same[0]} (same Qm, R)")
                continue
            cand = [k for k in range(len(mcs)) if av[k]]
            k = min(cand, key=lambda k: abs(k - m))
            shift = gap(*mcs[m]) - gap(*mcs[k])
            steps = shift / (snr[1] - snr[0])
            xs = np.arange(len(snr)) - steps
            a[m] = np.stack([np.interp(xs, np.arange(len(snr)), row, left=1.0, right=0.0 if row[-1] < 1e-9 else row[-1])
                             for row in a[k]])
            notes.append(f"{cat} t{t} MCS {m}: MCS {k} curve shifted {shift:+.2f} dB (Shannon-gap difference)")
        if t == 1:
            tabs[1] = (a, np.ones_like(av))
    a = tabs[mcs_table][0][: len(MCS_TABLES[mcs_table])]
    a = np.minimum.accumulate(np.clip(a, 0, 1), axis=-1)
    return a.astype(np.float32), snr.astype(np.float32), [n for n in notes if f" t{mcs_table} " in n]
