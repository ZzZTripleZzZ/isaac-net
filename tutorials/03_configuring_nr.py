# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Tutorial 03: Configuring NR
#
# One dataclass, `NRConfig`, configures every module of the package: the numerology and carrier, the TDD
# pattern, MAC timing, HARQ and RLC, the PHY tables, the radio and cell layout, and the application fields. This
# tutorial walks through the presets, the frame structure, HARQ, the downlink, multiple cells and strict mode.
# It runs on a CPU.
#
# A short glossary for readers from robotics: the *numerology* `mu` sets the subcarrier spacing
# (15 x 2^mu kHz) and the slot length (1 ms / 2^mu). A *TDD pattern* such as `DDDSU` says which slots carry
# downlink (D) or uplink (U) data, with a special slot (S) that switches between the two. A *PRB* (physical
# resource block) is 12 subcarriers of one slot, the unit the scheduler hands out, and an *RBG* is a group of
# PRBs scheduled together. *HARQ* retransmits a transport block that failed to decode, and several HARQ
# processes let a robot keep sending new data while an earlier block waits for its retransmission.

# %%
import torch

from isaaclab_net import NRConfig, Requests, make_engine
from isaaclab_net.core import lena_like, multicell, netslot_compat, oai_like, srsran_like
from isaaclab_net.core.config import fields_read_by

E, R, dev = 4, 6, torch.device("cpu")

# %% [markdown]
# ## The default config and its derived quantities
#
# Every derived quantity (PRBs, RBG size, slots per control step) is computed from the fields when you read it,
# following the 3GPP tables cited in `config.py`. `summary()` prints them in one line.

# %%
cfg = NRConfig()
print(cfg.summary())
print("PRBs:", cfg.nprb, "| RBG size:", cfg.rbg, "| subbands:", cfg.subband_prbs)
print("slot:", cfg.slot_ms, "ms | slots per control step:", cfg.slots_per_step,
      "| UL data slots per step:", cfg.ul_slots_per_step)

# %% [markdown]
# ## Presets
#
# The presets are plain functions that return an `NRConfig`, and each takes keyword overrides.
#
# - `netslot_compat()`: the geometry and timing of the legacy slot-level model with the 3GPP PHY.
# - `lena_like()` and `lena_validation()`: the ns-3 5G-LENA reference scenario. They use the 5G-LENA BLER
#   tables, which you generate locally (see the Licensing page).
# - `srsran_like()` and `oai_like()`: uplink latency fitted to public srsRAN and OAI measurements.
# - `multicell(n)`: a hexagonal cluster of `n` cells with interference, power control and handover.

# %%
for name, preset in [("netslot_compat", netslot_compat), ("srsran_like", srsran_like), ("oai_like", oai_like),
                     ("lena_like", lena_like)]:
    print(f"{name:15s}", preset().summary())

# %% [markdown]
# ## Frame structure: numerology, bandwidth and TDD pattern
#
# A 40 MHz carrier at 30 kHz subcarrier spacing with the uplink-heavier pattern `DSUUU` gives three times as
# many uplink data slots per control step as the default `DDDSU`. `slot_symbols(pos)` returns the downlink and
# uplink data symbols of each slot position in the pattern.

# %%
ul_heavy = NRConfig(mu=1, bandwidth_mhz=40, tdd_pattern="DSUUU")
print(ul_heavy.summary())
print("(DL, UL) data symbols per slot of the pattern:",
      [ul_heavy.slot_symbols(i) for i in range(len(ul_heavy.tdd_pattern))])

# %% [markdown]
# The configurable NR engine is level `L2`. The helper below drives it with the same traffic for any config and
# reports the delivered messages and the delay quantiles.

# %%
g = torch.Generator().manual_seed(3)
T = 40
sends = (torch.rand(T, E, R, generator=g) < 0.5).long() * 2          # large (30 kB) messages
pos = 20 + 60 * torch.rand(E, R, 2, generator=g)


def run(config, level="L2"):
    torch.manual_seed(0)
    net = make_engine(level, E, R, dev, config, seed=0)
    dlv, delays = 0, []
    for t in range(T):
        net.submit(None, Requests(sends[t]))
        out = net.step(None, pos)
        dlv += int(out["delivered"].sum())
        delays.append(out["delay"][out["delivered"]])
    d = torch.cat(delays) * config.control_step_ms
    q = [round(x, 1) for x in d.quantile(torch.tensor([0.5, 0.95])).tolist()] if d.numel() else None
    return {"delivered": dlv, "delay_p50_p95_ms": q}


print("DDDSU, 20 MHz:", run(NRConfig()))
print("DSUUU, 40 MHz:", run(ul_heavy))

# %% [markdown]
# ## HARQ processes
#
# `n_harq` sets the HARQ processes per robot and direction. With one process (`n_harq=1`) a robot waits for the
# outcome of each transport block before it sends the next one, which is head-of-line blocking. `max_harq_tx`
# caps the transmissions per block, and `harq_fail` chooses what happens after the last one: `"rlc_am"` resends
# the data after `rlc_retx_slots`, and `"drop"` loses it as RLC unacknowledged mode would.

# %%
for n in (1, 4, 16):
    print(f"n_harq={n:2d}:", run(NRConfig(n_harq=n)))
print("harq_fail='drop':", run(NRConfig(harq_fail="drop")))

# %% [markdown]
# How much these settings matter depends on the load and on how often transport blocks fail to decode. In this
# small example the numbers of processes give similar results, so treat the cell as a template for your own
# scenario rather than as a measurement.

# %% [markdown]
# ## Downlink
#
# With `dl=True` the NR engine also keeps per-robot downlink queues at the gNB. `add_dl_frames(t, nbytes)`
# enqueues `nbytes [E, R]` (0 = nothing) and the step output gains `dl_newest` and `dl_queue_len`. Below, every
# robot's 3 kB downlink message is captured at step 0 and delivered within that step, so `dl_newest` is 0
# everywhere (it would be -1 where nothing arrived) and the downlink queues are empty again.

# %%
torch.manual_seed(0)
net = make_engine("L2", E, R, dev, NRConfig(dl=True), seed=0)
net.add_dl_frames(None, torch.full((E, R), 3000.0))
out = net.step(None, pos)
print("downlink: newest delivered capture step per robot:\n", out["dl_newest"])
print("downlink queue length:\n", out["dl_queue_len"])

# %% [markdown]
# ## Multiple cells
#
# `multicell(n)` places `n` gNBs in a hexagonal cluster at 100 m inter-site distance, with thermal noise,
# same-slot uplink interference between cells, fractional uplink power control and A3 handover. Multi-cell
# configurations currently run on level `L2-legacy`, and the step output adds `serving_cell [E, R]`.

# %%
mc = multicell(3)
print("gNB positions (m):", [(round(x, 1), round(y, 1)) for x, y in mc.gnb_xy()])
torch.manual_seed(0)
net = make_engine("L2-legacy", E, R, dev, mc, seed=0)
arena_pos = 150 * torch.rand(E, R, 2, generator=g)
for _ in range(5):
    net.submit(None, Requests(sends[0]))
    out = net.step(None, arena_pos)
print("serving cell per robot:\n", out["serving_cell"])

try:
    make_engine("L2", E, R, dev, mc)
except NotImplementedError as err:
    print("L2 with 3 cells:", err)

# %% [markdown]
# ## Strict mode
#
# Every level reads only some of the config fields. The prototype levels, for example, read only the
# application fields and their own parameters, because their radio and MAC are fixed. By default a field that a
# level ignores is ignored silently. `unused_fields(level)` lists the fields that are set away from their
# defaults and that the level ignores, and `make_engine(..., strict=True)` raises instead of ignoring them.

# %%
custom = NRConfig(mcs_table=2, n_harq=8, l0_loss=0.1)
print("fields L0 reads:", sorted(fields_read_by("L0")))
print("ignored by L0:", custom.unused_fields("L0"))
print("ignored by L2:", custom.unused_fields("L2"))
try:
    make_engine("L0", E, R, dev, custom, strict=True)
except ValueError as err:
    print("strict:", err)

# %% [markdown]
# Some fields cannot be ignored at all. The prototype levels are compiled around a 16-message frame buffer, a
# 20-step timeout and a 100 ms control step, so they refuse a config that asks for anything else, strict or
# not. The NR engine reads these fields.

# %%
try:
    make_engine("L2-legacy", E, R, dev, NRConfig(timeout_steps=10))
except ValueError as err:
    print(err)
print("L2 with a 10-step timeout:", run(NRConfig(timeout_steps=10)))
