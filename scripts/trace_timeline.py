"""Generate the sample figures of docs/trace.md: a slot trace of one L2 environment, its timeline and slot map.

    python scripts/trace_timeline.py                    # writes docs/img/trace_timeline.png and trace_slots.png

Four robots share one cell (default NRConfig, CPU, reference backend). Every robot sends a 4000-byte message each
control step; robot 0 sits at the cell edge (low SNR), so its transport blocks use low MCS and fail more often.
"""
import argparse
import os

import torch

from isaac_net.core import NRConfig, Requests, make_engine
from isaac_net.core.trace import SlotTrace
from isaac_net.viz.trace import plot_slot_heatmap, plot_timeline


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "..", "docs", "img"))
    ap.add_argument("--steps", type=int, default=6)
    a = ap.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    E, R = 1, 4
    cfg = NRConfig(control_step_ms=20.0, msg_sizes=(2500,))
    net = make_engine("L2", E, R, "cpu", cfg, seed=7)
    trace = SlotTrace.attach(net, env=0)
    snr = torch.tensor([[5.0, 12.0, 15.0, 18.0]])
    for _ in range(a.steps):
        net.submit(None, Requests(torch.ones(E, R, dtype=torch.long)))
        net.step(None, snr)
    os.makedirs(a.out, exist_ok=True)
    plot_timeline(trace, robot=0, max_frames=4, path=os.path.join(a.out, "trace_timeline.png"))
    plot_slot_heatmap(trace, t1_ms=60.0, path=os.path.join(a.out, "trace_slots.png"))
    print(trace.summary().to_string())


if __name__ == "__main__":
    main()
