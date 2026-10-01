"""Regenerate the tiny synthetic radio map shipped in core/data/radio_map_synthetic.npz.

Two gNBs at (25, 75) and (125, 75) m in a 150 m arena, log-distance 40 + 35 log10(d), a 10 dB wall at x = 75 m,
16 x 16 grid. Use it with NRConfig(channel="radio_map", radio_map_path="synthetic", n_cells=2,
cell_positions_m=((25, 75), (125, 75))).

    python -m isaac_net.tools.make_synthetic_radio_map [out.npz]
"""
import sys

from isaac_net.core.channels.radio_map import SYNTHETIC_MAP, make_synthetic_map, synthetic_gnb_xy


def main(out=SYNTHETIC_MAP):
    m = make_synthetic_map(synthetic_gnb_xy())
    m.save(out)
    print(f"wrote {out}: {m.C} cells, {m.H} x {m.W}, bounds {m.bounds}")


if __name__ == "__main__":
    main(*sys.argv[1:])
