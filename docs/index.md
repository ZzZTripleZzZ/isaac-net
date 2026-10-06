# isaac-net

![A robot fleet in a warehouse at dusk, with 5G masts among the robot lanes](img/banner-plain.jpg){ .isaac-hero loading=lazy }

**`isaac-net` simulates the 5G (and Wi-Fi) uplink of thousands of robot-learning environments at once on one GPU, stepped in lockstep with the physics of Isaac Lab or MuJoCo Playground. A policy therefore trains against queues that build up when the team transmits together, links that degrade as robots move, and retransmissions that stretch the delay tail, instead of a fixed or random delay.**

![One control step of isaac-net](img/overview.png)

*One control step: the simulator submits message classes and robot poses, the engine runs the uplink slots of the NR MAC with all state in `[envs, robots, ...]` tensors, and per-robot deliveries, delays, age of information and SINR return as observations.*

## Quick start

```bash
pip install isaac-net            # install the PyTorch build you want first (2.7+); on Linux its CUDA build includes Triton
```

```python
import torch
from isaac_net import NRConfig, Requests, make_engine

E, R = 64, 8                                                   # 64 environments x 8 robots
net = make_engine("L2", E, R, "cpu", NRConfig(), seed=0)       # on a GPU: "cuda", backend="triton"
pos = torch.rand(E, R, 2) * 100.0                              # robot positions in metres
for t in range(20):
    net.submit(None, Requests((torch.rand(E, R) < 0.3).long()))   # 1 = one 4 kB message, 0 = nothing
    out = net.step(None, pos)                                  # one control step = 100 ms for every env
print(f"mean delay {out['delay'][out['delivered']].mean() * 100:.1f} ms, queued {int(out['queue_len'].sum())}")
```

No install at hand? [Open the quick start in Colab](https://colab.research.google.com/github/ZzZTripleZzZ/isaac-net/blob/main/docs/tutorials/00_quickstart_colab.ipynb): it runs on a free CPU or GPU runtime in under three minutes.

## Next steps

1. **[Choose a configuration](choosing.md)**: which fidelity level, backend, preset and simulator fit your experiment, with costs and what each one leaves out.
2. **[Work through the tutorials](tutorials/index.md)**: from a first network to fidelity levels, `NRConfig`, the Isaac Lab layer and validation against ns-3. The [Cookbook](cookbook.md) and the [FAQ](faq.md) answer the questions that come next.
3. **[Put the network into Isaac Lab](isaac-lab.md)**: four hook calls in a `DirectRLEnv`, on Windows or [kit-less on Linux](isaac-lab-linux.md).

[Concepts](concepts.md) explains the ideas behind the engine (slot-synchronous stepping, fixed shapes, per-env clocks, backends), and [Status](STATUS.md) lists what is done and what is open. The package is a research prototype: version 0.2.0 is on [PyPI](https://pypi.org/project/isaac-net/), and the code is on [GitHub](https://github.com/ZzZTripleZzZ/isaac-net).

## Citation

If you use `isaac-net`, please cite the paper, [arXiv:2610.02370](https://arxiv.org/abs/2610.02370). Zifan Zhang and Mingzhe Han contributed equally.

```bibtex
@article{zhang2026isaacnet,
  title   = {Network-in-the-Loop at Scale: GPU-Batched 5G Simulation for Massively Parallel Robot Learning},
  author  = {Zhang, Zifan and Han, Mingzhe and Athreya, Kannan and Liu, Yuchen},
  journal = {arXiv preprint arXiv:2610.02370},
  year    = {2026},
  note    = {Zifan Zhang and Mingzhe Han contributed equally}
}
```

## Building these docs

```bash
pip install -e ".[docs]"
mkdocs serve                     # live preview at http://127.0.0.1:8000
mkdocs build --strict            # what CI runs
```

The API reference is generated from the docstrings. Tutorials 01–05 are stored executed and rendered without running them, and `bash tutorials/build_notebooks.sh` regenerates them from the scripts in `tutorials/`. The Colab quick start is a notebook of its own, stored executed as well.
