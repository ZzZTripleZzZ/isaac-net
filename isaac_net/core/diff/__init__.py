"""Differentiable network models (exploratory research infrastructure, separate from the engine levels).

    relax.py      smooth relaxations of min / max / step / floor under a temperature (tau = 0 is exact)
    fluid.py      DiffFluid: mode "L1" = L1D (relaxed L1 NetFluid), mode "QA" = QAD (relaxed QA analytic queue),
                  relaxed_bernoulli, rollout (KPIs: delay, delivery, age of information, energy)
    reference.py  the same KPIs from a real engine level with binary sends (make_discrete, discrete_rollout)
    proxy.py      recipe: a small neural proxy fitted to L2-legacy rollouts, used as a differentiable stand-in

These models are not engine levels (no make_engine entry, no CUDA-graph backend): they exist to give gradients
of network KPIs with respect to send probability, message size, transmit power and position. They are fluid
models; the scheduler, HARQ and BLER of L2 are not differentiated. See docs/differentiable.md.
"""
from .fluid import DiffFluid, radio_snr_db, relaxed_bernoulli, rollout
from .reference import discrete_rollout, make_discrete

__all__ = ["DiffFluid", "radio_snr_db", "relaxed_bernoulli", "rollout", "discrete_rollout", "make_discrete"]
