"""Robot-body blockage: every other robot of the same env is a sphere of radius r; a link loses `loss_db` when the
segment from the robot's antenna to the gNB passes through at least one of them.

Fixed-shape pairwise test [E, R, R, C] (blockers j of link (i, c)): with a = robot i, d = gNB c - a and w = centre of
robot j - a, the projection t = (w . d) / |d|^2 must lie strictly inside (0, 1) and the distance |w - t d| must be
below r. The robot itself (j = i) never blocks. Cost O(E R^2 C) per call, fine for R up to a few hundred.
"""
from __future__ import annotations

import torch


def blocked_links(pos3, gnb3, radius):
    """pos3 [E,R,3] antenna positions (sphere centres), gnb3 [C,3] -> bool [E,R,C]: link (e, i, c) is blocked."""
    E, R, _ = pos3.shape
    d = gnb3[None, None] - pos3[:, :, None, :]                          # [E,R,C,3]
    w = pos3[:, None, :, :] - pos3[:, :, None, :]                       # [E,i,j,3]  centre j - robot i
    wd = torch.einsum("eijx,eicx->eijc", w, d)                          # [E,i,j,C]
    dd = (d * d).sum(-1).clamp(min=1e-9)[:, :, None, :]                 # [E,i,1,C]
    ww = (w * w).sum(-1)[..., None]                                     # [E,i,j,1]
    t = wd / dd
    dist2 = ww - 2 * t * wd + t * t * dd                                # |w - t d|^2
    hit = (t > 0) & (t < 1) & (dist2 < radius * radius)
    eye = torch.eye(R, dtype=torch.bool, device=pos3.device)[None, :, :, None]
    return (hit & ~eye).any(2)
