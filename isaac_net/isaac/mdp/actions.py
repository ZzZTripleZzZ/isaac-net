"""ActionTerm that lets the policy choose what each robot sends (manager-based workflow).

    from isaac_net.isaac.mdp.actions import NetSendActionCfg
    actions.net_send = NetSendActionCfg(asset_name="robots", num_classes=2)    # one action channel per robot

Each robot's channel in [-1, 1] is bucketed into num_classes + 1 equal bins: none, class 1, ..., class num_classes
(with 2 classes, as the fleet task: < -1/3 none, < 1/3 small frame, else large frame). The classes are written to
env.isaac_net (isaac/runtime.py) in process_actions, i.e. at the start of the env step, which is when the frames are
captured; the network step later in the same env step submits them. apply_actions does nothing (no actuation).

bucket_send is pure torch; the ActionTerm classes exist only when Isaac Lab imports (NetSendAction is None otherwise).
"""
from __future__ import annotations

import torch


def bucket_send(actions: torch.Tensor, num_classes: int) -> torch.Tensor:
    """[E,R] long classes 0..num_classes from actions [E,R] in [-1, 1] (num_classes + 1 equal bins)."""
    k = num_classes + 1
    edges = torch.linspace(-1.0, 1.0, k + 1, device=actions.device)[1:-1]
    return torch.bucketize(actions.clamp(-1.0, 1.0).contiguous(), edges)


try:
    from dataclasses import MISSING  # noqa: F401

    from isaaclab.managers import ActionTerm, ActionTermCfg
    from isaaclab.utils import configclass

    from ..runtime import get_runtime

    class NetSendAction(ActionTerm):
        cfg: "NetSendActionCfg"

        def __init__(self, cfg, env):
            super().__init__(cfg, env)
            self._rt = None
            self._R = None
            self._raw = None
            self._send = None

        def _runtime(self):
            if self._rt is None:
                self._rt = get_runtime(self._env)
                self._rt.send_from_action = True
                self._R = self._rt.R
                self._raw = torch.zeros(self.num_envs, self._R, device=self.device)
                self._send = torch.zeros(self.num_envs, self._R, dtype=torch.long, device=self.device)
            return self._rt

        @property
        def action_dim(self) -> int:
            if self.cfg.num_robots is not None:
                return int(self.cfg.num_robots)
            return self._runtime().R

        @property
        def raw_actions(self) -> torch.Tensor:
            self._runtime()
            return self._raw

        @property
        def processed_actions(self) -> torch.Tensor:
            self._runtime()
            return self._send

        def process_actions(self, actions: torch.Tensor):
            rt = self._runtime()
            self._raw[:] = actions
            self._send = bucket_send(actions, self.cfg.num_classes)
            rt.write_send(self._send)

        def apply_actions(self):
            pass

        def reset(self, env_ids=None) -> None:
            if self._raw is not None:
                self._raw[env_ids if env_ids is not None else slice(None)] = 0.0

    @configclass
    class NetSendActionCfg(ActionTermCfg):
        """One action channel per robot choosing none / class 1 / ... / class num_classes (module docstring)."""
        class_type: type = NetSendAction
        num_classes: int = 2
        num_robots: int | None = None       # None: the network's robot count

except ImportError:                          # Isaac Lab not installed: only bucket_send is available
    NetSendAction = None
    NetSendActionCfg = None
