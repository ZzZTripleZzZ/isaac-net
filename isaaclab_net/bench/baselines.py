"""Baselines: random, a scripted heuristic with a queue-aware send rule, and PPO with a shared MLP or GRU actor.

Every baseline acts through the same interface on a NetTask:

    pol.reset(task)                                   # before an evaluation or a training run
    cont, send = pol.act(task, obs, done)             # obs [E,R,D]; done [E] of the previous step (None at start)

PPO follows the trainer pattern of the earlier kill-test code: one actor shared by every robot (parameter sharing,
per-robot observations), a hybrid action (Gaussian velocity or controller parameters in [-1,1], categorical send
choice), GAE(0.99, 0.95), clip 0.2, 4 epochs x 8 minibatches, entropy 0.005, Adam 3e-4, gradient norm 0.5.
The GRU variant adds a recurrent trunk trained on the rollout sequences (hidden state reset at episode starts).
"""
from __future__ import annotations

import time
from typing import Optional

import torch
import torch.nn as nn
from torch.distributions import Categorical, Normal

BASELINES = ("random", "heuristic", "ppo_mlp", "ppo_gru")


# ------------------------------------------------------------------------------------------------ scripted
class RandomPolicy:
    """Uniform continuous action in [-1,1] and a uniform send choice (own generator)."""
    name = "random"

    def __init__(self, seed: int = 0):
        self.seed = seed

    def reset(self, task):
        self.gen = torch.Generator(device=task.dev)
        self.gen.manual_seed(int(self.seed) * 31 + 5)

    def act(self, task, obs, done=None):
        E, R, A = task.E, task.R, task.action_spec.cont_dim
        cont = torch.rand(E, R, A, device=task.dev, generator=self.gen) * 2 - 1
        send = torch.randint(0, task.action_spec.n_send, (E, R), device=task.dev, generator=self.gen)
        return cont, send


class HeuristicPolicy:
    """The task's scripted motion (task.heuristic(), privileged task state) and a queue-aware send rule: send the
    task's preferred choice when the robot's uplink queue is empty and at least min_interval steps have passed
    since its last send, else the task's idle choice."""
    name = "heuristic"

    def __init__(self, min_interval: int = 1):
        self.min_interval = int(min_interval)

    def reset(self, task):
        self.since = torch.full((task.E, task.R), 10 ** 6, dtype=torch.long, device=task.dev)

    def act(self, task, obs, done=None):
        if done is not None:
            self.since = torch.where(done[:, None], torch.full_like(self.since, 10 ** 6), self.since)
        cont, ready, busy = task.heuristic()
        ok = (task.queue_len == 0) & (self.since >= self.min_interval)
        send = torch.where(ok, ready, busy)
        self.since = torch.where(send != busy, torch.ones_like(self.since), self.since + 1)
        return cont, send


# ------------------------------------------------------------------------------------------------ PPO models
class MLPActorCritic(nn.Module):
    recurrent = False

    def __init__(self, obs_dim, cont_dim, n_send, h=128):
        super().__init__()
        self.body = nn.Sequential(nn.Linear(obs_dim, h), nn.Tanh(), nn.Linear(h, h), nn.Tanh())
        self.mu = nn.Linear(h, cont_dim)
        self.logit = nn.Linear(h, n_send)
        self.logstd = nn.Parameter(torch.full((cont_dim,), -0.5))
        self.v = nn.Sequential(nn.Linear(obs_dim, h), nn.Tanh(), nn.Linear(h, h), nn.Tanh(), nn.Linear(h, 1))

    def forward(self, o, h=None):
        z = self.body(o)
        return Normal(self.mu(z), self.logstd.exp()), Categorical(logits=self.logit(z)), self.v(o).squeeze(-1), h

    def init_hidden(self, n, device):
        return None


class GRUActorCritic(nn.Module):
    recurrent = True

    def __init__(self, obs_dim, cont_dim, n_send, h=128):
        super().__init__()
        self.hdim = h
        self.enc = nn.Sequential(nn.Linear(obs_dim, h), nn.Tanh())
        self.gru = nn.GRUCell(h, h)
        self.mu = nn.Linear(h, cont_dim)
        self.logit = nn.Linear(h, n_send)
        self.logstd = nn.Parameter(torch.full((cont_dim,), -0.5))
        self.vh = nn.Linear(h, 1)

    def init_hidden(self, n, device):
        return torch.zeros(n, self.hdim, device=device)

    def forward(self, o, h):
        h = self.gru(self.enc(o), h)
        return Normal(self.mu(h), self.logstd.exp()), Categorical(logits=self.logit(h)), self.vh(h).squeeze(-1), h

    def sequence(self, O, h0, starts):
        """O [T,N,D], h0 [N,H], starts [T,N] bool (hidden reset before step t) -> mu, logits, v, all [T,N,...]."""
        h, mus, logits, vs = h0, [], [], []
        for t in range(O.shape[0]):
            h = torch.where(starts[t][:, None], torch.zeros_like(h), h)
            h = self.gru(self.enc(O[t]), h)
            mus.append(self.mu(h))
            logits.append(self.logit(h))
            vs.append(self.vh(h).squeeze(-1))
        return torch.stack(mus), torch.stack(logits), torch.stack(vs)


class PPOPolicy:
    """A trained actor as a baseline: deterministic (mean velocity, most likely send choice) by default."""

    def __init__(self, model, name, stochastic=False):
        self.model, self.name, self.stochastic = model, name, stochastic

    def reset(self, task):
        self.h = self.model.init_hidden(task.E * task.R, task.dev)

    @torch.no_grad()
    def act(self, task, obs, done=None):
        E, R = task.E, task.R
        o = obs.reshape(E * R, -1)
        if self.h is not None and done is not None:
            self.h = torch.where(done.repeat_interleave(R)[:, None], torch.zeros_like(self.h), self.h)
        dn, dc, _, self.h = self.model(o, self.h)
        if self.stochastic:
            cont, send = dn.sample(), dc.sample()
        else:
            cont, send = dn.mean, dc.probs.argmax(-1)
        return cont.clamp(-1, 1).view(E, R, -1), send.view(E, R)


def make_model(arch, task, hidden=128):
    cls = {"mlp": MLPActorCritic, "gru": GRUActorCritic}[arch]
    return cls(task.obs_spec.dim, task.action_spec.cont_dim, task.action_spec.n_send, hidden).to(task.dev)


# ------------------------------------------------------------------------------------------------ PPO training
def train_ppo(task, arch="mlp", iters=30, horizon=32, lr=3e-4, seed=0, hidden=128, gamma=0.99, lam=0.95,
              epochs=4, minibatches=8, clip=0.2, ent=0.005, log=None):
    """Train a shared actor on `task` (a NetTask, reset here). Returns (model, info) with the learning curve
    (one row per finished env-episode batch: iteration, mean return and task metric) and the wall time."""
    torch.manual_seed(int(seed) * 97 + 11)
    dev = task.dev
    model = make_model(arch, task, hidden)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    E, R = task.E, task.R
    N = E * R
    obs = task.reset().reshape(N, -1)
    h = model.init_hidden(N, dev)
    start = torch.zeros(N, dtype=torch.bool, device=dev)
    curve, t0, steps = [], time.time(), 0
    key = task.METRIC.key
    for it in range(iters):
        bo, ba, bc, blp, bv, br, bd, bs = [], [], [], [], [], [], [], []
        h0 = h.detach() if h is not None else None
        eps = []
        for _ in range(horizon):
            with torch.no_grad():
                dn, dc, v, h = model(obs, h)
                a = dn.sample()
                c = dc.sample()
                lp = dn.log_prob(a).sum(-1) + dc.log_prob(c)
            o2, rew, done, info = task.step(a.clamp(-1, 1).view(E, R, -1), c.view(E, R))
            steps += 1
            dn_r = done.repeat_interleave(R)
            bo.append(obs); ba.append(a); bc.append(c); blp.append(lp); bv.append(v)
            br.append(rew.reshape(N)); bd.append(dn_r.float()); bs.append(start)
            eps += info["episodes"]
            start = dn_r
            if h is not None:
                h = torch.where(dn_r[:, None], torch.zeros_like(h), h)
            obs = o2.reshape(N, -1)
        with torch.no_grad():
            nv = model(obs, h)[2]
            adv = torch.zeros(horizon, N, device=dev)
            last = torch.zeros(N, device=dev)
            for s in reversed(range(horizon)):
                nxt = nv if s == horizon - 1 else bv[s + 1]
                nonterm = 1.0 - bd[s]
                delta = br[s] + gamma * nxt * nonterm - bv[s]
                last = delta + gamma * lam * nonterm * last
                adv[s] = last
            ret = adv + torch.stack(bv)
        O, A, C = torch.stack(bo), torch.stack(ba), torch.stack(bc)          # [T,N,...]
        LP, S = torch.stack(blp), torch.stack(bs)
        ADV = (adv - adv.mean()) / (adv.std() + 1e-8)
        for _ in range(epochs):
            if model.recurrent:
                perm = torch.randperm(N, device=dev)
                chunks = perm.chunk(minibatches)
            else:
                perm = torch.randperm(horizon * N, device=dev)
                chunks = perm.chunk(minibatches)
            for ix in chunks:
                if model.recurrent:
                    mu, logit, v = model.sequence(O[:, ix], h0[ix], S[:, ix])
                    dn, dc = Normal(mu, model.logstd.exp()), Categorical(logits=logit)
                    a_, c_, lp0, adv_, ret_ = A[:, ix], C[:, ix], LP[:, ix], ADV[:, ix], ret[:, ix]
                else:
                    o = O.reshape(horizon * N, -1)[ix]
                    dn, dc, v, _ = model(o, None)
                    a_ = A.reshape(horizon * N, -1)[ix]
                    c_, lp0 = C.reshape(-1)[ix], LP.reshape(-1)[ix]
                    adv_, ret_ = ADV.reshape(-1)[ix], ret.reshape(-1)[ix]
                lp = dn.log_prob(a_).sum(-1) + dc.log_prob(c_)
                ratio = (lp - lp0).exp()
                pg = -torch.min(ratio * adv_, ratio.clamp(1 - clip, 1 + clip) * adv_).mean()
                vl = (v - ret_).pow(2).mean()
                entropy = dn.entropy().sum(-1).mean() + dc.entropy().mean()
                loss = pg + 0.5 * vl - ent * entropy
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                opt.step()
        if eps:
            row = {"iter": it, "return": sum(e["return"] for e in eps) / len(eps),
                   key: sum(e[key] for e in eps) / len(eps), "episodes": len(eps)}
            curve.append(row)
            if log:
                log(f"it {it} return {row['return']:.3f} {key} {row[key]:.4f} t {time.time() - t0:.0f}s")
    sec = time.time() - t0
    return model, {"arch": arch, "iters": iters, "horizon": horizon, "lr": lr, "hidden": hidden, "curve": curve,
                   "train_sec": sec, "task_steps": steps, "samples": steps * N}


def make_policy(name: str, seed: int = 0, model=None) -> Optional[object]:
    if name == "random":
        return RandomPolicy(seed)
    if name == "heuristic":
        return HeuristicPolicy()
    if name in ("ppo_mlp", "ppo_gru"):
        if model is None:
            raise ValueError(f"{name} needs a trained model (bench.runner trains it)")
        return PPOPolicy(model, name)
    raise ValueError(f"unknown baseline {name!r}; one of {BASELINES}")
