"""NetModuleMJX: the network module inside jitted, vmapped MuJoCo Playground / MJX (JAX) code.

    net = NetModuleMJX("L2-legacy", E, R, "cuda", NRConfig(msg_sizes=(4000.0, 30000.0)), backend="triton", seed=0)
    # inside the per-env step of a Playground MjxEnv (vmapped over E envs by the Brax wrappers):
    out = net(poses_end[R,3], send[R], tag[R], cur_tag[], reset[])    # dict of JAX arrays (NET_OUTPUTS)
    obs = jnp.concatenate([..., out["feats"].reshape(-1)])            # [R,4]: AoI, SNR, queued frames, delivered

The engine is the same torch code as the Isaac Lab layer: NetModuleMJX owns an `isaac_net.isaac.NetModule`
(pure torch, no Isaac imports) built by make_engine(level, E, R, device, NRConfig, backend), so every level and
backend of the Isaac layer is available here. Only the glue differs:

JAX <-> torch interop
  jax.experimental.buffer_callback (JAX >= 0.7) runs a Python function in the middle of an XLA program and hands it
  the device buffers of its operands and results plus XLA's CUDA stream. The callback wraps them with
  torch.from_dlpack, which is zero-copy (same device pointer; checked in tests/mjx), and enqueues every torch
  kernel on XLA's stream through torch.cuda.ExternalStream. The network step is therefore ordered after the MJX
  physics of the same XLA program and before everything that reads its outputs, with no stream synchronization
  and no host round trip of the data. Under jax.vmap the callback uses vmap_method="broadcast_all": it is called
  once per step with the whole batch [E, ...], exactly what the engine wants.

Copies that remain (all on the device, O(E*R) elements, none through the host)
  * int32 -> int64 casts of send / tag / cur_tag (JAX runs without x64; the engine indexes with int64).
  * the z column appended to [E,R,2] poses (NetModule does it; pass [R,3] poses to skip it).
  * the results: NetModule returns fresh tensors, which are written into the XLA-owned result buffers with one
    device-to-device copy each (feats [E,R,4] plus a few [E,R] vectors).

Host synchronization
  One per step, and only for partial resets: the per-env reset flags are a device array, and the engine's reset
  draws (fading, shadowing) depend on how many envs reset, so reset() needs the reset indices on the host. The
  callback reads them with one nonzero(), which waits for the physics of the step to finish. Isaac Lab does the
  same when it turns reset_buf into env_ids. A graph-safe masked reset in the core (draw for all envs, keep the
  rows of the reset ones) would remove it; see docs/backends-mjx.md.

Resets
  Playground's BraxAutoResetWrapper resets a done env by restoring the cached first mjx.Data and observation, and
  leaves `info` alone, so the env tells the module about a reset through the `reset` flag of the next step (the
  fleet env uses data.time == 0, which holds only for data that come out of reset()). The module resets those envs
  before it submits the step's messages, which is the order of the Isaac layer (reset after the done step, then
  submit / step of the next one). Other envs are bitwise unaffected (the engine contract).

Constraints
  * One NetModuleMJX per env batch: its callback checks that the batch it sees has E envs. A Brax eval env needs its
    own instance (built with num_eval_envs).
  * The network state lives in torch, outside the JAX state. A jitted function that steps the env advances the
    network once per execution, so do not re-execute a step for the same state (e.g. jax.checkpoint recomputation);
    gradients through the network are not defined.
  * JAX and torch must see the same GPU (JAX device 0 = torch "cuda:0"). Set XLA_PYTHON_CLIENT_PREALLOCATE=false
    so that JAX does not take 75% of the GPU memory before torch allocates the engine.
"""
from __future__ import annotations

import atexit
import contextlib
import gc
import weakref
from typing import Optional

import jax
import jax.numpy as jnp
import torch
from jax.experimental.buffer_callback import buffer_callback

from ..core.config import NRConfig
from ..isaac.net_module import FAST_BACKENDS, NetModule, TrafficRequest, net_features

# name -> (trailing shape given R, dtype) of the per-env outputs
NET_OUTPUTS = {
    "feats": (lambda R: (R, 4), jnp.float32),       # net_features: AoI/5 s, SNR/40 dB, queued/F, delivered
    "delivered": (lambda R: (R,), jnp.bool_),       # at least one message of the robot delivered this step
    "newest_cap": (lambda R: (R,), jnp.int32),      # newest capture step delivered this step (-1: none), env clock
    "last_cap": (lambda R: (R,), jnp.int32),        # newest capture delivered this episode (0: the reset state)
    "aoi_s": (lambda R: (R,), jnp.float32),         # age of that information at the end of the step
    "queue_len": (lambda R: (R,), jnp.int32),       # frames queued after the step
    "sinr_db": (lambda R: (R,), jnp.float32),       # SNR of the step
    "tag_delivered": (lambda R: (), jnp.bool_),     # a message tagged with the env's current tag was delivered
}


def _torch_outputs(o: dict, step_dt: float, depth: int) -> dict:
    """The NET_OUTPUTS entries from a NetModule.step dict (torch, [E, ...])."""
    return dict(feats=net_features(o, step_dt, depth), delivered=o["delivered"], newest_cap=o["newest_cap"],
                last_cap=o["last_cap"], aoi_s=o["aoi_s"], queue_len=o["queue_len"], sinr_db=o["sinr_db"],
                tag_delivered=o["tag_delivered"])


_LIVE: "weakref.WeakSet[NetModule]" = weakref.WeakSet()


@atexit.register
def _release_graphs():
    """Free the CUDA graphs of the fast backends before JAX tears down the CUDA context at exit (otherwise the
    graph destructors run after it and abort with 'context is destroyed')."""
    for net in list(_LIVE):
        graphs = getattr(net.eng, "_graphs", None)
        if graphs:
            graphs.clear()
    gc.collect()


def build_module(level, E, R, device, config, backend, seed, **kwargs) -> NetModule:
    """The torch NetModule, built deterministically from `seed` (the triton backend draws its Philox seed from the
    global CPU generator at construction; that draw is taken under a forked, seeded CPU RNG)."""
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        net = NetModule(level, E, R, device, config, backend, seed=seed, **kwargs)
    prewarm(net)
    _LIVE.add(net)
    return net


def prewarm(net: NetModule):
    """Capture the CUDA graphs (and compile the triton kernels) of a fast backend before any JAX program runs.

    The fast backends capture lazily on the first submit / step, and a capture calls torch.cuda.synchronize and
    must not overlap other CUDA work of the process. Doing it here keeps it out of the XLA callback. One dummy
    step, then a full reset; the global CUDA RNG is restored, so the only trace of the warm-up is in the engine
    and radio generators (deterministic for a given seed, and identical in replay())."""
    if net.backend not in FAST_BACKENDS or net.dev.type != "cuda" or net.backend == "eager":
        return
    st = torch.cuda.get_rng_state(net.dev)
    z = torch.zeros(net.E, net.R, dtype=torch.long, device=net.dev)
    net.submit(None, TrafficRequest(send=z))
    net.step(None, torch.zeros(net.E, net.R, 3, device=net.dev), cur_tag=torch.zeros(net.E, dtype=torch.long,
                                                                                       device=net.dev))
    net.reset(None)
    torch.cuda.set_rng_state(st, net.dev)
    torch.cuda.synchronize(net.dev)


class NetModuleMJX:
    """The network of a batch of E envs x R robots, callable from jitted, vmapped JAX code (module docstring).

    level, num_envs, num_robots, device, config, backend and **kwargs (pose_chunks, gnb_pos, radio, ranges, params,
    inject) are those of isaac_net.isaac.NetModule. seed fixes the engine, the radio and the triton Philox seed.
    record=True keeps the inputs of every step (poses, sends, tags, resets, the CUDA RNG state) for replay().
    """

    def __init__(self, level="L2-legacy", num_envs: int = None, num_robots: int = None, device="cuda",
                 config: Optional[NRConfig] = None, backend: str = "graph", *, seed: int = 0, record: bool = False,
                 **kwargs):
        self.level, self.backend, self.seed = level, backend, int(seed)
        self.E, self.R, self.dev = int(num_envs), int(num_robots), torch.device(device)
        if self.dev.type == "cuda" and self.dev.index is None:
            self.dev = torch.device("cuda", torch.cuda.current_device())
        self.config = config if config is not None else NRConfig()
        self.kwargs = kwargs
        self.net = build_module(level, self.E, self.R, self.dev, self.config, backend, self.seed, **kwargs)
        self.step_dt, self.F = self.net.step_dt, self.net.F
        self.record = record
        self.records: list[dict] = []
        self.calls = 0
        self.resets = 0
        shapes = {k: jax.ShapeDtypeStruct(f(self.R), dt) for k, (f, dt) in NET_OUTPUTS.items()}
        self._fn = buffer_callback(self._callback, shapes, has_side_effect=True, vmap_method="broadcast_all")

    # ------------------------------------------------------------------ JAX side
    def __call__(self, poses_end, send, tag=None, cur_tag=None, reset=None) -> dict:
        """One env's network step, to be vmapped over the E envs (the Playground convention).

        poses_end [R,3] (or [R,2]) float32 positions at the END of the control step (radio coordinates);
        send [R] int (0 nothing, c >= 1 one message of class c); tag [R] int per-message label (-1 none);
        cur_tag [] int the env's current tag (-1 none); reset [] bool: reset this env before the step.
        Returns the NET_OUTPUTS dict of JAX arrays ([R, ...] per env; [E, R, ...] once vmapped)."""
        R = self.R
        send = jnp.asarray(send, jnp.int32)
        tag = jnp.full((R,), -1, jnp.int32) if tag is None else jnp.asarray(tag, jnp.int32)
        cur_tag = jnp.int32(-1) if cur_tag is None else jnp.asarray(cur_tag, jnp.int32)
        reset = jnp.bool_(False) if reset is None else jnp.asarray(reset, jnp.bool_)
        return self._fn(jnp.asarray(poses_end, jnp.float32), send, tag, cur_tag, reset)

    def step_batched(self, poses_end, send, tag=None, cur_tag=None, reset=None) -> dict:
        """The same step for arrays with the env dimension in front ([E,R,3], [E,R], [E,R], [E], [E])."""
        E, R = self.E, self.R
        tag = jnp.full((E, R), -1, jnp.int32) if tag is None else tag
        cur_tag = jnp.full((E,), -1, jnp.int32) if cur_tag is None else cur_tag
        reset = jnp.zeros((E,), jnp.bool_) if reset is None else reset
        return jax.vmap(self.__call__)(poses_end, send, tag, cur_tag, reset)

    # ------------------------------------------------------------------ torch side (runs inside XLA)
    def _stream(self, ctx):
        if self.dev.type != "cuda":
            return contextlib.nullcontext()
        return torch.cuda.stream(torch.cuda.ExternalStream(ctx.stream, device=self.dev))

    def _callback(self, ctx, out, poses, send, tag, cur_tag, reset):
        with torch.cuda.device(self.dev) if self.dev.type == "cuda" else contextlib.nullcontext(), self._stream(ctx):
            P = torch.from_dlpack(poses)                        # zero-copy views of XLA's buffers
            if P.shape[0] != self.E or P.shape[1] != self.R:
                raise ValueError(f"NetModuleMJX was built for E={self.E}, R={self.R}; the env step passed poses "
                                 f"{tuple(P.shape)} (one NetModuleMJX per env batch: the eval env needs its own)")
            ids = torch.from_dlpack(reset).nonzero(as_tuple=True)[0]    # the one host sync (module docstring)
            if ids.numel():
                self.net.reset(ids)
                self.resets += int(ids.numel())
            snd = torch.from_dlpack(send).long()
            tg = torch.from_dlpack(tag).long()
            cur = torch.from_dlpack(cur_tag).long()
            if self.record:
                rng = torch.cuda.get_rng_state(self.dev) if self.dev.type == "cuda" else torch.get_rng_state()
                self.records.append(dict(poses=P.clone(), send=snd.clone(), tag=tg.clone(), cur_tag=cur.clone(),
                                         reset_ids=ids.clone(), rng=rng))
            self.net.submit(None, TrafficRequest(send=snd, tag=tg))
            o = self.net.step(None, P, cur_tag=cur)
            res = _torch_outputs(o, self.step_dt, self.F)
            for k, buf in out.items():
                torch.from_dlpack(buf).copy_(res[k])            # one D2D copy per result, on XLA's stream
            self.calls += 1

    # ------------------------------------------------------------------ passthroughs
    @property
    def clock(self) -> torch.Tensor:
        return self.net.clock

    def set_params(self, env_ids=None, **values):
        self.net.set_params(env_ids, **values)

    def sample_params(self, env_ids=None, ranges=None):
        self.net.sample_params(env_ids, ranges)


def replay(module: NetModuleMJX, records=None, backend: Optional[str] = None) -> list:
    """Drive a fresh torch NetModule directly (no JAX) with the recorded inputs of `module` (record=True).

    backend: None = the module's backend. The replay module is built with the same level, config, seed and
    kwargs and the same warm-up; before each submit it restores the CUDA RNG state recorded in the env, so the
    same backend reproduces the in-env outputs bitwise, and "reference" replays an "eager" module bitwise (the
    two consume the RNG identically; tests/test_isaac_layer.py). Returns one NET_OUTPUTS dict (torch) per step."""
    records = module.records if records is None else records
    be = module.backend if backend is None else backend
    net = build_module(module.level, module.E, module.R, module.dev, module.config, be, module.seed, **module.kwargs)
    outs = []
    for r in records:
        if r["reset_ids"].numel():
            net.reset(r["reset_ids"])
        if module.dev.type == "cuda":
            torch.cuda.set_rng_state(r["rng"], module.dev)
        else:
            torch.set_rng_state(r["rng"])
        net.submit(None, TrafficRequest(send=r["send"], tag=r["tag"]))
        o = net.step(None, r["poses"], cur_tag=r["cur_tag"])
        outs.append({k: v.clone() for k, v in _torch_outputs(o, net.step_dt, net.F).items()})
    return outs
