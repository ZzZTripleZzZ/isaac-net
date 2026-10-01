"""Triton side of proto/rng.py: the same counter-based hash in uint32, as @triton.jit helpers (used inline by the
fused L2 slot kernel) and as a standalone draw kernel (one launch per draw, graph-capturable). Imported lazily;
needs Triton and CUDA."""
import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice


@triton.jit
def mix32(x):
    x = x ^ (x >> 16)
    x = x * tl.full([], 0x21F0AAAD, tl.uint32)
    x = x ^ (x >> 15)
    x = x * tl.full([], 0x735A2D97, tl.uint32)
    return x ^ (x >> 15)


@triton.jit
def salt(v):
    return mix32(v) + tl.full([], 0x9E3779B9, tl.uint32)


@triton.jit
def u32(x):
    """Low 32 bits of an int64 / int32 value as uint32."""
    return (x.to(tl.int64) & 0xFFFFFFFF).to(tl.uint32)


@triton.jit
def key(s0, env, ep, chs, ctr, sts):
    """Base key of (seed key s0, env, episode, salted channel, counter, salted stream); all uint32."""
    h = mix32(s0 ^ salt(env))
    h = mix32(h ^ salt(ep))
    h = mix32(h ^ chs)
    h = mix32(h ^ salt(ctr))
    return mix32(h ^ sts)


@triton.jit
def elem(base, i):
    """Element hash of index i (int32 / int64 tensor) under base (uint32)."""
    w = u32(i) * tl.full([], 0x9E3779B9, tl.uint32)
    return mix32(mix32(base + w) ^ base)


@triton.jit
def uniform(base, i):
    return (elem(base, i) >> 8).to(tl.float32) * 5.9604644775390625e-08


@triton.jit
def normal(base, i):
    h1 = elem(base, 2 * i)
    h2 = elem(base, 2 * i + 1)
    u1 = ((h1 >> 8) + 1).to(tl.float32) * 5.9604644775390625e-08
    u2 = (h2 >> 8).to(tl.float32) * 5.9604644775390625e-08
    return libdevice.sqrt(-2.0 * libdevice.log(u1)) * libdevice.cos(6.283185307179586 * u2)


@triton.jit
def _draw_kernel(out_ptr, env_ptr, ep_ptr, ctr_ptr, s0, chs, sts, N, NORMAL: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    i = tl.program_id(1).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    m = i < N
    base = key(u32(s0), u32(tl.load(env_ptr + row)), u32(tl.load(ep_ptr + row)), u32(chs),
               u32(tl.load(ctr_ptr + row)), u32(sts))
    if NORMAL:
        z = normal(base, i)
    else:
        z = uniform(base, i)
    tl.store(out_ptr + row * N + i, z, mask=m)


def launch_draw(env, ep, ctr, s0, chs, sts, n, normal_):
    rows = env.shape[0]
    out = torch.empty(rows, n, dtype=torch.float32, device=env.device)
    if rows == 0 or n == 0:
        return out
    BLOCK = 1024 if n >= 1024 else max(16, triton.next_power_of_2(n))
    _draw_kernel[(rows, triton.cdiv(n, BLOCK))](out, env, ep, ctr, s0, chs, sts, n, NORMAL=bool(normal_),
                                               BLOCK=BLOCK)
    return out
