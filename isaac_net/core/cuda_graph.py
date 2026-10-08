"""CUDA graph capture that no garbage collection can interrupt.

Engines hold their captured graphs (torch.cuda.CUDAGraph) and often sit in reference cycles (hooks on their links that
point back at the engine), so a discarded engine, and its graphs with their memory pool, is freed only by a later run
of Python's cyclic garbage collector. If that run lands inside another capture, the destroyed graph's pool releases
memory while the stream is capturing, which PyTorch's caching allocator cannot do (an internal assert on the capture
status, then cudaErrorStreamCaptureInvalidated). torch.cuda.graph only collects with
torch.compiler.config.force_cudagraph_gc, so every capture in isaac_net goes through graph() below: it collects
first, outside the capture, and keeps the collector off while the stream captures.
"""
from __future__ import annotations

import contextlib
import gc

import torch


@contextlib.contextmanager
def graph(g, pool=None):
    """torch.cuda.graph(g, pool=pool) with a garbage collection before it and the collector disabled during it."""
    gc.collect()
    was = gc.isenabled()
    gc.disable()
    try:
        with torch.cuda.graph(g, pool=pool):
            yield
    finally:
        if was:
            gc.enable()


def drop(graphs, dev):
    """Destroy the captured graphs of a dict now, outside any capture, after the device finished their replays."""
    if graphs and torch.device(dev).type == "cuda":
        torch.cuda.synchronize(dev)
    graphs.clear()
