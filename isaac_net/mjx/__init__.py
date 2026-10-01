"""MuJoCo Playground / MJX (JAX) backend of the network module (see docs/backends-mjx.md).

    net_module.py   NetModuleMJX: the torch NetModule (isaac_net.isaac.net_module, which has no Isaac imports)
                    driven from inside jitted, vmapped JAX code through jax.experimental.buffer_callback, with
                    zero-copy DLPack views of the JAX buffers on XLA's own CUDA stream; replay() for validation

The example env (isaac_net/examples/mjx_fleet_env.py, MJXFleetEnv) is a Playground MjxEnv with R
velocity-actuated planar bodies per env, the fleet task of the Isaac demo and the network in its step.

Importing this package imports JAX; `import isaac_net` alone never does.
"""
from .net_module import NET_OUTPUTS, NetModuleMJX, prewarm, replay  # noqa: F401

__all__ = ["NetModuleMJX", "NET_OUTPUTS", "prewarm", "replay"]
