# API reference

The public API is small. Most code needs three names from the top-level package:

```python
from isaaclab_net import make_engine, NRConfig, Requests
```

| Page | Module | What it covers |
|:---|:---|:---|
| [Engine](engine.md) | `isaaclab_net.core.engine` | `make_engine`, the engine contract every level honors, the output dict, `NREngine` |
| [Configuration](config.md) | `isaaclab_net.core.config` | `NRConfig` with every field, the presets, `unused_fields` and strict mode |
| [Traffic](traffic.md) | `isaaclab_net.core.traffic` | `Requests`, the messages robots hand to the network |
| [Fidelity levels](levels.md) | `isaaclab_net.core.levels` | the level list and semantics, surrogate parameters, the bounds |
| [Isaac Lab layer](isaac.md) | `isaaclab_net.isaac` | `NetModule`, `NetEnvMixin`, `MessageHistory`, the Isaac radio and the domain-randomization event term |
| [ns-3 bridges](bridges.md) | `isaaclab_net.bridges` | how to run the lockstep, pool and offline bridges (validation only) |

The pages combine hand-written descriptions of the contract with entries generated from the docstrings. Names that start with an underscore are internal and are not listed. `isaaclab_net.core` has no simulator imports, and nothing in `isaaclab_net.isaac` imports Isaac Lab when the package is imported, so every module on these pages imports on a CPU-only machine.
