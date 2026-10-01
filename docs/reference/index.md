# API reference

The public API is small. Most code needs three names from the top-level package:

```python
from isaac_net import make_engine, NRConfig, Requests
```

| Page | Module | What it covers |
|:---|:---|:---|
| [Engine](engine.md) | `isaac_net.core.engine` | `make_engine`, the engine contract every level honors, the output dict, `NREngine` |
| [Configuration](config.md) | `isaac_net.core.config` | `NRConfig` with every field, the presets, `unused_fields` and strict mode |
| [Traffic](traffic.md) | `isaac_net.core.traffic` | `Requests` (the policy's messages) and `TrafficModel` (generators in the `L2` step) |
| [Fidelity levels](levels.md) | `isaac_net.core.levels` | the level list and semantics, surrogate parameters, the bounds |
| [Isaac Lab layer](isaac.md) | `isaac_net.isaac` | `NetModule`, `NetEnvMixin`, `MessageHistory`, the Isaac radio and the domain-randomization event term |
| [ns-3 bridges](bridges.md) | `isaac_net.bridges` | how to run the lockstep, pool and offline bridges (validation only) |

The pages combine hand-written descriptions of the contract with entries generated from the docstrings. Names that start with an underscore are internal and are not listed. `isaac_net.core` has no simulator imports, and nothing in `isaac_net.isaac` imports Isaac Lab when the package is imported, so every module on these pages imports on a CPU-only machine.
