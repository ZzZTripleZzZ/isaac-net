# Traffic

`Requests` is what robots hand to the network in one control step. Every engine's `submit(t, requests)` takes it, or a bare `send` tensor, which is the same as `Requests(send)`.

```python
from isaaclab_net import Requests

send = torch.zeros(E, R, dtype=torch.long)     # 0 = nothing
send[:, 0] = 1                                 # robot 0 of every env sends a class-1 message
send[:, 1] = 2                                 # robot 1 sends a class-2 message
net.submit(None, Requests(send))
```

The class index selects the message size: class `c` has `NRConfig.msg_sizes[c - 1]` bytes, 4,000 and 30,000 bytes by default. At most one message per robot enters the queue per control step, and it is refused (the `accepted` mask returned by `submit` is `False`) when the robot's buffer of `NRConfig.frame_buffer` messages is full.

The optional `det` and `hid` fields carry one application event per environment through the network. `det[e, r] = True` marks a message that carries the environment's current event, for example a camera frame that captured a hazard, and `hid[e]` is that event's id. `step` then returns `det_env [E]`, which is `True` when a message carrying the current event id was delivered in that step. The Isaac layer exposes the same mechanism as a per-message `tag` (see [Isaac Lab layer](isaac.md)).

Periodic telemetry and control-packet generators are a roadmap item.

::: isaaclab_net.core.proto.netsim.Requests
    options:
      heading_level: 2
