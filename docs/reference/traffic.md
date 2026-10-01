# Traffic

`Requests` is what the policy hands to the network in one control step. Every engine's `submit(t, requests)` takes it, or a bare `send` tensor, which is the same as `Requests(send)`.

```python
from isaac_net import Requests

send = torch.zeros(E, R, dtype=torch.long)     # 0 = nothing
send[:, 0] = 1                                 # robot 0 of every env sends a class-1 message
send[:, 1] = 2                                 # robot 1 sends a class-2 message
net.submit(None, Requests(send))
```

The class index selects the message size: class `c` has `NRConfig.msg_sizes[c - 1]` bytes, 4,000 and 30,000 bytes by default. At most one message per robot enters the queue per control step, and it is refused (the `accepted` mask returned by `submit` is `False`) when the robot's buffer of `NRConfig.frame_buffer` messages is full.

The optional `det` and `hid` fields carry one application event per environment through the network. `det[e, r] = True` marks a message that carries the environment's current event, for example a camera frame that captured a hazard, and `hid[e]` is that event's id. `step` then returns `det_env [E]`, which is `True` when a message carrying the current event id was delivered in that step. The Isaac layer exposes the same mechanism as a per-message `tag` (see [Isaac Lab layer](isaac.md)).

## Traffic models

On level `L2`, `NRConfig(traffic=[...])` adds generators that run inside the engine step, next to the policy's `submit()`: `TrafficModel.periodic` (periods may be shorter than the control step), `.bursty` (Markov on/off), `.video` (I/P frame pattern), `.event` (task triggers through `step(..., triggers=)`) and `.policy()`. Each generated message carries an arrival slot inside the step, and its delay counts from that slot. Every other level refuses traffic models with a `ValueError`. The [configurability guide](../configurability.md#traffic-models) explains the models, the arrival offsets and the limits.

```python
from isaac_net.core.traffic import TrafficModel as TM

cfg = NRConfig(traffic=[TM.periodic(200, period_ms=10).on(range(4)), TM.event(4000, trigger="alarm")])
net = make_engine("L2", E, R, "cuda", cfg, seed=0)
out = net.step(None, poses, triggers={"alarm": alarm_mask})
```

::: isaac_net.core.traffic.TrafficModel
    options:
      heading_level: 2

::: isaac_net.core.proto.netsim.Requests
    options:
      heading_level: 2
