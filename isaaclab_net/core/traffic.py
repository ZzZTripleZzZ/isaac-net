"""Traffic: what robots hand to the network each control step.

Requests(send, det=None, hid=None): send [E,R] long, 0 = nothing, c >= 1 = one message of traffic class c
(size NRConfig.msg_sizes[c-1] bytes); det [E,R] bool marks a message that carries the env's current task event
(for example a hazard detection), hid [E] long is that event's id. Every engine's submit() takes it.
Periodic telemetry and control-packet generators are a roadmap item.
"""
from .proto.netsim import Requests  # noqa: F401

__all__ = ["Requests"]
