"""Step-dict schema: one registry of every key an engine's step(t, x) dict can return.

Every engine has output_schema() -> {key: {"shape", "dtype", "unit", "doc", "when"}} for the keys its step returns
under its current config, in the order below. The descriptions here are the single source: they repeat the step-key
lists of the module docstrings (proto/netsim.py NetBase.step, engine.py NREngine.step, edge.py, energy.py,
background.py, access.py, adaptive.py, wifi/engine.py, and docs/obstacles.md for los / blocked), and
docs/configurability.md "Output schema" is generated from STEP_KEYS (markdown_table).

Shapes use E (envs), R (robots per env), F (frame buffer, NRConfig.frame_buffer), C (cells) and R_bg (background
UEs per env). Units: "bool" (mask), "step" (control-step index of the env clock), "steps" (duration in control
steps), "slot" (slot index inside the control step), "bytes", "dB", "J", "W", "ms", "Mbit/s", "count", "fraction",
"class", "id", "state", "m" (metres). "when" says when the key is present beyond the config switches that select it
("always", or a condition on the input or the call, such as "pose input").
"""
from __future__ import annotations

# key: (shape, dtype, unit, doc)
STEP_KEYS = {
    # ---- every level (proto/netsim.py NetBase.step: the contract of ARCHITECTURE.md)
    "newest": ("[E,R]", "int64", "step", "Newest capture step delivered this step for each robot, -1 if none."),
    "det_env": ("[E]", "bool", "bool", "A message carrying the env's current hazard (hid of the last submit) "
                                       "arrived this step."),
    "delivered": ("[E,R,F]", "bool", "bool", "Message slot delivered during this step (slots as queued before the "
                                              "step)."),
    "timed_out": ("[E,R,F]", "bool", "bool", "Message slot dropped at its application deadline (timeout_steps) this "
                                              "step."),
    "cap": ("[E,R,F]", "int64", "step", "Capture step of the message in each slot, -1 for an empty slot."),
    "cls": ("[E,R,F]", "int64", "class", "Traffic class of the message in each slot (index into msg_sizes + 1), 0 "
                                         "for an empty slot."),
    "delay": ("[E,R,F]", "float32", "steps", "Delivery time minus capture (or arrival) time in control steps, NaN "
                                             "if not delivered."),
    "queue_len": ("[E,R]", "int64", "count", "Messages in the robot's uplink queue after the step."),
    "queue_bytes": ("[E,R]", "float32", "bytes", "Accepted uplink bytes not yet resolved in order after the step."),
    "sinr_db": ("[E,R]", "float32", "dB", "Wideband SNR or serving-link SINR used for this step."),
    "t": ("[E]", "int64", "step", "Env clock value of this step (the clock is t + 1 afterwards)."),
    # ---- NR engine (engine.py NREngine.step) and the multi-cell legacy engine
    "dropped": ("[E,R,F]", "bool", "bool", "Message lost under RLC UM (harq_fail='drop'), on a handover with "
                                            "ho_rlc='flush' or on radio link failure, resolved this step."),
    "serving_cell": ("[E,R]", "int64", "id", "Serving cell (access point at level WIFI), 0 with one cell."),
    "dl_newest": ("[E,R]", "int64", "step", "Newest capture step of a downlink message delivered this step, -1 if "
                                            "none."),
    "dl_queue_len": ("[E,R]", "int64", "count", "Messages in the robot's downlink queue after the step."),
    "rank": ("[E,R]", "int64", "count", "Rank (layers) of the robot's last new uplink transport block, 1 without "
                                        "uplink rank 2."),
    "dl_rank": ("[E,R]", "int64", "count", "Rank (layers) of the robot's last new downlink transport block."),
    "rlf": ("[E,R]", "bool", "bool", "Robot is in radio link failure (T310 expired, not yet re-established)."),
    "los": ("[E,R]", "bool", "bool", "Line-of-sight state of the serving link (docs/obstacles.md)."),
    "blocked": ("[E,R]", "bool", "bool", "A dynamic blocker (robot, screen, model-A region) is on the serving "
                                         "link's direct path."),
    # ---- per-message extras (traffic models or submit(tag=, priority=, deadline_ms=))
    "arrival": ("[E,R,F]", "float64", "steps", "Arrival time of the message in env-clock control steps, including "
                                               "the in-step offset; NaN for an empty slot."),
    "arrival_slot": ("[E,R,F]", "int64", "slot", "Slot of the control step in which the message arrived, -1 for "
                                                 "an empty slot."),
    "tag": ("[E,R,F]", "int64", "id", "Tag of the message (traffic model position + 1, 0 = policy), 0 when empty."),
    "priority": ("[E,R,F]", "int64", "class", "Priority of the message (QoS class with scheduler='qos')."),
    "bytes": ("[E,R,F]", "int64", "bytes", "Bytes of the message on the air (payload plus packet overheads)."),
    "deadline_miss": ("[E,R,F]", "bool", "bool", "Message delivered after its deadline_ms, or lost with a finite "
                                                 "deadline."),
    "gen_accepted": ("[E,R]", "int64", "count", "Uplink messages generated by the traffic models and accepted this "
                                                "step."),
    "gen_bytes": ("[E,R]", "int64", "bytes", "Uplink bytes on the air of the generated messages accepted this step."),
    # ---- downlink traffic models (traffic.py, direction='dl')
    "dl_delivered": ("[E,R,F]", "bool", "bool", "Downlink message slot delivered this step."),
    "dl_lost": ("[E,R,F]", "bool", "bool", "Downlink message slot timed out or dropped this step."),
    "dl_delay": ("[E,R,F]", "float32", "steps", "Downlink delivery time minus arrival time in control steps, NaN "
                                                "if not delivered."),
    "dl_tag": ("[E,R,F]", "int64", "id", "Tag of the downlink message, 0 when empty."),
    "dl_generated": ("[E,R,F]", "bool", "bool", "Downlink slot holds a message from a traffic model (not an edge "
                                                "command)."),
    "dl_bytes": ("[E,R,F]", "int64", "bytes", "Bytes of the downlink message on the air."),
    "dl_deadline_miss": ("[E,R,F]", "bool", "bool", "Downlink message delivered after its deadline, or lost with a "
                                                    "finite deadline."),
    "gen_dl_accepted": ("[E,R]", "int64", "count", "Downlink messages generated and accepted this step."),
    "gen_dl_bytes": ("[E,R]", "int64", "bytes", "Downlink bytes on the air of the generated messages accepted this "
                                                "step."),
    # ---- access state machine (access.py, rach / drx)
    "access_state": ("[E,R]", "int64", "state", "Access state at the last slot of the step: 0 IDLE, 1 RACH, 2 "
                                                "CONNECTED, 3 DORMANT."),
    "access_sleep_frac": ("[E,R]", "float32", "fraction", "Share of the step's engine slots in which the robot was "
                                                          "DRX-dormant."),
    "rach_attempts": ("[E,R]", "int64", "count", "Preamble transmissions of the robot this step."),
    # ---- Wi-Fi level (wifi/engine.py)
    "wifi_mcs": ("[E,R]", "int64", "id", "802.11 MCS index of the robot's link, -1 out of range."),
    "wifi_rate_mbps": ("[E,R]", "float32", "Mbit/s", "PHY rate of the robot's link."),
    "wifi_access_ms": ("[E,R]", "float32", "ms", "Mean channel-access time over the sub-steps the robot contended, "
                                                 "NaN if it did not."),
    "wifi_p_fail": ("[E,R]", "float32", "fraction", "Mean conditional failure probability of an attempt."),
    "wifi_busy": ("[E,R]", "float32", "fraction", "Fraction of time the channel is busy as the robot senses it."),
    # ---- adaptive fidelity (adaptive.py)
    "fidelity": ("[E]", "int64", "bool", "1 if the env's step ran on the expensive level."),
    "fidelity_indicator": ("[E]", "float32", "fraction", "Switching indicator after the step."),
    # ---- background users (background.py), per cell
    "bg_n": ("[E,C]", "int64", "count", "Background UEs attached to the cell."),
    "bg_offered_bytes": ("[E,C]", "float64", "bytes", "Bytes the background offered this step."),
    "bg_delivered_bytes": ("[E,C]", "float64", "bytes", "Background bytes delivered this step (ghost UEs, L2)."),
    "bg_lost_bytes": ("[E,C]", "float64", "bytes", "Background bytes timed out or dropped this step (ghost UEs, L2)."),
    "bg_queue_bytes": ("[E,C]", "float64", "bytes", "Background bytes queued after the step (ghost UEs, L2)."),
    "bg_util": ("[E,C]", "float64", "fraction", "Share of the cell's uplink resources the background used."),
    "bg_pos": ("[E,R_bg,2]", "float32", "m", "Background UE positions after the step."),
    "bg_dl_offered_bytes": ("[E,C]", "float64", "bytes", "Downlink background bytes offered this step."),
    "bg_dl_delivered_bytes": ("[E,C]", "float64", "bytes", "Downlink background bytes delivered this step."),
    "bg_dl_lost_bytes": ("[E,C]", "float64", "bytes", "Downlink background bytes lost this step."),
    "bg_dl_queue_bytes": ("[E,C]", "float64", "bytes", "Downlink background bytes queued after the step."),
    "bg_dl_util": ("[E,C]", "float64", "fraction", "Share of the DL carrier's PRB-slots the background used."),
    # ---- edge loop (edge.py)
    "edge_done": ("[E,R]", "int64", "count", "Results the edge completed this step."),
    "edge_done_cap": ("[E,R]", "int64", "step", "Capture step of the newest result completed this step, -1 if none."),
    "edge_done_time": ("[E,R]", "float32", "steps", "Completion time of that result (env clock), NaN if none."),
    "edge_dropped": ("[E,R]", "int64", "count", "Messages dropped at the edge this step (full + deadline)."),
    "edge_dropped_full": ("[E,R]", "int64", "count", "Messages dropped because the edge queue was full."),
    "edge_dropped_deadline": ("[E,R]", "int64", "count", "Messages dropped at their edge deadline."),
    "edge_queue_len": ("[E]", "int64", "count", "Messages at the edge after the step (in service, waiting, not yet "
                                                "admitted)."),
    "edge_in_service": ("[E]", "int64", "count", "Messages in service after the step."),
    "edge_lag": ("[E]", "bool", "bool", "The env ran out of its event budget and catches up next step."),
    "act_new": ("[E,R]", "bool", "bool", "A newer action (command) reached the robot this step."),
    "act_cap": ("[E,R]", "int64", "step", "Capture step of the robot's newest action, -1 before the first."),
    "act_time": ("[E,R]", "float32", "steps", "Arrival time of that action at the robot (env clock)."),
    "act_age": ("[E,R]", "float32", "steps", "Age t + 1 - act_cap of the action the robot holds, NaN before the "
                                             "first."),
    "act_latency": ("[E,R]", "float32", "steps", "Capture to uplink to edge to return latency of that action."),
    "act_ul_delay": ("[E,R]", "float32", "steps", "Uplink stage of act_latency."),
    "act_edge_delay": ("[E,R]", "float32", "steps", "Edge stage (waiting and service) of act_latency."),
    "act_ret_delay": ("[E,R]", "float32", "steps", "Return-path stage of act_latency."),
    "cmd_dropped": ("[E,R]", "int64", "count", "Commands lost this step (replaced in flight, DL loss or DL queue "
                                               "full)."),
    # ---- energy (energy.py)
    "energy_j": ("[E,R]", "float32", "J", "Radio energy of the robot this step."),
    "energy_tx_j": ("[E,R]", "float32", "J", "Transmit part of energy_j (PA and circuit)."),
    "energy_cum_j": ("[E,R]", "float32", "J", "Energy since the env's last reset."),
    "tx_slots": ("[E,R]", "float32", "count", "Transmissions (transport blocks or slot equivalents) this step."),
    "rx_slots": ("[E,R]", "float32", "count", "Receive slots this step."),
    "battery_j": ("[E,R]", "float32", "J", "Battery energy left."),
    "battery_frac": ("[E,R]", "float32", "fraction", "Battery state of charge."),
    "low_battery": ("[E,R]", "bool", "bool", "battery_frac below low_battery_frac."),
    "battery_empty": ("[E,R]", "bool", "bool", "Battery is empty (the engine keeps transmitting)."),
}

# keys by the feature that adds them, in the order the engines add them
GROUPS = {
    "base": ("newest", "det_env", "delivered", "timed_out", "cap", "cls", "delay", "queue_len", "queue_bytes",
             "sinr_db", "t"),
    "nr": ("dropped", "serving_cell"),
    "dl": ("dl_newest", "dl_queue_len"),
    "rank": ("rank",),
    "dl_rank": ("dl_rank",),
    "rlf": ("rlf",),
    "los": ("los",),
    "blocked": ("blocked",),
    "extras": ("arrival", "arrival_slot", "tag", "priority", "bytes", "deadline_miss"),
    "gen": ("gen_accepted", "gen_bytes"),
    "dl_traffic": ("dl_delivered", "dl_lost", "dl_delay", "dl_tag", "dl_generated", "dl_bytes", "dl_deadline_miss",
                   "gen_dl_accepted", "gen_dl_bytes"),
    "access": ("access_state", "access_sleep_frac", "rach_attempts"),
    "serving_cell": ("serving_cell",),
    "wifi": ("wifi_mcs", "wifi_rate_mbps", "wifi_access_ms", "wifi_p_fail", "wifi_busy"),
    "fidelity": ("fidelity", "fidelity_indicator"),
    "background_load": ("bg_n", "bg_offered_bytes", "bg_util", "bg_pos"),
    "background": ("bg_n", "bg_offered_bytes", "bg_delivered_bytes", "bg_lost_bytes", "bg_queue_bytes", "bg_util",
                   "bg_pos"),
    "background_dl": ("bg_dl_offered_bytes", "bg_dl_delivered_bytes", "bg_dl_lost_bytes", "bg_dl_queue_bytes",
                      "bg_dl_util"),
    "edge": ("edge_done", "edge_done_cap", "edge_done_time", "edge_dropped", "edge_dropped_full",
             "edge_dropped_deadline", "edge_queue_len", "edge_in_service", "edge_lag", "act_new", "act_cap",
             "act_time", "act_age", "act_latency", "act_ul_delay", "act_edge_delay", "act_ret_delay", "cmd_dropped"),
    "energy": ("energy_j", "energy_tx_j", "energy_cum_j", "tx_slots", "rx_slots", "battery_j", "battery_frac",
               "low_battery", "battery_empty"),
}
assert all(k in STEP_KEYS for g in GROUPS.values() for k in g)
assert {k for g in GROUPS.values() for k in g} == set(STEP_KEYS)


def entry(key, when="always", **override):
    """Schema entry of one key: {"shape", "dtype", "unit", "doc", "when"} (override replaces fields)."""
    shape, dtype, unit, doc = STEP_KEYS[key]
    e = {"shape": shape, "dtype": dtype, "unit": unit, "doc": doc, "when": when}
    e.update(override)
    return e


def schema(*groups, when=None, base=None):
    """Ordered {key: entry} of the keys of `groups` appended to `base` (a schema dict or None). when: {key: text} for
    keys that need a particular input or call; a key already in base keeps its place."""
    out = dict(base) if base is not None else {}
    when = when or {}
    for g in groups:
        for k in GROUPS[g]:
            if k not in out:
                out[k] = entry(k, when.get(k, "always"))
    return out


def nr_schema(eng):
    """Schema of an NREngine (reference, graph or triton backend) under its config and current state (submit
    extras enabled, traffic models, access, obstacle stack)."""
    cfg, net = eng.config, eng.net
    groups = ["base", "nr"]
    when = {}
    if net.dl is not None:
        groups.append("dl")
    if cfg.n_layers_max > 1:
        groups.append("rank")
        if net.dl is not None:
            groups.append("dl_rank")
    if net.C > 1 and cfg.rlf:
        groups.append("rlf")
    if eng._extras:
        groups.append("extras")
        if eng.traffic is None:
            when.update({k: "after submit(tag=, priority= or deadline_ms=)" for k in GROUPS["extras"]})
    if eng.traffic is not None:
        groups.append("gen")
    if eng.access is not None:
        groups.append("access")
    if cfg.los_source != "stochastic" or cfg.blockage:            # radio.RadioMC.obstacle_outputs
        if cfg.los_source != "stochastic" or cfg.channel == "tr38901":
            groups.append("los")
            when["los"] = "pose input (step(t, poses))"
        if cfg.blockage:
            groups.append("blocked")
            when["blocked"] = "pose input (step(t, poses))"
    if eng.traffic_dl is not None:
        groups.append("dl_traffic")
    # step() order: NRNet keys, then the extras, then sinr_db / serving_cell / t, gen, obstacles, DL traffic
    return schema(*groups, when=when)


def markdown_table(keys=None):
    """Markdown table of STEP_KEYS (docs/configurability.md "Output schema")."""
    rows = ["| Key | Shape | Dtype | Unit | Meaning |", "|:---|:---|:---|:---|:---|"]
    for k in keys or STEP_KEYS:
        shape, dtype, unit, doc = STEP_KEYS[k]
        rows.append(f"| `{k}` | `{shape}` | {dtype} | {unit} | {doc} |")
    return "\n".join(rows)


__all__ = ["STEP_KEYS", "GROUPS", "entry", "schema", "nr_schema", "markdown_table"]
