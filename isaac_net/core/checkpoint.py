"""Checkpoints of a running network: engine.state_dict() / load_state_dict(), and save / load to a file.

    sd = net.state_dict()                        # {key: tensor | int | float | str | bool | None | list}, flat
    net2 = make_engine(level, E, R, device, cfg, backend)
    net2.load_state_dict(sd)                     # in place: net2 now continues exactly like net

    from isaac_net.core import checkpoint
    checkpoint.save(net, "isaac_net_100.pt", extra={"iter": 100})
    checkpoint.load(net2, "isaac_net_100.pt")    # checks level / backend / E / R / config, then restores

What the state dict holds. Every engine, wrapper and module of the package keeps its state in attributes: tensors
(queues, MAC / HARQ state, fading, radio fields, timers, counters, per-env clocks, RNG episode counters), torch
Generators (traffic models, the engine generator), and host values (the NR engine's global slot clock, the last
fading slot, host counters). state_dict() walks the object graph from the engine (vars() of every isaac_net object,
dicts, lists, tuples) and records
  tensor attributes          a clone, under the dotted path of the attribute ("net.ul.q.cap", "net.ul.ctr[tb_ok]")
  torch.Generator            its get_state() byte tensor
  int / float / bool / str / None attributes, and lists or tuples of them   the value
except the names in SKIP (constants, caches, captured graphs and statistics; each with its reason) and the
per-class names in `_ckpt_skip`. Objects that have their own state_dict (an inner engine of a wrapper, the shards
of a ShardedEngine, the two levels of an AdaptiveEngine, the engine of a NetModule) contribute theirs under their
attribute path, so their load hooks run. Configuration dataclasses are never walked: the config is checked, not
restored.

Restoring. load_state_dict(sd, strict=True) walks the target the same way and writes every value in place: tensors
with copy_ into the existing tensor (no reallocation, so the static buffers of the graph backends stay valid and
captured CUDA graphs keep reading the right memory), generators with set_state, host values with setattr. Then
each class's `_ckpt_finish()` runs (the NR graph backends rebind their persistent buffers). Before it, a class that
replays captured CUDA graphs gets `_ckpt_host_changed(keys)` when the restore changed a host value that its steps do
not update outside the graphs (`_ckpt_step_host`), e.g. the counter RNG's seed key, or replaced a tensor object: a
captured graph bakes in such values and addresses, so the class drops its graphs and the next step captures again. A value whose attribute
does not exist yet on the target (state that a step creates on first use, e.g. a power-control backoff) is set as a
new attribute. strict=True raises on keys the target cannot place, on state of the target missing from sd, and on
shape or dtype mismatches; strict=False skips them.

Guarantee: a resume on the same backend and device is bitwise equal to the uninterrupted run (outputs, counters()
and the next state_dict()), with rng="engine" (the default). Across devices or backends the restored state is the
same, but the continuation is only as close as the backends are to each other (see docs/checkpoint.md). With
rng="global" the stepping draws come from the global torch RNG, which is not engine state: save() records it
(torch.get_rng_state and the CUDA generators) and load(..., global_rng=True) restores it.
"""
from __future__ import annotations

import dataclasses
import enum
import math
import os
import types
import warnings

import torch

FORMAT = 1                     # file format version of save()

# Attribute names never recorded, with the reason. These are not state: constants built from the config, caches
# that are rebuilt on demand, captured CUDA graphs, static scratch rewritten by every step, statistics and sinks.
SKIP = {
    "config": "configuration (checked by checkpoint.load, never restored)",
    "cfg": "configuration",
    "ncfg": "configuration",
    "_config": "configuration",
    "isaac": "configuration (IsaacNetCfg)",
    "bg": "configuration (BackgroundConfig)",
    "fid": "configuration (FidelityConfig)",
    "params": "fitted level parameters given to make_engine (constant, can be large)",
    "phy": "PHY tables (MCS / TBS / BLER tables, several MB, constant)",
    "_occ_cache": "NR schedule cache (host lists keyed by the TDD position)",
    "_sched_cache": "NR schedule cache (host lists keyed by the TDD position)",
    "table": "CDF lookup table of the sum-of-cosines random fields (constant)",
    "_calls": "graph warm-up call counts of the adaptive engine (it captures again after a load)",
    "_gin": "static graph inputs of the adaptive engine",
    "_gout": "static graph outputs of the adaptive engine",
    "_graphs": "captured CUDA graphs",
    "_graph": "captured CUDA graph and its static buffers",
    "_pool": "CUDA graph memory pool",
    "_in": "static graph inputs, refilled before every replay",
    "_out": "static graph outputs, rewritten by every replay",
    "_stash": "graph scratch of the statistics, rewritten by every step",
    "_ion_delta": "per-graph host counter increments (a property of the captured graph)",
    "_reg": "registry of the persistent buffers (the buffers themselves are recorded under their attributes)",
    "_weyl": "RNG element-index cache",
    "_sched_tabs": "triton schedule tables (cache)",
    "_tables": "triton constant tables",
    "_const": "triton kernel constants",
    "_nt": "the triton module",
    "_writers": "recorder output files",
    "_tb": "TensorBoard writer",
    "_wb": "wandb run",
    "_slot": "compiled step function",
    "_fluid": "compiled step function",
    "_add": "compiled step function",
    "_arrival": "compiled step function",
    "_finish": "compiled step function",
}
# "stats" is skipped when it holds the log_stats lists (statistics, not state: collect() after a load covers the steps
# since the load); a stats dict of tensors only (AdaptiveEngine's cumulative counts) is state and is recorded
STATS = "stats"
# classes whose instances hold only constants (walking them would store large tables)
SKIP_CLASSES = {"PHY", "RadioMap", "Phy", "WifiPhy"}
SCALARS = (bool, int, float, str, type(None))


class StateDictMixin:
    """state_dict() / load_state_dict() for the engines, wrappers and modules of isaac_net (module docstring).

    Subclasses may set `_ckpt_skip` (attribute names that are not state, with a comment why) and override
    `_ckpt_prepare(sd)` (build lazily created parts that sd has state for, before the restore) and `_ckpt_finish()`
    (after the restore, e.g. rebind static buffers)."""

    _ckpt_skip: frozenset = frozenset()
    # host values (attribute names, matched against every component of a key) that the steps update outside captured
    # CUDA graphs; a load that changes any other host value, or replaces a tensor object, calls _ckpt_host_changed
    _ckpt_step_host: frozenset = frozenset()

    def state_dict(self) -> dict:
        """Flat {key: value} of every state tensor, generator state and host value (a snapshot: tensors are cloned)."""
        return collect(self)

    def load_state_dict(self, sd: dict, strict: bool = True):
        """Restore a state_dict() in place (module docstring). Returns self."""
        restore(self, sd, strict=strict)
        return self

    def _ckpt_prepare(self, sd):
        """Default: build the engine's radio when sd has state for it and this engine has not made it yet (engines
        make their radio at the first step with poses). Engines with such a radio define _ckpt_make_radio()."""
        if "radio" in vars(self) and self.radio is None and any(k.startswith("radio.") for k in sd):
            make = getattr(self, "_ckpt_make_radio", None)
            if make is not None:
                make()


    def _ckpt_host_changed(self, keys):
        """Called by load_state_dict, before _ckpt_finish, with the keys of the host values (ints, floats, flags, ...)
        that the restore changed, outside _ckpt_step_host, and of the tensors it replaced by new objects. Engines that
        replay captured CUDA graphs drop them here: a graph bakes in the host values and the tensor addresses it was
        captured with (e.g. the counter RNG's seed key rng.s0), so it would continue the target's old run. Only
        classes that override this method pay for the comparison."""

    def _ckpt_finish(self):
        pass


def _skip_names(obj):
    names = set(SKIP)
    for c in type(obj).__mro__:
        names |= set(c.__dict__.get("_ckpt_skip", ()))
    return names


def _skipped(name, v, skip):
    if name in skip:
        return True
    return name == STATS and not (isinstance(v, dict) and all(torch.is_tensor(x) for x in v.values()))


def _walkable(v):
    """Objects whose attributes are walked: isaac_net objects (not config dataclasses) and SimpleNamespace."""
    if isinstance(v, types.SimpleNamespace):
        return True
    t = type(v)
    if not (t.__module__ or "").startswith("isaac_net"):
        return False
    if dataclasses.is_dataclass(v) or isinstance(v, (type, enum.Enum, torch.nn.Module)) or t.__name__ in SKIP_CLASSES:
        return False                   # configs, classes, network weights (the NN surrogate's: constant), tables
    return hasattr(v, "__dict__")


def _scalar_seq(v):
    return isinstance(v, (list, tuple)) and all(isinstance(x, SCALARS) for x in v)


# ---------------------------------------------------------------------------------------------- collect
def collect(root) -> dict:
    """state_dict of any object (module docstring); StateDictMixin.state_dict calls it."""
    out = {}
    _collect_obj(root, "", out, {id(root)})
    return out


def _collect_obj(obj, prefix, out, seen):
    skip = _skip_names(obj)
    for name, v in list(vars(obj).items()):
        if _skipped(name, v, skip):
            continue
        _collect_val(v, prefix + name, out, seen)


def _collect_val(v, key, out, seen):
    if torch.is_tensor(v):
        out[key] = v.detach().clone()
    elif isinstance(v, torch.Generator):
        out[key] = v.get_state()
    elif isinstance(v, SCALARS):
        out[key] = v
    elif isinstance(v, (list, tuple)):
        if _scalar_seq(v):
            out[key] = type(v)(v) if type(v) in (list, tuple) else list(v)
        else:
            for i, x in enumerate(v):
                _collect_val(x, f"{key}[{i}]", out, seen)
    elif isinstance(v, dict):
        for k, x in list(v.items()):
            _collect_val(x, f"{key}[{k}]", out, seen)
    elif isinstance(v, StateDictMixin):
        if id(v) in seen:
            return
        seen.add(id(v))
        for k, x in v.state_dict().items():
            out[f"{key}.{k}"] = x
    elif _walkable(v):
        if id(v) in seen:
            return
        seen.add(id(v))
        _collect_obj(v, key + ".", out, seen)


# ---------------------------------------------------------------------------------------------- restore
class _Slot:
    """One restorable place of the target: get() the current value, set(v) replace it."""
    __slots__ = ("get", "set")

    def __init__(self, get, set_):
        self.get, self.set = get, set_


def _attr_slot(obj, name):
    return _Slot(lambda: getattr(obj, name), lambda v: setattr(obj, name, v))


def _item_slot(d, k):
    def set_(v):
        d[k] = v
    return _Slot(lambda: d[k], set_)


def _tuple_slot(parent, i):
    def set_(v):
        t = list(parent.get())
        t[i] = v
        parent.set(tuple(t))
    return _Slot(lambda: parent.get()[i], set_)


def _targets(root):
    """Walk the target like collect: ({key: _Slot}, {prefix: object} of walked objects, [(prefix, child)] of children
    with their own state_dict)."""
    slots, objs, kids = {}, {"": root}, []
    _target_obj(root, "", slots, objs, kids, {id(root)})
    return slots, objs, kids


def _target_obj(obj, prefix, slots, objs, kids, seen):
    skip = _skip_names(obj)
    for name, v in list(vars(obj).items()):
        if _skipped(name, v, skip):
            continue
        _target_val(v, prefix + name, _attr_slot(obj, name), slots, objs, kids, seen)


def _target_val(v, key, slot, slots, objs, kids, seen):
    if torch.is_tensor(v) or isinstance(v, (torch.Generator,) + SCALARS):
        slots[key] = slot
    elif isinstance(v, (list, tuple)):
        if _scalar_seq(v):
            slots[key] = slot
        else:
            for i, x in enumerate(v):
                sub = _item_slot(v, i) if isinstance(v, list) else _tuple_slot(slot, i)
                _target_val(x, f"{key}[{i}]", sub, slots, objs, kids, seen)
    elif isinstance(v, dict):
        for k, x in list(v.items()):
            _target_val(x, f"{key}[{k}]", _item_slot(v, k), slots, objs, kids, seen)
    elif isinstance(v, StateDictMixin):
        if id(v) in seen:
            return
        seen.add(id(v))
        kids.append((key, v))
    elif _walkable(v):
        if id(v) in seen:
            return
        seen.add(id(v))
        objs[key] = v
        _target_obj(v, key + ".", slots, objs, kids, seen)


def _host_print(slots):
    """{key: value} of the host values of a target, {key: ("tensor", id)} of its tensors (identity, not content)."""
    out = {}
    for k, slot in slots.items():
        v = slot.get()
        if torch.is_tensor(v) or isinstance(v, torch.Generator):
            out[k] = ("tensor", id(v))
        else:
            out[k] = list(v) if isinstance(v, (list, tuple)) else v
    return out


def _same_host(a, b):
    if isinstance(a, float) and isinstance(b, float) and math.isnan(a) and math.isnan(b):
        return True
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(_same_host(x, y) for x, y in zip(a, b))
    return type(a) is type(b) and a == b


def _host_changes(before, after, step_host):
    """Keys whose host value or tensor object differs between two _host_print, except keys with a component in
    step_host."""
    out = []
    for k in sorted(set(before) | set(after)):
        if k in before and k in after and _same_host(before[k], after[k]):
            continue
        if step_host and not step_host.isdisjoint(k.replace("[", ".").replace("]", "").split(".")):
            continue
        out.append(k)
    return out


def _device(root):
    d = getattr(root, "__dict__", {}).get("dev")
    return d if isinstance(d, torch.device) else None


def _put(slot, val, key, dev, strict, errors):
    """Write val into slot: tensors in place, generators by set_state, host values by assignment."""
    cur = slot.get()
    if torch.is_tensor(val):
        if torch.is_tensor(cur):
            if cur.shape != val.shape or cur.dtype != val.dtype:
                if strict:
                    errors.append(f"{key}: shape/dtype {tuple(val.shape)}/{val.dtype} in the state dict, "
                                  f"{tuple(cur.shape)}/{cur.dtype} in the engine")
                return
            v = val.to(cur.device)
            try:
                cur.copy_(v)
            except RuntimeError:          # an expanded (stride 0) view: a constant, or a new tensor
                if not torch.equal(cur, v):
                    slot.set(v.clone())
            return
        if isinstance(cur, torch.Generator):
            cur.set_state(val.cpu())
            return
        slot.set(val.to(dev).clone() if dev is not None else val.clone())    # e.g. an int that a step makes a tensor
        return
    if isinstance(cur, torch.Generator):
        if strict:
            errors.append(f"{key}: the engine holds a torch.Generator, the state dict {type(val).__name__}")
        return
    if torch.is_tensor(cur) and cur.dim() == 0 and isinstance(val, (bool, int, float)):
        cur.fill_(val)                    # a host value that this backend keeps in a device scalar (e.g. _g0)
        return
    if torch.is_tensor(cur) and val is not None and strict:
        errors.append(f"{key}: the engine holds a tensor, the state dict {type(val).__name__}")
        return
    if isinstance(val, list):
        val = list(val)
    if cur is not val:
        slot.set(val)


def _new_attr(objs, key):
    """(object, attribute) for a key the target has no slot for, if it names a direct attribute of a walked object
    (state that a step creates on first use), else None."""
    best = None
    for p, o in objs.items():
        if p == "" or key.startswith(p + "."):
            if best is None or len(p) > len(best[0]):
                best = (p, o)
    if best is None:
        return None
    p, o = best
    rest = key[len(p) + 1:] if p else key
    if not rest or "." in rest or "[" in rest or not rest.isidentifier():
        return None
    return o, rest


def restore(root, sd: dict, strict: bool = True):
    """load_state_dict of any object (module docstring); StateDictMixin.load_state_dict calls it."""
    errors = []
    _restore(root, sd, strict, errors, "")
    if errors:
        raise KeyError(f"load_state_dict({type(root).__name__}) failed for {len(errors)} key(s):\n" + "\n".join(errors))


def _restore(root, sd, strict, errors, prefix):
    if isinstance(root, StateDictMixin):
        root._ckpt_prepare(sd)
    slots, objs, kids = _targets(root)
    watch = isinstance(root, StateDictMixin) and \
        type(root)._ckpt_host_changed is not StateDictMixin._ckpt_host_changed
    before = _host_print(slots) if watch else None
    dev = _device(root)
    n_err = len(errors)
    used = set()
    for key, slot in slots.items():
        if key in sd:
            _put(slot, sd[key], prefix + key, dev, strict, errors)
            used.add(key)
        elif strict and slot.get() is not None:      # an attribute set to None is the same as an absent one
            errors.append(f"{prefix}{key}: missing from the state dict")
    for kp, kid in kids:
        p = kp + "."
        sub = {k[len(p):]: v for k, v in sd.items() if k.startswith(p)}
        used.update(p + k for k in sub)
        if type(kid).load_state_dict is StateDictMixin.load_state_dict:
            _restore(kid, sub, strict, errors, prefix + p)
        else:
            try:
                kid.load_state_dict(sub, strict=strict)
            except KeyError as e:
                errors.append(f"{prefix}{p}: {e.args[0]}")
    for key in sd:
        if key in used:
            continue
        hit = _new_attr(objs, key)
        if hit is not None:
            o, name = hit
            v = sd[key]
            if torch.is_tensor(v):
                d = _device(o) or dev
                v = v.to(d).clone() if d is not None else v.clone()
            setattr(o, name, v)
        elif strict:
            errors.append(f"{prefix}{key}: the engine has no place for this key (a part that is built lazily and "
                          "was not built, or a different engine type)")
    if len(errors) == n_err and isinstance(root, StateDictMixin):
        if watch:
            changed = _host_changes(before, _host_print(_targets(root)[0]), root._ckpt_step_host)
            if changed:
                root._ckpt_host_changed(changed)
        root._ckpt_finish()


# ---------------------------------------------------------------------------------------------- files
def _encode(v):
    """Config value -> plain data (dicts, lists, str, int, float, bool, None), loadable with weights_only=True."""
    if dataclasses.is_dataclass(v) and not isinstance(v, type):
        d = {f.name: _encode(getattr(v, f.name)) for f in dataclasses.fields(v)}
        d["__type__"] = type(v).__name__
        return d
    if isinstance(v, dict):
        return {str(k): _encode(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_encode(x) for x in v]
    if isinstance(v, enum.Enum):
        return str(v)
    if isinstance(v, float) and math.isnan(v):
        return "nan"
    if isinstance(v, SCALARS):
        return v
    if torch.is_tensor(v):
        return v.detach().cpu().tolist()
    if callable(v):
        return f"<callable {getattr(v, '__qualname__', type(v).__name__)}>"
    return repr(v)


def config_dict(cfg):
    """Plain-data form of a config: NRConfig.to_dict() when the config has one, else the dataclass fields."""
    if cfg is None:
        return None
    to_dict = getattr(cfg, "to_dict", None)
    if callable(to_dict):
        try:
            return _encode(to_dict())
        except Exception:              # noqa: BLE001  (a to_dict that refuses, e.g. callables: fall back)
            pass
    return _encode(cfg)


def _config_of(engine):
    cfg = getattr(engine, "config", None)
    if cfg is None and hasattr(engine, "eng"):
        cfg = getattr(engine.eng, "config", None)
    return cfg


def _level_of(engine):
    o, n = engine, 0
    while o is not None and n < 16:
        lv = getattr(o, "__dict__", {}).get("level") or getattr(type(o), "level", None)
        if isinstance(lv, str):
            return lv
        if type(o).__name__ == "AdaptiveEngine":
            return "adaptive"
        nxt = o.__dict__.get("engine") or o.__dict__.get("eng")
        if nxt is None and "shards" in o.__dict__:
            nxt = o.shards[0]
        o, n = nxt, n + 1
    return type(engine).__name__


def _backend_of(engine):
    o, n = engine, 0
    while o is not None and n < 16:
        b = getattr(o, "__dict__", {}).get("backend") or getattr(type(o), "backend", None)
        if isinstance(b, str):
            return b
        nxt = o.__dict__.get("engine") or o.__dict__.get("eng")
        if nxt is None and "shards" in o.__dict__:
            nxt = o.shards[0]
        o, n = nxt, n + 1
    return "reference"


# backends of one level whose engines share their state layout and are bitwise equal to each other: a checkpoint of
# one loads into the other and the continuation stays bitwise (the NR engine's reference and graph backends)
BITWISE_FAMILIES = {"NREngine": ("reference", "eager", "graph")}


def _kind(engine):
    """The engine class, with the NR engine's reference and graph classes as one kind (they share the state)."""
    from .engine import NREngine
    t = type(engine)
    if issubclass(t, NREngine) and getattr(t, "backend", "reference") in BITWISE_FAMILIES["NREngine"]:
        return "NREngine"
    return t.__name__


def header(engine) -> dict:
    """The compatibility fields save() writes next to the state: version, level, backend, device, E, R, config."""
    from .. import __version__
    dev = getattr(engine, "dev", None)
    return {"isaac_net": __version__, "format": FORMAT, "level": _level_of(engine), "backend": _backend_of(engine),
            "device": str(dev) if dev is not None else None, "E": int(engine.E), "R": int(engine.R),
            "kind": _kind(engine), "config": config_dict(_config_of(engine))}


def _rng_state():
    st = {"cpu": torch.get_rng_state()}
    if torch.cuda.is_available() and torch.cuda.is_initialized():
        st["cuda"] = [s for s in torch.cuda.get_rng_state_all()]
    return st


def save(engine, path, extra=None, *, state=None):
    """Write a checkpoint of `engine` (any engine, wrapper, ShardedEngine, AdaptiveEngine or NetModule) to path.

    The file is torch.save({"isaac_net": version, "format", "config", "level", "backend", "E", "R", "kind",
    "state": engine.state_dict() on the CPU, "global_rng": the global torch RNG state, "extra": extra}). extra is
    any data of the caller (plain data or tensors keep the file loadable with weights_only=True). state: a state dict
    to write instead of engine.state_dict(). The file is written to path + ".tmp" and renamed. Returns path."""
    ck = header(engine)
    sd = engine.state_dict() if state is None else state
    ck["state"] = {k: (v.cpu() if torch.is_tensor(v) else v) for k, v in sd.items()}
    ck["global_rng"] = _rng_state()
    ck["extra"] = extra
    d = os.path.dirname(os.fspath(path))
    if d:
        os.makedirs(d, exist_ok=True)
    tmp = os.fspath(path) + ".tmp"
    torch.save(ck, tmp)
    os.replace(tmp, path)
    return path


def _diff(a, b, path=""):
    """Paths where two plain-data configs differ (NaN equals NaN)."""
    if isinstance(a, dict) and isinstance(b, dict):
        out = []
        for k in sorted(set(a) | set(b)):
            if k in ("seed", "__type__") and path == "":
                continue
            if k not in a or k not in b:
                out.append(f"{path}{k}")
            else:
                out += _diff(a[k], b[k], f"{path}{k}.")
        return out
    if isinstance(a, list) and isinstance(b, list) and len(a) == len(b):
        out = []
        for i, (x, y) in enumerate(zip(a, b)):
            out += _diff(x, y, f"{path}{i}.")
        return out
    return [] if a == b else [path.rstrip(".")]


def read(path, map_location="cpu"):
    """The checkpoint dict of a file written by save() (weights_only when possible)."""
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except Exception:                  # noqa: BLE001  (extra data that is not plain: the caller's own file)
        return torch.load(path, map_location=map_location, weights_only=False)


def check(engine, ck, strict=True):
    """Compare a checkpoint's header with `engine`; raise ValueError (strict) or warn on a mismatch."""
    h = header(engine)
    errs = []
    for k in ("kind", "level", "E", "R"):
        if ck.get(k) != h[k]:
            errs.append(f"{k}: checkpoint {ck.get(k)!r}, engine {h[k]!r}")
    fam = BITWISE_FAMILIES.get(h["kind"], ())
    if ck.get("backend") != h["backend"] and not (ck.get("backend") in fam and h["backend"] in fam):
        msg = (f"backend: checkpoint {ck.get('backend')!r}, engine {h['backend']!r} (the state restores, but the "
               "continuation is only as close as the two backends; see docs/checkpoint.md)")
        if strict:
            errs.append(msg)
        else:
            warnings.warn(msg, stacklevel=3)
    d0, d1 = ck.get("device"), h["device"]
    if d0 and d1 and torch.device(d0).type != torch.device(d1).type:
        warnings.warn(f"device: checkpoint {d0}, engine {d1}: the state restores, but the continuation equals the "
                      "uninterrupted run only to float rounding (the counter RNG's normals and some kernels differ "
                      "between CPU and CUDA; docs/checkpoint.md)", stacklevel=3)
    if ck.get("config") is not None and h["config"] is not None:
        bad = _diff(ck["config"], h["config"])
        if bad:
            errs.append("config fields differ: " + ", ".join(bad[:20]) + (" ..." if len(bad) > 20 else ""))
    if errs:
        msg = "checkpoint does not match the engine: " + "; ".join(errs)
        if strict:
            raise ValueError(msg)
        warnings.warn(msg, stacklevel=3)


def load(engine, path, strict=True, global_rng=False, map_location="cpu"):
    """Restore a checkpoint written by save() into `engine` (built with the same level, E, R and config; the seed
    may differ, the checkpoint's RNG state wins). strict: config / level / backend / E / R must match and every key
    must be placed, else ValueError / KeyError; strict=False warns and restores what fits. global_rng=True also
    restores the global torch RNG (needed for an exact resume with NRConfig(rng="global")). Returns `extra`."""
    ck = read(path, map_location)
    if not isinstance(ck, dict) or "state" not in ck:
        raise ValueError(f"{path} is not an isaac_net checkpoint (no 'state')")
    check(engine, ck, strict=strict)
    engine.load_state_dict(ck["state"], strict=strict)
    if global_rng:
        set_global_rng(ck.get("global_rng"))
    return ck.get("extra")


def set_global_rng(g):
    """Restore the global torch RNG state that save() recorded (no-op for None)."""
    if not g:
        return
    torch.set_rng_state(g["cpu"])
    if "cuda" in g and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(g["cuda"])


def nbytes(sd: dict) -> int:
    """Bytes of the tensors of a state dict (the size of a checkpoint, roughly)."""
    return sum(v.numel() * v.element_size() for v in sd.values() if torch.is_tensor(v))
