"""Per-packet one-way delay (OWD) and per-frame delay from timestamped UDP probes.

Two inputs are supported:

- the CSV logs of probe.py (sender tx.csv, receiver rx.csv): send times come from the sender log, receive times
  from the receiver log (kernel timestamps when available). Packets in tx.csv missing from rx.csv are lost.
  Without tx.csv the send time embedded in each datagram is used and losses are not visible.
- two pcaps of the same probe stream (sender-side interface, e.g. the UE's wwan0, and receiver-side, e.g. the
  UPF's N6 or the gNB host's tun interface): capture timestamps at both ends, probe headers to match packets.
  Use this when the probe sender cannot run on the UE host (a phone, a robot controller).

OWD = t_rx - t_tx - offset, where ``offset_ms`` is the receiver clock minus the sender clock (0 when both ends
share one clock; see "Clock synchronization" in docs/measurement-protocol.md for how to measure it). The summary
reports how many OWDs are negative, which is the first sign of an unsynchronized pair.

A negative t_tx_ns in a sender log marks a local send error (the OAI bridge's UE agent logs -t when sendto fails):
the packet never left the host, so it is neither an OWD sample nor a network loss. Such rows are left out of the owd
table (their frame then cannot complete) and counted in the summary as ``send_errors``.
"""
from __future__ import annotations

import csv
import math

from . import pcap as P
from .probe import unpack
from .schema import new_row


def _read_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def _key(r):
    return int(r["flow"]), int(r["seq"])


def from_probe_logs(rx_csv, tx_csv=None, run_id="", ue="", offset_ms=0.0, src=None):
    """probe.py logs -> (owd rows, frame rows, summary). src: keep only receiver rows from this source IP."""
    rx = [r for r in _read_csv(rx_csv) if src is None or r.get("src") == src]
    rxd = {}
    for r in rx:
        rxd.setdefault(_key(r), r)                                   # first copy wins (duplicates ignored)
    if tx_csv:
        tx = _read_csv(tx_csv)
        pk = [(r, rxd.get(_key(r))) for r in tx]
    else:
        pk = [(r, r) for r in rxd.values()]
    n_err = sum(1 for t, _ in pk if int(t["t_tx_ns"]) < 0)
    pk = [(t, r) for t, r in pk if int(t["t_tx_ns"]) >= 0]          # local send errors: never sent
    rows = []
    for t, r in pk:
        t_tx = int(t["t_tx_ns"]) / 1e9
        t_rx = int(r["t_rx_ns"]) / 1e9 if r is not None else float("nan")
        rows.append(new_row(
            "owd", run_id=run_id, ue=ue, flow=int(t["flow"]), seq=int(t["seq"]), frame_id=int(t["frame_id"]),
            frag=int(t["frag"]), n_frag=int(t["n_frag"]), frame_bytes=int(t["frame_bytes"]),
            pkt_bytes=int(t["pkt_bytes"]), t_tx_s=t_tx, t_rx_s=t_rx,
            owd_ms=(int(r["t_rx_ns"]) - int(t["t_tx_ns"])) / 1e6 - offset_ms if r is not None else float("nan"),
            lost=0 if r is not None else 1))
    return rows, frames_from_owd(rows, offset_ms), summarize(rows) | {"send_errors": n_err}


def _probe_packets(path, port=None):
    out = {}
    for t, lt, fr in P.iter_pcap(path):
        u = P.udp_payload(lt, fr)
        if u is None or (port is not None and u[3] != port):
            continue
        h = unpack(u[4])
        if h is not None:
            out.setdefault((h["flow"], h["seq"]), (t, h, u[0]))
    return out


def from_pcaps(tx_pcap, rx_pcap, run_id="", ue="", offset_ms=0.0, port=None):
    """Sender-side and receiver-side captures of the same probe stream -> (owd rows, frame rows, summary)."""
    tx, rx = _probe_packets(tx_pcap, port), _probe_packets(rx_pcap, port)
    rows = []
    for k in sorted(tx, key=lambda k: tx[k][0]):
        t_tx, h, _ = tx[k]
        got = rx.get(k)
        t_rx = got[0] if got else float("nan")
        rows.append(new_row(
            "owd", run_id=run_id, ue=ue, flow=h["flow"], seq=h["seq"], frame_id=h["frame_id"], frag=h["frag"],
            n_frag=h["n_frag"], frame_bytes=h["frame_bytes"], pkt_bytes=h["pkt_bytes"], t_tx_s=t_tx, t_rx_s=t_rx,
            owd_ms=(t_rx - t_tx) * 1e3 - offset_ms if got else float("nan"), lost=0 if got else 1))
    return rows, frames_from_owd(rows, offset_ms), summarize(rows)


def frames_from_owd(rows, offset_ms=0.0):
    """Group packets into application frames: delay = last fragment received - first fragment sent."""
    fr = {}
    for r in rows:
        fr.setdefault((r["ue"], r["flow"], r["frame_id"]), []).append(r)
    out = []
    for (ue, flow, fid), ps in sorted(fr.items(), key=lambda kv: min(p["t_tx_s"] for p in kv[1])):
        n = ps[0]["n_frag"]
        rxs = [p for p in ps if not p["lost"]]
        complete = int(len({p["frag"] for p in rxs}) == n)
        t0 = min(p["t_tx_s"] for p in ps)
        tl = max(p["t_rx_s"] for p in rxs) if complete else float("nan")
        out.append(new_row(
            "frames", run_id=ps[0]["run_id"], ue=ue, flow=flow, frame_id=fid, frame_bytes=ps[0]["frame_bytes"],
            n_frag=n, n_rx=len(rxs), t_tx_first_s=t0, t_rx_last_s=tl,
            delay_ms=(tl - t0) * 1e3 - offset_ms if complete else float("nan"), complete=complete))
    return out


def _q(xs, p):
    if not xs:
        return float("nan")
    xs = sorted(xs)
    i = (len(xs) - 1) * p
    lo = int(math.floor(i))
    hi = min(lo + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (i - lo)


def summarize(rows):
    d = [r["owd_ms"] for r in rows if not r["lost"]]
    return {"n": len(rows), "send_errors": 0, "lost": sum(r["lost"] for r in rows),
            "negative": sum(1 for x in d if x < 0),
            "p50_ms": _q(d, 0.5), "p95_ms": _q(d, 0.95), "p99_ms": _q(d, 0.99),
            "min_ms": min(d) if d else float("nan"), "max_ms": max(d) if d else float("nan")}
