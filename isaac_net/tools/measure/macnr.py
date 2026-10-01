"""MAC-NR frames in pcaps: the Wireshark "mac-nr" context header and the UL MAC PDU subheaders.

Framing (Wireshark epan/dissectors/packet-mac-nr.h, read 2026-09-29): an optional "mac-nr" signature, three
mandatory bytes radioType (1 FDD, 2 TDD), direction (0 UL, 1 DL), rntiType (3 = C-RNTI), then tagged optional
fields and the payload tag 0x01 followed by the MAC PDU up to the end of the frame:

  0x02 RNTI (2 B), 0x03 UE id (2 B), 0x04 frame/subframe (2 B: SFN << 4 | subframe, deprecated),
  0x05 PHR type2 other cell (1 B), 0x06 HARQ id (1 B), 0x07 frame/slot (4 B: SFN, slot).

Who writes what (source read 2026-09-29):
- srsRAN Project lib/pcap/mac_pcap_impl.cpp writes DLT 252 (exported PDU). With ``mac_type: udp`` the dissector
  name is "udp" and the record is a dummy UDP header (ports 0xbeef -> 0xdead) + "mac-nr" + context + PDU; with
  ``mac_type: dlt`` the dissector name is "mac-nr-framed" and the context follows directly. Its context carries
  tag 0x04 (SFN and *subframe*), not 0x07, so the slot inside the subframe is lost at mu >= 1.
- OAI openair2/UTIL/OPT/probe.c (``--opt.type pcap`` or ``wireshark``) writes "mac-nr" over IPv4/UDP port 9999
  (DLT 228 in its own pcap files) with tag 0x07 (SFN and slot).

The UL PDU walk follows TS 38.321 Sec. 6.1.2 and Table 6.2.1-2 (UL-SCH LCIDs).
"""
from __future__ import annotations

import struct

from . import pcap as P

SIG = b"mac-nr"
TAG_PAYLOAD, TAG_RNTI, TAG_UEID, TAG_FRAME_SUBFRAME, TAG_PHR2, TAG_HARQID, TAG_FRAME_SLOT = 1, 2, 3, 4, 5, 6, 7
TAG_LEN = {TAG_RNTI: 2, TAG_UEID: 2, TAG_FRAME_SUBFRAME: 2, TAG_PHR2: 1, TAG_HARQID: 1, TAG_FRAME_SLOT: 4}
C_RNTI = 3

# UL-SCH fixed-size MAC CEs without an L field (TS 38.321 Table 6.2.1-2): LCID -> size in bytes
UL_FIXED = {0: 6, 52: 8, 53: 2, 55: 0, 57: 2, 58: 2, 59: 1, 61: 1}
UL_PADDING = 63
UL_SHORT_BSR, UL_SHORT_TRUNC_BSR, UL_LONG_BSR, UL_LONG_TRUNC_BSR = 61, 59, 62, 60

# Short BSR buffer-size levels, TS 38.321 Table 6.1.3.1-1 (5-bit index -> upper bound in bytes; 31 = > 150000).
# Transcribed from the specification; verify against your copy of 38.321 before relying on absolute values.
BSR5 = [0, 10, 14, 20, 28, 38, 53, 74, 102, 142, 198, 276, 384, 535, 745, 1038, 1446, 2014, 2806, 3909, 5446,
        7587, 10570, 14726, 20516, 28581, 39818, 55474, 77284, 107669, 150000, 150000]


def parse_context(buf):
    """Parse radioType/direction/rntiType + tags. Returns (info dict, pdu bytes) or (None, b"")."""
    if len(buf) < 4:
        return None, b""
    info = {"radio_type": buf[0], "direction": buf[1], "rnti_type": buf[2]}
    o = 3
    while o < len(buf):
        tag = buf[o]
        o += 1
        if tag == TAG_PAYLOAD:
            return info, buf[o:]
        n = TAG_LEN.get(tag)
        if n is None or o + n > len(buf):
            return None, b""
        v = buf[o:o + n]
        if tag == TAG_RNTI:
            info["rnti"] = struct.unpack(">H", v)[0]
        elif tag == TAG_UEID:
            info["ueid"] = struct.unpack(">H", v)[0]
        elif tag == TAG_HARQID:
            info["harq_id"] = v[0]
        elif tag == TAG_FRAME_SLOT:
            info["sfn"], info["slot"] = struct.unpack(">HH", v)
            info["slot_exact"] = 1
        elif tag == TAG_FRAME_SUBFRAME:
            x = struct.unpack(">H", v)[0]
            info["sfn"], info["subframe"] = x >> 4, x & 0x0F
            info["slot_exact"] = 0
        o += n
    return None, b""


def frame_to_macnr(linktype, frame):
    """Extract (info, pdu) from one pcap record of any supported link type, or (None, b"")."""
    if linktype == P.DLT_EXPORTED_PDU:
        tags, body = P.exported_pdu(frame)
        name = tags.get(P.EXP_PDU_TAG_DISSECTOR_NAME, b"").rstrip(b"\x00")
        if name == b"mac-nr-framed":
            return parse_context(body)
        if name == b"udp" and body[8:8 + len(SIG)] == SIG:
            return parse_context(body[8 + len(SIG):])
        i = body.find(SIG)
        return parse_context(body[i + len(SIG):]) if i >= 0 else (None, b"")
    u = P.udp_payload(linktype, frame)
    if u is not None and u[4][:len(SIG)] == SIG:
        return parse_context(u[4][len(SIG):])
    if linktype >= 147 and linktype <= 162 and frame[:len(SIG)] == SIG:      # DLT_USER*: bare signature
        return parse_context(frame[len(SIG):])
    return None, b""


def walk_ul_pdu(pdu):
    """Walk the subheaders of an UL MAC PDU. Returns dict(data_bytes, bsr_idx, bsr_bytes, lcids, ok).

    data_bytes counts SDU bytes of LCID 1..32; the last short / short-truncated BSR gives bsr_idx and bsr_bytes
    (Table 6.1.3.1-1 upper bound; LCG field ignored), a long BSR gives the first buffer-size octet as bsr_idx
    (8-bit table not transcribed, bsr_bytes NaN). ok is False if the walk ran off the PDU (a malformed or
    unsupported PDU; the counts are then partial)."""
    o, n = 0, len(pdu)
    res = {"data_bytes": 0, "bsr_idx": -1, "bsr_bytes": float("nan"), "lcids": [], "ok": True}
    while o < n:
        b0 = pdu[o]
        f, lcid = (b0 >> 6) & 1, b0 & 0x3F
        o += 1
        if lcid in (33, 34):                               # eLCID: 1 or 2 extra octets, then as a SDU with L
            o += 1 if lcid == 34 else 2
            f_len = True
        else:
            f_len = lcid not in UL_FIXED and lcid != UL_PADDING
        res["lcids"].append(lcid)
        if lcid == UL_PADDING:
            break
        if not f_len:
            size = UL_FIXED[lcid]
        else:
            if o + (2 if f else 1) > n:
                res["ok"] = False
                break
            size = struct.unpack(">H", pdu[o:o + 2])[0] if f else pdu[o]
            o += 2 if f else 1
        if o + size > n:
            res["ok"] = False
            break
        body = pdu[o:o + size]
        if 1 <= lcid <= 32:
            res["data_bytes"] += size
        elif lcid in (UL_SHORT_BSR, UL_SHORT_TRUNC_BSR) and size == 1:
            idx = body[0] & 0x1F
            res["bsr_idx"], res["bsr_bytes"] = idx, float(BSR5[idx])
        elif lcid in (UL_LONG_BSR, UL_LONG_TRUNC_BSR) and size >= 2:
            res["bsr_idx"], res["bsr_bytes"] = body[1], float("nan")
        o += size
    return res


def build_context(direction, rnti, harq_id, sfn, slot=None, subframe=None, radio_type=2, ueid=0):
    """Context header as srsRAN (subframe given) or OAI (slot given) writes it, ending with the payload tag."""
    b = bytes([radio_type, direction, C_RNTI, TAG_RNTI]) + struct.pack(">H", rnti)
    b += bytes([TAG_UEID]) + struct.pack(">H", ueid) + bytes([TAG_HARQID, harq_id])
    if subframe is not None:
        b += bytes([TAG_PHR2, 0, TAG_FRAME_SUBFRAME]) + struct.pack(">H", (sfn << 4) | subframe)
    else:
        b += bytes([TAG_FRAME_SLOT]) + struct.pack(">HH", sfn, slot)
    return b + bytes([TAG_PAYLOAD])


def srsran_record(context, pdu, mode="udp"):
    """One DLT 252 record exactly as srsRAN's backend_pcap_writer lays it out (for fixtures and tests)."""
    name = b"udp" if mode == "udp" else b"mac-nr-framed"
    padded = name + b"\x00" * ((4 - len(name) % 4) % 4)
    hdr = struct.pack(">HH", P.EXP_PDU_TAG_DISSECTOR_NAME, len(padded)) + padded + b"\x00\x00\x00\x00"
    if mode == "udp":
        body = SIG + context + pdu
        return hdr + struct.pack(">HHHH", 0xBEEF, 0xDEAD, 8 + len(body), 0) + body
    return hdr + context + pdu
