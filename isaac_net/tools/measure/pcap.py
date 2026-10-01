"""Minimal pcap / pcapng reader and link-layer decoder, standard library only.

Enough to pull UDP payloads (probe packets, UDP-framed MAC-NR) and Wireshark "exported PDU" records out of the
captures a gNB, tcpdump or Wireshark writes. Not a general dissector: IPv4/IPv6 without extension headers, UDP only.
"""
from __future__ import annotations

import struct

# link types (https://www.tcpdump.org/linktypes.html)
DLT_NULL, DLT_EN10MB, DLT_RAW, DLT_LINUX_SLL, DLT_IPV4, DLT_IPV6 = 0, 1, 101, 113, 228, 229
DLT_LOOP, DLT_LINUX_SLL2, DLT_EXPORTED_PDU = 108, 276, 252
DLT_RAW_ALT = (12, 14)          # DLT_RAW on some BSDs / old libpcap

EXP_PDU_TAG_END, EXP_PDU_TAG_PROTO_NAME, EXP_PDU_TAG_DISSECTOR_NAME = 0, 12, 12


def iter_pcap(path):
    """Yield (t_s, linktype, frame_bytes) for every record of a pcap or pcapng file."""
    with open(path, "rb") as f:
        data = f.read()
    if len(data) < 4:
        return
    magic = data[:4]
    if magic == b"\x0a\x0d\x0d\x0a":
        yield from _iter_pcapng(data)
        return
    m = struct.unpack("<I", magic)[0]
    if m in (0xA1B2C3D4, 0xA1B23C4D):
        e = "<"
    elif m in (0xD4C3B2A1, 0x4D3CB2A1):
        e = ">"
    else:
        raise ValueError(f"{path}: not a pcap/pcapng file (magic {magic.hex()})")
    nano = struct.unpack(e + "I", data[:4])[0] == 0xA1B23C4D
    linktype = struct.unpack(e + "I", data[20:24])[0] & 0x0FFFFFFF
    off = 24
    while off + 16 <= len(data):
        sec, frac, incl, _orig = struct.unpack(e + "IIII", data[off:off + 16])
        off += 16
        yield sec + frac * (1e-9 if nano else 1e-6), linktype, data[off:off + incl]
        off += incl


def _iter_pcapng(data):
    off, e, ifaces = 0, "<", []
    while off + 12 <= len(data):
        btype = struct.unpack(e + "I", data[off:off + 4])[0]
        if btype == 0x0A0D0D0A:                            # section header: byte order
            bom = data[off + 8:off + 12]
            e = "<" if bom == b"\x4d\x3c\x2b\x1a" else ">"
            ifaces = []
        blen = struct.unpack(e + "I", data[off + 4:off + 8])[0]
        if blen < 12:
            break
        body = data[off + 8:off + blen - 4]
        if btype == 1:                                     # interface description
            lt = struct.unpack(e + "H", body[:2])[0]
            tsres = 1e-6
            o = 8
            while o + 4 <= len(body):
                code, ln = struct.unpack(e + "HH", body[o:o + 4])
                if code == 0:
                    break
                if code == 9 and ln >= 1:                  # if_tsresol
                    v = body[o + 4]
                    tsres = 2.0 ** -(v & 0x7F) if v & 0x80 else 10.0 ** -v
                o += 4 + ((ln + 3) & ~3)
            ifaces.append((lt, tsres))
        elif btype == 6:                                   # enhanced packet
            iid, th, tl, cap, _ = struct.unpack(e + "IIIII", body[:20])
            lt, tsres = ifaces[iid] if iid < len(ifaces) else (DLT_EN10MB, 1e-6)
            yield ((th << 32) | tl) * tsres, lt, body[20:20 + cap]
        elif btype == 3:                                   # simple packet
            lt, _ = ifaces[0] if ifaces else (DLT_EN10MB, 1e-6)
            yield float("nan"), lt, body[4:]
        off += blen


def ip_payload(linktype, frame):
    """(ip_version, ip_packet) of a link-layer frame, or (0, b"") if it carries no IP."""
    if linktype in (DLT_RAW, DLT_IPV4, DLT_IPV6) or linktype in DLT_RAW_ALT:
        pkt = frame
    elif linktype == DLT_EN10MB:
        et, o = struct.unpack(">H", frame[12:14])[0], 14
        while et in (0x8100, 0x88A8) and len(frame) >= o + 4:          # VLAN tags
            et, o = struct.unpack(">H", frame[o + 2:o + 4])[0], o + 4
        if et not in (0x0800, 0x86DD):
            return 0, b""
        pkt = frame[o:]
    elif linktype == DLT_LINUX_SLL:
        if struct.unpack(">H", frame[14:16])[0] not in (0x0800, 0x86DD):
            return 0, b""
        pkt = frame[16:]
    elif linktype == DLT_LINUX_SLL2:
        if struct.unpack(">H", frame[0:2])[0] not in (0x0800, 0x86DD):
            return 0, b""
        pkt = frame[20:]
    elif linktype in (DLT_NULL, DLT_LOOP):
        pkt = frame[4:]
    else:
        return 0, b""
    if not pkt:
        return 0, b""
    return pkt[0] >> 4, pkt


def udp_payload(linktype, frame):
    """(src_ip, dst_ip, sport, dport, payload) of a UDP datagram, or None."""
    v, pkt = ip_payload(linktype, frame)
    if v == 4 and len(pkt) >= 20:
        ihl = (pkt[0] & 0x0F) * 4
        if pkt[9] != 17:
            return None
        src, dst = ".".join(map(str, pkt[12:16])), ".".join(map(str, pkt[16:20]))
        u = pkt[ihl:]
    elif v == 6 and len(pkt) >= 40:
        if pkt[6] != 17:
            return None
        src, dst = pkt[8:24].hex(), pkt[24:40].hex()
        u = pkt[40:]
    else:
        return None
    if len(u) < 8:
        return None
    sport, dport, ulen = struct.unpack(">HHH", u[:6])
    return src, dst, sport, dport, u[8:ulen] if ulen >= 8 else u[8:]


def exported_pdu(frame):
    """Split a DLT 252 (Wireshark exported PDU) record into ({tag: value}, payload)."""
    tags, o = {}, 0
    while o + 4 <= len(frame):
        tag, ln = struct.unpack(">HH", frame[o:o + 4])
        o += 4
        if tag == EXP_PDU_TAG_END:
            break
        tags[tag] = frame[o:o + ln]
        o += (ln + 3) & ~3 if ln % 4 else ln
    return tags, frame[o:]


# ---------------------------------------------------------------- writers (fixtures, synthetic tests)
def write_pcap(path, linktype, records):
    """Write a classic microsecond pcap: records = [(t_s, frame_bytes)]."""
    with open(path, "wb") as f:
        f.write(struct.pack("<IHHiIII", 0xA1B2C3D4, 2, 4, 0, 0, 65535, linktype))
        for t, fr in records:
            sec = int(t)
            f.write(struct.pack("<IIII", sec, int(round((t - sec) * 1e6)), len(fr), len(fr)))
            f.write(fr)


def ipv4_udp(src, dst, sport, dport, payload):
    """A raw IPv4/UDP packet (checksums zero)."""
    ip = lambda s: bytes(int(x) for x in s.split("."))
    udp = struct.pack(">HHHH", sport, dport, 8 + len(payload), 0) + payload
    hdr = struct.pack(">BBHHHBBH4s4s", 0x45, 0, 20 + len(udp), 0, 0, 64, 17, 0, ip(src), ip(dst))
    return hdr + udp
