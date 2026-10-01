"""Wire protocol shared by all transports (TCP, Unix socket, ns3-ai shared memory).

Every message is framed as a 12-byte header <u32 magic 'NSB1', u32 type, u32 payload_len> followed by
the payload. All fields are little endian and packed (no padding). N = nEnv * nUe UEs of one ns-3
process, UE index i = env * nUe + ue. Sizes and field order are documented in README.md.
"""
import struct

import numpy as np

MAGIC = 0x3142534E
HDR = struct.Struct("<III")
HELLO, STEP, RESULT, RESET, CLOSE, ERROR = 1, 2, 3, 4, 5, 6
PROTO_VERSION = 1
STEP_HAS_SHADOW = 1
STEP_INTERP_POS = 2
RESET_HAS_POS = 1
RESET_HAS_SHADOW = 2

FRAME_IN = np.dtype([("env", "<u2"), ("ue", "<u2"), ("fid", "<u4"), ("bytes", "<u4")])      # 12 B
FRAME_DONE = np.dtype([("env", "<u2"), ("ue", "<u2"), ("fid", "<u4"), ("t", "<f8")])        # 16 B
_HELLO = struct.Struct("<IIIIdd")
_STEP_HDR = struct.Struct("<iII")
_RES_HDR = struct.Struct("<iIIIddd")
_RESET_HDR = struct.Struct("<II")

RESULT_F32 = ("sinr_db", "rsrp_dbm", "rlc_bytes", "mcs")
RESULT_U32 = ("n_tb", "n_retx", "n_corrupt", "n_tb_lost", "ok_bytes")


def frame(mtype, payload=b""):
    return HDR.pack(MAGIC, mtype, len(payload)) + payload


def parse_hello(p):
    ver, n_env, n_ue, run, t0, step_s = _HELLO.unpack_from(p)
    if ver != PROTO_VERSION:
        raise RuntimeError(f"protocol version {ver} != {PROTO_VERSION}")
    return {"n_env": n_env, "n_ue": n_ue, "run": run, "t0": t0, "step_s": step_s}


def pack_step(t, pos, frames, shadow=None, interp=False):
    """pos float32 [N,3] (NaN = keep), frames FRAME_IN array, shadow float32 [N] or None."""
    flags = (STEP_HAS_SHADOW if shadow is not None else 0) | (STEP_INTERP_POS if interp else 0)
    parts = [_STEP_HDR.pack(int(t), flags, len(frames)), np.ascontiguousarray(pos, "<f4").tobytes()]
    if shadow is not None:
        parts.append(np.ascontiguousarray(shadow, "<f4").tobytes())
    parts.append(np.ascontiguousarray(frames, FRAME_IN).tobytes())
    return b"".join(parts)


def parse_result(p):
    t, n, n_done, k, t0, wall_run, wall_other = _RES_HDR.unpack_from(p)
    off = _RES_HDR.size
    out = {"t": t, "k": k, "t0": t0, "wall_run": wall_run, "wall_other": wall_other}
    for name in RESULT_F32:
        out[name] = np.frombuffer(p, "<f4", n, off)
        off += 4 * n
    for name in RESULT_U32:
        out[name] = np.frombuffer(p, "<u4", n, off)
        off += 4 * n
    out["done"] = np.frombuffer(p, FRAME_DONE, n_done, off)
    return out


def pack_reset(run, pos=None, shadow=None):
    flags = (RESET_HAS_POS if pos is not None else 0) | (RESET_HAS_SHADOW if shadow is not None else 0)
    parts = [_RESET_HDR.pack(int(run), flags)]
    if pos is not None:
        parts.append(np.ascontiguousarray(pos, "<f4").tobytes())
    if shadow is not None:
        parts.append(np.ascontiguousarray(shadow, "<f4").tobytes())
    return b"".join(parts)
