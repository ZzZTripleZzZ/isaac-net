"""Generate the OAI rfsim deployment files for the OAI bridge from an OpenAirInterface checkout.

OAI's configuration files are not copied into this repository. This script reads them from a local checkout of
openairinterface5g (tag 2026.w39 was used, see docs/bridges-oai.md), applies the overrides of one profile and writes a
run directory that ``docker compose`` can start (docker-compose.yaml of this directory, .env written here).

    python make_configs.py --oai ~/oai-src --out ~/oai_rfsim/run --profile lena_match --n-ue 2

Profiles (the rfsim cell; the core network is always OAI CN5G "mini" without NRF, as in OAI's CI):

    oai_default  OAI's CI cell: band 78, 106 PRB at 30 kHz (40 MHz), TDD 5 ms DDDDDDDSUU with a (6 DL, 4 UL) mixed
                 slot, min_rxtxtime 6.
    lena_match   the engine's lena_match / oai_like frame: 51 PRB at 30 kHz (20 MHz), TDD 2.5 ms DDDSU with a
                 (10 DL, 2 guard, 2 UL) special slot, min_rxtxtime 6 (the rfsim UE cannot meet shorter K1/K2).

MAC options go through ``--macrlc KEY=VALUE`` (for example ``ulsch_max_frame_inactivity=1000`` to force SR-based
access, ``pusch_TargetSNRx10=300`` to keep the UE at maximum power, ``ul_max_mcs=9``). They are written into the
MACRLCs section of the gNB YAML.

Channel models: rfsim's channel simulation (``chanmod``) is always enabled. The gNB, which is the rfsim server,
applies ``rfsimu_channel_ue<k>`` to the uplink of the k-th UE that connects; each UE applies
``rfsimu_channel_enB0`` to its downlink. All models are AWGN with ``ploss_dB`` = 0 at start, and the bridge changes
``ploss`` at run time through the telnet servers (``channelmod modify <id> ploss <dB>``).
"""
from __future__ import annotations

import argparse
import os
import re
import shutil

PROFILES = {
    "oai_default": dict(prb=106, ssb=621312, point_a=620040, coreset0=11, period=6, dl_slots=7, dl_sym=6, ul_slots=2,
                        ul_sym=4, min_rxtxtime=6, carrier_hz=3319680000, ssb_sc=516, pattern="DDDDDDDSUU", special=(6, 4, 4),
                        bandwidth_mhz=40),
    # 51 PRB band 78 values from OAI's gnb.sa.band78.51prb.usrpb200.conf; center = pointA + 51*12*30 kHz / 2; the UE's
    # --ssb is the SSB's first subcarrier above point A (the gNB prints the UE command line at start-up)
    "lena_match": dict(prb=51, ssb=640704, point_a=639996, coreset0=12, period=5, dl_slots=3, dl_sym=10, ul_slots=1,
                       ul_sym=2, min_rxtxtime=6, carrier_hz=3609120000, ssb_sc=234, pattern="DDDSU", special=(10, 2, 2),
                       bandwidth_mhz=20),
}

GNB_SRC = "ci-scripts/conf_files/gnb.sa.band78.106prb.rfsim.yaml"
UE_SRC = "ci-scripts/conf_files/nrue.uicc.yaml"
CN_SRC = "ci-scripts/yaml_files/5g_rfsimulator"
CN_FILES = ("oai_db.sql", "mini_nonrf_config.yaml", "mysql-healthcheck.sh")
MAX_UE = 4


def set_key(text, key, value, count=1):
    """Replace the value of the first `count` YAML lines `key: ...` (keys of the OAI files are unique)."""
    pat = re.compile(rf"^(\s*-?\s*){re.escape(key)}:\s*.*$", re.M)
    new, n = pat.subn(lambda m: f"{m.group(1)}{key}: {value}", text, count=count)
    if n == 0:
        raise KeyError(f"{key} not found")
    return new


def gnb_yaml(src, prof, macrlc):
    t = src
    p = PROFILES[prof]
    t = set_key(t, "min_rxtxtime", p["min_rxtxtime"])
    t = set_key(t, "absoluteFrequencySSB", p["ssb"])
    t = set_key(t, "dl_absoluteFrequencyPointA", p["point_a"])
    for k in ("dl_carrierBandwidth", "ul_carrierBandwidth"):
        t = set_key(t, k, p["prb"])
    loc = 275 * (p["prb"] - 1)
    for k in ("initialDLBWPlocationAndBandwidth", "initialULBWPlocationAndBandwidth"):
        t = set_key(t, k, loc)
    t = set_key(t, "initialDLBWPcontrolResourceSetZero", p["coreset0"])
    t = set_key(t, "dl_UL_TransmissionPeriodicity", p["period"])
    t = set_key(t, "nrofDownlinkSlots", p["dl_slots"])
    t = set_key(t, "nrofDownlinkSymbols", p["dl_sym"])
    t = set_key(t, "nrofUplinkSlots", p["ul_slots"])
    t = set_key(t, "nrofUplinkSymbols", p["ul_sym"])
    # MACRLC options: indent like pusch_TargetSNRx10
    m = re.search(r"^(\s*)pusch_TargetSNRx10:.*$", t, re.M)
    ind = m.group(1)
    for k, v in macrlc.items():
        if re.search(rf"^\s*{re.escape(k)}:", t, re.M):
            t = set_key(t, k, v)
        else:
            t = t[:m.end()] + f"\n{ind}{k}: {v}" + t[m.end():]
    # logs: keep info level but quiet the chatty layers
    for k in ("rlc_log_level", "pdcp_log_level", "ngap_log_level", "f1ap_log_level"):
        t = set_key(t, k, "warn")
    models = "\n".join(
        f"    - model_name: rfsimu_channel_ue{k}\n      type: AWGN\n      ploss_dB: 0\n      noise_power_dB: -50\n"
        f"      forgetfact: 0\n      offset: 0\n      ds_tdl: 0" for k in range(MAX_UE))
    t += f"\nchannelmod:\n  max_chan: 10\n  modellist: modellist_rfsimu_1\n  modellist_rfsimu_1:\n{models}\n"
    return t


def ue_yaml(src):
    t = src
    t = re.sub(r"^channelmod:.*", "", t, flags=re.M | re.S)       # the checkout's block, replaced below
    t += ("channelmod:\n  max_chan: 10\n  modellist: modellist_rfsimu_1\n  modellist_rfsimu_1:\n"
          "    - model_name: rfsimu_channel_enB0\n      type: AWGN\n      ploss_dB: 0\n      noise_power_dB: -50\n"
          "      forgetfact: 0\n      offset: 0\n      ds_tdl: 0\n")
    return t


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--oai", required=True, help="openairinterface5g checkout (tag 2026.w39 tested)")
    ap.add_argument("--out", required=True, help="run directory to write")
    ap.add_argument("--profile", default="lena_match", choices=sorted(PROFILES))
    ap.add_argument("--n-ue", type=int, default=1)
    ap.add_argument("--tag", default="2026.w39", help="oai-gnb / oai-nr-ue image tag")
    ap.add_argument("--cn-tag", default="v2.2.1", help="CN5G image tag")
    ap.add_argument("--macrlc", action="append", default=[], metavar="KEY=VALUE")
    ap.add_argument("--gnb-extra", default="", help="extra nr-softmodem options")
    ap.add_argument("--ue-extra", default="", help="extra nr-uesoftmodem options")
    a = ap.parse_args(argv)
    p = PROFILES[a.profile]
    os.makedirs(a.out, exist_ok=True)
    here = os.path.dirname(os.path.abspath(__file__))
    for f in CN_FILES:
        shutil.copy(os.path.join(a.oai, CN_SRC, f), os.path.join(a.out, f))
    shutil.copy(os.path.join(here, "docker-compose.yaml"), os.path.join(a.out, "docker-compose.yaml"))
    os.makedirs(os.path.join(a.out, "agent"), exist_ok=True)       # mounted read-only into the sidecars
    shutil.copy(os.path.join(here, "..", "agent.py"), os.path.join(a.out, "agent", "agent.py"))
    shutil.copy(os.path.join(here, "..", "..", "..", "tools", "measure", "probe.py"),
                os.path.join(a.out, "agent", "probe.py"))
    macrlc = dict(kv.split("=", 1) for kv in a.macrlc)
    with open(os.path.join(a.oai, GNB_SRC)) as f:
        g = gnb_yaml(f.read(), a.profile, macrlc)
    with open(os.path.join(a.out, "gnb.yaml"), "w") as f:
        f.write(g)
    with open(os.path.join(a.oai, UE_SRC)) as f:
        u = ue_yaml(f.read())
    with open(os.path.join(a.out, "nrue.yaml"), "w") as f:
        f.write(u)
    ue_args = f"-r {p['prb']} --numerology 1 --band 78 -C {p['carrier_hz']} --ssb {p['ssb_sc']}"
    with open(os.path.join(a.out, ".env"), "w") as f:
        f.write(f"TAG={a.tag}\nCN_TAG={a.cn_tag}\nUE_RADIO={ue_args}\nGNB_EXTRA={a.gnb_extra}\nUE_EXTRA={a.ue_extra}\n"
                f"COMPOSE_PROFILES={','.join(f'ue{k}' for k in range(2, a.n_ue + 1))}\n")
    meta = {"profile": a.profile, **{k: v for k, v in p.items()}, "macrlc": macrlc, "tag": a.tag, "cn_tag": a.cn_tag,
            "n_ue": a.n_ue}
    import json
    with open(os.path.join(a.out, "profile.json"), "w") as f:
        json.dump(meta, f, indent=1)
    print(f"wrote {a.out} ({a.profile}, {a.n_ue} UE)")


if __name__ == "__main__":
    main()
