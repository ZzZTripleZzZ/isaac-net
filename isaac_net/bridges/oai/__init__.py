"""Full-protocol-stack validation tier: a bridge to OpenAirInterface 5G in rfsim mode (real gNB, UE and core
software with a simulated RF interface). Validation only: never in the training loop. See docs/bridges-oai.md.

    deploy/            docker-compose.yaml (CN5G + gNB + up to 4 nr-UEs + traffic sidecars) and make_configs.py,
                       which writes a run directory from an OAI checkout (profiles oai_default, lena_match)
    agent.py           stdlib-only traffic agents (UE sender, sink with kernel receive timestamps, fake relay)
    telnet.py          OaiTelnet: channel path loss per UE and the rfsim virtual clock through the telnet servers
    vclock.py          VClock: wall time -> rfsim virtual time
    stack.py           DockerOaiStack (the running deployment) and FakeStack (local UDP relay, no OAI needed)
    bridge.py          OaiBridge: per-step frame injection and collection in wall or virtual lockstep, probe logs
    net.py             OaiNet (NetBase drop-in) and make_oai_netmodule (Isaac NetModule API)
"""
from .bridge import OaiBridge  # noqa: F401
from .stack import DockerOaiStack, FakeStack  # noqa: F401
from .vclock import VClock  # noqa: F401
