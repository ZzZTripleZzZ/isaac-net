// netslot-bridge: netslot-ref (ns3ref/scratch/netslot-ref) plus an external step interface.
//
// --io=""            standalone, identical to netslot-ref (internal Bernoulli traffic).
// --io=tcp:PORT      serve one client on TCP PORT (bind 127.0.0.1, or 0.0.0.0 with --bindAll).
// --io=stdio         commands on stdin, replies on stdout.
// --io=file:IN:OUT   offline: read commands from IN, write replies to OUT.
//
// Protocol (one line each way per control step; ASCII, space separated):
//   client -> worker   S <t> <np> {<ue> <x> <y> <lossDb>}*np <nf> {<ue> <fid> <bytes>}*nf
//                      Q
//   worker -> client   D <t> <wallUs> <nd> {<ue> <fid> <genS> <lastS>}*nd
// Step t covers sim time [appStart + t*period, appStart + (t+1)*period).  Positions and
// path-loss overrides (lossDb < 0 means "no override") apply from the start of the step; the
// nf frames are sent by the UE FrameSender Tick at the start of the step, exactly where the
// standalone sender would send them (same event order, see RunUntilNextTick).  The reply lists
// the frames whose last packet reached the sink during the step.
// In offline file mode with --mobility=waypoint, the whole trace is parsed first and UE
// positions follow WaypointMobilityModel (linear interpolation between step positions).
//
// Original netslot-ref header follows.
//
// netslot-ref: 5G-LENA fidelity reference for the batched NetSlot uplink model.
//
// One gNB at the corner (0,0,10 m) of a 150 m x 150 m square, N UEs, UL-only frame traffic
// to a remote host.  Every Period (100 ms) each UE generates, with probability p, a frame of
// S bytes that is segmented into UDP packets carrying a FrameHeader (ue, frame id, packet
// index, packet count, generation time).  The remote-host sink reassembles frames and the
// program writes frames.csv (one row per generated frame) plus packets.csv (optional).
//
// Radio configuration mirrors NetSlot (netsim.py) as far as 5G-LENA allows:
//   TDD DDDSU, numerology 1 (30 kHz), 20 MHz carrier trimmed to 50 PRBs, RBG = 10 PRBs (5 RBGs),
//   UE Tx 23 dBm spread over allocated RBs, no UL power control,
//   path loss 40 + 35 log10(d2D) with 6 dB log-normal shadowing (custom model below),
//   3GPP TR 38.901 UMi-StreetCanyon NLOS small-scale fading (optional),
//   gNB noise figure chosen so noise+interference = -90 dBm per 10-PRB subband,
//   OFDMA PF UL scheduler, SR/BSR, HARQ (max 3 retx = 4 tx), EESM IR error model (MCS table 1),
//   RLC UM with a PDCP discard timer equal to the 2 s app deadline.

#include "ns3/antenna-module.h"
#include "ns3/applications-module.h"
#include "ns3/core-module.h"
#include "ns3/flow-monitor-module.h"
#include "ns3/internet-module.h"
#include "ns3/mobility-module.h"
#include "ns3/network-module.h"
#include "ns3/nr-module.h"
#include "ns3/point-to-point-module.h"
#include "ns3/propagation-loss-model.h"

#include <chrono>
#include <cmath>
#include <fstream>
#include <map>
#include <sstream>
#include <cstdio>
#include <cstring>
#include <deque>
#include <arpa/inet.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <sys/socket.h>
#include <unistd.h>
#include <atomic>
#include <mutex>
#include <thread>
#include <algorithm>

using namespace ns3;

NS_LOG_COMPONENT_DEFINE("NetSlotBridge");

// Per-UE shadowing drawn in main() (coverage-conditioned drop); keyed by UE node id.
static std::map<uint32_t, double> g_ueShadowDb;
// Bridge: per-UE total path loss override (dB), keyed by UE node id; wins over everything.
static std::map<uint32_t, double> g_lossOverride;

// ---------------------------------------------------------------------------------------------
// Log-distance path loss on the 2D distance with per-link frozen log-normal shadowing.
// ---------------------------------------------------------------------------------------------
class LogDistShadowLoss : public PropagationLossModel
{
  public:
    static TypeId GetTypeId()
    {
        static TypeId tid =
            TypeId("ns3::LogDistShadowLoss")
                .SetParent<PropagationLossModel>()
                .AddConstructor<LogDistShadowLoss>()
                .AddAttribute("ReferenceLoss",
                              "Loss at 1 m (dB)",
                              DoubleValue(40.0),
                              MakeDoubleAccessor(&LogDistShadowLoss::m_refLoss),
                              MakeDoubleChecker<double>())
                .AddAttribute("Exponent",
                              "Path-loss exponent",
                              DoubleValue(3.5),
                              MakeDoubleAccessor(&LogDistShadowLoss::m_exp),
                              MakeDoubleChecker<double>())
                .AddAttribute("ShadowingStd",
                              "Std of log-normal shadowing (dB), 0 disables",
                              DoubleValue(6.0),
                              MakeDoubleAccessor(&LogDistShadowLoss::m_shStd),
                              MakeDoubleChecker<double>(0.0))
                .AddAttribute("MinDistance",
                              "Distance clamp (m)",
                              DoubleValue(1.0),
                              MakeDoubleAccessor(&LogDistShadowLoss::m_minD),
                              MakeDoubleChecker<double>(0.0));
        return tid;
    }

    LogDistShadowLoss()
    {
        m_rv = CreateObject<NormalRandomVariable>();
        m_rv->SetAttribute("Mean", DoubleValue(0.0));
        m_rv->SetAttribute("Variance", DoubleValue(1.0));
    }

    double GetLossDb(Ptr<MobilityModel> a, Ptr<MobilityModel> b) const
    {
        Vector pa = a->GetPosition();
        Vector pb = b->GetPosition();
        double d = std::max(m_minD, std::hypot(pa.x - pb.x, pa.y - pb.y));
        double loss = m_refLoss + 10.0 * m_exp * std::log10(d);
        if (!g_lossOverride.empty())
        {
            for (auto* m : {PeekPointer(a), PeekPointer(b)})
            {
                Ptr<Node> n = m->GetObject<Node>();
                if (n)
                {
                    auto it = g_lossOverride.find(n->GetId());
                    if (it != g_lossOverride.end())
                    {
                        return it->second;
                    }
                }
            }
        }
        for (auto* m : {PeekPointer(a), PeekPointer(b)})
        {
            Ptr<Node> n = m->GetObject<Node>();
            if (n)
            {
                auto it = g_ueShadowDb.find(n->GetId());
                if (it != g_ueShadowDb.end())
                {
                    return loss + it->second;
                }
            }
        }
        if (m_shStd > 0)
        {
            auto key = std::make_pair(std::min(PeekPointer(a), PeekPointer(b)),
                                      std::max(PeekPointer(a), PeekPointer(b)));
            auto it = m_shadow.find(key);
            if (it == m_shadow.end())
            {
                it = m_shadow.emplace(key, m_shStd * m_rv->GetValue()).first;
            }
            loss += it->second;
        }
        return loss;
    }

  private:
    double DoCalcRxPower(double txPowerDbm, Ptr<MobilityModel> a, Ptr<MobilityModel> b) const override
    {
        return txPowerDbm - GetLossDb(a, b);
    }

    int64_t DoAssignStreams(int64_t stream) override
    {
        m_rv->SetStream(stream);
        return 1;
    }

    double m_refLoss{40.0};
    double m_exp{3.5};
    double m_shStd{6.0};
    double m_minD{1.0};
    Ptr<NormalRandomVariable> m_rv;
    mutable std::map<std::pair<MobilityModel*, MobilityModel*>, double> m_shadow;
};

NS_OBJECT_ENSURE_REGISTERED(LogDistShadowLoss);


// ---------------------------------------------------------------------------------------------
// Optional speed-up (--ueUeFilter=1): drop UE->UE signals in the spectrum channel.  In this
// single-cell TDD scenario a UE only transmits in UL slots, when no UE is receiving, so these
// signals never affect decoding, but the channel still copies them and runs propagation and 3GPP
// fading for every UE pair (O(R^2) per UL transmission).
// ---------------------------------------------------------------------------------------------
class UeUeFilter : public SpectrumTransmitFilter
{
  public:
    static TypeId GetTypeId()
    {
        static TypeId tid = TypeId("ns3::UeUeFilter")
                                .SetParent<SpectrumTransmitFilter>()
                                .AddConstructor<UeUeFilter>();
        return tid;
    }

  private:
    bool DoFilter(Ptr<const SpectrumSignalParameters> params,
                  Ptr<const SpectrumPhy> receiverPhy) override
    {
        auto rx = receiverPhy->GetDevice();
        auto tx = params->txPhy ? params->txPhy->GetDevice() : nullptr;
        return rx && tx && DynamicCast<NrUeNetDevice>(rx) && DynamicCast<NrUeNetDevice>(tx);
    }

    int64_t DoAssignStreams(int64_t stream) override
    {
        return 0;
    }
};

NS_OBJECT_ENSURE_REGISTERED(UeUeFilter);
// ---------------------------------------------------------------------------------------------
// Frame header carried in every UDP packet.
// ---------------------------------------------------------------------------------------------
class FrameHeader : public Header
{
  public:
    uint16_t ue{0};
    uint32_t fid{0};
    uint16_t idx{0};
    uint16_t npk{0};
    uint64_t genNs{0};

    static TypeId GetTypeId()
    {
        static TypeId tid =
            TypeId("ns3::FrameHeader").SetParent<Header>().AddConstructor<FrameHeader>();
        return tid;
    }

    TypeId GetInstanceTypeId() const override
    {
        return GetTypeId();
    }

    uint32_t GetSerializedSize() const override
    {
        return 18;
    }

    void Serialize(Buffer::Iterator i) const override
    {
        i.WriteHtonU16(ue);
        i.WriteHtonU32(fid);
        i.WriteHtonU16(idx);
        i.WriteHtonU16(npk);
        i.WriteHtonU64(genNs);
    }

    uint32_t Deserialize(Buffer::Iterator i) override
    {
        ue = i.ReadNtohU16();
        fid = i.ReadNtohU32();
        idx = i.ReadNtohU16();
        npk = i.ReadNtohU16();
        genNs = i.ReadNtohU64();
        return 18;
    }

    void Print(std::ostream& os) const override
    {
        os << "ue=" << ue << " fid=" << fid << " idx=" << idx << "/" << npk << " gen=" << genNs;
    }
};

// Global bookkeeping shared by senders and the sink.
struct FrameRec
{
    uint16_t ue;
    uint32_t fid;
    double gen;
    uint32_t bytes;
    uint16_t npk;
    uint16_t rxpk{0};
    uint32_t rxbytes{0};
    double last{-1};
};

static std::map<std::pair<uint16_t, uint32_t>, FrameRec> g_frames;
static std::ofstream g_pktOut;
// Bridge: frames completed since the last reply.
static bool g_bridge = false;
static std::vector<FrameRec> g_done;
// Real-time mode: deliveries are streamed as they happen.
static FILE* g_rtOut = nullptr;
static std::chrono::steady_clock::time_point g_rtWall0;

static double
RtWallNow()
{
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - g_rtWall0).count();
}

// ---------------------------------------------------------------------------------------------
// On/off frame sender.
// ---------------------------------------------------------------------------------------------
class FrameSender : public Application
{
  public:
    static TypeId GetTypeId()
    {
        static TypeId tid =
            TypeId("ns3::FrameSender").SetParent<Application>().AddConstructor<FrameSender>();
        return tid;
    }

    void Setup(Address remote,
               uint16_t ue,
               uint32_t frameBytes,
               double p,
               Time period,
               Time firstTick,
               uint32_t nTicks,
               uint32_t payload,
               int64_t stream)
    {
        m_remote = remote;
        m_ue = ue;
        m_bytes = frameBytes;
        m_p = p;
        m_period = period;
        m_first = firstTick;
        m_nTicks = nTicks;
        m_payload = payload;
        m_u = CreateObject<UniformRandomVariable>();
        m_u->SetStream(stream);
    }

  private:
    void StartApplication() override
    {
        m_sock = Socket::CreateSocket(GetNode(), UdpSocketFactory::GetTypeId());
        m_sock->Bind();
        m_sock->Connect(m_remote);
        Simulator::Schedule(m_first - Simulator::Now(), &FrameSender::Tick, this);
    }

    void StopApplication() override
    {
        if (m_sock)
        {
            m_sock->Close();
        }
    }

  public:
    // Bridge: queue a frame to be sent by the next Tick (the start of the next control step).
    void SetExternal()
    {
        m_ext = true;
    }

    void QueueExt(uint32_t fid, uint32_t bytes)
    {
        m_extQ.emplace_back(fid, bytes);
    }

    // Real-time mode: send a frame now (called in the simulator thread).
    void SendNow(uint32_t fid, uint32_t bytes)
    {
        SendFrame(fid, bytes);
    }

  private:
    void Tick()
    {
        if (m_ext)
        {
            m_tick++;
            for (const auto& [fid, bytes] : m_extQ)
            {
                SendFrame(fid, bytes);
            }
            m_extQ.clear();
            Simulator::Schedule(m_period, &FrameSender::Tick, this);
            return;
        }
        if (m_tick >= m_nTicks)
        {
            return;
        }
        m_tick++;
        if (m_u->GetValue() < m_p)
        {
            SendFrame(m_fid, m_bytes);
            m_fid++;
        }
        Simulator::Schedule(m_period, &FrameSender::Tick, this);
    }

    void SendFrame(uint32_t fid, uint32_t bytes)
    {
        uint16_t n = static_cast<uint16_t>((bytes + m_payload - 1) / m_payload);
        uint64_t now = Simulator::Now().GetNanoSeconds();
        FrameRec rec{m_ue, fid, Simulator::Now().GetSeconds(), bytes, n};
        g_frames[{m_ue, fid}] = rec;
        for (uint16_t i = 0; i < n; ++i)
        {
            uint32_t sz = std::min(m_payload, bytes - i * m_payload);
            Ptr<Packet> pkt = Create<Packet>(sz);
            FrameHeader h;
            h.ue = m_ue;
            h.fid = fid;
            h.idx = i;
            h.npk = n;
            h.genNs = now;
            pkt->AddHeader(h);
            m_sock->Send(pkt);
        }
    }

    Ptr<Socket> m_sock;
    Address m_remote;
    uint16_t m_ue{0};
    uint32_t m_bytes{4000};
    double m_p{0.5};
    Time m_period{MilliSeconds(100)};
    Time m_first;
    uint32_t m_nTicks{0};
    uint32_t m_tick{0};
    uint32_t m_payload{1400};
    uint32_t m_fid{0};
    Ptr<UniformRandomVariable> m_u;
    bool m_ext{false};
    std::vector<std::pair<uint32_t, uint32_t>> m_extQ;
};

// ---------------------------------------------------------------------------------------------
// Frame sink on the remote host.
// ---------------------------------------------------------------------------------------------
class FrameSink : public Application
{
  public:
    static TypeId GetTypeId()
    {
        static TypeId tid =
            TypeId("ns3::FrameSink").SetParent<Application>().AddConstructor<FrameSink>();
        return tid;
    }

    void Setup(uint16_t port)
    {
        m_port = port;
    }

  private:
    void StartApplication() override
    {
        m_sock = Socket::CreateSocket(GetNode(), UdpSocketFactory::GetTypeId());
        m_sock->Bind(InetSocketAddress(Ipv4Address::GetAny(), m_port));
        m_sock->SetRecvCallback(MakeCallback(&FrameSink::Rx, this));
    }

    void StopApplication() override
    {
        if (m_sock)
        {
            m_sock->Close();
        }
    }

    void Rx(Ptr<Socket> s)
    {
        Ptr<Packet> pkt;
        Address from;
        while ((pkt = s->RecvFrom(from)))
        {
            FrameHeader h;
            pkt->RemoveHeader(h);
            double now = Simulator::Now().GetSeconds();
            auto it = g_frames.find({h.ue, h.fid});
            if (it != g_frames.end())
            {
                it->second.rxpk++;
                it->second.rxbytes += pkt->GetSize();
                it->second.last = now;
                if (g_bridge && it->second.rxpk == it->second.npk)
                {
                    if (g_rtOut)
                    {
                        std::fprintf(g_rtOut, "D %u %u %.9f %.9f %.6f\n", it->second.ue,
                                     it->second.fid, it->second.gen, it->second.last, RtWallNow());
                        std::fflush(g_rtOut);
                    }
                    else
                    {
                        g_done.push_back(it->second);
                    }
                    g_frames.erase(it);
                }
            }
            if (g_pktOut.is_open())
            {
                g_pktOut << h.ue << "," << h.fid << "," << h.idx << "," << h.genNs * 1e-9 << ","
                         << now << "\n";
            }
        }
    }

    Ptr<Socket> m_sock;
    uint16_t m_port{9000};
};

// ---------------------------------------------------------------------------------------------
// gNB slot utilization (PRB usage) and UE MAC SR/BSR state + RLC buffer.
// ---------------------------------------------------------------------------------------------
static std::ofstream g_posOut;
static std::ofstream g_slotOut;
static std::ofstream g_bufOut;
static std::map<uint16_t, std::pair<int, uint32_t>> g_lastBuf; // nodeId -> (srState, bytes)

static void
SlotDataStats(const SfnSf& sfn,
              uint32_t scheduledUe,
              uint32_t usedReg,
              uint32_t usedSym,
              uint32_t availableRb,
              uint32_t availableSym,
              uint16_t bwpId,
              uint16_t cellId)
{
    if (scheduledUe == 0)
    {
        return;
    }
    g_slotOut << Simulator::Now().GetSeconds() << "," << sfn.GetFrame() << ","
              << +sfn.GetSubframe() << "," << +sfn.GetSlot() << "," << scheduledUe << ","
              << usedReg << "," << usedSym << "," << availableRb << "," << availableSym << "\n";
}

static void
UeMacState(uint64_t imsi,
           const SfnSf sfn,
           const uint16_t nodeId,
           const uint16_t rnti,
           const uint8_t bwpId,
           const NrUeMac::SrBsrMachine srState,
           std::unordered_map<uint8_t, NrMacSapProvider::BufferStatusReportParameters> bsr,
           int retx,
           std::string nameFunc)
{
    uint32_t bytes = 0;
    for (const auto& [lcid, b] : bsr)
    {
        bytes += b.txQueueSize + b.retxQueueSize + b.statusPduSize;
    }
    auto cur = std::make_pair(static_cast<int>(srState), bytes);
    auto it = g_lastBuf.find(nodeId);
    if (it != g_lastBuf.end() && it->second == cur)
    {
        return; // log changes only
    }
    g_lastBuf[nodeId] = cur;
    g_bufOut << Simulator::Now().GetSeconds() << "," << nodeId << "," << rnti << ","
             << static_cast<int>(srState) << "," << bytes << "," << retx << "," << nameFunc
             << "\n";
}

// ---------------------------------------------------------------------------------------------
// Bridge driver.
// ---------------------------------------------------------------------------------------------
static int
OpenListen(int port, bool bindAll)
{
    int fd = socket(AF_INET, SOCK_STREAM, 0);
    NS_ABORT_MSG_IF(fd < 0, "socket");
    int one = 1;
    setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));
    sockaddr_in a{};
    a.sin_family = AF_INET;
    a.sin_port = htons(port);
    a.sin_addr.s_addr = bindAll ? htonl(INADDR_ANY) : htonl(INADDR_LOOPBACK);
    NS_ABORT_MSG_IF(bind(fd, reinterpret_cast<sockaddr*>(&a), sizeof(a)) < 0,
                    "bind port " << port << ": " << std::strerror(errno));
    NS_ABORT_MSG_IF(listen(fd, 1) < 0, "listen");
    return fd;
}

static const char*
SkipWs(const char* p)
{
    while (*p == ' ' || *p == '\t')
    {
        ++p;
    }
    return p;
}

static long
NextLong(const char*& p)
{
    char* e = nullptr;
    long v = std::strtol(SkipWs(p), &e, 10);
    NS_ABORT_MSG_IF(e == p, "protocol: integer expected at '" << std::string(p).substr(0, 40) << "'");
    p = e;
    return v;
}

static double
NextDouble(const char*& p)
{
    char* e = nullptr;
    double v = std::strtod(SkipWs(p), &e);
    NS_ABORT_MSG_IF(e == p, "protocol: number expected at '" << std::string(p).substr(0, 40) << "'");
    p = e;
    return v;
}

static int
RunBridge(const std::string& io,
          bool bindAll,
          const std::string& mobility,
          bool flowmon,
          NodeContainer ueNodes,
          Ptr<Node> remoteHost,
          std::vector<Ptr<FrameSender>>& apps,
          double appStart,
          Time period,
          int listenFd)
{
    FILE* in = nullptr;
    FILE* out = nullptr;
    int connFd = -1;
    if (io == "stdio")
    {
        in = stdin;
        out = stdout;
    }
    else if (io.rfind("file:", 0) == 0)
    {
        std::string rest = io.substr(5);
        auto c = rest.find(':');
        NS_ABORT_MSG_IF(c == std::string::npos, "--io=file:IN:OUT");
    }
    else if (io.rfind("tcp:", 0) != 0)
    {
        NS_ABORT_MSG("unknown --io " << io);
    }

    // File mode streams the trace line by line, opened at the same point where TCP mode
    // accepts its client, so both modes have the same heap history.  5G-LENA's outcome depends
    // on heap layout (see README: some UE containers are ordered by pointer), so the two modes are
    // only bit-identical this way.  Waypoint mobility needs the whole trace up front.
    std::vector<std::string> script;
    const bool fileMode = io.rfind("file:", 0) == 0;
    const bool waypoint = fileMode && mobility == "waypoint";
    std::string inPath;
    std::string outPath;
    if (fileMode)
    {
        std::string rest = io.substr(5);
        auto c = rest.find(':');
        inPath = rest.substr(0, c);
        outPath = rest.substr(c + 1);
    }
    if (waypoint)
    {
        in = std::fopen(inPath.c_str(), "r");
        char* buf = nullptr;
        size_t cap = 0;
        while (getline(&buf, &cap, in) > 0)
        {
            script.emplace_back(buf);
        }
        std::free(buf);
        std::fclose(in);
        in = nullptr;
    }
    if (waypoint)
    {
        for (const auto& line : script)
        {
            const char* p = line.c_str();
            p = SkipWs(p);
            if (*p != 'S')
            {
                continue;
            }
            ++p;
            long t = NextLong(p);
            long np = NextLong(p);
            Time at = Seconds(appStart) + period * t;
            for (long k = 0; k < np; ++k)
            {
                long ue = NextLong(p);
                double x = NextDouble(p);
                double y = NextDouble(p);
                NextDouble(p);
                ueNodes.Get(ue)->GetObject<WaypointMobilityModel>()->AddWaypoint(
                    Waypoint(at, Vector(x, y, 1.5)));
            }
        }
    }

    Ptr<FlowMonitor> fm;
    FlowMonitorHelper fmh;
    if (flowmon)
    {
        NodeContainer ends;
        ends.Add(remoteHost);
        ends.Add(ueNodes);
        fm = fmh.Install(ends);
    }

    // Run up to the first control-step boundary.  The Stop event is scheduled before the
    // FrameSender's first Tick, so it fires first and Run returns just before Tick_0.
    auto w0 = std::chrono::steady_clock::now();
    Simulator::Stop(Seconds(appStart));
    Simulator::Run();
    double setupWall = std::chrono::duration<double>(std::chrono::steady_clock::now() - w0).count();

    if (fileMode)
    {
        if (!waypoint)
        {
            in = std::fopen(inPath.c_str(), "r");
        }
        out = std::fopen(outPath.c_str(), "w");
        NS_ABORT_MSG_IF((!waypoint && !in) || !out, "cannot open " << inPath << " / " << outPath);
    }
    if (listenFd >= 0)
    {
        connFd = accept(listenFd, nullptr, nullptr);
        NS_ABORT_MSG_IF(connFd < 0, "accept");
        int one = 1;
        setsockopt(connFd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));
        in = fdopen(connFd, "r");
        out = fdopen(dup(connFd), "w");
    }
    std::fprintf(out,
                 "H %u %.9f %.9f %.6f\n",
                 ueNodes.GetN(),
                 appStart,
                 period.GetSeconds(),
                 setupWall);
    std::fflush(out);

    long expect = 0;
    size_t lineNo = 0;
    char* buf = nullptr;
    size_t cap = 0;
    std::string reply;
    reply.reserve(1 << 16);
    while (true)
    {
        const char* p = nullptr;
        if (waypoint)
        {
            if (lineNo >= script.size())
            {
                break;
            }
            p = script[lineNo++].c_str();
        }
        else
        {
            if (getline(&buf, &cap, in) <= 0)
            {
                break;
            }
            p = buf;
        }
        p = SkipWs(p);
        if (*p == 'Q')
        {
            break;
        }
        if (*p != 'S')
        {
            continue;
        }
        ++p;
        long t = NextLong(p);
        NS_ABORT_MSG_IF(t != expect, "step " << t << " out of order, expected " << expect);
        long np = NextLong(p);
        for (long k = 0; k < np; ++k)
        {
            long ue = NextLong(p);
            double x = NextDouble(p);
            double y = NextDouble(p);
            double l = NextDouble(p);
            Ptr<Node> n = ueNodes.Get(ue);
            if (!waypoint)
            {
                Ptr<MobilityModel> mm = n->GetObject<MobilityModel>();
                Vector v = mm->GetPosition();
                if (std::abs(v.x - x) > 1e-9 || std::abs(v.y - y) > 1e-9)
                {
                    mm->SetPosition(Vector(x, y, 1.5));
                }
            }
            if (l >= 0)
            {
                g_lossOverride[n->GetId()] = l;
            }
            else
            {
                g_lossOverride.erase(n->GetId());
            }
        }
        long nf = NextLong(p);
        for (long k = 0; k < nf; ++k)
        {
            long ue = NextLong(p);
            long fid = NextLong(p);
            long bytes = NextLong(p);
            apps.at(ue)->QueueExt(static_cast<uint32_t>(fid), static_cast<uint32_t>(bytes));
        }
        auto r0 = std::chrono::steady_clock::now();
        Simulator::Stop(period);
        Simulator::Run();
        double wall = std::chrono::duration<double>(std::chrono::steady_clock::now() - r0).count();
        reply.clear();
        char tmp[128];
        std::snprintf(tmp, sizeof(tmp), "D %ld %.0f %zu", t, wall * 1e6, g_done.size());
        reply += tmp;
        for (const auto& r : g_done)
        {
            std::snprintf(tmp, sizeof(tmp), " %u %u %.9f %.9f", r.ue, r.fid, r.gen, r.last);
            reply += tmp;
        }
        reply += "\n";
        g_done.clear();
        std::fputs(reply.c_str(), out);
        std::fflush(out);
        expect++;
    }
    std::free(buf);
    if (out)
    {
        std::fclose(out);
    }
    if (in && in != stdin)
    {
        std::fclose(in);
    }
    if (listenFd >= 0)
    {
        close(listenFd);
    }
    Simulator::Destroy();
    return 0;
}

// ---------------------------------------------------------------------------------------------
// Real-time emulation driver (RealtimeSimulatorImpl, BestEffort).  The client streams, in wall
// clock time:   F <ue> <fid> <bytes>        send a frame from UE ue now
//               P <ue> <x> <y> <lossDb>     move UE / set its path-loss override now
//               Q                           stop
// The worker streams back  D <ue> <fid> <genS> <lastS> <wallS>  per delivered frame and, at the
// end, a lag summary  L <n> <p50Ms> <p99Ms> <maxMs> <finalMs> <fracOver10ms> <warmupMaxMs>  where
// lag = wall - sim time, sampled every 10 ms of sim time from 0.1 s; the stats cover sim >= 1.5 s
// (after attach and first channel generation), the warm-up is reported as its max only.
// ---------------------------------------------------------------------------------------------
static std::vector<double> g_lagMs;

static void
RtStop()
{
    Simulator::Stop();
}

static std::vector<double> g_lagWarmMs; // lag samples during the warm-up (attach, first channels)
static const double kRtWarmupS = 1.5;
static std::string g_rtLagFile; // --rtLagFile: dump the post-warm-up lag series

static void
LagProbe()
{
    double lag = 1e3 * (RtWallNow() - Simulator::Now().GetSeconds());
    (Simulator::Now().GetSeconds() < kRtWarmupS ? g_lagWarmMs : g_lagMs).push_back(lag);
    Simulator::Schedule(MilliSeconds(10), &LagProbe);
}

static void
RtApplyPos(Ptr<Node> n, double x, double y, double l)
{
    n->GetObject<MobilityModel>()->SetPosition(Vector(x, y, 1.5));
    if (l >= 0)
    {
        g_lossOverride[n->GetId()] = l;
    }
}

static int
RunRealtime(NodeContainer ueNodes, std::vector<Ptr<FrameSender>>& apps, int listenFd)
{
    int connFd = accept(listenFd, nullptr, nullptr);
    NS_ABORT_MSG_IF(connFd < 0, "accept");
    int one = 1;
    setsockopt(connFd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));
    FILE* in = fdopen(connFd, "r");
    g_rtOut = fdopen(dup(connFd), "w");
    std::fprintf(g_rtOut, "H %u rt\n", ueNodes.GetN());
    std::fflush(g_rtOut);
    std::atomic<bool> quit{false};
    std::thread reader([&]() {
        char* buf = nullptr;
        size_t cap = 0;
        while (getline(&buf, &cap, in) > 0)
        {
            const char* p = SkipWs(buf);
            if (*p == 'Q')
            {
                break;
            }
            char c = *p++;
            if (c == 'F')
            {
                long ue = NextLong(p);
                long fid = NextLong(p);
                long bytes = NextLong(p);
                Simulator::ScheduleWithContext(ueNodes.Get(ue)->GetId(), Time(0),
                                               &FrameSender::SendNow, apps.at(ue),
                                               static_cast<uint32_t>(fid),
                                               static_cast<uint32_t>(bytes));
            }
            else if (c == 'P')
            {
                long ue = NextLong(p);
                double x = NextDouble(p);
                double y = NextDouble(p);
                double l = NextDouble(p);
                Simulator::ScheduleWithContext(ueNodes.Get(ue)->GetId(), Time(0), &RtApplyPos,
                                               ueNodes.Get(ue), x, y, l);
            }
        }
        std::free(buf);
        quit = true;
        Simulator::ScheduleWithContext(0xffffffff, Time(0), &RtStop);
    });
    g_rtWall0 = std::chrono::steady_clock::now();
    Simulator::Schedule(MilliSeconds(100), &LagProbe);
    Simulator::Stop(Seconds(3600));
    Simulator::Run();
    reader.join();
    if (!g_rtLagFile.empty())
    {
        std::ofstream lf(g_rtLagFile);
        lf << "sim_s,lag_ms\n";
        for (size_t i = 0; i < g_lagMs.size(); ++i)
        {
            lf << kRtWarmupS + 0.01 * i << "," << g_lagMs[i] << "\n";   // approx. sim time of the sample
        }
    }
    std::vector<double> v = g_lagMs;
    std::sort(v.begin(), v.end());
    auto q = [&](double f) { return v.empty() ? 0.0 : v[std::min(v.size() - 1, size_t(f * v.size()))]; };
    size_t over10 = std::count_if(v.begin(), v.end(), [](double x) { return x > 10.0; });
    double warmMax = g_lagWarmMs.empty() ? 0.0 : *std::max_element(g_lagWarmMs.begin(), g_lagWarmMs.end());
    std::fprintf(g_rtOut, "L %zu %.3f %.3f %.3f %.3f %.5f %.3f\n", v.size(), q(0.5), q(0.99),
                 v.empty() ? 0.0 : v.back(), g_lagMs.empty() ? 0.0 : g_lagMs.back(),
                 v.empty() ? 0.0 : double(over10) / v.size(), warmMax);
    std::fflush(g_rtOut);
    std::fclose(g_rtOut);
    std::fclose(in);
    close(listenFd);
    Simulator::Destroy();
    return 0;
}

int
main(int argc, char* argv[])
{
    uint32_t nUe = 8;
    uint32_t frameBytes = 4000;
    double p = 0.5;
    double trafficTime = 20.0; // seconds of frame generation
    double appStart = 0.5;
    double deadline = 2.0;
    double periodMs = 100.0;
    uint32_t payload = 1400;
    std::string placement = "random"; // random | dists
    std::string dists = "";           // comma list of distances (m) when placement=dists
    double side = 150.0;
    double freq = 3.5e9;
    double bw = 20e6;
    double rbOverhead = 0.1; // 20 MHz * 0.9 / 360 kHz = 50 PRBs
    uint32_t rbgSize = 10;
    double ueTxPower = 23.0;
    double gnbTxPower = 30.0;
    double niPerSubbandDbm = -90.0;
    double shadowStd = 6.0;
    double minSnrDb = -1000.0; // redraw a UE whose wideband per-RB SNR (power over all RBGs) is below this
    bool fading = true;
    double vScatt = 3.0; // m/s: NetSlot RHO=0.93 per UL slot corresponds to 3 m/s at 3.5 GHz
    std::string errorModel = "ns3::NrEesmIrT1";
    std::string sched = "ns3::NrMacSchedulerOfdmaPF";
    std::string ulPowerAlloc = "UniformPowerAllocUsed"; // or UniformPowerAllocBw
    std::string rlc = "UM"; // UM (with PDCP discard = deadline) | AM
    bool harq = true;
    bool srs = false; // SRS in UL slots; off by default (avoids an EESM empty-RB abort in v5.1)
    bool syncPhase = true;
    bool pktLog = true;
    bool macTraces = true;
    uint32_t run = 1;
    std::string outDir = ".";
    std::string io = "";         // "" standalone | tcp:PORT | stdio | file:IN:OUT
    bool bindAll = false;
    std::string initPos = "";    // x:y:loss,... one per UE
    std::string mobility = "hold"; // hold (SetPosition per step) | waypoint (file mode only)
    bool flowmon = true;
    bool ueUeFilter = false;
    std::string rtLagFile = "";

    CommandLine cmd(__FILE__);
    cmd.AddValue("nUe", "Number of UEs", nUe);
    cmd.AddValue("frameBytes", "Frame size S in bytes", frameBytes);
    cmd.AddValue("p", "Per-period send probability", p);
    cmd.AddValue("trafficTime", "Seconds of frame generation", trafficTime);
    cmd.AddValue("appStart", "Traffic start (s)", appStart);
    cmd.AddValue("deadline", "App deadline (s); sim runs this long past the last frame", deadline);
    cmd.AddValue("periodMs", "Frame period (ms)", periodMs);
    cmd.AddValue("payload", "UDP payload bytes per packet", payload);
    cmd.AddValue("placement", "random | dists", placement);
    cmd.AddValue("dists", "Comma-separated UE distances (m) for placement=dists", dists);
    cmd.AddValue("side", "Square side (m) for random drop", side);
    cmd.AddValue("freq", "Carrier frequency (Hz)", freq);
    cmd.AddValue("bw", "Channel bandwidth (Hz)", bw);
    cmd.AddValue("rbOverhead", "NrHelper RB overhead (0.1 gives 50 PRBs at 20 MHz, mu=1)", rbOverhead);
    cmd.AddValue("rbgSize", "PRBs per RBG (scheduler granularity)", rbgSize);
    cmd.AddValue("ueTxPower", "UE Tx power (dBm)", ueTxPower);
    cmd.AddValue("gnbTxPower", "gNB Tx power (dBm)", gnbTxPower);
    cmd.AddValue("niPerSubbandDbm", "Noise+interference per 10-PRB subband (dBm)", niPerSubbandDbm);
    cmd.AddValue("shadowStd", "Log-normal shadowing std (dB)", shadowStd);
    cmd.AddValue("minSnrDb",
                 "Coverage-conditioned drop: redraw UEs whose SNR with 23 dBm spread over all RBGs "
                 "(incl. shadowing, no fading) is below this (dB)",
                 minSnrDb);
    cmd.AddValue("fading", "Enable 3GPP UMi NLOS small-scale fading", fading);
    cmd.AddValue("vScatt", "Scatterer speed for Doppler (m/s)", vScatt);
    cmd.AddValue("errorModel", "UL error model TypeId", errorModel);
    cmd.AddValue("sched", "Scheduler TypeId", sched);
    cmd.AddValue("rlc", "UM | AM", rlc);
    cmd.AddValue("ulPowerAlloc",
                 "UE PSD: UniformPowerAllocUsed (23 dBm over allocated RBs, NetSlot-like) or "
                 "UniformPowerAllocBw (23 dBm over all 50 RBs, SINR independent of allocation)",
                 ulPowerAlloc);
    cmd.AddValue("harq", "Enable HARQ retransmissions", harq);
    cmd.AddValue("srs", "Transmit SRS in UL slots (UL CQI then also from SRS)", srs);
    cmd.AddValue("syncPhase", "All UEs tick at the same instants (NetSlot control steps)", syncPhase);
    cmd.AddValue("pktLog", "Write packets.csv", pktLog);
    cmd.AddValue("macTraces", "Enable NR PHY/MAC/RLC trace files", macTraces);
    cmd.AddValue("run", "RNG run number", run);
    cmd.AddValue("outDir", "Output directory for csv files", outDir);
    cmd.AddValue("io", "Bridge I/O: empty (standalone), tcp:PORT, stdio, file:IN:OUT", io);
    cmd.AddValue("bindAll", "tcp: bind 0.0.0.0 instead of 127.0.0.1", bindAll);
    cmd.AddValue("init", "Bridge: initial UE x:y:lossDb list (lossDb<0 = model loss)", initPos);
    cmd.AddValue("mobility", "Bridge file mode: hold | waypoint", mobility);
    cmd.AddValue("flowmon", "Install FlowMonitor and write flowmon.xml", flowmon);
    cmd.AddValue("ueUeFilter", "Skip UE->UE signals in the spectrum channel (speed-up)", ueUeFilter);
    cmd.AddValue("rtLagFile", "rt mode: write the lag series (csv) here", rtLagFile);
    cmd.Parse(argc, argv);
    g_bridge = !io.empty();
    g_rtLagFile = rtLagFile;
    NS_ABORT_MSG_IF(g_bridge && !syncPhase, "bridge mode needs --syncPhase=1");
    if (io.rfind("rt:", 0) == 0)
    {
        GlobalValue::Bind("SimulatorImplementationType",
                          StringValue("ns3::RealtimeSimulatorImpl"));
        Config::SetDefault("ns3::RealtimeSimulatorImpl::SynchronizationMode",
                           StringValue("BestEffort"));
    }
    int listenFd = -1;
    if (io.rfind("tcp:", 0) == 0 || io.rfind("rt:", 0) == 0)
    {
        // listen before the (slow) scenario setup so that clients can connect right away
        listenFd = OpenListen(std::stoi(io.substr(io.find(':') + 1)), bindAll);
    }

    RngSeedManager::SetSeed(1);
    RngSeedManager::SetRun(run);

    // ---- RLC / PDCP -------------------------------------------------------------------------
    Config::SetDefault("ns3::NrRlcUm::MaxTxBufferSize", UintegerValue(100000000));
    Config::SetDefault("ns3::NrRlcAm::MaxTxBufferSize", UintegerValue(100000000));
    if (rlc == "UM")
    {
        Config::SetDefault("ns3::NrRlcUm::EnablePdcpDiscarding", BooleanValue(true));
        Config::SetDefault("ns3::NrRlcUm::DiscardTimerMs",
                           UintegerValue(static_cast<uint32_t>(deadline * 1000)));
        Config::SetDefault("ns3::NrGnbRrc::QosFlowToRlcMapping",
                           EnumValue(NrGnbRrc::RLC_UM_ALWAYS));
    }
    else
    {
        Config::SetDefault("ns3::NrGnbRrc::QosFlowToRlcMapping",
                           EnumValue(NrGnbRrc::RLC_AM_ALWAYS));
    }

    // ---- Nodes and positions ----------------------------------------------------------------
    NodeContainer gnbNodes;
    gnbNodes.Create(1);
    NodeContainer ueNodes;
    ueNodes.Create(nUe);

    MobilityHelper mob;
    mob.SetMobilityModel("ns3::ConstantPositionMobilityModel");
    mob.Install(gnbNodes);
    if (g_bridge && mobility == "waypoint")
    {
        MobilityHelper wmob;
        wmob.SetMobilityModel("ns3::WaypointMobilityModel");
        wmob.Install(ueNodes);
    }
    else
    {
        mob.Install(ueNodes);
    }
    gnbNodes.Get(0)->GetObject<MobilityModel>()->SetPosition(Vector(0, 0, 10));

    std::vector<double> dvec;
    if (placement == "dists")
    {
        std::stringstream ss(dists);
        std::string tok;
        while (std::getline(ss, tok, ','))
        {
            dvec.push_back(std::stod(tok));
        }
        NS_ABORT_MSG_IF(dvec.empty(), "placement=dists needs --dists");
    }
    Ptr<UniformRandomVariable> posRv = CreateObject<UniformRandomVariable>();
    posRv->SetStream(1000);
    g_posOut.open(outDir + "/ues.csv");
    Ptr<NormalRandomVariable> shRv = CreateObject<NormalRandomVariable>();
    shRv->SetStream(1001);
    const double nRbg = 5.0; // 50 PRBs / 10
    const double subbandHzDrop = rbgSize * 12 * 30e3;
    g_posOut << "ue,nodeId,x,y,d2d,shadow_db,snr1_db,snr_bw_db,redraws\n";
    for (uint32_t i = 0; i < nUe; ++i)
    {
        double x = 0;
        double y = 0;
        double sh = 0;
        double snr1 = 0;
        uint32_t tries = 0;
        do
        {
            if (placement == "dists")
            {
                double d = dvec[i % dvec.size()];
                double a = posRv->GetValue(0, M_PI / 2); // inside the first quadrant
                x = d * std::cos(a);
                y = d * std::sin(a);
            }
            else
            {
                x = posRv->GetValue(0, side);
                y = posRv->GetValue(0, side);
            }
            sh = shadowStd * shRv->GetValue();
            double d2 = std::max(1.0, std::hypot(x, y));
            // single-subband, full-power SNR (NetSlot's snr_db convention)
            snr1 = ueTxPower - (40.0 + 35.0 * std::log10(d2)) - sh - niPerSubbandDbm;
            tries++;
        } while (snr1 - 10 * std::log10(nRbg) < minSnrDb && tries < 10000);
        (void)subbandHzDrop;
        g_ueShadowDb[ueNodes.Get(i)->GetId()] = sh;
        ueNodes.Get(i)->GetObject<MobilityModel>()->SetPosition(Vector(x, y, 1.5));
        g_posOut << i << "," << ueNodes.Get(i)->GetId() << "," << x << "," << y << ","
                 << std::hypot(x, y) << "," << sh << "," << snr1 << ","
                 << snr1 - 10 * std::log10(nRbg) << "," << tries - 1 << "\n";
    }
    g_posOut.close();
    if (!initPos.empty())
    {
        // Bridge: initial UE positions and loss overrides "x:y:loss,x:y:loss,..." (loss < 0: none)
        std::stringstream ss(initPos);
        std::string tok;
        uint32_t i = 0;
        while (std::getline(ss, tok, ',') && i < nUe)
        {
            double x = 0;
            double y = 0;
            double l = -1;
            NS_ABORT_MSG_IF(std::sscanf(tok.c_str(), "%lf:%lf:%lf", &x, &y, &l) != 3, "bad --init");
            ueNodes.Get(i)->GetObject<MobilityModel>()->SetPosition(Vector(x, y, 1.5));
            if (l >= 0)
            {
                g_lossOverride[ueNodes.Get(i)->GetId()] = l;
            }
            i++;
        }
        NS_ABORT_MSG_IF(i != nUe, "--init needs one entry per UE");
    }

    // ---- NR helpers -------------------------------------------------------------------------
    Ptr<NrPointToPointEpcHelper> epc = CreateObject<NrPointToPointEpcHelper>();
    Ptr<IdealBeamformingHelper> bf = CreateObject<IdealBeamformingHelper>();
    Ptr<NrHelper> nr = CreateObject<NrHelper>();
    nr->SetBeamformingHelper(bf);
    nr->SetEpcHelper(epc);
    epc->SetAttribute("S1uLinkDelay", TimeValue(MilliSeconds(0)));
    nr->SetAttribute("RbOverhead", DoubleValue(rbOverhead));
    nr->SetAttribute("NumRbPerRbg", UintegerValue(rbgSize));

    CcBwpCreator ccBwp;
    CcBwpCreator::SimpleOperationBandConf bandConf(freq, bw, 1);
    OperationBandInfo band = ccBwp.CreateOperationBandContiguousCc(bandConf);

    Ptr<NrChannelHelper> ch = CreateObject<NrChannelHelper>();
    ch->ConfigureFactories("UMi", "NLOS", "ThreeGpp");
    ch->ConfigurePropagationFactory(LogDistShadowLoss::GetTypeId());
    ch->SetPathlossAttribute("ShadowingStd", DoubleValue(shadowStd));
    if (fading)
    {
        Config::SetDefault("ns3::ThreeGppChannelModel::vScatt", DoubleValue(vScatt));
        ch->AssignChannelsToBands({band});
    }
    else
    {
        ch->AssignChannelsToBands({band}, NrChannelHelper::INIT_PROPAGATION);
    }
    BandwidthPartInfoPtrVector allBwps = CcBwpCreator::GetAllBwps({band});

    bf->SetAttribute("BeamformingMethod", TypeIdValue(DirectPathBeamforming::GetTypeId()));
    nr->SetUeAntennaAttribute("NumRows", UintegerValue(1));
    nr->SetUeAntennaAttribute("NumColumns", UintegerValue(1));
    nr->SetUeAntennaAttribute("AntennaElement", PointerValue(CreateObject<IsotropicAntennaModel>()));
    nr->SetGnbAntennaAttribute("NumRows", UintegerValue(1));
    nr->SetGnbAntennaAttribute("NumColumns", UintegerValue(1));
    nr->SetGnbAntennaAttribute("AntennaElement",
                               PointerValue(CreateObject<IsotropicAntennaModel>()));

    // Scheduler, AMC and error model
    nr->SetSchedulerTypeId(TypeId::LookupByName(sched));
    nr->SetSchedulerAttribute("EnableHarqReTx", BooleanValue(harq));
    nr->SetSchedulerAttribute("EnableSrsInUlSlots", BooleanValue(srs));
    nr->SetSchedulerAttribute("EnableSrsInFSlots", BooleanValue(srs));
    nr->SetUlErrorModel(errorModel);
    nr->SetDlErrorModel(errorModel);
    nr->SetGnbUlAmcAttribute("AmcModel", EnumValue(NrAmc::ErrorModel));
    nr->SetGnbDlAmcAttribute("AmcModel", EnumValue(NrAmc::ErrorModel));

    // PHY: TDD DDDSU at numerology 1; noise figure absorbs the -90 dBm/subband N+I floor
    double subbandHz = rbgSize * 12 * 30e3;
    double nf = niPerSubbandDbm - (-174.0 + 10 * std::log10(subbandHz));
    nr->SetGnbPhyAttribute("Pattern", StringValue("DL|DL|DL|S|UL|DL|DL|DL|S|UL|"));
    nr->SetGnbPhyAttribute("Numerology", UintegerValue(1));
    nr->SetGnbPhyAttribute("TxPower", DoubleValue(gnbTxPower));
    nr->SetGnbPhyAttribute("NoiseFigure", DoubleValue(nf));
    nr->SetUePhyAttribute("TxPower", DoubleValue(ueTxPower));
    nr->SetUePhyAttribute("EnableUplinkPowerControl", BooleanValue(false));
    nr->SetUePhyAttribute("PowerAllocationType", StringValue(ulPowerAlloc));

    NetDeviceContainer gnbDev = nr->InstallGnbDevice(gnbNodes, allBwps);
    NetDeviceContainer ueDev = nr->InstallUeDevice(ueNodes, allBwps);
    if (ueUeFilter)
    {
        NrHelper::GetGnbPhy(gnbDev.Get(0), 0)
            ->GetSpectrumPhy()
            ->GetSpectrumChannel()
            ->AddSpectrumTransmitFilter(CreateObject<UeUeFilter>());
    }
    int64_t stream = 1;
    stream += nr->AssignStreams(gnbDev, stream);
    stream += nr->AssignStreams(ueDev, stream);

    // ---- IP / core --------------------------------------------------------------------------
    auto [remoteHost, remoteAddr] = epc->SetupRemoteHost("100Gb/s", 2500, Seconds(0.0));
    InternetStackHelper internet;
    internet.Install(ueNodes);
    Ipv4InterfaceContainer ueIf = epc->AssignUeIpv4Address(ueDev);
    Ipv4StaticRoutingHelper srh;
    for (uint32_t i = 0; i < nUe; ++i)
    {
        srh.GetStaticRouting(ueNodes.Get(i)->GetObject<Ipv4>())
            ->SetDefaultRoute(epc->GetUeDefaultGatewayAddress(), 1);
    }
    for (uint32_t i = 0; i < nUe; ++i)
    {
        nr->AttachToGnb(ueDev.Get(i), gnbDev.Get(0));
    }

    // ---- Applications -----------------------------------------------------------------------
    const uint16_t port = 9000;
    Ptr<FrameSink> sink = CreateObject<FrameSink>();
    sink->Setup(port);
    remoteHost->AddApplication(sink);
    sink->SetStartTime(Seconds(0.0));

    uint32_t nTicks = static_cast<uint32_t>(std::floor(trafficTime * 1000.0 / periodMs + 1e-9));
    Ptr<UniformRandomVariable> phaseRv = CreateObject<UniformRandomVariable>();
    phaseRv->SetStream(2000);
    std::vector<Ptr<FrameSender>> apps;
    for (uint32_t i = 0; i < nUe; ++i)
    {
        Ptr<FrameSender> app = CreateObject<FrameSender>();
        double off = syncPhase ? 0.0 : phaseRv->GetValue(0, periodMs / 1000.0);
        app->Setup(InetSocketAddress(remoteAddr, port),
                   static_cast<uint16_t>(i),
                   frameBytes,
                   p,
                   MilliSeconds(periodMs),
                   Seconds(appStart + off),
                   nTicks,
                   payload,
                   3000 + i);
        ueNodes.Get(i)->AddApplication(app);
        app->SetStartTime(Seconds(0.1));
        if (g_bridge)
        {
            app->SetExternal();
        }
        apps.push_back(app);
    }

    if (pktLog)
    {
        g_pktOut.open(outDir + "/packets.csv");
        g_pktOut << "ue,fid,idx,gen,rx\n";
    }
    if (macTraces)
    {
        nr->EnableUlPhyTraces();
        nr->EnableUlMacSchedTraces();
        nr->EnableGnbMacCtrlMsgsTraces();
        nr->EnableUeMacCtrlMsgsTraces();
        nr->EnableRlcSimpleTraces();
    }

    if (macTraces)
    {
        g_slotOut.open(outDir + "/gnb_slots.csv");
        g_slotOut << "t,frame,subframe,slot,nUe,usedReg,usedSym,availRb,availSym\n";
        NrHelper::GetGnbPhy(gnbDev.Get(0), 0)
            ->TraceConnectWithoutContext("SlotDataStats", MakeCallback(&SlotDataStats));
        g_bufOut.open(outDir + "/ue_mac_buf.csv");
        g_bufOut << "t,nodeId,rnti,srState,bufBytes,retx,func\n";
        for (uint32_t i = 0; i < nUe; ++i)
        {
            NrHelper::GetUeMac(ueDev.Get(i), 0)
                ->TraceConnectWithoutContext("UeMacStateMachineTrace", MakeCallback(&UeMacState));
        }
    }

    if (io.rfind("rt:", 0) == 0)
    {
        return RunRealtime(ueNodes, apps, listenFd);
    }
    if (g_bridge)
    {
        return RunBridge(io, bindAll, mobility, flowmon, ueNodes, remoteHost, apps, appStart,
                         MilliSeconds(periodMs), listenFd);
    }

    FlowMonitorHelper fmh;
    NodeContainer ends;
    ends.Add(remoteHost);
    ends.Add(ueNodes);
    Ptr<FlowMonitor> fm = fmh.Install(ends);

    double stopTime = appStart + trafficTime + deadline + 0.2;
    Simulator::Stop(Seconds(stopTime));
    auto t0 = std::chrono::steady_clock::now();
    Simulator::Run();
    double wall = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();

    // ---- Outputs ----------------------------------------------------------------------------
    std::ofstream fo(outDir + "/frames.csv");
    fo << "ue,fid,gen,bytes,npk,rxpk,rxbytes,last,delay,ok\n";
    uint64_t nOk = 0;
    uint64_t nAll = 0;
    for (const auto& [k, r] : g_frames)
    {
        bool complete = (r.rxpk >= r.npk);
        double delay = complete ? r.last - r.gen : -1.0;
        bool ok = complete && delay <= deadline;
        nAll++;
        nOk += ok;
        fo << r.ue << "," << r.fid << "," << r.gen << "," << r.bytes << "," << r.npk << ","
           << r.rxpk << "," << r.rxbytes << "," << r.last << "," << delay << "," << ok << "\n";
    }
    fo.close();
    for (auto* f : {&g_pktOut, &g_slotOut, &g_bufOut})
    {
        if (f->is_open())
        {
            f->close();
        }
    }
    fm->CheckForLostPackets();
    fm->SerializeToXmlFile(outDir + "/flowmon.xml", false, false);

    std::ofstream mo(outDir + "/meta.txt");
    mo << "appStart " << appStart << "\nnUe " << nUe << "\nframeBytes " << frameBytes << "\np " << p << "\ntrafficTime "
       << trafficTime << "\nsimTime " << stopTime << "\nwallSeconds " << wall
       << "\nwallPerSimSecond " << wall / stopTime << "\nnoiseFigureDb " << nf << "\nframes "
       << nAll << "\nframesOk " << nOk << "\nrun " << run << "\nfading " << fading
       << "\nerrorModel " << errorModel << "\nsched " << sched << "\nrlc " << rlc << "\nulPowerAlloc " << ulPowerAlloc << "\nminSnrDb " << minSnrDb << "\nside " << side
       << "\nniPerSubbandDbm " << niPerSubbandDbm << "\n";
    mo.close();
    std::cout << "frames " << nAll << " ok " << nOk << " wall " << wall << " s for " << stopTime
              << " sim s (" << wall / stopTime << " wall/sim)" << std::endl;

    Simulator::Destroy();
    return 0;
}
