// netslot-bridge: lockstep co-simulation server around the netslot-ref 5G-LENA scenario.
//
// The radio/MAC/RLC/core configuration is copied verbatim from scratch/netslot-ref (ns3ref), so a
// bridge run with the same CLI, the same seed and the same frame schedule reproduces the standalone
// run event for event (see tests/test_correctness.py).  What changes:
//   * nEnv independent cells in one process (one gNB + nUe UEs each, own operation band and own
//     SpectrumChannel, cells offset by 20 km so nothing couples them), default nEnv = 1;
//   * FrameSender does not draw frames: it sends, at every 100 ms tick, the frames the client queued
//     for it (frame id and size chosen by the client);
//   * the simulator is advanced one control step per STEP message: Run() stops 1 ns before the next
//     tick, so the tick of step t+1 fires only after the client has provided the frames of step t+1;
//   * per-step reporting: frames completed at the sink (last packet received), per-UE UL SINR / MCS /
//     HARQ counters from RxPacketTraceGnb, DL RSRP from ReportRsrp, RLC buffer bytes from the UE MAC
//     state-machine trace;
//   * RESET rebuilds the whole scenario in-process (Simulator::Destroy + rebuild) with a new run number.
//
// Transports (same byte framing, see README "Message schema"):
//   --bridge=tcp:PORT      TCP server on 0.0.0.0:PORT (Windows <-> WSL via localhost forwarding)
//   --bridge=unix:PATH     Unix domain socket server (WSL only)
//   --bridge=shm:NAME      ns3-ai msg-interface (Boost shared memory, Python is the segment creator)
//   --bridge=none          behave like netslot-ref (no client), for debugging only

#include "ns3/antenna-module.h"
#include "ns3/applications-module.h"
#include "ns3/core-module.h"
#include "ns3/flow-monitor-module.h"
#include "ns3/internet-module.h"
#include "ns3/ipv4-address-generator.h"
#include "ns3/mobility-module.h"
#include "ns3/network-module.h"
#include "ns3/nr-module.h"
#include "ns3/point-to-point-module.h"
#include "ns3/propagation-loss-model.h"

#include <arpa/inet.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>

#include <chrono>
#include <cmath>
#include <cstring>
#include <fstream>
#include <map>
#include <memory>
#include <set>
#include <sstream>
#include <tuple>
#include <vector>

#ifdef WITH_NS3AI
#include "ns3-ai-msg-interface.h"
#endif

using namespace ns3;

NS_LOG_COMPONENT_DEFINE("NetSlotBridge");

// ---------------------------------------------------------------------------------------------
// Wire protocol
// ---------------------------------------------------------------------------------------------
static const uint32_t MAGIC = 0x3142534E; // "NSB1" little endian
enum MsgType : uint32_t
{
    MSG_HELLO = 1,
    MSG_STEP = 2,
    MSG_RESULT = 3,
    MSG_RESET = 4,
    MSG_CLOSE = 5,
    MSG_ERROR = 6,
};
static const uint32_t PROTO_VERSION = 1;
static const uint32_t STEP_HAS_SHADOW = 1;
static const uint32_t STEP_INTERP_POS = 2; // positions are END-of-step poses, move linearly during step
static const uint32_t RESET_HAS_POS = 1;
static const uint32_t RESET_HAS_SHADOW = 2;

#pragma pack(push, 1)
struct WireFrameIn
{
    uint16_t env;
    uint16_t ue;
    uint32_t fid;
    uint32_t bytes;
};

struct WireFrameDone
{
    uint16_t env;
    uint16_t ue;
    uint32_t fid;
    double t;
};
#pragma pack(pop)

class Transport
{
  public:
    virtual ~Transport() = default;
    virtual void Send(uint32_t type, const std::string& payload) = 0;
    virtual bool Recv(uint32_t& type, std::string& payload) = 0;
};

class StreamTransport : public Transport
{
  public:
    // "tcp:PORT" or "unix:PATH"
    explicit StreamTransport(const std::string& spec)
    {
        int ls = -1;
        if (spec.rfind("tcp:", 0) == 0)
        {
            int port = std::stoi(spec.substr(4));
            ls = socket(AF_INET, SOCK_STREAM, 0);
            int one = 1;
            setsockopt(ls, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));
            sockaddr_in a{};
            a.sin_family = AF_INET;
            a.sin_addr.s_addr = htonl(INADDR_ANY);
            a.sin_port = htons(port);
            NS_ABORT_MSG_IF(bind(ls, (sockaddr*)&a, sizeof(a)) != 0, "bind tcp " << port);
        }
        else
        {
            std::string path = spec.substr(5);
            unlink(path.c_str());
            ls = socket(AF_UNIX, SOCK_STREAM, 0);
            sockaddr_un a{};
            a.sun_family = AF_UNIX;
            std::strncpy(a.sun_path, path.c_str(), sizeof(a.sun_path) - 1);
            NS_ABORT_MSG_IF(bind(ls, (sockaddr*)&a, sizeof(a)) != 0, "bind unix " << path);
            m_unlink = path;
        }
        NS_ABORT_MSG_IF(listen(ls, 1) != 0, "listen");
        std::cout << "LISTENING " << spec << std::endl;
        m_fd = accept(ls, nullptr, nullptr);
        NS_ABORT_MSG_IF(m_fd < 0, "accept");
        close(ls);
        if (spec.rfind("tcp:", 0) == 0)
        {
            int one = 1;
            setsockopt(m_fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));
        }
    }

    ~StreamTransport() override
    {
        if (m_fd >= 0)
        {
            close(m_fd);
        }
        if (!m_unlink.empty())
        {
            unlink(m_unlink.c_str());
        }
    }

    void Send(uint32_t type, const std::string& payload) override
    {
        uint32_t hdr[3] = {MAGIC, type, static_cast<uint32_t>(payload.size())};
        std::string buf(reinterpret_cast<char*>(hdr), sizeof(hdr));
        buf += payload;
        WriteAll(buf.data(), buf.size());
    }

    bool Recv(uint32_t& type, std::string& payload) override
    {
        uint32_t hdr[3];
        if (!ReadAll(reinterpret_cast<char*>(hdr), sizeof(hdr)))
        {
            return false;
        }
        NS_ABORT_MSG_IF(hdr[0] != MAGIC, "bad magic");
        type = hdr[1];
        payload.resize(hdr[2]);
        return hdr[2] == 0 || ReadAll(payload.data(), hdr[2]);
    }

  private:
    void WriteAll(const char* p, size_t n)
    {
        while (n > 0)
        {
            ssize_t k = write(m_fd, p, n);
            NS_ABORT_MSG_IF(k <= 0, "socket write failed");
            p += k;
            n -= k;
        }
    }

    bool ReadAll(char* p, size_t n)
    {
        while (n > 0)
        {
            ssize_t k = read(m_fd, p, n);
            if (k <= 0)
            {
                return false;
            }
            p += k;
            n -= k;
        }
        return true;
    }

    int m_fd{-1};
    std::string m_unlink;
};

#ifdef WITH_NS3AI
// One fixed-size byte buffer per direction; carries exactly the same framed bytes as the sockets.
static const uint32_t SHM_CAP = 1u << 20;

struct ShmMsg
{
    uint32_t type;
    uint32_t len;
    uint8_t data[SHM_CAP];
};

class ShmAiTransport : public Transport
{
  public:
    explicit ShmAiTransport(const std::string& name)
        : m_seg(name),
          m_c2p(name + "_c2p"),
          m_p2c(name + "_p2c"),
          m_lock(name + "_lock")
    {
        // Python created the segment (ns3-ai convention); the C++ side opens it.
        m_if = std::make_unique<Ns3AiMsgInterfaceImpl<ShmMsg, ShmMsg>>(false,
                                                                      false,
                                                                      true,
                                                                      4096,
                                                                      m_seg.c_str(),
                                                                      m_c2p.c_str(),
                                                                      m_p2c.c_str(),
                                                                      m_lock.c_str());
        std::cout << "SHM_OPEN " << name << std::endl;
    }

    void Send(uint32_t type, const std::string& payload) override
    {
        NS_ABORT_MSG_IF(payload.size() > SHM_CAP, "payload too large for shm");
        m_if->CppSendBegin();
        ShmMsg* m = m_if->GetCpp2PyStruct();
        m->type = type;
        m->len = payload.size();
        std::memcpy(m->data, payload.data(), payload.size());
        m_if->CppSendEnd();
    }

    bool Recv(uint32_t& type, std::string& payload) override
    {
        m_if->CppRecvBegin();
        ShmMsg* m = m_if->GetPy2CppStruct();
        type = m->type;
        payload.assign(reinterpret_cast<char*>(m->data), m->len);
        m_if->CppRecvEnd();
        return true;
    }

  private:
    std::string m_seg, m_c2p, m_p2c, m_lock;
    std::unique_ptr<Ns3AiMsgInterfaceImpl<ShmMsg, ShmMsg>> m_if;
};
#endif

// ---------------------------------------------------------------------------------------------
// Scenario state (rebuilt on RESET)
// ---------------------------------------------------------------------------------------------
static std::map<uint32_t, double> g_ueShadowDb; // node id -> shadowing (dB)
// Last commanded end-of-step pose per UE (global UE index), the start of the next STEP_INTERP_POS interpolation.
// The scheduled waypoints stop at 7/8 of the way, so the mobility model's own position lags the command.
static std::vector<Vector> g_lastEnd;
static std::vector<bool> g_hasLastEnd;

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
        os << "ue=" << ue << " fid=" << fid << " idx=" << idx << "/" << npk;
    }
};

struct FrameRec
{
    uint16_t ue;
    uint32_t fid;
    double gen;
    uint32_t bytes;
    uint16_t npk;
    uint16_t rxpk{0};
    double last{-1};
};

static std::map<std::pair<uint16_t, uint32_t>, FrameRec> g_frames;
static std::vector<WireFrameDone> g_done;
static uint32_t g_nUePerEnv = 1;

class FrameSender : public Application
{
  public:
    static TypeId GetTypeId()
    {
        static TypeId tid =
            TypeId("ns3::BridgeFrameSender").SetParent<Application>().AddConstructor<FrameSender>();
        return tid;
    }

    void Setup(Address remote, uint16_t ue, Time period, Time firstTick, uint32_t payload, int64_t stream)
    {
        m_remote = remote;
        m_ue = ue;
        m_period = period;
        m_first = firstTick;
        m_payload = payload;
        // Created only to keep the automatic RNG stream counter aligned with netslot-ref.
        m_u = CreateObject<UniformRandomVariable>();
        m_u->SetStream(stream);
    }

    void Queue(uint32_t fid, uint32_t bytes)
    {
        m_pending.emplace_back(fid, bytes);
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

    void Tick()
    {
        for (auto [fid, bytes] : m_pending)
        {
            SendFrame(fid, bytes);
        }
        m_pending.clear();
        Simulator::Schedule(m_period, &FrameSender::Tick, this);
    }

    void SendFrame(uint32_t fid, uint32_t bytes)
    {
        uint16_t n = static_cast<uint16_t>((bytes + m_payload - 1) / m_payload);
        uint64_t now = Simulator::Now().GetNanoSeconds();
        g_frames[{m_ue, fid}] = FrameRec{m_ue, fid, Simulator::Now().GetSeconds(), bytes, n};
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
    Time m_period{MilliSeconds(100)};
    Time m_first;
    uint32_t m_payload{1400};
    Ptr<UniformRandomVariable> m_u;
    std::vector<std::pair<uint32_t, uint32_t>> m_pending;
};

class FrameSink : public Application
{
  public:
    static TypeId GetTypeId()
    {
        static TypeId tid =
            TypeId("ns3::BridgeFrameSink").SetParent<Application>().AddConstructor<FrameSink>();
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
            if (it == g_frames.end())
            {
                continue;
            }
            FrameRec& r = it->second;
            r.rxpk++;
            r.last = now;
            if (r.rxpk == r.npk)
            {
                g_done.push_back(WireFrameDone{static_cast<uint16_t>(h.ue / g_nUePerEnv),
                                               static_cast<uint16_t>(h.ue % g_nUePerEnv),
                                               h.fid,
                                               now});
                g_frames.erase(it);
            }
        }
    }

    Ptr<Socket> m_sock;
    uint16_t m_port{9000};
};

// Per-UE per-step accumulators (global UE index = env * nUe + ue).
struct UeStepStats
{
    double sinrLinSum{0};
    double mcsSum{0};
    uint32_t nTb{0};
    uint32_t nRetx{0};
    uint32_t nCorrupt{0};
    uint32_t nLost{0};
    uint64_t okBytes{0};
    float rsrp{NAN};
    uint32_t bufBytes{0};
};

static std::vector<UeStepStats> g_st;
static std::map<std::pair<uint64_t, uint16_t>, uint32_t> g_cellRntiToUe;
static std::map<uint64_t, uint32_t> g_imsiToUe;
static std::map<uint32_t, uint32_t> g_nodeToUe;
static std::set<std::tuple<uint64_t, uint16_t, uint32_t, uint8_t, uint16_t, uint8_t, uint8_t>> g_seenTb;

static void
RxTbGnb(RxPacketTraceParams p)
{
    auto it = g_cellRntiToUe.find({p.m_cellId, p.m_rnti});
    if (it == g_cellRntiToUe.end())
    {
        return;
    }
    // LENA fires some UL rows twice (see ns3ref report): de-duplicate on the TB identity.
    auto key = std::make_tuple(p.m_cellId, p.m_rnti, p.m_frameNum, p.m_subframeNum, p.m_slotNum,
                               p.m_symStart, p.m_rv);
    if (!g_seenTb.insert(key).second)
    {
        return;
    }
    UeStepStats& s = g_st[it->second];
    s.nTb++;
    s.sinrLinSum += p.m_sinr;
    s.mcsSum += p.m_mcs;
    s.nRetx += (p.m_rv > 0);
    if (p.m_corrupt)
    {
        s.nCorrupt++;
        s.nLost += (p.m_rv >= 3);
    }
    else
    {
        s.okBytes += p.m_tbSize;
    }
}

static void
UeRsrp(uint16_t cellId, uint16_t imsi, uint16_t rnti, double rsrp, uint8_t bwp)
{
    auto it = g_imsiToUe.find(imsi);
    if (it != g_imsiToUe.end())
    {
        g_st[it->second].rsrp = static_cast<float>(rsrp);
    }
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
    auto it = g_nodeToUe.find(nodeId);
    if (it == g_nodeToUe.end())
    {
        return;
    }
    uint32_t bytes = 0;
    for (const auto& [lcid, b] : bsr)
    {
        bytes += b.txQueueSize + b.retxQueueSize + b.statusPduSize;
    }
    g_st[it->second].bufBytes = bytes;
}

// ---------------------------------------------------------------------------------------------
// Configuration and world
// ---------------------------------------------------------------------------------------------
struct Cfg
{
    uint32_t nEnv = 1;
    uint32_t nUe = 8;
    double appStart = 0.5;
    double deadline = 2.0;
    double periodMs = 100.0;
    uint32_t payload = 1400;
    std::string placement = "random";
    std::string dists = "";
    double side = 150.0;
    double freq = 3.5e9;
    double bw = 20e6;
    double rbOverhead = 0.1;
    uint32_t rbgSize = 10;
    double ueTxPower = 23.0;
    double gnbTxPower = 30.0;
    double niPerSubbandDbm = -90.0;
    double shadowStd = 6.0;
    double minSnrDb = -1000.0;
    bool fading = true;
    double vScatt = 3.0;
    double chanUpdateMs = 0.0;
    std::string errorModel = "ns3::NrEesmIrT1";
    std::string sched = "ns3::NrMacSchedulerOfdmaPF";
    std::string ulPowerAlloc = "UniformPowerAllocUsed";
    std::string rlc = "UM";
    bool harq = true;
    bool srs = false;
    bool flowmon = false;
    uint32_t run = 1;
    double envSpacing = 20000.0;
    double ueHeight = 1.5;
    std::string bridge = "tcp:5555";
    std::string framesOut = "";
};

struct World
{
    NodeContainer gnbNodes;
    NodeContainer ueNodes;
    NetDeviceContainer ueDev;
    std::vector<Ptr<FrameSender>> senders;
    Ptr<FlowMonitor> fm;
    FlowMonitorHelper* fmh{nullptr};
    // Helpers must outlive the run (netslot-ref keeps them alive in main()).
    Ptr<NrPointToPointEpcHelper> epc;
    Ptr<IdealBeamformingHelper> bf;
    Ptr<NrHelper> nr;
    std::vector<Ptr<NrChannelHelper>> ch;
};

static Cfg g_cfg;
static World g_w;

static void
BuildWorld(const Cfg& c, uint32_t run)
{
    RngSeedManager::SetSeed(1);
    RngSeedManager::SetRun(run);
    // Automatic stream indices are process-global: restart them so a rebuilt world with the same
    // run number draws exactly the same random numbers as a fresh process.
    RngSeedManager::ResetNextStreamIndex();
    g_ueShadowDb.clear();
    g_frames.clear();
    g_done.clear();
    g_nUePerEnv = c.nUe;
    uint32_t nTot = c.nEnv * c.nUe;
    g_st.assign(nTot, UeStepStats{});

    Config::SetDefault("ns3::NrRlcUm::MaxTxBufferSize", UintegerValue(100000000));
    Config::SetDefault("ns3::NrRlcAm::MaxTxBufferSize", UintegerValue(100000000));
    if (c.rlc == "UM")
    {
        Config::SetDefault("ns3::NrRlcUm::EnablePdcpDiscarding", BooleanValue(true));
        Config::SetDefault("ns3::NrRlcUm::DiscardTimerMs",
                           UintegerValue(static_cast<uint32_t>(c.deadline * 1000)));
        Config::SetDefault("ns3::NrGnbRrc::QosFlowToRlcMapping", EnumValue(NrGnbRrc::RLC_UM_ALWAYS));
    }
    else
    {
        Config::SetDefault("ns3::NrGnbRrc::QosFlowToRlcMapping", EnumValue(NrGnbRrc::RLC_AM_ALWAYS));
    }

    World& w = g_w;
    w = World{};
    w.gnbNodes.Create(c.nEnv);
    w.ueNodes.Create(nTot);
    MobilityHelper mob;
    mob.SetMobilityModel("ns3::ConstantPositionMobilityModel");
    mob.Install(w.gnbNodes);
    mob.Install(w.ueNodes);
    for (uint32_t e = 0; e < c.nEnv; ++e)
    {
        w.gnbNodes.Get(e)->GetObject<MobilityModel>()->SetPosition(Vector(e * c.envSpacing, 0, 10));
    }

    // Initial drop: identical draws to netslot-ref (so nEnv=1 reproduces its geometry).
    std::vector<double> dvec;
    if (c.placement == "dists")
    {
        std::stringstream ss(c.dists);
        std::string tok;
        while (std::getline(ss, tok, ','))
        {
            dvec.push_back(std::stod(tok));
        }
    }
    Ptr<UniformRandomVariable> posRv = CreateObject<UniformRandomVariable>();
    posRv->SetStream(1000);
    Ptr<NormalRandomVariable> shRv = CreateObject<NormalRandomVariable>();
    shRv->SetStream(1001);
    const double nRbg = 5.0;
    for (uint32_t i = 0; i < nTot; ++i)
    {
        double x = 0, y = 0, sh = 0, snr1 = 0;
        uint32_t tries = 0;
        do
        {
            if (c.placement == "dists")
            {
                double d = dvec[(i % c.nUe) % dvec.size()];
                double a = posRv->GetValue(0, M_PI / 2);
                x = d * std::cos(a);
                y = d * std::sin(a);
            }
            else
            {
                x = posRv->GetValue(0, c.side);
                y = posRv->GetValue(0, c.side);
            }
            sh = c.shadowStd * shRv->GetValue();
            double d2 = std::max(1.0, std::hypot(x, y));
            snr1 = c.ueTxPower - (40.0 + 35.0 * std::log10(d2)) - sh - c.niPerSubbandDbm;
            tries++;
        } while (snr1 - 10 * std::log10(nRbg) < c.minSnrDb && tries < 10000);
        g_ueShadowDb[w.ueNodes.Get(i)->GetId()] = sh;
        uint32_t e = i / c.nUe;
        w.ueNodes.Get(i)->GetObject<MobilityModel>()->SetPosition(
            Vector(e * c.envSpacing + x, y, c.ueHeight));
    }

    Ptr<NrPointToPointEpcHelper> epc = CreateObject<NrPointToPointEpcHelper>();
    Ptr<IdealBeamformingHelper> bf = CreateObject<IdealBeamformingHelper>();
    Ptr<NrHelper> nr = CreateObject<NrHelper>();
    w.epc = epc;
    w.bf = bf;
    w.nr = nr;
    nr->SetBeamformingHelper(bf);
    nr->SetEpcHelper(epc);
    epc->SetAttribute("S1uLinkDelay", TimeValue(MilliSeconds(0)));
    nr->SetAttribute("RbOverhead", DoubleValue(c.rbOverhead));
    nr->SetAttribute("NumRbPerRbg", UintegerValue(c.rbgSize));

    if (c.fading)
    {
        Config::SetDefault("ns3::ThreeGppChannelModel::vScatt", DoubleValue(c.vScatt));
        if (c.chanUpdateMs > 0)
        {
            Config::SetDefault("ns3::ThreeGppChannelModel::UpdatePeriod",
                               TimeValue(MilliSeconds(c.chanUpdateMs)));
        }
    }
    // One operation band (own CcBwpCreator -> BWP id 0, own SpectrumChannel) per env.
    std::vector<OperationBandInfo> bands(c.nEnv);
    std::vector<BandwidthPartInfoPtrVector> bwps(c.nEnv);
    for (uint32_t e = 0; e < c.nEnv; ++e)
    {
        CcBwpCreator ccBwp;
        CcBwpCreator::SimpleOperationBandConf bandConf(c.freq, c.bw, 1);
        bands[e] = ccBwp.CreateOperationBandContiguousCc(bandConf);
        Ptr<NrChannelHelper> ch = CreateObject<NrChannelHelper>();
        ch->ConfigureFactories("UMi", "NLOS", "ThreeGpp");
        ch->ConfigurePropagationFactory(LogDistShadowLoss::GetTypeId());
        ch->SetPathlossAttribute("ShadowingStd", DoubleValue(c.shadowStd));
        w.ch.push_back(ch);
        if (c.fading)
        {
            ch->AssignChannelsToBands({bands[e]});
        }
        else
        {
            ch->AssignChannelsToBands({bands[e]}, NrChannelHelper::INIT_PROPAGATION);
        }
        bwps[e] = CcBwpCreator::GetAllBwps({bands[e]});
    }

    bf->SetAttribute("BeamformingMethod", TypeIdValue(DirectPathBeamforming::GetTypeId()));
    nr->SetUeAntennaAttribute("NumRows", UintegerValue(1));
    nr->SetUeAntennaAttribute("NumColumns", UintegerValue(1));
    nr->SetUeAntennaAttribute("AntennaElement", PointerValue(CreateObject<IsotropicAntennaModel>()));
    nr->SetGnbAntennaAttribute("NumRows", UintegerValue(1));
    nr->SetGnbAntennaAttribute("NumColumns", UintegerValue(1));
    nr->SetGnbAntennaAttribute("AntennaElement", PointerValue(CreateObject<IsotropicAntennaModel>()));

    nr->SetSchedulerTypeId(TypeId::LookupByName(c.sched));
    nr->SetSchedulerAttribute("EnableHarqReTx", BooleanValue(c.harq));
    nr->SetSchedulerAttribute("EnableSrsInUlSlots", BooleanValue(c.srs));
    nr->SetSchedulerAttribute("EnableSrsInFSlots", BooleanValue(c.srs));
    nr->SetUlErrorModel(c.errorModel);
    nr->SetDlErrorModel(c.errorModel);
    nr->SetGnbUlAmcAttribute("AmcModel", EnumValue(NrAmc::ErrorModel));
    nr->SetGnbDlAmcAttribute("AmcModel", EnumValue(NrAmc::ErrorModel));

    double subbandHz = c.rbgSize * 12 * 30e3;
    double nf = c.niPerSubbandDbm - (-174.0 + 10 * std::log10(subbandHz));
    nr->SetGnbPhyAttribute("Pattern", StringValue("DL|DL|DL|S|UL|DL|DL|DL|S|UL|"));
    nr->SetGnbPhyAttribute("Numerology", UintegerValue(1));
    nr->SetGnbPhyAttribute("TxPower", DoubleValue(c.gnbTxPower));
    nr->SetGnbPhyAttribute("NoiseFigure", DoubleValue(nf));
    nr->SetUePhyAttribute("TxPower", DoubleValue(c.ueTxPower));
    nr->SetUePhyAttribute("EnableUplinkPowerControl", BooleanValue(false));
    nr->SetUePhyAttribute("PowerAllocationType", StringValue(c.ulPowerAlloc));

    NetDeviceContainer gnbDev;
    for (uint32_t e = 0; e < c.nEnv; ++e)
    {
        gnbDev.Add(nr->InstallGnbDevice(NodeContainer(w.gnbNodes.Get(e)), bwps[e]));
    }
    for (uint32_t e = 0; e < c.nEnv; ++e)
    {
        NodeContainer ues;
        for (uint32_t r = 0; r < c.nUe; ++r)
        {
            ues.Add(w.ueNodes.Get(e * c.nUe + r));
        }
        w.ueDev.Add(nr->InstallUeDevice(ues, bwps[e]));
    }
    int64_t stream = 1;
    stream += nr->AssignStreams(gnbDev, stream);
    stream += nr->AssignStreams(w.ueDev, stream);

    auto [remoteHost, remoteAddr] = epc->SetupRemoteHost("100Gb/s", 2500, Seconds(0.0));
    InternetStackHelper internet;
    internet.Install(w.ueNodes);
    epc->AssignUeIpv4Address(w.ueDev);
    Ipv4StaticRoutingHelper srh;
    for (uint32_t i = 0; i < nTot; ++i)
    {
        srh.GetStaticRouting(w.ueNodes.Get(i)->GetObject<Ipv4>())
            ->SetDefaultRoute(epc->GetUeDefaultGatewayAddress(), 1);
    }
    for (uint32_t i = 0; i < nTot; ++i)
    {
        nr->AttachToGnb(w.ueDev.Get(i), gnbDev.Get(i / c.nUe));
    }

    const uint16_t port = 9000;
    Ptr<FrameSink> sink = CreateObject<FrameSink>();
    sink->Setup(port);
    remoteHost->AddApplication(sink);
    sink->SetStartTime(Seconds(0.0));

    Ptr<UniformRandomVariable> phaseRv = CreateObject<UniformRandomVariable>();
    phaseRv->SetStream(2000);
    for (uint32_t i = 0; i < nTot; ++i)
    {
        Ptr<FrameSender> app = CreateObject<FrameSender>();
        app->Setup(InetSocketAddress(remoteAddr, port),
                   static_cast<uint16_t>(i),
                   MilliSeconds(c.periodMs),
                   Seconds(c.appStart),
                   c.payload,
                   3000 + i);
        w.ueNodes.Get(i)->AddApplication(app);
        app->SetStartTime(Seconds(0.1));
        w.senders.push_back(app);
    }

    // Reporting hooks.
    g_imsiToUe.clear();
    g_nodeToUe.clear();
    for (uint32_t i = 0; i < nTot; ++i)
    {
        auto dev = DynamicCast<NrUeNetDevice>(w.ueDev.Get(i));
        g_imsiToUe[dev->GetImsi()] = i;
        g_nodeToUe[w.ueNodes.Get(i)->GetId()] = i;
        NrHelper::GetUeMac(w.ueDev.Get(i), 0)
            ->TraceConnectWithoutContext("UeMacStateMachineTrace", MakeCallback(&UeMacState));
        NrHelper::GetUePhy(w.ueDev.Get(i), 0)
            ->TraceConnectWithoutContext("ReportRsrp", MakeCallback(&UeRsrp));
    }
    for (uint32_t e = 0; e < c.nEnv; ++e)
    {
        NrHelper::GetGnbPhy(gnbDev.Get(e), 0)
            ->GetSpectrumPhy()
            ->TraceConnectWithoutContext("RxPacketTraceGnb", MakeCallback(&RxTbGnb));
    }

    if (c.flowmon)
    {
        w.fmh = new FlowMonitorHelper();
        NodeContainer ends;
        ends.Add(remoteHost);
        ends.Add(w.ueNodes);
        w.fm = w.fmh->Install(ends);
    }
}

static void
RefreshRntiMap()
{
    g_cellRntiToUe.clear();
    for (uint32_t i = 0; i < g_w.ueDev.GetN(); ++i)
    {
        auto dev = DynamicCast<NrUeNetDevice>(g_w.ueDev.Get(i));
        auto rrc = dev->GetRrc();
        g_cellRntiToUe[{rrc->GetCellId(), rrc->GetRnti()}] = i;
    }
}

static void
RunTo(Time target)
{
    Time now = Simulator::Now();
    if (target > now)
    {
        Simulator::Stop(target - now);
        Simulator::Run();
    }
}

static Time
StepStart(uint32_t k)
{
    return Seconds(g_cfg.appStart) + MilliSeconds(g_cfg.periodMs) * static_cast<int64_t>(k);
}

static void
SetUePos(uint32_t i, double x, double y, double z)
{
    uint32_t e = i / g_cfg.nUe;
    g_w.ueNodes.Get(i)->GetObject<MobilityModel>()->SetPosition(
        Vector(e * g_cfg.envSpacing + x, y, z));
}

static void
TeardownWorld()
{
    if (g_w.fmh)
    {
        delete g_w.fmh;
        g_w.fmh = nullptr;
    }
    g_w = World{};
    Simulator::Destroy();
    Ipv4AddressGenerator::Reset();
}

template <typename T>
static const T*
Take(const std::string& buf, size_t& off, size_t n)
{
    NS_ABORT_MSG_IF(off + n * sizeof(T) > buf.size(), "short message");
    const T* p = reinterpret_cast<const T*>(buf.data() + off);
    off += n * sizeof(T);
    return p;
}

template <typename T>
static void
Put(std::string& buf, const T* p, size_t n)
{
    buf.append(reinterpret_cast<const char*>(p), n * sizeof(T));
}

static std::string
HelloPayload(uint32_t run)
{
    std::string h;
    uint32_t u[4] = {PROTO_VERSION, g_cfg.nEnv, g_cfg.nUe, run};
    double d[2] = {g_cfg.appStart, g_cfg.periodMs / 1000.0};
    Put(h, u, 4);
    Put(h, d, 2);
    return h;
}

int
main(int argc, char* argv[])
{
    Cfg& c = g_cfg;
    CommandLine cmd(__FILE__);
    cmd.AddValue("nEnv", "Independent cells (envs) in this process", c.nEnv);
    cmd.AddValue("nUe", "UEs (robots) per env", c.nUe);
    cmd.AddValue("appStart", "Sim time of control step 0 (s)", c.appStart);
    cmd.AddValue("deadline", "PDCP discard timer (s)", c.deadline);
    cmd.AddValue("periodMs", "Control step (ms)", c.periodMs);
    cmd.AddValue("payload", "UDP payload bytes per packet", c.payload);
    cmd.AddValue("placement", "Initial drop: random | dists", c.placement);
    cmd.AddValue("dists", "Comma-separated UE distances (m) for placement=dists", c.dists);
    cmd.AddValue("side", "Square side (m) for random drop", c.side);
    cmd.AddValue("freq", "Carrier frequency (Hz)", c.freq);
    cmd.AddValue("bw", "Channel bandwidth (Hz)", c.bw);
    cmd.AddValue("rbOverhead", "NrHelper RB overhead", c.rbOverhead);
    cmd.AddValue("rbgSize", "PRBs per RBG", c.rbgSize);
    cmd.AddValue("ueTxPower", "UE Tx power (dBm)", c.ueTxPower);
    cmd.AddValue("gnbTxPower", "gNB Tx power (dBm)", c.gnbTxPower);
    cmd.AddValue("niPerSubbandDbm", "Noise+interference per 10-PRB subband (dBm)", c.niPerSubbandDbm);
    cmd.AddValue("shadowStd", "Log-normal shadowing std (dB) of the initial drop", c.shadowStd);
    cmd.AddValue("minSnrDb", "Coverage-conditioned initial drop (see netslot-ref)", c.minSnrDb);
    cmd.AddValue("fading", "3GPP UMi NLOS small-scale fading", c.fading);
    cmd.AddValue("vScatt", "Scatterer speed for Doppler (m/s)", c.vScatt);
    cmd.AddValue("chanUpdateMs", "ThreeGppChannelModel UpdatePeriod (ms), 0 = never (netslot-ref)", c.chanUpdateMs);
    cmd.AddValue("errorModel", "UL error model TypeId", c.errorModel);
    cmd.AddValue("sched", "Scheduler TypeId", c.sched);
    cmd.AddValue("rlc", "UM | AM", c.rlc);
    cmd.AddValue("ulPowerAlloc", "UniformPowerAllocUsed | UniformPowerAllocBw", c.ulPowerAlloc);
    cmd.AddValue("harq", "HARQ retransmissions", c.harq);
    cmd.AddValue("srs", "SRS in UL slots", c.srs);
    cmd.AddValue("flowmon", "Install FlowMonitor (only to mirror netslot-ref event order)", c.flowmon);
    cmd.AddValue("run", "RNG run number of the first episode", c.run);
    cmd.AddValue("envSpacing", "Distance between env cells (m)", c.envSpacing);
    cmd.AddValue("ueHeight", "UE antenna height (m) when the client sends 2D poses", c.ueHeight);
    cmd.AddValue("bridge", "tcp:PORT | unix:PATH | shm:NAME | none", c.bridge);
    cmd.AddValue("framesOut", "Optional csv of every completed frame (debug)", c.framesOut);
    cmd.Parse(argc, argv);

    NS_ABORT_MSG_IF(c.nEnv * c.nUe > 65535, "too many UEs");
    std::unique_ptr<Transport> tr;
    if (c.bridge.rfind("tcp:", 0) == 0 || c.bridge.rfind("unix:", 0) == 0)
    {
        tr = std::make_unique<StreamTransport>(c.bridge);
    }
#ifdef WITH_NS3AI
    else if (c.bridge.rfind("shm:", 0) == 0)
    {
        tr = std::make_unique<ShmAiTransport>(c.bridge.substr(4));
    }
#endif
    else
    {
        NS_FATAL_ERROR("unsupported --bridge " << c.bridge);
    }

    std::ofstream fo;
    if (!c.framesOut.empty())
    {
        fo.open(c.framesOut);
        fo << "env,ue,fid,done\n";
    }

    uint32_t run = c.run;
    BuildWorld(c, run);
    RunTo(StepStart(0) - NanoSeconds(1));
    tr->Send(MSG_HELLO, HelloPayload(run));

    uint32_t k = 0; // control steps done since the last (re)build
    const uint32_t N = c.nEnv * c.nUe;
    std::string in, out;
    uint32_t type = 0;
    while (tr->Recv(type, in))
    {
        if (type == MSG_CLOSE)
        {
            break;
        }
        if (type == MSG_RESET)
        {
            size_t off = 0;
            const uint32_t* h = Take<uint32_t>(in, off, 2);
            run = h[0];
            uint32_t flags = h[1];
            const float* pos = (flags & RESET_HAS_POS) ? Take<float>(in, off, N * 3) : nullptr;
            const float* sh = (flags & RESET_HAS_SHADOW) ? Take<float>(in, off, N) : nullptr;
            TeardownWorld();
            BuildWorld(c, run);
            g_lastEnd.assign(N, Vector());
            g_hasLastEnd.assign(N, false); // no command yet: interpolate from the rebuilt world's pose
            for (uint32_t i = 0; i < N; ++i)
            {
                if (pos && std::isfinite(pos[3 * i]))
                {
                    double z = std::isfinite(pos[3 * i + 2]) ? pos[3 * i + 2] : c.ueHeight;
                    SetUePos(i, pos[3 * i], pos[3 * i + 1], z);
                }
                if (sh && std::isfinite(sh[i]))
                {
                    g_ueShadowDb[g_w.ueNodes.Get(i)->GetId()] = sh[i];
                }
            }
            k = 0;
            RunTo(StepStart(0) - NanoSeconds(1));
            tr->Send(MSG_HELLO, HelloPayload(run));
            continue;
        }
        NS_ABORT_MSG_IF(type != MSG_STEP, "unexpected message type " << type);
        auto w0 = std::chrono::steady_clock::now();
        size_t off = 0;
        const int32_t* tt = Take<int32_t>(in, off, 1);
        const uint32_t* hdr = Take<uint32_t>(in, off, 2);
        int32_t tClient = tt[0];
        uint32_t flags = hdr[0];
        uint32_t nIn = hdr[1];
        const float* pos = Take<float>(in, off, N * 3);
        const float* sh = (flags & STEP_HAS_SHADOW) ? Take<float>(in, off, N) : nullptr;
        const WireFrameIn* fr = Take<WireFrameIn>(in, off, nIn);

        Time t0 = StepStart(k);
        Time t1 = StepStart(k + 1);
        if (g_lastEnd.size() != N)
        {
            g_lastEnd.assign(N, Vector());
            g_hasLastEnd.assign(N, false);
        }
        for (uint32_t i = 0; i < N; ++i)
        {
            if (sh && std::isfinite(sh[i]))
            {
                g_ueShadowDb[g_w.ueNodes.Get(i)->GetId()] = sh[i];
            }
            if (!std::isfinite(pos[3 * i]))
            {
                continue;
            }
            double z = std::isfinite(pos[3 * i + 2]) ? pos[3 * i + 2] : c.ueHeight;
            if (flags & STEP_INTERP_POS)
            {
                // Waypoint: 4 linear sub-steps from the previous commanded end pose (the mobility model's
                // position before the first command) to this step's end pose.
                Ptr<MobilityModel> mm = g_w.ueNodes.Get(i)->GetObject<MobilityModel>();
                Vector p0 = g_hasLastEnd[i] ? g_lastEnd[i] : mm->GetPosition();
                double ex = (i / c.nUe) * c.envSpacing;
                Vector p1(ex + pos[3 * i], pos[3 * i + 1], z);
                g_lastEnd[i] = p1;
                g_hasLastEnd[i] = true;
                const int C = 4;
                for (int j = 0; j < C; ++j)
                {
                    double a = (j + 0.5) / C;
                    Vector pj(p0.x + a * (p1.x - p0.x), p0.y + a * (p1.y - p0.y), p0.z + a * (p1.z - p0.z));
                    Time at = t0 + (t1 - t0) * (static_cast<double>(j) / C);
                    Simulator::Schedule(at - Simulator::Now(), &MobilityModel::SetPosition, mm, pj);
                }
            }
            else
            {
                SetUePos(i, pos[3 * i], pos[3 * i + 1], z);
                g_lastEnd[i] = g_w.ueNodes.Get(i)->GetObject<MobilityModel>()->GetPosition();
                g_hasLastEnd[i] = true;
            }
        }
        for (uint32_t j = 0; j < nIn; ++j)
        {
            uint32_t i = fr[j].env * c.nUe + fr[j].ue;
            NS_ABORT_MSG_IF(i >= N, "frame for unknown ue");
            g_w.senders[i]->Queue(fr[j].fid, fr[j].bytes);
        }
        for (auto& s : g_st)
        {
            float rsrp = s.rsrp;
            uint32_t buf = s.bufBytes;
            s = UeStepStats{};
            s.rsrp = rsrp;
            s.bufBytes = buf;
        }
        g_seenTb.clear();
        g_done.clear();
        RefreshRntiMap();

        auto w1 = std::chrono::steady_clock::now();
        RunTo(t1 - NanoSeconds(1));
        auto w2 = std::chrono::steady_clock::now();
        k++;

        // RESULT
        out.clear();
        int32_t ti[1] = {tClient};
        uint32_t u[3] = {N, static_cast<uint32_t>(g_done.size()), k};
        double d[3] = {t0.GetSeconds(),
                       std::chrono::duration<double>(w2 - w1).count(),
                       0.0};
        std::vector<float> sinr(N), rsrp(N), qb(N), mcs(N);
        std::vector<uint32_t> ntb(N), nretx(N), ncor(N), nlost(N), okb(N);
        for (uint32_t i = 0; i < N; ++i)
        {
            const UeStepStats& s = g_st[i];
            sinr[i] = s.nTb ? static_cast<float>(10 * std::log10(s.sinrLinSum / s.nTb)) : NAN;
            mcs[i] = s.nTb ? static_cast<float>(s.mcsSum / s.nTb) : NAN;
            rsrp[i] = s.rsrp;
            qb[i] = static_cast<float>(s.bufBytes);
            ntb[i] = s.nTb;
            nretx[i] = s.nRetx;
            ncor[i] = s.nCorrupt;
            nlost[i] = s.nLost;
            okb[i] = static_cast<uint32_t>(s.okBytes);
        }
        auto w3 = std::chrono::steady_clock::now();
        d[2] = std::chrono::duration<double>((w1 - w0) + (w3 - w2)).count();
        Put(out, ti, 1);
        Put(out, u, 3);
        Put(out, d, 3);
        Put(out, sinr.data(), N);
        Put(out, rsrp.data(), N);
        Put(out, qb.data(), N);
        Put(out, mcs.data(), N);
        Put(out, ntb.data(), N);
        Put(out, nretx.data(), N);
        Put(out, ncor.data(), N);
        Put(out, nlost.data(), N);
        Put(out, okb.data(), N);
        Put(out, g_done.data(), g_done.size());
        if (fo.is_open())
        {
            for (auto& f : g_done)
            {
                fo << f.env << "," << f.ue << "," << f.fid << "," << f.t << "\n";
            }
        }
        tr->Send(MSG_RESULT, out);

        // Bound the registry: frames older than 10 s can no longer complete in any useful sense.
        double horizon = Simulator::Now().GetSeconds() - 10.0;
        for (auto it = g_frames.begin(); it != g_frames.end();)
        {
            it = (it->second.gen < horizon) ? g_frames.erase(it) : std::next(it);
        }
    }
    tr.reset();
    Simulator::Destroy();
    return 0;
}
