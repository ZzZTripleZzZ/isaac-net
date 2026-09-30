// Python side of the ns3-ai shared-memory transport (pybind11). Python is the segment creator.
// ns3-ai's Ns3AiMsgInterfaceImpl keeps its managed_shared_memory in a function-local static, so one
// template instantiation can own only one segment per process. We instantiate it NSLOT times with
// distinct tag types so one Python process can open up to NSLOT channels over its lifetime.
#include "ns3-ai-msg-interface.h"

#include <pybind11/pybind11.h>

#include <cstring>
#include <memory>
#include <string>

namespace py = pybind11;

static const uint32_t SHM_CAP = 1u << 20;
// ns3-ai keeps the segment in a function-local static: a slot (template instance) can be used once per
// process lifetime, so the loader hands out each slot only once.
static constexpr int NSLOT = 64;

template <int K>
struct ShmMsgK
{
    uint32_t type;
    uint32_t len;
    uint8_t data[SHM_CAP];
};

struct ChanBase
{
    virtual ~ChanBase() = default;
    virtual void send(uint32_t type, py::bytes payload) = 0;
    virtual py::tuple recv() = 0;
};

template <int K>
struct Chan : ChanBase
{
    using M = ShmMsgK<K>;
    std::unique_ptr<ns3::Ns3AiMsgInterfaceImpl<M, M>> impl;
    std::string seg, c2p, p2c, lock;

    explicit Chan(const std::string& name)
        : seg(name), c2p(name + "_c2p"), p2c(name + "_p2c"), lock(name + "_lock")
    {
        impl = std::make_unique<ns3::Ns3AiMsgInterfaceImpl<M, M>>(
            true, false, true, 4u << 20, seg.c_str(), c2p.c_str(), p2c.c_str(), lock.c_str());
    }

    void send(uint32_t type, py::bytes payload) override
    {
        std::string_view v = payload;
        if (v.size() > SHM_CAP)
        {
            throw std::runtime_error("payload larger than SHM_CAP");
        }
        {
            py::gil_scoped_release nogil;
            impl->PySendBegin();
        }
        M* m = impl->GetPy2CppStruct();
        m->type = type;
        m->len = static_cast<uint32_t>(v.size());
        std::memcpy(m->data, v.data(), v.size());
        impl->PySendEnd();
    }

    py::tuple recv() override
    {
        {
            py::gil_scoped_release nogil;
            impl->PyRecvBegin();
        }
        M* m = impl->GetCpp2PyStruct();
        uint32_t type = impl->PyGetFinished() ? 0u : m->type;
        py::bytes out(reinterpret_cast<const char*>(m->data), impl->PyGetFinished() ? 0 : m->len);
        impl->PyRecvEnd();
        return py::make_tuple(type, out);
    }
};

template <int K>
static std::unique_ptr<ChanBase>
make(int slot, const std::string& name)
{
    if constexpr (K < NSLOT)
    {
        if (slot == K)
        {
            return std::make_unique<Chan<K>>(name);
        }
        return make<K + 1>(slot, name);
    }
    else
    {
        throw std::runtime_error("shm slots exhausted (NSLOT per Python process lifetime)");
    }
}

PYBIND11_MODULE(ns3ai_bridge_py, m)
{
    py::class_<ChanBase>(m, "Chan").def("send", &ChanBase::send).def("recv", &ChanBase::recv);
    m.def("create", [](int slot, const std::string& name) { return make<0>(slot, name); });
    m.attr("NSLOT") = NSLOT;
}
