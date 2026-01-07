#pragma once
// lu_net.h - minimal Winsock2 helpers + length-prefixed messaging for MPI-lite.
//
// This header encapsulates a tiny message framing layer and a few convenient
// socket helpers to keep the driver/worker code readable.
//
// Design notes & assumptions:
//  - Implemented specifically for Windows/Winsock2. The code uses raw POD
//    serialization via memcpy and pod<T> pushing; this requires driver and
//    worker to run on same-architecture, same-endianness machines (typical
//    in a lab cluster). If multi-arch support is required, adapt to a stable
//    network serialization format (e.g., protobuf, flatbuffers) or ensure
//    network (big-endian) byte order.
//  - The message frame format is:
//       uint32 payload_bytes   // includes 4 bytes used to carry msg+reserved
//       uint16 msg             // net::Msg enum value
//       uint16 reserved
//       uint8  payload[payload_bytes-4]
//    This packing allows a simple, length-prefixed read on the receiver.
//  - ByteWriter/ByteReader perform simple POD appends and reads. They do not
//    perform bounds-checking on writes and assume the user encodes sizes where needed.
//  - send_all / recv_all are blocking and loop until requested bytes are sent/received.
//    They raise std::runtime_error on socket errors and on peer closure.
// Authors: Morteza Farrokhnejad, Ali Farrokhnejad

#ifdef _WIN32
  #ifndef NOMINMAX
    #define NOMINMAX
  #endif
  #include <winsock2.h>
  #include <ws2tcpip.h>
  #pragma comment(lib, "Ws2_32.lib")
#else
  #error "This MPI-lite networking layer is implemented for Windows (Winsock2)."
#endif

#include <cstdint>
#include <vector>
#include <string>
#include <stdexcept>
#include <sstream>
#include <cstring>
#include <iostream>

namespace net {

// RAII wrapper for WSAStartup / WSACleanup. Construct one instance before using Winsock,
// it will call WSACleanup() automatically on destruction.
struct WSAInit {
    WSAInit() {
        WSADATA wsa;
        int rc = WSAStartup(MAKEWORD(2, 2), &wsa);
        if (rc != 0) throw std::runtime_error("WSAStartup failed: " + std::to_string(rc));
    }
    ~WSAInit() { WSACleanup(); }
};

// Format a last-error string using WSAGetLastError. Helpful when throwing detailed runtime_error.
inline std::string last_error(const char* where) {
    int e = WSAGetLastError();
    std::ostringstream oss;
    oss << where << " failed (WSAGetLastError=" << e << ")";
    return oss.str();
}

// Safe close: call closesocket only if socket is valid.
inline void closesocket_safe(SOCKET s) {
    if (s != INVALID_SOCKET) closesocket(s);
}

// send_all: repeatedly call ::send until all bytes have been transmitted.
// Throws runtime_error on error.
inline void send_all(SOCKET s, const void* data, size_t n) {
    const char* p = (const char*)data;
    while (n > 0) {
        int sent = ::send(s, p, (int)std::min<size_t>(n, 1u<<30), 0);
        if (sent == SOCKET_ERROR) throw std::runtime_error(last_error("send"));
        p += sent;
        n -= (size_t)sent;
    }
}

// recv_all: repeatedly call ::recv until exactly n bytes have been read.
// Throws runtime_error on error or on peer closure (recv returns 0).
inline void recv_all(SOCKET s, void* data, size_t n) {
    char* p = (char*)data;
    while (n > 0) {
        int recvd = ::recv(s, p, (int)std::min<size_t>(n, 1u<<30), 0);
        if (recvd == 0) throw std::runtime_error("recv failed: peer closed connection");
        if (recvd == SOCKET_ERROR) throw std::runtime_error(last_error("recv"));
        p += recvd;
        n -= (size_t)recvd;
    }
}

// set TCP_NODELAY to reduce small-packet latency (disable Nagle).
inline void set_nodelay(SOCKET s, bool on=true) {
    int flag = on ? 1 : 0;
    setsockopt(s, IPPROTO_TCP, TCP_NODELAY, (const char*)&flag, sizeof(flag));
}

// Establish a TCP connection to host:port and return connected socket.
// Throws runtime_error on DNS resolution or connect failure.
inline SOCKET connect_tcp(const std::string& host, uint16_t port) {
    addrinfo hints{};
    hints.ai_family = AF_INET;
    hints.ai_socktype = SOCK_STREAM;
    hints.ai_protocol = IPPROTO_TCP;

    char port_str[16];
    std::snprintf(port_str, sizeof(port_str), "%u", (unsigned)port);

    addrinfo* res = nullptr;
    int rc = getaddrinfo(host.c_str(), port_str, &hints, &res);
    if (rc != 0 || !res) throw std::runtime_error("getaddrinfo failed for " + host + ":" + port_str);

    SOCKET s = INVALID_SOCKET;
    for (addrinfo* p = res; p; p = p->ai_next) {
        s = socket(p->ai_family, p->ai_socktype, p->ai_protocol);
        if (s == INVALID_SOCKET) continue;
        if (::connect(s, p->ai_addr, (int)p->ai_addrlen) == 0) break;
        closesocket_safe(s);
        s = INVALID_SOCKET;
    }
    freeaddrinfo(res);
    if (s == INVALID_SOCKET) throw std::runtime_error("connect failed to " + host + ":" + port_str);
    set_nodelay(s, true);
    return s;
}

// Create a listening socket bound to bind_ip:port, accept a single connection and return the accepted socket.
// The listening socket is closed before returning the accepted socket (so the caller receives only the accepted peer).
inline SOCKET listen_and_accept(const std::string& bind_ip, uint16_t port) {
    addrinfo hints{};
    hints.ai_family = AF_INET;
    hints.ai_socktype = SOCK_STREAM;
    hints.ai_protocol = IPPROTO_TCP;
    hints.ai_flags = AI_PASSIVE;

    char port_str[16];
    std::snprintf(port_str, sizeof(port_str), "%u", (unsigned)port);

    addrinfo* res = nullptr;
    int rc = getaddrinfo(bind_ip.empty() ? nullptr : bind_ip.c_str(), port_str, &hints, &res);
    if (rc != 0 || !res) throw std::runtime_error("getaddrinfo(listen) failed");

    SOCKET ls = socket(res->ai_family, res->ai_socktype, res->ai_protocol);
    if (ls == INVALID_SOCKET) { freeaddrinfo(res); throw std::runtime_error(last_error("socket(listen)")); }

    // Allow quick restart by setting SO_REUSEADDR
    BOOL yes = 1;
    setsockopt(ls, SOL_SOCKET, SO_REUSEADDR, (const char*)&yes, sizeof(yes));

    if (::bind(ls, res->ai_addr, (int)res->ai_addrlen) == SOCKET_ERROR) {
        freeaddrinfo(res);
        closesocket_safe(ls);
        throw std::runtime_error(last_error("bind"));
    }
    freeaddrinfo(res);

    if (::listen(ls, 1) == SOCKET_ERROR) {
        closesocket_safe(ls);
        throw std::runtime_error(last_error("listen"));
    }

    SOCKET cs = ::accept(ls, nullptr, nullptr);
    closesocket_safe(ls);
    if (cs == INVALID_SOCKET) throw std::runtime_error(last_error("accept"));
    set_nodelay(cs, true);
    return cs;
}

// ---------------------- Message packing ----------------------
//
// The RPC protocol uses small integer Msg codes and a simple length-prefixed frame.
// Each frame contains:
//    uint32 payload_bytes   // length of (msg+reserved+payload), so payload_bytes >= 4
//    uint16 msg             // net::Msg enum
//    uint16 reserved        // unused (future extension)
//    uint8  payload[]       // zero or more bytes
//
// ByteWriter: append POD values and raw bytes to a std::vector<uint8_t> buffer.
// ByteReader: read POD values and raw bytes sequentially from a received buffer.
//
// Note: This implementation assumes identical sizeof(T) and binary layout on sender and receiver
// for POD types. This is fine for homogeneous lab machines but not portable across different
// architectures or compilers with different ABI assumptions.
enum class Msg : uint16_t {
    INIT = 1,
    PIVOT_SCAN = 2,
    GET_ROW = 3,
    PUT_ROW = 4,
    LOCAL_SWAP = 5,
    GET_PIVOT_TAIL = 6,
    BCAST_PIVOT_TAIL = 7,
    ELIMINATE = 8,
    GET_BLOCK = 9,
    GET_STATS = 10,
    SHUTDOWN = 11,

    OK = 200,
    ERR = 500
};

struct ByteWriter {
    std::vector<uint8_t> buf;
    // Append a POD value's raw bytes (little-endian/native order).
    template <typename T>
    void pod(const T& v) {
        const uint8_t* p = reinterpret_cast<const uint8_t*>(&v);
        buf.insert(buf.end(), p, p + sizeof(T));
    }
    // Append raw bytes (e.g., an array of doubles).
    void bytes(const void* data, size_t n) {
        const uint8_t* p = reinterpret_cast<const uint8_t*>(data);
        buf.insert(buf.end(), p, p + n);
    }
};

struct ByteReader {
    const uint8_t* p;
    const uint8_t* e;
    ByteReader(const std::vector<uint8_t>& b) : p(b.data()), e(b.data() + b.size()) {}
    // Read a POD value (throws runtime_error on underflow).
    template <typename T>
    T pod() {
        if ((size_t)(e - p) < sizeof(T)) throw std::runtime_error("ByteReader: underflow");
        T v;
        std::memcpy(&v, p, sizeof(T));
        p += sizeof(T);
        return v;
    }
    // Read raw bytes into out (throws runtime_error if not enough bytes remain).
    void bytes(void* out, size_t n) {
        if ((size_t)(e - p) < n) throw std::runtime_error("ByteReader: underflow(bytes)");
        std::memcpy(out, p, n);
        p += n;
    }
};

// send_msg: write the framed message to socket s. The protocol stores a uint32 payload_bytes
// that equals (4 + payload.size()), because the 4 bytes are the msg+reserved fields (2+2).
inline void send_msg(SOCKET s, Msg type, const std::vector<uint8_t>& payload) {
    uint32_t payload_bytes = (uint32_t)(payload.size() + 4); // msg+reserved included in payload area
    uint16_t t = (uint16_t)type;
    uint16_t rsv = 0;
    send_all(s, &payload_bytes, sizeof(payload_bytes));
    send_all(s, &t, sizeof(t));
    send_all(s, &rsv, sizeof(rsv));
    if (!payload.empty()) send_all(s, payload.data(), payload.size());
}

// recv_msg: read a framed message from socket s. Fills type_out and payload_out.
// Throws on framing errors, recv errors, or underflow.
inline void recv_msg(SOCKET s, Msg& type_out, std::vector<uint8_t>& payload_out) {
    uint32_t payload_bytes = 0;
    recv_all(s, &payload_bytes, sizeof(payload_bytes));
    if (payload_bytes < 4) throw std::runtime_error("Bad frame: payload_bytes<4");
    uint16_t t=0, rsv=0;
    recv_all(s, &t, sizeof(t));
    recv_all(s, &rsv, sizeof(rsv));
    (void)rsv;
    type_out = (Msg)t;
    payload_out.resize(payload_bytes - 4);
    if (!payload_out.empty()) recv_all(s, payload_out.data(), payload_out.size());
}

// parse a host:port string. If no colon present, return port=0 to indicate error/unspecified.
inline std::pair<std::string,uint16_t> parse_hostport(const std::string& s) {
    auto pos = s.find(':');
    if (pos == std::string::npos) return {s, 0};
    return {s.substr(0,pos), (uint16_t)std::stoi(s.substr(pos+1))};
}

} // namespace net
