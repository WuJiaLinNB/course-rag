#include <rag/llm_sse.hpp>

#if defined(CPPHTTPLIB_OPENSSL_SUPPORT)

#include <core/log.hpp>

#include <openssl/ssl.h>
#include <openssl/err.h>
#include <openssl/x509.h>
#include <openssl/x509v3.h>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#ifdef _WIN32
#include <winsock2.h>
#include <ws2tcpip.h>
#pragma comment(lib, "ws2_32.lib")
using SockT = SOCKET;
#define CLOSE_SOCK(s) ::closesocket(s)
#define SOCK_ERRNO WSAGetLastError()
#define SOCK_WOULD_BLOCK WSAEWOULDBLOCK
#define SOCK_IN_PROGRESS WSAEWOULDBLOCK
#define INVALID_SOCKET_VAL INVALID_SOCKET
#else
#include <arpa/inet.h>
#include <fcntl.h>
#include <netdb.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <sys/select.h>
#include <sys/socket.h>
#include <unistd.h>
using SockT = int;
#define CLOSE_SOCK(s) ::close(s)
#define SOCK_ERRNO errno
#define SOCK_WOULD_BLOCK EWOULDBLOCK
#define SOCK_IN_PROGRESS EINPROGRESS
#define INVALID_SOCKET_VAL (-1)
#endif

namespace rag {
namespace {

// ---- 平台小件 ----

#ifdef _WIN32
struct WsaInit {
    WsaInit() { WSADATA w; WSAStartup(MAKEWORD(2, 2), &w); }
};
WsaInit g_wsa;   // 进程级一次初始化（main 前完成）

// 从 Windows 系统证书存储（ROOT）加载根证书到 OpenSSL store——与 httplib 0.18 的
// load_system_certs_on_windows 同方案；vendored OpenSSL 的 OPENSSLDIR 在 Windows
// 上没有 /etc/ssl/certs，只能走系统存储。
bool load_system_certs(X509_STORE* store) {
    HCERTSTORE h = CertOpenSystemStoreW(0, L"ROOT");
    if (!h) return false;
    PCCERT_CONTEXT c = nullptr;
    int n = 0;
    while ((c = CertEnumCertificatesInStore(h, c)) != nullptr) {
        const unsigned char* p = c->pbCertEncoded;
        X509* x = d2i_X509(nullptr, &p, static_cast<long>(c->cbCertEncoded));
        if (x) {
            if (X509_STORE_add_cert(store, x) == 1) ++n;
            X509_free(x);
        }
    }
    CertFreeCertificateContext(c);
    CertCloseStore(h, 0);
    return n > 0;
}
#endif

// 阻塞等待 fd 可读/可写；超时返回 false
bool wait_io(SockT fd, int timeout_ms, bool for_write) {
    fd_set fds;
    FD_ZERO(&fds);
    FD_SET(fd, &fds);
    timeval tv;
    tv.tv_sec = timeout_ms / 1000;
    tv.tv_usec = (timeout_ms % 1000) * 1000;
    const int r = ::select(static_cast<int>(fd + 1),
                           for_write ? nullptr : &fds,
                           for_write ? &fds : nullptr,
                           nullptr, &tv);
    return r > 0;
}

// 读一层（TLS 解密后或裸 TCP）；返回 >0 字节数、0 连接关闭、-1 超时/错误
int sock_read(SSL* ssl, SockT fd, char* buf, size_t len, int timeout_ms) {
    if (!wait_io(fd, timeout_ms, false)) return -1;
    if (ssl) {
        const int n = SSL_read(ssl, buf, static_cast<int>(len));
        if (n > 0) return n;
        return SSL_get_error(ssl, n) == SSL_ERROR_ZERO_RETURN ? 0 : -1;
    }
    const int n = ::recv(fd, buf, static_cast<int>(len), 0);
    return n > 0 ? n : (n == 0 ? 0 : -1);
}

// 读一行（到 \r\n），累积进 leftover；超时返回 false
bool read_line(SSL* ssl, SockT fd, std::string& leftover, std::string& line,
               int timeout_ms) {
    line.clear();
    for (;;) {
        const size_t p = leftover.find('\n');
        if (p != std::string::npos) {
            line = leftover.substr(0, p);
            if (!line.empty() && line.back() == '\r') line.pop_back();
            leftover.erase(0, p + 1);
            return true;
        }
        char buf[1024];
        const int n = sock_read(ssl, fd, buf, sizeof buf, timeout_ms);
        if (n <= 0) return false;
        leftover.append(buf, static_cast<size_t>(n));
    }
}

// 缓冲读：优先从 leftover 取（read_line 可能已把后续数据一并读入 leftover），
// 取不够再从 socket 补。body 的所有读取必须走这里——直接 sock_read 会绕过 leftover
// 里已缓冲的数据，导致 chunk 边界与 leftover 内容错位（曾引发只收到第一帧就误把
// JSON 字节当 chunk size 解析、提前结束流的问题）。
int sock_read_buf(SSL* ssl, SockT fd, std::string& leftover,
                  char* out, size_t len, int timeout_ms) {
    if (!leftover.empty()) {
        // 不用 std::min(a,b) 裸调：MSVC 下 windows.h 的 min 宏会把它展开成 (a)<(b)?…
        // 导致 C2589；显式三目无此问题
        const size_t take = len < leftover.size() ? len : leftover.size();
        std::memcpy(out, leftover.data(), take);
        leftover.erase(0, take);
        return static_cast<int>(take);
    }
    return sock_read(ssl, fd, out, len, timeout_ms);
}

// ---- SSE 增量解析：喂原始字节，吐出 choices[0].delta.content 给 on_delta ----

struct SseParser {
    std::string buf;
    const std::function<bool(const std::string&)>& on_delta;
    bool got_any = false;

    bool feed(const char* data, size_t len) {
        buf.append(data, len);
        size_t pos;
        while ((pos = buf.find("\n\n")) != std::string::npos) {
            const std::string ev = buf.substr(0, pos);
            buf.erase(0, pos + 2);
            const size_t dp = ev.find("data:");
            if (dp == std::string::npos) continue;
            std::string payload = ev.substr(dp + 5);
            const size_t b = payload.find_first_not_of(" \t\r");
            if (b == std::string::npos) continue;
            const size_t e = payload.find_last_not_of(" \t\r");
            payload = payload.substr(b, e - b + 1);
            if (payload == "[DONE]") continue;
            try {
                // 免依赖 JSON 解析：只按固定形状取 choices[0].delta.content，
                // 任何形状不对都跳过该帧（外部数据不做假设）
                const std::string t = extract_delta_content(payload);
                if (!t.empty()) {
                    got_any = true;
                    if (!on_delta(t)) return false;   // 客户端断开 → 中止
                }
            } catch (...) { /* 单帧解析失败跳过，不炸整个流 */ }
        }
        return true;
    }

private:
    // 手写轻量抽取：{"choices":[{"delta":{"content":"..."}}],...}
    // 关键字段都在行内且顺序固定（OpenAI 兼容协议），避免引入完整 JSON 库依赖
    static std::string extract_delta_content(const std::string& s) {
        const std::string key = "\"delta\"";
        const std::string ck = "\"content\"";
        const size_t d = s.find(key);
        if (d == std::string::npos) return "";
        const size_t c = s.find(ck, d);
        if (c == std::string::npos) return "";
        const size_t q1 = s.find('"', c + ck.size());
        if (q1 == std::string::npos) return "";
        const size_t q2 = s.find('"', q1 + 1);
        if (q2 == std::string::npos) return "";
        std::string out = s.substr(q1 + 1, q2 - q1 - 1);
        // 反转义 JSON 字符串（\n \" \\ \uXXXX 常见）
        std::string un;
        un.reserve(out.size());
        for (size_t i = 0; i < out.size(); ++i) {
            if (out[i] != '\\' || i + 1 >= out.size()) { un += out[i]; continue; }
            const char c2 = out[++i];
            switch (c2) {
                case 'n': un += '\n'; break;
                case 'r': un += '\r'; break;
                case 't': un += '\t'; break;
                case '"': un += '"'; break;
                case '\\': un += '\\'; break;
                case '/': un += '/'; break;
                case 'u': {   // \uXXXX → UTF-8（只处理 BMP，代理对罕见可忽略）
                    if (i + 4 < out.size()) {
                        unsigned cp = 0;
                        bool ok = true;
                        for (int k = 1; k <= 4; ++k) {
                            const char h = out[i + k];
                            cp <<= 4;
                            if (h >= '0' && h <= '9') cp |= static_cast<unsigned>(h - '0');
                            else if (h >= 'a' && h <= 'f') cp |= static_cast<unsigned>(h - 'a' + 10);
                            else if (h >= 'A' && h <= 'F') cp |= static_cast<unsigned>(h - 'A' + 10);
                            else { ok = false; break; }
                        }
                        i += 4;
                        if (ok && cp < 0x80) un += static_cast<char>(cp);
                        else if (ok && cp < 0x800) {
                            un += static_cast<char>(0xC0 | (cp >> 6));
                            un += static_cast<char>(0x80 | (cp & 0x3F));
                        } else if (ok) {
                            un += static_cast<char>(0xE0 | (cp >> 12));
                            un += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
                            un += static_cast<char>(0x80 | (cp & 0x3F));
                        }
                    }
                    break;
                }
                default: un += c2; break;
            }
        }
        return un;
    }
};

// ---- 主流程 ----

bool connect_and_tls(const std::string& cli_base, SockT& fd, SSL*& ssl,
                     std::string& host_out) {
    // 拆 scheme://host[:port]（IPv6 字面量不在支持范围，LLM 端点不会用）
    std::string scheme = "https", rest = cli_base;
    const size_t se = cli_base.find("://");
    if (se != std::string::npos) {
        scheme = cli_base.substr(0, se);
        rest = cli_base.substr(se + 3);
    }
    const bool tls = (scheme == "https");
    std::string host = rest, port = tls ? "443" : "80";
    const size_t colon = rest.rfind(':');
    if (colon != std::string::npos) {
        host = rest.substr(0, colon);
        port = rest.substr(colon + 1);
    }
    host_out = host;

    addrinfo hints;
    std::memset(&hints, 0, sizeof hints);
    hints.ai_family = AF_UNSPEC;
    hints.ai_socktype = SOCK_STREAM;
    addrinfo* res = nullptr;
    if (getaddrinfo(host.c_str(), port.c_str(), &hints, &res) != 0) {
        LOG_WARN("llm sse: getaddrinfo failed for %s", host.c_str());
        return false;
    }
    fd = INVALID_SOCKET_VAL;
    for (addrinfo* ai = res; ai != nullptr; ai = ai->ai_next) {
        const SockT s = ::socket(ai->ai_family, ai->ai_socktype, ai->ai_protocol);
        if (s == INVALID_SOCKET_VAL) continue;
        // 非阻塞 connect + select（超时 5s，硬约束 embedding/LLM 连接的 fail-fast）
#ifdef _WIN32
        u_long nb = 1;
        ioctlsocket(s, FIONBIO, &nb);
#else
        const int fl = fcntl(s, F_GETFL, 0);
        fcntl(s, F_SETFL, fl | O_NONBLOCK);
#endif
        if (::connect(s, ai->ai_addr, static_cast<int>(ai->ai_addrlen)) == 0 ||
            (SOCK_ERRNO == SOCK_IN_PROGRESS && wait_io(s, 5000, true))) {
            int soerr = 0;
#ifdef _WIN32
            int elen = sizeof soerr;
            getsockopt(s, SOL_SOCKET, SO_ERROR, reinterpret_cast<char*>(&soerr), &elen);
#else
            socklen_t elen = sizeof soerr;
            getsockopt(s, SOL_SOCKET, SO_ERROR, &soerr, &elen);
#endif
            if (soerr == 0) {
                // 恢复阻塞模式
#ifdef _WIN32
                nb = 0;
                ioctlsocket(s, FIONBIO, &nb);
#else
                fcntl(s, F_SETFL, fl);
#endif
                fd = s;
                break;
            }
        }
        CLOSE_SOCK(s);
    }
    freeaddrinfo(res);
    if (fd == INVALID_SOCKET_VAL) {
        LOG_WARN("llm sse: connect to %s:%s failed", host.c_str(), port.c_str());
        return false;
    }
    if (!tls) return true;

    SSL_CTX* ctx = SSL_CTX_new(TLS_client_method());
    if (!ctx) return false;
    SSL_CTX_set_verify(ctx, SSL_VERIFY_PEER, nullptr);
#ifdef _WIN32
    if (!load_system_certs(SSL_CTX_get_cert_store(ctx))) {
        SSL_CTX_set_default_verify_paths(ctx);   // 兜底（Windows 上通常无目录）
    }
#else
    SSL_CTX_set_default_verify_paths(ctx);
#endif
    ssl = SSL_new(ctx);
    SSL_CTX_free(ctx);   // SSL 持有 ctx 引用
    if (!ssl) return false;
    SSL_set_fd(ssl, static_cast<int>(fd));
    SSL_set_tlsext_host_name(ssl, host.c_str());   // SNI
    X509_VERIFY_PARAM* vp = SSL_get0_param(ssl);
    X509_VERIFY_PARAM_set1_host(vp, host.c_str(), host.size());   // hostname 校验
    SSL_set_connect_state(ssl);
    if (SSL_connect(ssl) != 1) {
        LOG_WARN("llm sse: tls handshake failed: %s",
                 ERR_error_string(ERR_get_error(), nullptr));
        return false;
    }
    return true;
}

bool write_all(SSL* ssl, SockT fd, const char* data, size_t len, int timeout_ms) {
    size_t off = 0;
    while (off < len) {
        if (ssl) {
            if (!wait_io(fd, timeout_ms, true)) return false;
            const int n = SSL_write(ssl, data + off, static_cast<int>(len - off));
            if (n <= 0) return false;
            off += static_cast<size_t>(n);
        } else {
            const int n = ::send(fd, data + off, static_cast<int>(len - off), 0);
            if (n <= 0) return false;
            off += static_cast<size_t>(n);
        }
    }
    return true;
}

} // namespace

bool llm_sse_post(const std::string& cli_base, const std::string& path_prefix,
                  const std::string& api_key, const std::string& body_json,
                  const std::function<bool(const std::string&)>& on_delta) {
    if (on_delta == nullptr) return false;
    SockT fd = INVALID_SOCKET_VAL;
    SSL* ssl = nullptr;
    std::string host;
    if (!connect_and_tls(cli_base, fd, ssl, host)) {
        if (fd != INVALID_SOCKET_VAL) CLOSE_SOCK(fd);
        return false;
    }

    // 请求行/头：Connection: close 简化收尾（读完整响应即断，无需 keep-alive）
    std::string req = "POST " + path_prefix + "/chat/completions HTTP/1.1\r\n"
                      "Host: " + host + "\r\n"
                      "Authorization: Bearer " + api_key + "\r\n"
                      "Content-Type: application/json\r\n"
                      "Content-Length: " + std::to_string(body_json.size()) + "\r\n"
                      "Connection: close\r\n"
                      "\r\n" + body_json;
    if (!write_all(ssl, fd, req.data(), req.size(), 5000)) {
        LOG_WARN("llm sse: write request failed");
        goto fail;
    }

    // 响应头
    {
        std::string leftover;
        std::string line;
        int status = 0;
        std::string transfer_encoding;
        long long content_length = -1;
        for (;;) {
            if (!read_line(ssl, fd, leftover, line, 30000)) goto fail;
            if (line.empty()) break;                    // 头结束
            if (line.rfind("HTTP/1.", 0) == 0) {
                // HTTP/1.1 200 OK
                if (line.size() > 12 && line[9] >= '0' && line[9] <= '9' &&
                    line[10] >= '0' && line[10] <= '9' && line[11] >= '0' &&
                    line[11] <= '9') {
                    status = (line[9] - '0') * 100 + (line[10] - '0') * 10 +
                             (line[11] - '0');
                }
            } else if (line.rfind("transfer-encoding:", 0) == 0 ||
                       line.rfind("Transfer-Encoding:", 0) == 0) {
                transfer_encoding = line.substr(line.find(':') + 1);
            } else if (line.rfind("content-length:", 0) == 0 ||
                       line.rfind("Content-Length:", 0) == 0) {
                content_length = std::atoll(line.substr(line.find(':') + 1).c_str());
            }
        }
        if (status < 200 || status >= 300) {
            LOG_WARN("llm sse: http status %d", status);
            goto fail;
        }
        SseParser parser{std::string(), on_delta};
        if (transfer_encoding.find("chunked") != std::string::npos) {
            // 逐 chunk：hex size 行 → size 字节 → CRLF
            for (;;) {
                std::string sz;
                if (!read_line(ssl, fd, leftover, sz, 30000)) goto fail;
                const size_t semi = sz.find(';');      // 去掉分块扩展
                if (semi != std::string::npos) sz = sz.substr(0, semi);
                const long long n = std::strtoll(sz.c_str(), nullptr, 16);
                if (n <= 0) break;                      // 最后一个 chunk
                long long got = 0;
                char buf[8192];
                while (got < n) {
                    const int r = sock_read_buf(ssl, fd, leftover, buf,
                        static_cast<size_t>(std::min<long long>(sizeof buf, n - got)),
                        30000);
                    if (r <= 0) goto fail;
                    if (!parser.feed(buf, static_cast<size_t>(r))) goto fail;
                    got += r;
                }
                char crlf[2];
                if (sock_read_buf(ssl, fd, leftover, crlf, 2, 30000) != 2) goto fail;
            }
        } else if (content_length >= 0) {
            long long got = 0;
            char buf[8192];
            while (got < content_length) {
                const int r = sock_read_buf(ssl, fd, leftover, buf,
                    static_cast<size_t>(std::min<long long>(sizeof buf,
                                                            content_length - got)),
                    30000);
                if (r <= 0) goto fail;
                if (!parser.feed(buf, static_cast<size_t>(r))) goto fail;
                got += r;
            }
        } else {
            // 无长度无 chunked：读到连接关闭（Connection: close）
            char buf[8192];
            for (;;) {
                const int r = sock_read_buf(ssl, fd, leftover, buf, sizeof buf, 30000);
                if (r <= 0) break;
                if (!parser.feed(buf, static_cast<size_t>(r))) goto fail;
            }
        }
        if (!parser.got_any) {
            LOG_WARN("llm sse: stream ended without any content");
            goto fail;
        }
        if (ssl) SSL_shutdown(ssl);
        SSL_free(ssl);
        CLOSE_SOCK(fd);
        return true;
    }

fail:
    if (ssl) { SSL_shutdown(ssl); SSL_free(ssl); }
    CLOSE_SOCK(fd);
    return false;
}

} // namespace rag

#else   // !CPPHTTPLIB_OPENSSL_SUPPORT（Linux CI / 无 OpenSSL 构建 → 降级）

namespace rag {
bool llm_sse_post(const std::string&, const std::string&, const std::string&,
                  const std::string&,
                  const std::function<bool(const std::string&)>&) {
    return false;
}
} // namespace rag

#endif
