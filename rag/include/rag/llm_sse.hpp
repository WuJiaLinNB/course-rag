#pragma once
#include <functional>
#include <string>

namespace rag {

// 极简 HTTPS SSE 客户端（OpenAI 兼容 /chat/completions?stream=true 专用）。
//
// 为什么不用 httplib Client：0.15.3 / 0.18.0 只有 Get 支持流式 ContentReceiver，
// Post 返回时响应体已被 detail::read_content 完整读入内存，而 OpenAI 兼容接口是
// POST——httplib 无法流式读 LLM 增量。此实现是"HTTP/1.1 + TLS(OpenSSL) +
// chunked 解码 + SSE 边界解析"的最小集，范围刻意收窄：单次请求、
// Connection: close、无重定向、无压缩、失败即终止（fail-fast）。
//
// 仅当编译期定义 CPPHTTPLIB_OPENSSL_SUPPORT 时提供真实实现（Windows vendored
// OpenSSL 由 CMake 定义，证书校验走 Windows 系统证书存储，与 httplib 0.18 同方案）；
// 未定义时（Linux CI 等）恒返回 false，调用方走 llm_ok=false 降级路径。
//
// 返回 true = 流完整结束且至少收到一个增量块；false = 连接失败/超时/非 2xx/
// 零增量/on_delta 中止。on_delta 收到完整增量文本，返回 false 立即断开。
bool llm_sse_post(const std::string& cli_base,
                  const std::string& path_prefix,
                  const std::string& api_key,
                  const std::string& body_json,
                  const std::function<bool(const std::string&)>& on_delta);

} // namespace rag
