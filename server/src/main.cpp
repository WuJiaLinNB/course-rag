// course-rag HTTP 服务（Task 12 + Task 15）：httplib + nlohmann 单头，对接 core::Engine 与 rag::Pipeline。
//
// 启动流程：server_config.json → data/chunks.json 内存副本（下标 = id）→ 读 vectors.bin 16 字节头
// 拿 dim → Engine 构造 + load（一致性校验在 Engine 内，失败退出）→ 监听 127.0.0.1。
// 配置全部收敛在 server_config.json（j.at() 语义：字段缺失/类型错直接抛异常，fail fast；
// 真实配置本地持有，模板 server_config.example.json 进 git，真实文件被 .gitignore 排除）。
//
// 鉴权（Task 15）：/search /ask /documents 包 Bearer token 中间件（常数时间比较，缺/错一律
// 401 不区分）；/healthz 与静态页不鉴权。静态页 web/index.html 经 set_mount_point("/", "web")
// 托管，API 路由优先于静态文件。
//
// 优雅停机：控制台回调（独立线程）只调 svr.stop() 停止接收新请求；listen 返回后主线程
// wait_rebuild → dirty 时先原子写 chunks.json 再 persist vectors.bin。回调里绝不做 flush。
//
// v1 限制已解除（Task 16 前置）：CMake 探测到 third_party/openssl 时定义
// CPPHTTPLIB_OPENSSL_SUPPORT，httplib 客户端可调 https 的 embedding/LLM API
// （仅出站客户端；入站 TLS 由公网隧道层终结，server 本体继续只说 HTTP）。
#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>   // SetConsoleCtrlHandler / MoveFileExA
#endif

#include <httplib.h>
#include <json.hpp>

#include <core/engine.hpp>
#include <core/log.hpp>
#include <core/types.hpp>
#include <rag/pipeline.hpp>
#include <rag/llm_sse.hpp>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <fstream>
#include <functional>
#include <iterator>
#include <limits>
#include <mutex>
#include <shared_mutex>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr size_t kPayloadMax = 64 * 1024 * 1024 + 1024;   // 全局入库包上限，超限 httplib 自动 413
constexpr size_t kQueryBodyMax = 64 * 1024;               // /search /ask 请求体上限（路由内自查）
constexpr size_t kQueryCharMax = 2000;                    // query 字符数上限（按 UTF-8 字符计）
constexpr size_t kHistoryTurnMax = 20;                    // 多轮历史最大轮数
constexpr size_t kHistoryMsgCharMax = 4000;               // 历史单条消息最大字符数（按 UTF-8 字符计）
constexpr const char* kConfigPath = "server_config.json"; // 真实配置（.gitignore 排除），模板见 server_config.example.json

// chunks.json 内存副本的并发保护：/documents 追加会让 vector 重新分配，与 Pipeline
// 的 fetch_*（按 id 读）并发是数据竞争 → 写用 unique_lock、读用 shared_lock。
// 追加只改尾部，已有 id 的数据不变，锁内访问即可保证安全。
std::shared_mutex g_store_mtx;

httplib::Server* g_svr = nullptr;      // 控制台回调里只调 svr.stop()（线程安全）
std::atomic<bool> g_shutdown{false};
// 不可服务标志：/documents 入库中途异常会让 engine 与内存副本 id 永久错位（无法就地
// 修复），置位后三个业务路由一律 503，进程保持存活等待停机，且不再 flush（见停机处）
std::atomic<bool> g_service_broken{false};

// ---- 启动辅助 ------------------------------------------------------------

// 配置结构 + 加载：server_config.json，j.at() 语义（字段缺失/类型错抛异常），
// fail fast——清晰报错优于带错启动
struct AppConfig {
    int port = 0;
    std::string token;
    std::string emb_base, emb_key, emb_model;
    std::string llm_base, llm_key, llm_model;
    std::string vectors_path, chunks_path;
};

AppConfig load_config(const std::string& path) {
    std::ifstream f(path);
    if (!f) throw std::runtime_error("cannot open config: " + path);
    const auto j = nlohmann::json::parse(f);
    AppConfig c;
    c.port         = j.at("server").at("port").get<int>();
    c.token        = j.at("server").at("auth_token").get<std::string>();
    c.emb_base     = j.at("embedding").at("base_url").get<std::string>();
    c.emb_key      = j.at("embedding").at("api_key").get<std::string>();
    c.emb_model    = j.at("embedding").at("model").get<std::string>();
    c.llm_base     = j.at("llm").at("base_url").get<std::string>();
    c.llm_key      = j.at("llm").at("api_key").get<std::string>();
    c.llm_model    = j.at("llm").at("model").get<std::string>();
    c.vectors_path = j.at("data").at("vectors").get<std::string>();
    c.chunks_path  = j.at("data").at("chunks").get<std::string>();
    return c;
}

// 鉴权中间件（Task 15，设计文档第 11 节）：横切面收在包装函数里，三个业务路由
// 各包一层，路由内部代码零改动。/healthz 与静态页不包鉴权。

// 常数时间比较：逐位异或累计差，无提前返回——防时序攻击
bool ct_equal(const std::string& a, const std::string& b) {
    if (a.size() != b.size()) return false;
    volatile uint8_t diff = 0;
    for (size_t i = 0; i < a.size(); ++i) diff |= static_cast<uint8_t>(a[i]) ^ static_cast<uint8_t>(b[i]);
    return diff == 0;
}

static std::atomic<uint64_t> g_unauthorized{0};

// 缺 token / 错 token 一律 401，不区分原因（不向未授权者泄露信息）
template <typename F>
auto with_auth(F handler, const std::string& token) {
    return [handler, token](const httplib::Request& req, httplib::Response& res) {
        static const std::string prefix = "Bearer ";
        auto it = req.headers.find("Authorization");
        if (it == req.headers.end() || it->second.rfind(prefix, 0) != 0
            || !ct_equal(it->second.substr(prefix.size()), token)) {
            ++g_unauthorized;                    // 401 计数（不记 token 内容）
            LOG_WARN("401 unauthorized, total=%llu",
                     static_cast<unsigned long long>(g_unauthorized.load()));
            res.status = 401;
            return;
        }
        handler(req, res);
    };
}

// 只读 vectors.bin 16 字节头拿 dim（Engine 构造函数需要）；magic/版本/字节数/条数的
// 完整校验在 Engine::load 内做，这里只负责"拿得到 dim"并给出明确报错
uint32_t read_dim_from_header(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) {
        LOG_ERROR("cannot open %s", path.c_str());
        std::exit(1);
    }
    unsigned char hdr[16];
    f.read(reinterpret_cast<char*>(hdr), static_cast<std::streamsize>(sizeof(hdr)));
    if (!f) {
        LOG_ERROR("%s header incomplete (need 16 bytes)", path.c_str());
        std::exit(1);
    }
    if (std::memcmp(hdr, "CRV1", 4) != 0) {
        LOG_ERROR("%s bad magic: not a CRV1 vectors file", path.c_str());
        std::exit(1);
    }
    // 头部小端显式拼装，不依赖本机字节序：magic(4) version(4) dim(4) count(4)
    const uint32_t dim = static_cast<uint32_t>(hdr[8]) |
                         static_cast<uint32_t>(hdr[9]) << 8 |
                         static_cast<uint32_t>(hdr[10]) << 16 |
                         static_cast<uint32_t>(hdr[11]) << 24;
    if (dim == 0) {
        LOG_ERROR("%s header dim == 0", path.c_str());
        std::exit(1);
    }
    return dim;
}

// 解析 chunks.json 为内存副本：JSON 数组第 i 项 = 第 i 个向量的元数据 + 原文，
// 数组下标即向量 id（tools/embedder.py 生成的格式；JSON "type" 映射 ChunkMeta::type_）
void load_chunks(const std::string& path,
                 std::vector<core::ChunkMeta>& metas,
                 std::vector<std::string>& contents) {
    std::ifstream f(path, std::ios::binary);
    if (!f) {
        LOG_ERROR("cannot open %s", path.c_str());
        std::exit(1);
    }
    const std::string buf((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    // 外部数据解析不抛异常（allow_exceptions=false），形状不对 fail fast
    const auto j = nlohmann::json::parse(buf, nullptr, /*allow_exceptions=*/false);
    if (j.is_discarded() || !j.is_array()) {
        LOG_ERROR("%s is not a JSON array", path.c_str());
        std::exit(1);
    }
    metas.reserve(j.size());
    contents.reserve(j.size());
    for (size_t i = 0; i < j.size(); ++i) {
        const auto& it = j[i];
        if (!it.is_object()) {
            LOG_ERROR("%s item %zu is not an object", path.c_str(), i);
            std::exit(1);
        }
        for (const char* k : {"course", "semester", "type", "title", "content"}) {
            if (!it.contains(k) || !it[k].is_string()) {
                LOG_ERROR("%s item %zu missing or non-string field \"%s\"", path.c_str(), i, k);
                std::exit(1);
            }
        }
        metas.push_back({it["course"].get<std::string>(), it["semester"].get<std::string>(),
                         it["type"].get<std::string>(), it["title"].get<std::string>()});
        contents.push_back(it["content"].get<std::string>());
    }
    LOG_INFO("loaded %zu chunk metas from %s", metas.size(), path.c_str());
}

// chunks.json 落盘：与 core/src/vector_store.cpp 的 atomic_replace 同一套路——
// 全部写完后一步替换（Windows MoveFileExA + MOVEFILE_REPLACE_EXISTING），绝不留半个正式文件
void atomic_write_chunks(const std::string& path,
                         const std::vector<core::ChunkMeta>& metas,
                         const std::vector<std::string>& contents) {
    nlohmann::json arr = nlohmann::json::array();
    for (size_t i = 0; i < metas.size(); ++i) {
        arr.push_back({
            {"course", metas[i].course},
            {"semester", metas[i].semester},
            {"type", metas[i].type_},   // ChunkMeta::type_ ↔ JSON "type"
            {"title", metas[i].title},
            {"content", contents[i]}});
    }
    const std::string tmp = path + ".tmp";
    {
        std::ofstream f(tmp, std::ios::binary | std::ios::trunc);
        if (!f) throw std::runtime_error("cannot write: " + tmp);
        f << arr.dump(1);   // indent=1 且非 ASCII 不转义，与 tools/embedder.py 的输出一致
        f.flush();
        if (!f) throw std::runtime_error("write failed: " + tmp);
    }                       // 离开作用域 = 文件确实关闭落盘，之后才替换
    // Windows-only：MoveFileExA 同卷原子替换已存在目标文件。项目不构建 POSIX 目标
    // （console handler 等处本就无 _WIN32 守卫），不留 std::rename 的"看似可移植实则
    // 编译不过"死分支
    if (!MoveFileExA(tmp.c_str(), path.c_str(), MOVEFILE_REPLACE_EXISTING))
        throw std::runtime_error("atomic replace failed: " + path);
}

// ---- 请求处理辅助 --------------------------------------------------------

// UTF-8 字符数（只数首字节）："query 长度 > 2000 字符"按字符而非字节计，
// 否则 2000 字节的中文只有约 666 个字，限制会严苛三倍
size_t utf8_length(const std::string& s) {
    size_t n = 0;
    for (const unsigned char c : s)
        if ((c & 0xC0) != 0x80) ++n;
    return n;
}

// 统一错误响应：所有错误都带 JSON body，信息里点名具体字段
void send_error(httplib::Response& res, int status, const std::string& msg) {
    res.status = status;
    res.set_content(nlohmann::json{{"error", msg}}.dump(), "application/json");
}

// /search 与 /ask 共用的请求校验：请求体大小 + JSON 解析 + query/top_k 合法性。
// 失败时填好 400 响应并返回 false；成功时通过出参带出 query/top_k 与解析后的 JSON
// （j_out 供 /ask 路由继续读可选 history 字段，/search 不用）。
bool parse_search_like_body(const httplib::Request& req, httplib::Response& res,
                            std::string& query, long long& top_k,
                            nlohmann::json& j_out) {
    if (req.body.size() > kQueryBodyMax) {
        send_error(res, 400, "request body exceeds " + std::to_string(kQueryBodyMax)
                               + " bytes limit");
        return false;
    }
    const auto j = nlohmann::json::parse(req.body, nullptr, /*allow_exceptions=*/false);
    if (j.is_discarded() || !j.is_object()) {
        send_error(res, 400, "invalid JSON: expected object with fields \"query\" and \"top_k\"");
        return false;
    }
    if (!j.contains("query") || !j["query"].is_string()) {
        send_error(res, 400, "missing or non-string field \"query\"");
        return false;
    }
    query = j["query"].get<std::string>();
    if (utf8_length(query) > kQueryCharMax) {
        send_error(res, 400, "field \"query\" exceeds 2000 characters");
        return false;
    }
    if (!j.contains("top_k") || !j["top_k"].is_number_integer()) {
        send_error(res, 400, "missing or non-integer field \"top_k\"");
        return false;
    }
    // 无符号超范围值直接钳到 LLONG_MAX（get<long long> 对越界无符号是实现定义行为）
    if (j["top_k"].is_number_unsigned()) {
        const uint64_t u = j["top_k"].get<uint64_t>();
        top_k = u > static_cast<uint64_t>(std::numeric_limits<long long>::max())
                    ? std::numeric_limits<long long>::max()
                    : static_cast<long long>(u);
    } else {
        top_k = j["top_k"].get<long long>();
    }
    if (top_k <= 0) {
        send_error(res, 400, "field \"top_k\" must be a positive integer");
        return false;
    }
    j_out = j;
    return true;
}

// 解析可选多轮历史（功能 D）：/ask /ask/stream 请求体里的 "history" 字段，缺省 = 单轮。
// 格式：JSON 数组，每项 {user, assistant} 均为字符串，按时间序先到先放。
// 校验硬约束（外部输入边界）：轮数 ≤ kHistoryTurnMax、单条消息 ≤ kHistoryMsgCharMax
// （按字符计，与 query 同口径）——历史随每轮请求透传进 LLM 上下文，无界历史会让
// 请求体与 LLM 输入一起膨胀。成功返回 true（缺省也为 true，history 为空）。
bool parse_history(const nlohmann::json& j, rag::ChatHistory& history) {
    if (!j.contains("history")) return true;
    const auto& h = j["history"];
    if (!h.is_array() || h.size() > kHistoryTurnMax) return false;
    for (size_t i = 0; i < h.size(); ++i) {
        const auto& it = h[i];
        if (!it.is_object() || !it.contains("user") || !it["user"].is_string() ||
            !it.contains("assistant") || !it["assistant"].is_string()) {
            return false;
        }
        const std::string u = it["user"].get<std::string>();
        const std::string a = it["assistant"].get<std::string>();
        if (utf8_length(u) > kHistoryMsgCharMax || utf8_length(a) > kHistoryMsgCharMax)
            return false;
        history.emplace_back(std::move(u), std::move(a));
    }
    return true;
}

// 在线单条 embedding：与 rag/src/pipeline.cpp 的 call_llm 同款 fail fast——
// 连接/读超时 5s、失败不重试；任何失败返回空 vector，由路由层转 502。
// 请求体用 nlohmann 构造（query 含任意文本，手拼 JSON 必漏转义），响应解析不抛异常。
std::vector<float> embed_query_once(const std::string& base_url,
                                    const std::string& api_key,
                                    const std::string& model,
                                    const std::string& query) {
    // httplib 0.15.3 Client 构造不接受带路径的 base_url（见 rag/pipeline.hpp
    // split_base_url 注释）：base_url 如 "https://api.siliconflow.cn/v1" 时先拆出
    // 可连部分 + 路径前缀，请求路径 = prefix + "/embeddings"
    const auto [cli_base, prefix] = rag::split_base_url(base_url);
    httplib::Client cli(cli_base);
    cli.set_connection_timeout(5);
    cli.set_read_timeout(5);
    if (!api_key.empty()) cli.set_bearer_token_auth(api_key);

    nlohmann::json body;
    body["model"] = model;
    body["input"] = nlohmann::json::array({query});
    const auto res = cli.Post(prefix + "/embeddings", body.dump(-1, ' ', false,
                              nlohmann::json::error_handler_t::replace), "application/json");
    if (!res) {   // 连接失败/超时/不支持 http scheme
        LOG_WARN("embed request failed: %s", httplib::to_string(res.error()).c_str());
        return {};
    }
    if (res->status < 200 || res->status >= 300) {
        LOG_WARN("embed http status %d", static_cast<int>(res->status));
        return {};
    }
    const auto jr = nlohmann::json::parse(res->body, nullptr, /*allow_exceptions=*/false);
    if (jr.is_discarded() || jr.contains("error")) {
        LOG_WARN("embed response not valid json or has error");
        return {};
    }
    if (!jr.contains("data") || !jr["data"].is_array() || jr["data"].empty()) {
        LOG_WARN("embed response missing data");
        return {};
    }
    const auto& first = jr["data"][0];
    if (!first.is_object() || !first.contains("embedding") || !first["embedding"].is_array()
        || first["embedding"].empty()) {
        LOG_WARN("embed response missing embedding");
        return {};
    }
    std::vector<float> vec;
    vec.reserve(first["embedding"].size());
    for (const auto& x : first["embedding"]) {
        if (!x.is_number()) {
            LOG_WARN("embedding element is not a number");
            return {};
        }
        vec.push_back(x.get<float>());
    }
    return vec;
}

// 优雅停机回调：运行在系统分配的独立线程，只做 svr.stop()（线程安全）；
// flush 链路（wait_rebuild → 写 chunks.json → persist）留在 listen 返回后的主线程。
// CTRL_CLOSE/LOGOFF/SHUTDOWN 同样只调 stop()：这三类事件 Windows 给的约 5s 宽限期
// 属于 handler 线程的执行时间，handler 一返回进程即被强杀——本实现 handler 立即返回，
// 主线程 flush 是与强杀"赛跑"，不保证完成；但 flush 本身事务性（tmp+原子替换），
// 中途被杀不留半截文件，最坏退化为"本次新增未落盘"，不会写出损坏数据
BOOL WINAPI console_ctrl_handler(DWORD ctrl_type) {
    switch (ctrl_type) {
    case CTRL_C_EVENT:
    case CTRL_BREAK_EVENT:
    case CTRL_CLOSE_EVENT:
    case CTRL_LOGOFF_EVENT:
    case CTRL_SHUTDOWN_EVENT:
        if (!g_shutdown.exchange(true)) {
            LOG_INFO("console ctrl %lu received, stopping accept loop...",
                     static_cast<unsigned long>(ctrl_type));
            if (g_svr) g_svr->stop();
        }
        return TRUE;   // 已处理，屏蔽默认强杀，让主线程走优雅 flush
    default:
        return FALSE;
    }
}

} // namespace

int main() {
    // ---- 1. 配置文件（server_config.json；缺失/字段错即 fail fast）----
    AppConfig cfg;
    try {
        cfg = load_config(kConfigPath);
    } catch (const std::exception& e) {
        LOG_ERROR("config load failed: %s", e.what());
        LOG_ERROR("recovery hint: copy server_config.example.json to server_config.json "
                  "and fill in real values");
        return 1;
    }
    if (cfg.port <= 0 || cfg.port > 65535) {
        LOG_ERROR("invalid server.port in %s: %d", kConfigPath, cfg.port);
        return 1;
    }

    // ---- 2. chunks.json 内存副本（下标 = id，供 fetch_chunk / fetch_meta / 结果回填）----
    std::vector<core::ChunkMeta> metas;
    std::vector<std::string> contents;
    load_chunks(cfg.chunks_path, metas, contents);

    // ---- 3. vectors.bin 头部取 dim → Engine 构造 + load（校验失败即退出）----
    const uint32_t dim = read_dim_from_header(cfg.vectors_path);
    core::Engine engine(dim, core::Engine::Mode::Auto);
    try {
        engine.load(cfg.vectors_path, metas);   // magic/版本/维度/字节数/跨文件条数校验都在 Engine 内
    } catch (const std::exception& e) {
        LOG_ERROR("startup aborted: %s", e.what());
        // 恢复提示：count mismatch 是最常见的数据不一致，指向重建路径
        LOG_ERROR("recovery hint: if this is a count mismatch, re-run tools/embedder.py "
                  "to rebuild data/, or fix vectors.bin / chunks.json so entry counts match");
        return 1;
    }
    LOG_INFO("engine loaded %zu vectors (dim=%u, index=%s)", engine.size(), dim,
             engine.active_index_name().c_str());

    // ---- 4. 共用 embed_query（/search 路由与 Pipeline 共用同一函数对象）----
    // 线程内记忆化：/ask 路由要先探测 embedding 可用性（失败 502），Pipeline::ask 内部
    // 还会再调一次——thread_local 缓存让同一请求内的第二次调用命中缓存，避免对 embedding
    // 服务连发两次。httplib 每连接一线程、单请求全程在同一线程内处理，缓存无竞争。
    // 失败结果不缓存（空向量不命中），下次请求会重试真实服务。
    // 隐含前提 (a)：命中判定是整串相等，依赖 Pipeline 原样透传同一 query 字符串；未来若
    // Pipeline 对 query 做 trim/改写，只是退化为多一次网络调用，不影响正确性。
    // 隐含前提 (b)：缓存键只有 query、未含模型/服务配置，当前配置进程级固定所以正确；
    // 未来若支持 per-request 换模型，缓存键需把配置一并纳入。
    std::function<std::vector<float>(const std::string&)> embed_query =
        [emb_base = cfg.emb_base, emb_key = cfg.emb_key,
         emb_model = cfg.emb_model](const std::string& q) -> std::vector<float> {
        static thread_local std::string cached_q;
        static thread_local std::vector<float> cached_v;
        if (!cached_v.empty() && cached_q == q) return cached_v;
        std::vector<float> v = embed_query_once(emb_base, emb_key, emb_model, q);
        if (!v.empty()) {
            cached_q = q;
            cached_v = v;
        }
        return v;
    };

    // ---- 5. Pipeline（engine 声明在前，保证比 pipeline 活得久）----
    // 流式 LLM 出口（/ask/stream 用）：rag::llm_sse_post 完成 TLS + HTTP/1.1 +
    // chunked + SSE 解析（httplib 客户端不支持流式 POST，见 rag/llm_sse.hpp 注释）。
    // 超时语义沿用硬约束：连接 5s、读等待 30s = 两次数据块之间的最长等待，长回答靠
    // 每块间隔 < 30s 维持不断；超过即 fail-fast 终止，不重试。
    // on_delta 返回 false（客户端断开）时立即断开 LLM 流。
    // history（功能 D）：历史轮次按 user/assistant 原样前置，当前轮 prompt 收尾
    // ——与 Pipeline::call_llm 的 messages 构造保持一致（见 pipeline.cpp）。
    rag::Pipeline::LlmStreamFn llm_stream =
        [llm_base = cfg.llm_base, llm_key = cfg.llm_key, llm_model = cfg.llm_model](
            const std::string& prompt, const rag::ChatHistory& history,
            const std::function<bool(const std::string&)>& on_delta) -> bool {
        const auto [cli_base, prefix] = rag::split_base_url(llm_base);
        nlohmann::json body;
        body["model"] = llm_model;
        nlohmann::json messages = nlohmann::json::array();
        for (const auto& [q, a] : history) {
            messages.push_back({{"role", "user"}, {"content", q}});
            messages.push_back({{"role", "assistant"}, {"content", a}});
        }
        messages.push_back({{"role", "user"}, {"content", prompt}});
        body["messages"] = std::move(messages);
        body["stream"] = true;
        body["temperature"] = 0.3;
        return rag::llm_sse_post(cli_base, prefix, llm_key,
                                 body.dump(-1, ' ', false,
                                           nlohmann::json::error_handler_t::replace),
                                 on_delta);
    };
    rag::Pipeline pipeline(engine,
        rag::LlmConfig{cfg.llm_base, cfg.llm_key, cfg.llm_model},
        embed_query,
        [&contents](uint32_t id) -> std::string {
            std::shared_lock lk(g_store_mtx);
            if (id >= contents.size())   // id 恒有效，越界属编程错误，抛异常兜底
                throw std::out_of_range("fetch_chunk: id " + std::to_string(id) + " out of range");
            return contents[id];
        },
        [&metas](uint32_t id) -> core::ChunkMeta {
            std::shared_lock lk(g_store_mtx);
            if (id >= metas.size())
                throw std::out_of_range("fetch_meta: id " + std::to_string(id) + " out of range");
            return metas[id];
        }, llm_stream);

    // ---- 6. 路由 ----
    httplib::Server svr;
    g_svr = &svr;

    // 404/405/413 等 httplib 生成的错误响应也带 JSON body（不覆盖 handler 已写的错误体）
    svr.set_error_handler([](const auto&, auto& res) {
        if (res.body.empty())
            res.set_content(nlohmann::json{{"error", "http " + std::to_string(res.status)}}.dump(),
                            "application/json");
    });
    // 兜底：handler 内未捕获的异常不让整个进程 terminate
    svr.set_exception_handler([](const auto&, auto& res, std::exception_ptr ep) {
        try {
            if (ep) std::rethrow_exception(ep);
        } catch (const std::exception& e) {
            LOG_ERROR("unhandled exception: %s", e.what());
        } catch (...) {
            // 非 std 异常（原始类型 / 第三方库异常）也记一笔，避免静默 terminate 风险
            LOG_ERROR("unhandled non-std exception");
        }
        send_error(res, 500, "internal error");
    });
    svr.set_logger([](const auto& req, const auto& res) {
        LOG_INFO("%s %s -> %d", req.method.c_str(), req.path.c_str(), static_cast<int>(res.status));
    });
    // 全局放行入库包：超限由 httplib 自动 413（/search /ask 另有路由内 64KiB 自查）
    svr.set_payload_max_length(kPayloadMax);

    // GET /healthz：无鉴权存活探针
    svr.Get("/healthz", [&](const httplib::Request&, httplib::Response& res) {
        nlohmann::json out = {
            {"status", "ok"},
            {"index", engine.active_index_name()},
            {"index_ready", engine.index_ready()}};
        res.set_content(out.dump(), "application/json");
    });

    // POST /search：embed → engine.search → 结果带元数据与原文（包鉴权）
    auto search_handler = [&](const httplib::Request& req, httplib::Response& res) {
        if (g_service_broken.load()) {   // 入库中途异常后的不可服务状态（见 g_service_broken）
            send_error(res, 503, "server state inconsistent, restart required");
            return;
        }
        std::string query;
        long long top_k = 0;
        nlohmann::json body_j;
        if (!parse_search_like_body(req, res, query, top_k, body_j)) return;

        // 超容量 clamp 到当前条数；空索引时 k=0，直接返回空结果，不进 engine.search
        // （不依赖 core 的边界语义）
        const size_t k = static_cast<size_t>(
            std::min<long long>(top_k, static_cast<long long>(engine.size())));

        // 空索引：k 必为 0，跳过 embed 直接返回空结果——省一次注定失败的外网调用。
        // 与 /ask 的语义差异：/ask 必须产出 LLM 答案，对空资料区发起只会白费一次 LLM
        // 调用故返回 503；/search 空集是合法的"0 命中"，无需外呼即可完整响应
        if (engine.size() == 0) {
            nlohmann::json out = {
                {"index", engine.active_index_name()},
                {"index_ready", engine.index_ready()},
                {"top_k_effective", k},
                {"results", nlohmann::json::array()}};
            res.set_content(out.dump(), "application/json");
            return;
        }

        const std::vector<float> qv = embed_query(query);
        if (qv.empty()) {
            send_error(res, 502, "embedding request failed");
            return;
        }
        if (qv.size() != static_cast<size_t>(dim)) {   // 防 BruteIndex::search 的 dim 校验抛异常
            send_error(res, 502, "embedding dimension mismatch");
            return;
        }

        nlohmann::json results = nlohmann::json::array();
        double search_ms = 0.0;
        if (k > 0) {
            const auto t0 = std::chrono::steady_clock::now();
            const auto hits = engine.search(qv, k, {});   // v1：不做元数据过滤
            search_ms = std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - t0).count();
            std::shared_lock lk(g_store_mtx);
            for (const auto& h : hits) {
                results.push_back({
                    {"id", h.id},
                    {"similarity", h.similarity},
                    {"course", metas[h.id].course},
                    {"semester", metas[h.id].semester},
                    {"type", metas[h.id].type_},
                    {"title", metas[h.id].title},
                    {"content", contents[h.id]}});
            }
        }
        nlohmann::json out = {
            {"index", engine.active_index_name()},
            {"index_ready", engine.index_ready()},
            {"top_k_effective", k},
            {"n_vectors", engine.size()},
            {"search_ms", search_ms},
            {"results", std::move(results)}};
        res.set_content(out.dump(), "application/json");
    };
    svr.Post("/search", with_auth(search_handler, cfg.token));

    // POST /ask：embed（失败 502）→ Pipeline::ask（LLM 失败内部降级，仍 200）（包鉴权）
    // 引用列表序列化（/ask 与 /ask/stream 共用）
    const auto build_citations_json = [](const std::vector<rag::Citation>& citations) {
        nlohmann::json arr = nlohmann::json::array();
        for (const auto& c : citations) {
            arr.push_back({
                {"id", c.id},
                {"course", c.course},
                {"semester", c.semester},
                {"type", c.type_},
                {"title", c.title},
                {"similarity", c.similarity}});
        }
        return arr;
    };
    auto ask_handler = [&](const httplib::Request& req, httplib::Response& res) {
        if (g_service_broken.load()) {
            send_error(res, 503, "server state inconsistent, restart required");
            return;
        }
        std::string query;
        long long top_k = 0;
        nlohmann::json body_j;
        if (!parse_search_like_body(req, res, query, top_k, body_j)) return;
        rag::ChatHistory history;
        if (!parse_history(body_j, history)) {
            send_error(res, 400, "invalid field \"history\": expected array of "
                       "{\"user\", \"assistant\"} string pairs, at most "
                       + std::to_string(kHistoryTurnMax) + " turns, each message at most "
                       + std::to_string(kHistoryMsgCharMax) + " characters");
            return;
        }

        // 空索引直接 503：问答必须产出 LLM 答案，对空资料区发起只会白费一次 LLM 调用
        // （/search 空集可无外呼返回 200 空结果，两路由语义不同故处理不同）
        const size_t total = engine.size();
        if (total == 0) {
            send_error(res, 503, "index is empty");
            return;
        }
        const size_t k = static_cast<size_t>(
            std::min<long long>(top_k, static_cast<long long>(total)));

        // 先探测 embedding：失败 502；成功则 Pipeline 内的第二次调用命中线程内缓存
        const std::vector<float> qv = embed_query(query);
        if (qv.empty()) {
            send_error(res, 502, "embedding request failed");
            return;
        }
        if (qv.size() != static_cast<size_t>(dim)) {
            send_error(res, 502, "embedding dimension mismatch");
            return;
        }

        const rag::AskResult r = pipeline.ask(query, k, history);
        nlohmann::json out = {
            {"answer", r.answer},
            {"llm_ok", r.llm_ok},
            {"citations", build_citations_json(r.citations)},
            {"trace", {{"index", r.trace.index},
                       {"n_vectors", r.trace.n_vectors},
                       {"search_ms", r.trace.search_ms}}}};
        res.set_content(out.dump(), "application/json");
    };
    svr.Post("/ask", with_auth(ask_handler, cfg.token));

    // POST /ask/stream：SSE 流式问答（包鉴权）。事件序：meta（citations+trace，先于
    // 首字到达，上层先渲染引用）→ delta（LLM 增量文本，多帧）→ done（{llm_ok}）。
    // llm_ok=false 时前端展示降级文案。实现用 chunked transfer encoding 推 SSE 帧；
    // provider 运行在 httplib worker 线程，handler 返回后才执行，故 handler 只做校验。
    auto ask_stream_handler = [&](const httplib::Request& req, httplib::Response& res) {
        if (g_service_broken.load()) {
            send_error(res, 503, "server state inconsistent, restart required");
            return;
        }
        std::string query;
        long long top_k = 0;
        nlohmann::json body_j;
        if (!parse_search_like_body(req, res, query, top_k, body_j)) return;
        rag::ChatHistory history;
        if (!parse_history(body_j, history)) {
            send_error(res, 400, "invalid field \"history\": expected array of "
                       "{\"user\", \"assistant\"} string pairs, at most "
                       + std::to_string(kHistoryTurnMax) + " turns, each message at most "
                       + std::to_string(kHistoryMsgCharMax) + " characters");
            return;
        }
        const size_t total = engine.size();
        if (total == 0) {
            send_error(res, 503, "index is empty");
            return;
        }
        const size_t k = static_cast<size_t>(
            std::min<long long>(top_k, static_cast<long long>(total)));
        const std::vector<float> qv = embed_query(query);
        if (qv.empty()) {
            send_error(res, 502, "embedding request failed");
            return;
        }
        if (qv.size() != static_cast<size_t>(dim)) {
            send_error(res, 502, "embedding dimension mismatch");
            return;
        }
        // history/query 按值捕获进 provider lambda：chunked provider 运行在 httplib
        // worker 线程、handler 返回后才执行，值捕获保证其生命周期独立于本 handler 栈
        res.set_chunked_content_provider("text/event-stream",
            [&pipeline, &build_citations_json, query, k, history](
                size_t, httplib::DataSink& sink) -> bool {
                pipeline.ask_stream(query, k, history,
                    // on_meta：检索结果先于 LLM 首字推送，前端立即渲染引用与 trace
                    [&sink, &build_citations_json](const rag::AskResult& meta) -> bool {
                        nlohmann::json j;
                        j["citations"] = build_citations_json(meta.citations);
                        j["trace"] = {{"index", meta.trace.index},
                                      {"n_vectors", meta.trace.n_vectors},
                                      {"search_ms", meta.trace.search_ms}};
                        const std::string frame = "event: meta\ndata: " + j.dump() + "\n\n";
                        return sink.write(frame.data(), frame.size());
                    },
                    [&sink](const std::string& t) -> bool {
                        nlohmann::json j = {{"text", t}};
                        const std::string frame = "event: delta\ndata: " + j.dump() + "\n\n";
                        return sink.write(frame.data(), frame.size());
                    },
                    [&sink](bool ok) {
                        nlohmann::json j = {{"llm_ok", ok}};
                        const std::string frame = "event: done\ndata: " + j.dump() + "\n\n";
                        sink.write(frame.data(), frame.size());
                    });
                sink.done();
                return true;
            });
    };
    svr.Post("/ask/stream", with_auth(ask_stream_handler, cfg.token));

    // GET /chunk?id=N：按向量 id 取原文 + 元数据（引用卡片懒加载用）（包鉴权）。
    // id 来自 /ask 返回的 citations[].id；原文含任意文本，经 JSON 序列化返回，
    // 前端必须 textContent/转义渲染，禁止 innerHTML 直接拼接（XSS 约束）。
    auto chunk_handler = [&](const httplib::Request& req, httplib::Response& res) {
        auto it = req.params.find("id");
        if (it == req.params.end()) {
            send_error(res, 400, "missing id");
            return;
        }
        uint32_t id = 0;
        try {
            const unsigned long v = std::stoul(it->second);
            if (v > std::numeric_limits<uint32_t>::max()) throw std::out_of_range("id overflow");
            id = static_cast<uint32_t>(v);
        } catch (...) {
            send_error(res, 400, "invalid id");
            return;
        }
        std::shared_lock lk(g_store_mtx);
        if (id >= contents.size()) {
            send_error(res, 404, "chunk not found");
            return;
        }
        const auto& m = metas[id];
        nlohmann::json out = {
            {"id", id},
            {"course", m.course},
            {"semester", m.semester},
            {"type", m.type_},
            {"title", m.title},
            {"content", contents[id]}};
        res.set_content(out.dump(), "application/json");
    };
    svr.Get("/chunk", with_auth(chunk_handler, cfg.token));

    // POST /documents：收 chunk 包，先整体校验再逐条入库（要么全收要么全拒）（包鉴权）
    auto documents_handler = [&](const httplib::Request& req, httplib::Response& res) {
        if (g_service_broken.load()) {
            send_error(res, 503, "server state inconsistent, restart required");
            return;
        }
        // 重建窗口禁止写（Engine 契约：load 后台重建期间无写冲突，由本 503 保证）；
        // 拒绝时不得改动内存副本
        if (!engine.index_ready()) {
            send_error(res, 503, "index rebuilding, documents temporarily rejected");
            return;
        }
        const auto j = nlohmann::json::parse(req.body, nullptr, /*allow_exceptions=*/false);
        if (j.is_discarded() || !j.is_object()) {
            send_error(res, 400, "invalid JSON: expected object with field \"items\"");
            return;
        }
        if (!j.contains("items") || !j["items"].is_array()) {
            send_error(res, 400, "missing or non-array field \"items\"");
            return;
        }
        const auto& items = j["items"];

        // 先整体校验（维度、字段齐全、数值类型、零向量），任何一条不合格整体拒绝
        std::vector<std::vector<float>> vecs;
        std::vector<core::ChunkMeta> new_metas;
        std::vector<std::string> new_contents;
        vecs.reserve(items.size());
        new_metas.reserve(items.size());
        new_contents.reserve(items.size());
        for (size_t i = 0; i < items.size(); ++i) {
            const auto& it = items[i];
            const std::string at = "items[" + std::to_string(i) + "]";
            if (!it.is_object()) {
                send_error(res, 400, at + " is not an object");
                return;
            }
            if (!it.contains("vector") || !it["vector"].is_array()) {
                send_error(res, 400, at + " missing or non-array field \"vector\"");
                return;
            }
            if (it["vector"].size() != static_cast<size_t>(dim)) {
                send_error(res, 400, at + ".vector dimension "
                           + std::to_string(it["vector"].size()) + " != engine dimension "
                           + std::to_string(dim));
                return;
            }
            if (!it.contains("meta") || !it["meta"].is_object()) {
                send_error(res, 400, at + " missing or non-object field \"meta\"");
                return;
            }
            for (const char* k : {"course", "semester", "type", "title"}) {
                if (!it["meta"].contains(k) || !it["meta"][k].is_string()) {
                    send_error(res, 400,
                               at + ".meta missing or non-string field \"" + k + "\"");
                    return;
                }
            }
            if (!it.contains("content") || !it["content"].is_string()) {
                send_error(res, 400, at + " missing or non-string field \"content\"");
                return;
            }
            std::vector<float> v;
            v.reserve(dim);
            // 判零必须与 VectorStore::add 的 float 累加判定（n2 += x*x 后判 n2 == 0.0f）
            // 逐位一致：若这里用 double 累加，全 1e-30f 这类向量 double 侧非零、float 侧
            // 为零，会绕过前置校验，在下面锁内 add 循环中途抛 "zero vector rejected"，
            // 造成 engine 与内存副本 id 永久错位
            float n2 = 0.0f;
            for (const auto& x : it["vector"]) {
                if (!x.is_number()) {
                    send_error(res, 400, at + ".vector contains non-number element");
                    return;
                }
                const float f = x.get<float>();
                n2 += f * f;
                v.push_back(f);
            }
            // 零向量前置校验：VectorStore::add 会拒绝零向量，先拦下才能保证"全收或全拒"
            if (n2 == 0.0f) {
                send_error(res, 400, at + ".vector is a zero vector");
                return;
            }
            vecs.push_back(std::move(v));
            new_metas.push_back({it["meta"]["course"].get<std::string>(),
                                 it["meta"]["semester"].get<std::string>(),
                                 it["meta"]["type"].get<std::string>(),
                                 it["meta"]["title"].get<std::string>()});
            new_contents.push_back(it["content"].get<std::string>());
        }

        // 全部通过才入库：engine 按调用顺序分配 id，内存副本追加必须与 add 同锁串行，
        // 否则并发 /documents 会造成 id 与副本下标错位
        {
            std::unique_lock lk(g_store_mtx);
            try {
                for (size_t i = 0; i < vecs.size(); ++i)
                    engine.add(std::move(vecs[i]), new_metas[i]);
                metas.insert(metas.end(), new_metas.begin(), new_metas.end());
                contents.insert(contents.end(), std::make_move_iterator(new_contents.begin()),
                                std::make_move_iterator(new_contents.end()));
            } catch (const std::exception& e) {
                // TODO: 批量 add 的原子性根治在 core 侧，超出本任务；这里只能止损——
                // 中途异常（修完零向量前置校验后现实路径主要是 bad_alloc）意味着部分
                // 条目已入 engine 而内存副本未追加，两者 id 已永久错位且无法就地修复，
                // 必须整体停服，靠重启从磁盘一致状态恢复
                LOG_ERROR("documents add failed mid-batch: %s; engine and in-memory copy "
                          "are now inconsistent, restart required", e.what());
                g_service_broken.store(true);
                send_error(res, 503, "server state inconsistent, restart required");
                return;
            }
        }
        LOG_INFO("documents added: %zu (total %zu)", items.size(), engine.size());
        nlohmann::json out = {
            {"added", items.size()},
            {"index", engine.active_index_name()},
            {"index_ready", engine.index_ready()}};
        res.set_content(out.dump(), "application/json");
    };
    svr.Post("/documents", with_auth(documents_handler, cfg.token));

    // 静态页托管（Task 15.4）：GET / 返回 web/index.html；API 路由先注册，httplib
    // 精确路径优先于挂载点，业务路由不会被静态文件遮蔽
    svr.set_mount_point("/", "web");

    // ---- 7. 监听 + 优雅停机 ----
    SetConsoleCtrlHandler(console_ctrl_handler, TRUE);
    // 关闭"注册 handler 与 listen 之间"的窄窗口竞态：若窗口内已收到停机信号，
    // 不再进 listen（此刻 stop 已把内部状态置停，再 listen 会重新绑定端口开始服务），
    // 直接落到下面的 flush 流程——此时必无脏数据，等价空跑一遍优雅停机
    if (g_shutdown.load()) {
        LOG_INFO("shutdown signal received before listen, exiting gracefully");
    } else {
        LOG_INFO("course-rag server listening on 127.0.0.1:%d", cfg.port);
        if (!svr.listen("127.0.0.1", cfg.port)) {
            LOG_ERROR("listen failed on 127.0.0.1:%d", cfg.port);
            return 1;
        }
    }
    // listen 返回 = 已停止接收新请求；flush 在主线程做（回调里绝不 flush）。
    // 此刻无锁读 metas/contents 是安全的：httplib 0.15.3 的 listen_internal 在 accept
    // 循环退出后调 task_queue->shutdown()，其中置停机标志、notify_all 后 join 全部
    // worker，worker 排空剩余任务才退出——listen 返回后不再有任何 handler 并发访问
    // 内存副本。勿在此处额外加锁或改动该排空顺序
    LOG_INFO("accept loop stopped, flushing...");
    engine.wait_rebuild();   // 等后台重建结束，避免 persist 与重建线程竞争

    if (g_service_broken.load()) {
        // 不可服务状态下绝不 flush：engine 已含中途入库的部分向量而内存副本没有，
        // 此时落盘会写出 vectors.bin 与 chunks.json 条数不一致的数据，重启必然
        // count mismatch 拒载；不 flush 则磁盘保持最后一次一致状态，重启即恢复
        LOG_ERROR("service broken flag set, skip flush to keep last consistent on-disk state");
    } else if (engine.dirty()) {    // 只有 /documents 新增过才置脏（load 本身不置脏）
        try {
            // 顺序按计划规定：先原子写 chunks.json，再 persist vectors.bin；任一失败非零码退出
            atomic_write_chunks(cfg.chunks_path, metas, contents);
            engine.persist(cfg.vectors_path);
            LOG_INFO("flushed %zu chunks + vectors to disk", metas.size());
        } catch (const std::exception& e) {
            LOG_ERROR("flush failed: %s", e.what());
            return 3;
        }
    }
    LOG_INFO("server exited cleanly");
    return 0;
}
