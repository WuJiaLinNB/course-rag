// course-rag HTTP 服务（Task 12）：httplib + nlohmann 单头，对接 core::Engine 与 rag::Pipeline。
//
// 启动流程：环境变量 → data/chunks.json 内存副本（下标 = id）→ 读 vectors.bin 16 字节头
// 拿 dim → Engine 构造 + load（一致性校验在 Engine 内，失败退出）→ 监听 127.0.0.1。
// 配置全部走环境变量，fail fast：必需变量缺失/数据文件读不了 → stderr 明确报错后 exit 1，
// 清晰报错优于带错启动。
//
// 优雅停机：控制台回调（独立线程）只调 svr.stop() 停止接收新请求；listen 返回后主线程
// wait_rebuild → dirty 时先原子写 chunks.json 再 persist vectors.bin。回调里绝不做 flush。
//
// v1 限制（与 rag/src/pipeline.cpp 的 call_llm 一致）：未定义 CPPHTTPLIB_OPENSSL_SUPPORT，
// httplib 只能发 http，https 端点会连接失败走 502；TLS 策略留给 Task 15/16。
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

#include <algorithm>
#include <atomic>
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
constexpr const char* kChunksPath = "data/chunks.json";
constexpr const char* kVectorsPath = "data/vectors.bin";

// chunks.json 内存副本的并发保护：/documents 追加会让 vector 重新分配，与 Pipeline
// 的 fetch_*（按 id 读）并发是数据竞争 → 写用 unique_lock、读用 shared_lock。
// 追加只改尾部，已有 id 的数据不变，锁内访问即可保证安全。
std::shared_mutex g_store_mtx;

httplib::Server* g_svr = nullptr;      // 控制台回调里只调 svr.stop()（线程安全）
std::atomic<bool> g_shutdown{false};

// ---- 启动辅助 ------------------------------------------------------------

// 必需环境变量：缺失即报错退出（fail fast）
std::string require_env(const char* name) {
    const char* v = std::getenv(name);
    if (!v || !*v) {
        LOG_ERROR("missing required environment variable: %s", name);
        std::exit(1);
    }
    return std::string(v);
}

// 可选环境变量（EMBED_API_KEY / LLM_API_KEY）：有则走 Bearer 鉴权，无则放行
std::string optional_env(const char* name) {
    const char* v = std::getenv(name);
    return (v && *v) ? std::string(v) : std::string();
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
#ifdef _WIN32
    if (!MoveFileExA(tmp.c_str(), path.c_str(), MOVEFILE_REPLACE_EXISTING))
        throw std::runtime_error("atomic replace failed: " + path);
#else
    if (std::rename(tmp.c_str(), path.c_str()) != 0)
        throw std::runtime_error("atomic replace failed: " + path);
#endif
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
// 失败时填好 400 响应并返回 false；成功时通过出参带出 query/top_k
bool parse_search_like_body(const httplib::Request& req, httplib::Response& res,
                            std::string& query, long long& top_k) {
    if (req.body.size() > kQueryBodyMax) {
        send_error(res, 400, "request body exceeds 65536 bytes limit");
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
    return true;
}

// 在线单条 embedding：与 rag/src/pipeline.cpp 的 call_llm 同款 fail fast——
// 连接/读超时 5s、失败不重试；任何失败返回空 vector，由路由层转 502。
// 请求体用 nlohmann 构造（query 含任意文本，手拼 JSON 必漏转义），响应解析不抛异常。
std::vector<float> embed_query_once(const std::string& base_url,
                                    const std::string& api_key,
                                    const std::string& model,
                                    const std::string& query) {
    httplib::Client cli(base_url);
    cli.set_connection_timeout(5);
    cli.set_read_timeout(5);
    if (!api_key.empty()) cli.set_bearer_token_auth(api_key);

    nlohmann::json body;
    body["model"] = model;
    body["input"] = nlohmann::json::array({query});
    const auto res = cli.Post("/embeddings", body.dump(-1, ' ', false,
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
// flush 链路（wait_rebuild → 写 chunks.json → persist）留在 listen 返回后的主线程
BOOL WINAPI console_ctrl_handler(DWORD ctrl_type) {
    if (ctrl_type == CTRL_C_EVENT || ctrl_type == CTRL_BREAK_EVENT) {
        if (!g_shutdown.exchange(true)) {
            LOG_INFO("console ctrl %lu received, stopping accept loop...",
                     static_cast<unsigned long>(ctrl_type));
            if (g_svr) g_svr->stop();
        }
        return TRUE;   // 已处理，屏蔽默认强杀，让主线程走优雅 flush
    }
    return FALSE;
}

} // namespace

int main() {
    // ---- 1. 环境变量（PORT 默认 8080；必需变量缺失 fail fast）----
    int port = 8080;
    if (const char* p = std::getenv("PORT"); p && *p) {
        char* end = nullptr;
        const long v = std::strtol(p, &end, 10);
        if (!end || *end != '\0' || v <= 0 || v > 65535) {
            LOG_ERROR("invalid PORT value: %s", p);
            return 1;
        }
        port = static_cast<int>(v);
    }
    const std::string embed_base = require_env("EMBED_API_BASE");
    const std::string embed_model = require_env("EMBED_MODEL");
    const std::string embed_key = optional_env("EMBED_API_KEY");
    const std::string llm_base = require_env("LLM_API_BASE");
    const std::string llm_model = require_env("LLM_MODEL");
    const std::string llm_key = optional_env("LLM_API_KEY");

    // ---- 2. chunks.json 内存副本（下标 = id，供 fetch_chunk / fetch_meta / 结果回填）----
    std::vector<core::ChunkMeta> metas;
    std::vector<std::string> contents;
    load_chunks(kChunksPath, metas, contents);

    // ---- 3. vectors.bin 头部取 dim → Engine 构造 + load（校验失败即退出）----
    const uint32_t dim = read_dim_from_header(kVectorsPath);
    core::Engine engine(dim, core::Engine::Mode::Auto);
    try {
        engine.load(kVectorsPath, metas);   // magic/版本/维度/字节数/跨文件条数校验都在 Engine 内
    } catch (const std::exception& e) {
        LOG_ERROR("startup aborted: %s", e.what());
        return 1;
    }
    LOG_INFO("engine loaded %zu vectors (dim=%u, index=%s)", engine.size(), dim,
             engine.active_index_name().c_str());

    // ---- 4. 共用 embed_query（/search 路由与 Pipeline 共用同一函数对象）----
    // 线程内记忆化：/ask 路由要先探测 embedding 可用性（失败 502），Pipeline::ask 内部
    // 还会再调一次——thread_local 缓存让同一请求内的第二次调用命中缓存，避免对 embedding
    // 服务连发两次。httplib 每连接一线程、单请求全程在同一线程内处理，缓存无竞争。
    // 失败结果不缓存（空向量不命中），下次请求会重试真实服务。
    std::function<std::vector<float>(const std::string&)> embed_query =
        [embed_base, embed_key, embed_model](const std::string& q) -> std::vector<float> {
        static thread_local std::string cached_q;
        static thread_local std::vector<float> cached_v;
        if (!cached_v.empty() && cached_q == q) return cached_v;
        std::vector<float> v = embed_query_once(embed_base, embed_key, embed_model, q);
        if (!v.empty()) {
            cached_q = q;
            cached_v = v;
        }
        return v;
    };

    // ---- 5. Pipeline（engine 声明在前，保证比 pipeline 活得久）----
    rag::Pipeline pipeline(engine,
        rag::LlmConfig{llm_base, llm_key, llm_model},
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
        });

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

    // POST /search：embed → engine.search → 结果带元数据与原文
    svr.Post("/search", [&](const httplib::Request& req, httplib::Response& res) {
        std::string query;
        long long top_k = 0;
        if (!parse_search_like_body(req, res, query, top_k)) return;

        // 超容量 clamp 到当前条数；条数为 0 时 k=0 会触发 BruteIndex 空堆 UB，
        // 直接返回空结果（不进 engine.search）
        const size_t k = static_cast<size_t>(
            std::min<long long>(top_k, static_cast<long long>(engine.size())));

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
        if (k > 0) {
            const auto hits = engine.search(qv, k, {});   // v1：不做元数据过滤
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
            {"results", std::move(results)}};
        res.set_content(out.dump(), "application/json");
    });

    // POST /ask：embed（失败 502）→ Pipeline::ask（LLM 失败内部降级，仍 200）
    svr.Post("/ask", [&](const httplib::Request& req, httplib::Response& res) {
        std::string query;
        long long top_k = 0;
        if (!parse_search_like_body(req, res, query, top_k)) return;

        const size_t total = engine.size();
        if (total == 0) {   // 空索引无法问答，也避开 k=0 的空堆 UB
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

        const rag::AskResult r = pipeline.ask(query, k);
        nlohmann::json citations = nlohmann::json::array();
        for (const auto& c : r.citations) {
            citations.push_back({
                {"course", c.course},
                {"semester", c.semester},
                {"type", c.type_},
                {"title", c.title},
                {"similarity", c.similarity}});
        }
        nlohmann::json out = {
            {"answer", r.answer},
            {"llm_ok", r.llm_ok},
            {"citations", std::move(citations)}};
        res.set_content(out.dump(), "application/json");
    });

    // POST /documents：收 chunk 包，先整体校验再逐条入库（要么全收要么全拒）
    svr.Post("/documents", [&](const httplib::Request& req, httplib::Response& res) {
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
            double n2 = 0.0;
            for (const auto& x : it["vector"]) {
                if (!x.is_number()) {
                    send_error(res, 400, at + ".vector contains non-number element");
                    return;
                }
                const float f = x.get<float>();
                n2 += static_cast<double>(f) * f;
                v.push_back(f);
            }
            // 零向量前置校验：VectorStore::add 会拒绝零向量，先拦下才能保证"全收或全拒"
            if (n2 == 0.0) {
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
            for (size_t i = 0; i < vecs.size(); ++i)
                engine.add(std::move(vecs[i]), new_metas[i]);
            metas.insert(metas.end(), new_metas.begin(), new_metas.end());
            contents.insert(contents.end(), std::make_move_iterator(new_contents.begin()),
                            std::make_move_iterator(new_contents.end()));
        }
        LOG_INFO("documents added: %zu (total %zu)", items.size(), engine.size());
        nlohmann::json out = {
            {"added", items.size()},
            {"index", engine.active_index_name()},
            {"index_ready", engine.index_ready()}};
        res.set_content(out.dump(), "application/json");
    });

    // ---- 7. 监听 + 优雅停机 ----
    SetConsoleCtrlHandler(console_ctrl_handler, TRUE);
    LOG_INFO("course-rag server listening on 127.0.0.1:%d", port);
    if (!svr.listen("127.0.0.1", port)) {
        LOG_ERROR("listen failed on 127.0.0.1:%d", port);
        return 1;
    }
    // listen 返回 = 已停止接收新请求；flush 在主线程做（回调里绝不 flush）
    LOG_INFO("accept loop stopped, flushing...");
    engine.wait_rebuild();   // 等后台重建结束，避免 persist 与重建线程竞争

    if (engine.dirty()) {    // 只有 /documents 新增过才置脏（load 本身不置脏）
        try {
            // 顺序按计划规定：先原子写 chunks.json，再 persist vectors.bin；任一失败非零码退出
            atomic_write_chunks(kChunksPath, metas, contents);
            engine.persist(kVectorsPath);
            LOG_INFO("flushed %zu chunks + vectors to disk", metas.size());
        } catch (const std::exception& e) {
            LOG_ERROR("flush failed: %s", e.what());
            return 3;
        }
    }
    LOG_INFO("server exited cleanly");
    return 0;
}
