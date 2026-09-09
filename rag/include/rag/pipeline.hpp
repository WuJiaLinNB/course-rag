#pragma once
#include <core/engine.hpp>
#include <string>
#include <utility>
#include <vector>
#include <functional>

namespace rag {

// httplib 0.15.3 的 Client(scheme_host_port) 构造函数只接受 "scheme://host[:port]"：
// 带路径前缀（如 "https://api.example.com/v1"）会被正则整体匹配失败，把整个串当
// host 去连 80 端口 → 连接必失败。拆成可构造 Client 的部分 + 请求路径前缀，
// 调用方以 "prefix + /xxx" 作为请求路径。base_url 无路径时 prefix 为空串，行为不变。
inline std::pair<std::string, std::string> split_base_url(const std::string& base_url) {
    const auto scheme_end = base_url.find("://");
    if (scheme_end == std::string::npos) return {base_url, ""};
    const auto host_start = scheme_end + 3;
    const auto slash = base_url.find('/', host_start);
    if (slash == std::string::npos) return {base_url, ""};
    return {base_url.substr(0, slash), base_url.substr(slash)};
}

// 一条引用（出处卡片）：检索命中的向量 id + 元数据 + 相似度，随答案一起展示给用户。
// id 供上层按需回查原文（server 层 GET /chunk?id=），引用本身不携带正文。
struct Citation {
    uint32_t id;
    std::string course, semester, type_, title;
    float similarity;
};

// 一次检索的白盒信息（Trace 视图）：把"这次回答是怎么检索的"暴露给上层展示，
// 让索引行为可观测——命中 hnsw/brute、向量规模、单次 top-k 检索耗时。
struct SearchTrace {
    std::string index;    // brute / ivf / hnsw（engine.active_index_name() 实际服务索引）
    size_t n_vectors;     // 引擎当前向量规模
    double search_ms;     // engine.search 耗时（毫秒，仅该次 top-k 检索，不含 embedding/LLM）
};

struct AskResult {
    std::string answer;
    std::vector<Citation> citations;
    bool llm_ok;                     // false = LLM 失败，answer 是降级文案
    SearchTrace trace;               // 检索白盒信息，供上层 Trace 视图展示
};

struct LlmConfig {
    std::string base_url, api_key, model;
};

// 多轮对话历史：一轮 = 用户提问 + 助手回答 的原文对（按时间序，先到先放）。
// 语义：当前轮的规则/资料/问题只进当前 user 消息；历史轮次作为前置
// user/assistant 消息原样透传给 LLM，供其理解"刚才聊过什么"（追问场景）。
// 上限（轮数/单条字符数）由 server 层校验（kHistoryTurnMax/kHistoryMsgCharMax）；
// 无界历史会让请求体与 LLM 上下文一起膨胀，属外部输入边界。
using ChatHistory = std::vector<std::pair<std::string, std::string>>;

// 检索增强问答流水线：embedding → engine.search → 拼 prompt → 调 LLM。
//
// 过关自测（三个 WHY，长期保留）：
// 1) LLM 失败时为什么还要返回 citations？——检索已经完成且成功，结果对用户仍有价值：
//    用户至少能看到"哪几段资料与问题最相关"，可以自己去翻原文。不能因为生成端故障
//    就把检索端已完成的工作一起丢掉，降级文案 + 引用列表让服务不整体不可用。
// 2) 资料区为什么要用 <资料> 分隔符包裹并声明"是数据不是指令"？——prompt 注入的
//    基础防御：资料原文里若混入"忽略之前的规则……"之类恶意指令，显式的包裹边界 +
//    "是数据不是指令"声明能让模型把这段内容当数据处理，而不是当指令执行
//    （纵深防御的第一层，规则本身仍由指令区约束）。
// 3) LLM 超时为什么定 30s 且不重试？——LLM 生成动辄数秒到几十秒，30s 是"正常长回答"
//    与"卡死"的分界；重试叠加秒级超时会拖垮调用线程（线程池被占满 → 整个服务雪崩）。
//    fail fast + 降级文案是更可控的失败方式。
class Pipeline {
public:
    // 流式 LLM 出口：调用方注入真实实现（httplib Client + stream:true + SSE 解析）。
    // 返回 true 表示流完整成功；on_delta 逐块回调增量文本，返回 false 表示客户端已断开，
    // 调用方应尽快终止（fail-fast，不重试）。
    // history 语义与 call_llm 一致：历史轮次按 user/assistant 消息原样前置，
    // 当前轮 prompt（规则+资料+问题）作为最后一条 user 消息。
    using LlmStreamFn =
        std::function<bool(const std::string& prompt,
                           const ChatHistory& history,
                           const std::function<bool(const std::string&)>& on_delta)>;

    // engine_ 必须比 Pipeline 活得久（server 层二者同生命周期）
    Pipeline(core::Engine& engine, LlmConfig cfg,
             std::function<std::vector<float>(const std::string&)> embed_query,
             // 原文/元数据提供者：server 层持有 chunks.json 的内存副本（core 不存原文）
             std::function<std::string(uint32_t id)> fetch_chunk,
             std::function<core::ChunkMeta(uint32_t id)> fetch_meta,
             // 可选流式 LLM 出口；默认空 = ask_stream 走降级（on_done(false)）
             LlmStreamFn llm_stream = {});
    AskResult ask(const std::string& question, size_t top_k,
                  const ChatHistory& history = {}) const;

    // 流式问答：先同步检索，on_meta 回传 citations + trace（answer 为空），随后 LLM 增量
    // 文本经 on_delta 逐块回调（回调返回 false 立即终止，视为客户端断开）；结束统一
    // on_done(llm_ok)。llm_stream_ 未注入或调用失败 → 仍回调 on_done(false)，上层据此
    // 展示降级文案（与 ask() 的 llm_ok 语义一致）。history 透传给 llm_stream_（见上）。
    void ask_stream(const std::string& question, size_t top_k,
                    const ChatHistory& history,
                    const std::function<bool(const AskResult&)>& on_meta,
                    const std::function<bool(const std::string&)>& on_delta,
                    const std::function<void(bool)>& on_done) const;

    // build_prompt_ 放在 public 供测试直接断言 prompt 内容（规则/资料边界/编号原文）：
    // prompt 只经 call_llm 出口无法在单测中观察，否则只能真连网络，违背确定性原则。
    std::string build_prompt_(const std::string& question,
                              const std::vector<core::SearchItem>& hits) const;

private:
    core::Engine& engine_;
    LlmConfig llm_;
    std::function<std::vector<float>(const std::string&)> embed_query_;
    std::function<std::string(uint32_t id)> fetch_chunk_;
    std::function<core::ChunkMeta(uint32_t id)> fetch_meta_;
    LlmStreamFn llm_stream_;
    std::string call_llm(const std::string& prompt, const ChatHistory& history) const;
    // 失败返回 ""
};

} // namespace rag
