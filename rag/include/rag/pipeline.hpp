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

// 一条引用（出处卡片）：检索命中的元数据 + 相似度，随答案一起展示给用户
struct Citation {
    std::string course, semester, type_, title;
    float similarity;
};

struct AskResult {
    std::string answer;
    std::vector<Citation> citations;
    bool llm_ok;                     // false = LLM 失败，answer 是降级文案
};

struct LlmConfig {
    std::string base_url, api_key, model;
};

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
    // engine_ 必须比 Pipeline 活得久（server 层二者同生命周期）
    Pipeline(core::Engine& engine, LlmConfig cfg,
             std::function<std::vector<float>(const std::string&)> embed_query,
             // 原文/元数据提供者：server 层持有 chunks.json 的内存副本（core 不存原文）
             std::function<std::string(uint32_t id)> fetch_chunk,
             std::function<core::ChunkMeta(uint32_t id)> fetch_meta);
    AskResult ask(const std::string& question, size_t top_k) const;

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
    std::string call_llm(const std::string& prompt) const;   // 失败返回 ""
};

} // namespace rag
