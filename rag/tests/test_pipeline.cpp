#include <gtest/gtest.h>
#include <rag/pipeline.hpp>
#include <map>
#include <string>
#include <vector>

using namespace rag;

namespace {

// 夹具：8 维 Engine（BruteOnly：无后台重建线程、暴力检索逐位确定）+ 三条正交单位向量。
// 全部依赖用注入式 lambda + 本地查表实现——单测不碰真实网络；LLM 用"必然立刻连接失败"
// 的 127.0.0.1:1（端口 1 直接拒绝）走降级路径，无等待开销。
struct Fixture {
    static constexpr size_t DIM = 8;
    core::Engine engine{DIM, core::Engine::Mode::BruteOnly};
    std::map<uint32_t, core::ChunkMeta> metas;
    std::map<uint32_t, std::string> chunks;

    Fixture() {
        for (uint32_t i = 0; i < 3; ++i) {
            std::vector<float> v(DIM, 0.0f);
            v[i] = 1.0f;                       // e0/e1/e2 两两正交；store 入库时归一化
            const core::ChunkMeta m{"C" + std::to_string(i), "S" + std::to_string(i),
                                    "T" + std::to_string(i), "T_" + std::to_string(i)};
            engine.add(v, m);
            chunks[i] = "CHUNK_TEXT_" + std::to_string(i);
            metas[i] = m;
        }
    }

    core::ChunkMeta meta_of(uint32_t id) const { return metas.at(id); }
    std::string chunk_of(uint32_t id) const { return chunks.at(id); }
    // 与 id=0 的向量完全一致 → 余弦相似度恒为 1.0，稳定排第一（同分 tie 由 id 升序兜底）
    std::vector<float> query_e0() const {
        std::vector<float> v(DIM, 0.0f);
        v[0] = 1.0f;
        return v;
    }
};

Pipeline make_pipeline(Fixture& f) {
    return Pipeline(f.engine, {"http://127.0.0.1:1", "sk-test", "test-model"},
                    [&f](const std::string&) { return f.query_e0(); },
                    [&f](uint32_t id) { return f.chunk_of(id); },
                    [&f](uint32_t id) { return f.meta_of(id); });
}

} // namespace

// prompt 组装：规则区、资料区边界、编号原文、问题区都要在，且顺序固定
TEST(Pipeline, PromptContainsRulesAndSources) {
    Fixture f;
    Pipeline p = make_pipeline(f);
    const auto hits = f.engine.search(f.query_e0(), 2, {});
    ASSERT_EQ(hits.size(), 2u);
    EXPECT_EQ(hits[0].id, 0u);                 // 查询向量与 id0 完全一致 → 稳居第一
    const std::string prompt = p.build_prompt_("什么是进程？", hits);
    EXPECT_NE(prompt.find("规则"), std::string::npos);
    EXPECT_NE(prompt.find("<资料>"), std::string::npos);
    EXPECT_NE(prompt.find("[1] CHUNK_TEXT_0"), std::string::npos);
    EXPECT_NE(prompt.find("</资料>"), std::string::npos);
    EXPECT_NE(prompt.find("问题：什么是进程？"), std::string::npos);
    // 结构顺序：资料区整体夹在 <资料>…</资料> 之间，问题区在资料区之后
    EXPECT_LT(prompt.find("<资料>"), prompt.find("</资料>"));
    EXPECT_LT(prompt.find("</资料>"), prompt.find("问题：什么是进程？"));
}

// LLM 不可达 → 降级文案 + llm_ok=false；但检索已完成，citations 必须原样返回
TEST(Pipeline, LlmFailureDegradesWithCitations) {
    Fixture f;
    Pipeline p = make_pipeline(f);
    const auto r = p.ask("什么是进程？", 2);
    EXPECT_FALSE(r.llm_ok);
    EXPECT_EQ(r.answer, "AI 服务暂不可用，以下为检索到的原文片段");
    ASSERT_EQ(r.citations.size(), 2u);
    for (const auto& c : r.citations) {
        EXPECT_GE(c.similarity, -1.0f);        // 余弦相似度合法区间
        EXPECT_LE(c.similarity, 1.0f);
        EXPECT_FALSE(c.course.empty());        // fetch_meta 查表结果确实被填进来了
    }
    EXPECT_EQ(r.citations[0].title, "T_0");    // hits 确定性：首命中 id=0 → 首引用 T_0
}

// citations 与 engine.search 的 hits 一一对应：id → 元数据字段、相似度原样透传
TEST(Pipeline, CitationsMatchHits) {
    Fixture f;
    Pipeline p = make_pipeline(f);
    const auto hits = f.engine.search(f.query_e0(), 2, {});
    const auto r = p.ask("什么是进程？", 2);
    ASSERT_EQ(r.citations.size(), hits.size());
    EXPECT_FALSE(r.trace.index.empty());            // Trace：索引名非空（brute/ivf/hnsw 之一）
    EXPECT_EQ(r.trace.n_vectors, f.engine.size());  // Trace：向量规模与引擎一致
    EXPECT_GE(r.trace.search_ms, 0.0);              // Trace：耗时非负
    for (size_t i = 0; i < hits.size(); ++i) {
        const auto& m = f.meta_of(hits[i].id);  // 同一查询两次结果逐位一致（double 累加器 + tie 按 id）
        EXPECT_EQ(r.citations[i].id, hits[i].id);      // 引用携带向量 id，供上层回查原文
        EXPECT_EQ(r.citations[i].course, m.course);
        EXPECT_EQ(r.citations[i].semester, m.semester);
        EXPECT_EQ(r.citations[i].type_, m.type_);
        EXPECT_EQ(r.citations[i].title, m.title);
        EXPECT_FLOAT_EQ(r.citations[i].similarity, hits[i].similarity);
    }
}
