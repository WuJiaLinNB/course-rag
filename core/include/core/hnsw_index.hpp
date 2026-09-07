#pragma once
#include <core/index.hpp>
#include <core/vector_store.hpp>
#include <cstdint>
#include <random>
#include <vector>

namespace core {

// HNSW 分层可导航小世界图索引（Malkov & Yashunin, TPAMI 2018，按论文编号对照实现）。
// 相似度 = 余弦：向量入库即归一化，点积即余弦；打分用 double 累加器，
// 与 BruteIndex（全项目 ground truth）同口径，保证分数逐位一致。
// 第一版过滤策略为 post-filter：先按图探索取 ef 条候选，再按元数据过滤——
// 结果可能不足 k（BruteIndex 是 pre-filter，无此限制）。
class HnswIndex : public Index {
public:
    HnswIndex(size_t dim, size_t M = 16, size_t ef_construction = 200);
    void add(std::vector<float> v, const ChunkMeta& meta) override;
    size_t size() const override;
    std::vector<SearchItem> search(const std::vector<float>& q,
                                   size_t k,
                                   const MetaFilter& filter) const override;
    std::string name() const override { return "hnsw"; }
    void set_ef_search(size_t ef);
    // DEBUG 断言：第 0 层 BFS 连通、各层度数 ≤ 上限、无自环/重复邻居/越界邻居、
    // 层数与 node_level_ 一致。add() 在 #ifdef _DEBUG 下每次插入后调用
    void check_graph_invariants() const;

private:
    // 带分数候选：sim 为 double 累加器口径，search 的最终分数由此而来
    struct SimItem { uint32_t id; double sim; };
    // a 比 b 更接近：sim 大者优先，同分 id 小者优先——与 BruteIndex::Better 同规则
    struct Nearer {
        bool operator()(const SimItem& a, const SimItem& b) const {
            return a.sim > b.sim || (a.sim == b.sim && a.id < b.id);
        }
    };
    // 候选堆比较器：priority_queue 堆顶 = 比较器下的“最大”者，
    // 令“更远者为小” → 堆顶恰为最近候选（Algorithm 2 每轮先扩展最近者）
    struct FartherFirst {
        bool operator()(const SimItem& a, const SimItem& b) const {
            return a.sim < b.sim || (a.sim == b.sim && a.id > b.id);
        }
    };

    size_t random_level_();               // 层高：floor(-ln(unif) * mL)，mL = 1/ln(M)
    void insert_(uint32_t id);            // 论文 Algorithm 1
    std::vector<uint32_t> search_layer_(const std::vector<float>& qn,
                                        std::vector<uint32_t> entry,
                                        size_t ef, size_t layer) const;
    // 带分数版（论文 Algorithm 2）。qn 为已归一化查询指针；search 需要分数做
    // post-filter 与最终排序，故保留 plan 签名的 search_layer_ 作为薄包装
    std::vector<SimItem> search_layer_scored_(const float* qn,
                                              const std::vector<uint32_t>& entry,
                                              size_t ef, size_t layer) const;
    std::vector<uint32_t> select_neighbors_(uint32_t cand,
        const std::vector<uint32_t>& cands, size_t M) const;   // 论文 Algorithm 4 启发式
    double sim_query_node_(const float* qn, uint32_t b) const; // qn 已归一化
    double sim_node_node_(uint32_t a, uint32_t b) const;       // 库内向量均已归一化

    VectorStore store_;
    std::vector<ChunkMeta> metas_;
    std::vector<std::vector<std::vector<uint32_t>>> graph_;  // [节点][层][邻居]
    std::vector<size_t> node_level_;
    size_t M_, Mmax_, Mmax0_, efC_, efS_ = 16;
    uint32_t entry_point_ = 0;
    size_t max_level_ = 0;
    mutable std::mt19937 rng_{42};   // 固定种子：层高序列可复现，问题可复现
};

} // namespace core
