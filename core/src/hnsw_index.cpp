#include <core/hnsw_index.hpp>
#include <algorithm>
#include <cmath>
#include <limits>
#include <queue>
#include <stdexcept>
#include <string>

namespace core {

HnswIndex::HnswIndex(size_t dim, size_t M, size_t ef_construction)
    : store_(dim) {
    // 论文 mL = 1/ln(M) 要求 M ≥ 2；efConstruction 至少 1（否则建不出边）
    M_ = std::max<size_t>(2, M);
    Mmax_ = M_;          // 上层邻居上限 = M（论文标准）
    Mmax0_ = 2 * M_;     // 第 0 层邻居上限 = 2M（论文标准：底层最稠密、负责全部召回）
    efC_ = std::max<size_t>(1, ef_construction);
}

void HnswIndex::add(std::vector<float> v, const ChunkMeta& meta) {
    store_.add(std::move(v));   // 归一化 + 维度/零向量校验，失败不留中间态
    metas_.push_back(meta);
    graph_.emplace_back();
    node_level_.push_back(0);
    insert_((uint32_t)(metas_.size() - 1));
#ifdef _DEBUG
    check_graph_invariants();   // DEBUG 构建下每次插入后做图不变量断言
#endif
}

size_t HnswIndex::size() const { return store_.size(); }

void HnswIndex::set_ef_search(size_t ef) { efS_ = ef; }

// 层高 = floor(-ln(uniform(0,1)) * mL)，mL = 1/ln(M)。
// 【过关自测】和跳表是同一个思想：每升一层节点数按 1/M 几何递减，
// 顶层稀疏图是“高速公路”，负责长距离贪心跳步；底层稠密图负责精细召回。
// 与跳表不同的是邻居不是链表而是小世界图，且每层用贪心搜索定位。
size_t HnswIndex::random_level_() {
    std::uniform_real_distribution<float> unif(0.0f, 1.0f);
    float u = unif(rng_);
    if (u <= 0.0f) u = std::numeric_limits<float>::min();   // 防 log(0) = -inf
    const double mL = 1.0 / std::log((double)M_);
    return (size_t)std::floor(-std::log((double)u) * mL);
}

// 与 BruteIndex 完全同口径的 double 累加器点积：分数逐位一致，召回对照才有意义
double HnswIndex::sim_query_node_(const float* qn, uint32_t b) const {
    const float* v = store_.raw() + (size_t)b * store_.dim();
    double s = 0.0;
    for (size_t d = 0; d < store_.dim(); ++d) s += (double)qn[d] * (double)v[d];
    return s;
}

double HnswIndex::sim_node_node_(uint32_t a, uint32_t b) const {
    const float* va = store_.raw() + (size_t)a * store_.dim();
    const float* vb = store_.raw() + (size_t)b * store_.dim();
    double s = 0.0;
    for (size_t d = 0; d < store_.dim(); ++d) s += (double)va[d] * (double)vb[d];
    return s;
}

// 论文 Algorithm 2（单层检索）。本项目相似度越大越近，与论文“距离越小越近”
// 方向相反，所有比较反转：候选堆堆顶 = 最近者；结果堆堆顶 = 已保留 ef 条中最差者。
// qn 为已归一化查询（search() 内归一化一次；insert_ 传已归一化的库内向量的 SoA 指针）
std::vector<HnswIndex::SimItem>
HnswIndex::search_layer_scored_(const float* qn,
                                const std::vector<uint32_t>& entry,
                                size_t ef, size_t layer) const {
    std::vector<char> visited(store_.size(), 0);
    std::priority_queue<SimItem, std::vector<SimItem>, Nearer> results;          // 守门员
    std::priority_queue<SimItem, std::vector<SimItem>, FartherFirst> candidates; // 堆顶最近

    for (uint32_t e : entry) {
        if (e >= store_.size() || visited[e]) continue;
        visited[e] = 1;
        SimItem it{e, sim_query_node_(qn, e)};
        candidates.push(it);
        results.push(it);
        if (results.size() > ef) results.pop();
    }

    while (!candidates.empty()) {
        SimItem c = candidates.top(); candidates.pop();
        // 收敛：最近的未探索候选比守门员还差且结果已满 → 之后只会更差，停
        if (results.size() == ef && c.sim < results.top().sim) break;
        for (uint32_t e : graph_[c.id][layer]) {
            if (visited[e]) continue;
            visited[e] = 1;
            SimItem it{e, sim_query_node_(qn, e)};
            const SimItem& worst = results.top();
            if (results.size() < ef || it.sim > worst.sim ||
                (it.sim == worst.sim && it.id < worst.id)) {
                candidates.push(it);
                results.push(it);
                if (results.size() > ef) results.pop();
            }
        }
    }

    std::vector<SimItem> out;
    out.reserve(results.size());
    while (!results.empty()) { out.push_back(results.top()); results.pop(); }
    return out;
}

// plan 签名的薄包装：仅丢弃分数（search 的贪心降层只需要 id 作下一层入口）
std::vector<uint32_t> HnswIndex::search_layer_(const std::vector<float>& qn,
                                               std::vector<uint32_t> entry,
                                               size_t ef, size_t layer) const {
    auto w = search_layer_scored_(qn.data(), entry, ef, layer);
    std::vector<uint32_t> ids;
    ids.reserve(w.size());
    for (const auto& s : w) ids.push_back(s.id);
    return ids;
}

// 论文 Algorithm 4（启发式选邻居，保连通性）。
// 【过关自测】比“取最近 M 个”好在哪：只取最近 M 个会把同一方向的点全占满——
// 一旦某方向上有个更近的点，方向相近的其余点都被挤掉，远处簇没有边可达，
// 图断连、召回塌方。启发式按“离 cand 由近到远”贪心，仅当 e 离 cand 比离任何
// 已选邻居都近（sim(cand,e) > sim(r,e)）时保留 → 各方向都有代表，保住跨簇桥边。
std::vector<uint32_t> HnswIndex::select_neighbors_(uint32_t cand,
        const std::vector<uint32_t>& cands, size_t M) const {
    std::vector<SimItem> scored;
    scored.reserve(cands.size());
    for (uint32_t e : cands)
        if (e != cand) scored.push_back({e, sim_node_node_(cand, e)});
    std::sort(scored.begin(), scored.end(), [](const SimItem& a, const SimItem& b) {
        return a.sim > b.sim || (a.sim == b.sim && a.id < b.id);
    });

    std::vector<SimItem> selected;
    selected.reserve(M);
    for (const auto& e : scored) {
        if (selected.size() >= M) break;
        bool keep = true;
        for (const auto& r : selected)
            if (sim_node_node_(r.id, e.id) >= e.sim) { keep = false; break; }
        if (keep) selected.push_back(e);
    }

    std::vector<uint32_t> out;
    out.reserve(selected.size());
    for (const auto& s : selected) out.push_back(s.id);
    return out;
}

// 论文 Algorithm 1（插入）
void HnswIndex::insert_(uint32_t id) {
    const size_t level = random_level_();
    graph_[id].resize(level + 1);
    node_level_[id] = level;

    if (id == 0) {                       // 第一个节点：直接作为入口
        entry_point_ = id;
        max_level_ = level;
        return;
    }

    // 新节点向量已入库并归一化，直接以它为查询（不能再归一化，会引入浮点误差）
    const float* qn = store_.raw() + (size_t)id * store_.dim();

    std::vector<uint32_t> eps{entry_point_};
    // 1) 顶层贪心下降到 level+1 层：ef=1，每层只带走最近一个点作下层入口
    for (size_t lc = max_level_; lc > level; --lc) {
        auto w = search_layer_scored_(qn, eps, 1, lc);
        eps.clear();
        for (const auto& s : w) eps.push_back(s.id);
    }
    // 2) 从 min(level, max_level) 层向下逐层：efC 检索 → 启发式选邻居 →
    //    双向连边 → 超度数邻居收缩（lc-- > 0 处理 size_t 下溢）
    for (size_t lc = std::min(level, max_level_) + 1; lc-- > 0;) {
        auto W = search_layer_scored_(qn, eps, efC_, lc);
        std::vector<uint32_t> cands;
        cands.reserve(W.size());
        for (const auto& s : W) cands.push_back(s.id);

        auto neighbors = select_neighbors_(id, cands, M_);
        for (uint32_t nb : neighbors) {
            graph_[id][lc].push_back(nb);
            graph_[nb][lc].push_back(id);
            // 邻居可能超度数：对它的邻居集合重跑启发式收缩（论文的 shrink 步骤）
            const size_t cap = (lc == 0) ? Mmax0_ : Mmax_;
            auto& nb_list = graph_[nb][lc];
            if (nb_list.size() > cap) {
                auto kept = select_neighbors_(nb, nb_list, cap);
                nb_list.assign(kept.begin(), kept.end());
            }
        }
        // 论文：ep ← W（本层找到的全部元素作为下一层入口）
        eps = std::move(cands);
    }
    if (level > max_level_) {
        entry_point_ = id;
        max_level_ = level;
    }
}

// 论文 Algorithm 5（k-NN 检索）
std::vector<SearchItem> HnswIndex::search(const std::vector<float>& q,
                                          size_t k,
                                          const MetaFilter& filter) const {
    if (q.size() != store_.dim()) throw std::invalid_argument("query dim mismatch");
    if (store_.size() == 0) return {};

    // 与 BruteIndex 完全一致的查询归一化（float 累加 + float 逆范数）→ 分数逐位一致
    float n2 = 0.0f;
    for (float x : q) n2 += x * x;
    std::vector<float> qn(q);
    if (n2 > 0.0f) { float inv = 1.0f / std::sqrt(n2); for (float& x : qn) x *= inv; }

    // 自顶层贪心下降到第 1 层（Algorithm 5）：每层 ef=1，只带走最近点作下层入口
    std::vector<uint32_t> eps{entry_point_};
    for (size_t lc = max_level_; lc >= 1; --lc)
        eps = search_layer_(qn, eps, 1, lc);

    // 第 0 层：ef 必须 ≥ k 才可能返回 k 条。
    // 【过关自测】efConstruction 只影响建图质量（候选池越大图越准、建得越慢）；
    // efSearch 只影响查询（候选池越大召回越高、查询越慢），两者互不替代。
    const size_t ef = std::max(efS_, k);
    auto W = search_layer_scored_(qn.data(), eps, ef, 0);

    // post-filter：先探索后过滤，结果可能不足 k（第一版声明限制）
    std::vector<SearchItem> out;
    out.reserve(W.size());
    for (const auto& w : W)
        if (match(metas_[w.id], filter)) out.push_back({w.id, (float)w.sim});
    std::sort(out.begin(), out.end(), [](const SearchItem& a, const SearchItem& b) {
        return a.similarity > b.similarity ||
               (a.similarity == b.similarity && a.id < b.id);
    });
    if (out.size() > k) out.resize(k);
    return out;
}

// DEBUG 图不变量断言（违例抛 runtime_error）：
//   1) graph_[i].size() == node_level_[i]+1（层数一致）
//   2) 各层度数 ≤ 上限（第 0 层 Mmax0_=2M，其余 Mmax_=M）
//   3) 邻居不越界、无自环、邻居层级不低于边所在层、无重复邻居
//   4) 第 0 层从 entry_point_ BFS 可达全部节点（连通性）
void HnswIndex::check_graph_invariants() const {
    const size_t n = store_.size();
    if (n == 0) return;
    for (size_t i = 0; i < n; ++i) {
        if (graph_[i].size() != node_level_[i] + 1)
            throw std::runtime_error("hnsw invariant: node " + std::to_string(i) +
                                     " has " + std::to_string(graph_[i].size()) +
                                     " layers but level " + std::to_string(node_level_[i]));
        for (size_t lc = 0; lc < graph_[i].size(); ++lc) {
            const size_t cap = (lc == 0) ? Mmax0_ : Mmax_;
            if (graph_[i][lc].size() > cap)
                throw std::runtime_error("hnsw invariant: node " + std::to_string(i) +
                                         " layer " + std::to_string(lc) + " degree " +
                                         std::to_string(graph_[i][lc].size()) +
                                         " > cap " + std::to_string(cap));
            for (size_t j = 0; j < graph_[i][lc].size(); ++j) {
                const uint32_t nb = graph_[i][lc][j];
                if (nb >= n)
                    throw std::runtime_error("hnsw invariant: neighbor out of range");
                if (nb == (uint32_t)i)
                    throw std::runtime_error("hnsw invariant: self loop at node " +
                                             std::to_string(i));
                if (lc > node_level_[nb])
                    throw std::runtime_error("hnsw invariant: edge at layer " +
                                             std::to_string(lc) + " to node " +
                                             std::to_string(nb) + " below its level");
                for (size_t t = j + 1; t < graph_[i][lc].size(); ++t)
                    if (graph_[i][lc][t] == nb)
                        throw std::runtime_error("hnsw invariant: duplicate neighbor " +
                                                 std::to_string(nb) + " at node " +
                                                 std::to_string(i));
            }
        }
    }
    std::vector<char> seen(n, 0);
    std::vector<uint32_t> stack{entry_point_};
    seen[entry_point_] = 1;
    size_t reached = 0;
    while (!stack.empty()) {
        uint32_t cur = stack.back(); stack.pop_back();
        ++reached;
        for (uint32_t nb : graph_[cur][0])
            if (!seen[nb]) { seen[nb] = 1; stack.push_back(nb); }
    }
    if (reached != n)
        throw std::runtime_error("hnsw invariant: layer-0 disconnected, reachable " +
                                 std::to_string(reached) + " of " + std::to_string(n));
}

} // namespace core
