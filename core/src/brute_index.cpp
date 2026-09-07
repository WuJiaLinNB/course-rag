#include <core/brute_index.hpp>
#include <algorithm>
#include <cmath>
#include <queue>
#include <stdexcept>

namespace core {

namespace {

// “更好”比较器：similarity 更高者更好；同分时 id 更小者更好。
// std::priority_queue 的堆顶是“其余所有元素都比它更好”的那一条，
// 以 Better 为比较器 → 堆顶恰是当前保留 K 条中最差的一条（守门员）：
//   - 保留规则：sim 大者胜；同 sim 时 id 小者胜 → 反转后最终输出“同分按 id 升序”
//   - 注意不能用 pair<float,uint32_t> 加 greater<> 的字典序：
//     同分时 id 小的 pair 反而更小、会被当最差者先淘汰，恰好丢掉想保留的 id
struct Better {
    bool operator()(const SearchItem& a, const SearchItem& b) const {
        return a.similarity > b.similarity ||
               (a.similarity == b.similarity && a.id < b.id);
    }
};

} // namespace

BruteIndex::BruteIndex(size_t dim) : store_(dim) {}

void BruteIndex::add(std::vector<float> v, const ChunkMeta& meta) {
    store_.add(std::move(v));          // VectorStore 内部归一化
    metas_.push_back(meta);
}

size_t BruteIndex::size() const { return store_.size(); }

std::vector<SearchItem> BruteIndex::search(const std::vector<float>& q,
                                           size_t k,
                                           const MetaFilter& filter) const {
    const size_t n = store_.size();
    const size_t dim = store_.dim();
    if (q.size() != dim) throw std::invalid_argument("query dim mismatch");
    const float* raw = store_.raw();

    // pre-filter：先筛候选 id（元数据不满足的直接不参与算分）
    std::vector<uint32_t> cand;
    for (size_t i = 0; i < n; ++i)
        if (match(metas_[i], filter)) cand.push_back((uint32_t)i);

    // 查询向量归一化（库内向量已归一化，点积即余弦）
    float n2 = 0.0f;
    for (float x : q) n2 += x * x;
    std::vector<float> qn(q);
    if (n2 > 0.0f) { float inv = 1.0f / std::sqrt(n2); for (float& x : qn) x *= inv; }

    // K 容量堆守门员：堆顶是当前 Top-K 中最差的，来更好的就淘汰堆顶
    std::priority_queue<SearchItem, std::vector<SearchItem>, Better> heap;
    for (uint32_t id : cand) {
        const float* v = raw + (size_t)id * dim;
        double s = 0.0;              // double 累加器：float 顺序误差在 1024 维可见，基准必须逐位可复现
        for (size_t d = 0; d < dim; ++d) s += (double)qn[d] * (double)v[d];
        SearchItem e{id, (float)s};
        if (heap.size() < k) heap.push(e);
        else if (Better()(e, heap.top())) { heap.pop(); heap.push(e); }
        // 守门员规则：新条目比堆顶（最差者）更好 → 淘汰堆顶、收下新条目
    }

    // 弹出顺序为“最差在前”：sim 升序、同分 id 降序；反转为 sim 降序、同分 id 升序
    std::vector<SearchItem> r;
    r.reserve(heap.size());
    while (!heap.empty()) { r.push_back(heap.top()); heap.pop(); }
    std::reverse(r.begin(), r.end());
    return r;
}

} // namespace core
