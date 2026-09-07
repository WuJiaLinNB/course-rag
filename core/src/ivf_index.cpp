#include <core/ivf_index.hpp>
#include <core/kmeans.hpp>
#include <algorithm>
#include <cmath>
#include <numeric>
#include <queue>
#include <stdexcept>

// 过关自测：
// 1. nprobe = 全部桶时 IVF 和暴力等价，为什么？
//    nprobe 覆盖所有桶 → 每个点都会被扫描；算分与排序规则（pre-filter + double
//    累加点积 + 同一守门员堆 + 同分 id 升序）与 BruteIndex 完全一致 → 结果逐位相同，
//    IVF 只是多了“查中心选桶”这一步开销。
// 2. 数据从 100 长到 120 要不要重训？
//    不要。重训条件是 size > trained_at_ * 1.5（100*1.5=150），120 未达阈值；
//    新点只做增量分配（挂到最近中心），避免频繁重训抖动。
namespace core {

namespace {

// 与 BruteIndex 完全相同的“更好”比较器：sim 高者胜，同分 id 小者胜。
// 不能用 pair 字典序：同分时 id 小的 pair 反而被当最差者先淘汰（见 brute_index.cpp 注释）
struct Better {
    bool operator()(const SearchItem& a, const SearchItem& b) const {
        return a.similarity > b.similarity ||
               (a.similarity == b.similarity && a.id < b.id);
    }
};

} // namespace

IvfIndex::IvfIndex(size_t dim, size_t min_train, size_t k_centroids, size_t nprobe)
    : store_(dim), min_train_(min_train), k_(k_centroids), nprobe_(nprobe) {}

void IvfIndex::add(std::vector<float> v, const ChunkMeta& meta) {
    store_.add(std::move(v));              // VectorStore 内部归一化
    metas_.push_back(meta);
    maybe_train();
    // 桶覆盖语义：已训练时给新点按最近中心增量补分配——重训之前新数据也可检索，
    // 保证 bucket_of_ 始终覆盖全部点（重训时会被 kmeans 的 assign 整体重建）
    if (ready() && bucket_of_.size() < store_.size())
        assign_new_points();
}

size_t IvfIndex::size() const { return store_.size(); }

bool IvfIndex::ready() const { return !centroids_.empty(); }   // 有中心 = 已训练

void IvfIndex::set_nprobe(size_t n) { nprobe_ = n; }   // 允许 > k_，search 时再收敛

void IvfIndex::maybe_train() {
    const size_t n = store_.size();
    if (n < min_train_) return;                 // 未达门槛：保持未训练（search 自动降级）
    // 首训一次后，数据量超过上次训练时的 1.5 倍才重训
    //（double 比较，避免 size_t 乘 1.5 截断出错）
    if (!centroids_.empty() && !((double)n > (double)trained_at_ * 1.5)) return;

    // 取出全量数据（store_ 为 SoA 扁平存储）喂给 kmeans
    const size_t dim = store_.dim();
    const float* raw = store_.raw();
    std::vector<std::vector<float>> data;
    data.reserve(n);
    for (size_t i = 0; i < n; ++i)
        data.emplace_back(raw + i * dim, raw + (i + 1) * dim);

    // k 取 min(k_, n)：不给 kmeans 传 k > n（k 过大时中心大量重复、桶退化）。
    // 常规配置 min_train_ >= k_，训练时 n >= min_train_ >= k_ → 桶数恰为 k_。
    const size_t kk = std::min(k_, n);
    KMeansResult r = kmeans(data, kk, /*iters=*/10);   // 固定种子 → 可复现

    centroids_ = std::move(r.centroids);
    bucket_of_ = std::move(r.assign);           // 与全量点对齐，∈ [0, centroids_.size())
    buckets_.assign(centroids_.size(), {});
    for (size_t i = 0; i < n; ++i)
        buckets_[bucket_of_[i]].push_back((uint32_t)i);   // 桶内 id 升序 → 扫描顺序确定
    trained_at_ = n;
}

void IvfIndex::assign_new_points() {
    const size_t dim = store_.dim();
    const float* raw = store_.raw();
    for (size_t i = bucket_of_.size(); i < store_.size(); ++i) {
        const float* v = raw + i * dim;
        float best = -2.0f; size_t bi = 0;      // -2.0 哨兵：零范数中心永不选中（同 kmeans）
        for (size_t c = 0; c < centroids_.size(); ++c) {
            double s = 0.0, vc = 0.0, cc = 0.0;
            for (size_t d = 0; d < dim; ++d) {
                s += (double)v[d] * (double)centroids_[c][d];
                vc += (double)v[d] * (double)v[d];
                cc += (double)centroids_[c][d] * (double)centroids_[c][d];
            }
            if (cc <= 0.0) continue;            // 零范数中心跳过（成员相消时可能出现）
            float sim = (float)(s / std::sqrt(vc * cc));   // 余弦（中心非单位长）
            if (sim > best) { best = sim; bi = c; }
        }
        bucket_of_.push_back((uint32_t)bi);
        buckets_[bi].push_back((uint32_t)i);    // i 递增 → 桶内仍保持 id 升序
    }
}

std::vector<SearchItem> IvfIndex::search(const std::vector<float>& q,
                                         size_t k,
                                         const MetaFilter& filter) const {
    const size_t n = store_.size();
    const size_t dim = store_.dim();
    if (q.size() != dim) throw std::invalid_argument("query dim mismatch");
    const float* raw = store_.raw();

    // 查询向量归一化（库内向量已归一化，点积即余弦）
    float n2 = 0.0f;
    for (float x : q) n2 += x * x;
    std::vector<float> qn(q);
    if (n2 > 0.0f) { float inv = 1.0f / std::sqrt(n2); for (float& x : qn) x *= inv; }

    // K 容量堆守门员：与 BruteIndex 同一比较器、同一规则 → 分数逐位一致
    std::priority_queue<SearchItem, std::vector<SearchItem>, Better> heap;
    auto consider = [&](uint32_t id) {
        if (!match(metas_[id], filter)) return;         // pre-filter：不满足不算分
        const float* v = raw + (size_t)id * dim;
        double s = 0.0;          // double 累加器：float 顺序误差在 1024 维可见
        for (size_t d = 0; d < dim; ++d) s += (double)qn[d] * (double)v[d];
        SearchItem e{id, (float)s};
        if (heap.size() < k) heap.push(e);
        else if (Better()(e, heap.top())) { heap.pop(); heap.push(e); }
    };

    if (!ready()) {
        // 数据不足阈值：IVF 自动降级为全量暴力扫描——算分/排序与 BruteIndex 逐位一致，
        // Engine 层（Task 8）据此可无感降级
        for (size_t i = 0; i < n; ++i) consider((uint32_t)i);
    } else {
        // 1) 查询 vs 各中心算余弦，取最相似的 nprobe 个桶
        //（中心是成员均值、非单位向量，需除范数；零范数中心 -2.0 永不进 top）
        std::vector<float> cscore(centroids_.size(), -2.0f);
        double qc = 0.0;
        for (float x : qn) qc += (double)x * (double)x;
        for (size_t c = 0; c < centroids_.size(); ++c) {
            double s = 0.0, cc = 0.0;
            for (size_t d = 0; d < dim; ++d) {
                s += (double)qn[d] * (double)centroids_[c][d];
                cc += (double)centroids_[c][d] * (double)centroids_[c][d];
            }
            if (cc > 0.0 && qc > 0.0) cscore[c] = (float)(s / std::sqrt(qc * cc));
        }
        std::vector<uint32_t> order(centroids_.size());
        std::iota(order.begin(), order.end(), 0u);
        std::stable_sort(order.begin(), order.end(),    // 稳定排序：同分时中心编号小者在前
                         [&cscore](uint32_t a, uint32_t b) { return cscore[a] > cscore[b]; });

        // 2) 只扫被选中的桶：nprobe 收敛到桶数上限（set_nprobe 可传大值），空桶自然跳过
        const size_t nprobe = std::min(nprobe_, buckets_.size());
        for (size_t t = 0; t < nprobe; ++t)
            for (uint32_t id : buckets_[order[t]]) consider(id);
    }

    // 弹出顺序“最差在前”→ 反转为 sim 降序、同分 id 升序（与 BruteIndex 一致）
    std::vector<SearchItem> r;
    r.reserve(heap.size());
    while (!heap.empty()) { r.push_back(heap.top()); heap.pop(); }
    std::reverse(r.begin(), r.end());
    return r;
}

} // namespace core
