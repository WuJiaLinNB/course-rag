#include <core/kmeans.hpp>
#include <core/distance.hpp>
#include <random>
#include <cmath>

// 过关自测：
// 1. K-Means 两步循环：①分配——每个点归到余弦相似度最大的中心；
//    ②更新——每个中心替换为其成员的均值，交替迭代直到收敛/轮数用完。
// 2. 空簇不删掉：assign 里存的是中心编号，删掉某个中心会让已有编号悬空
//    （外部按编号取中心会错位），保留旧中心即可维持编号稳定。
namespace core {

namespace {

// 任一方范数为 0 时返回 -2.0f（压不过 best 哨兵 -2.0f）→ 该点永不选此簇，
// 空簇保留旧中心的规则自然接管；否则 cosine 会抛异常，IVF 训练直接崩。
float cosine_or_floor(const std::vector<float>& a, const std::vector<float>& b) {
    float na = 0.0f, nb = 0.0f;
    for (float x : a) na += x * x;
    for (float x : b) nb += x * x;
    if (na <= 0.0f || nb <= 0.0f) return -2.0f;
    return cosine(a, b);
}

// 分配：每点归余弦相似度最大的中心
void assign_points(const std::vector<std::vector<float>>& data,
                   const std::vector<std::vector<float>>& centroids,
                   std::vector<uint32_t>& assign) {
    for (size_t i = 0; i < data.size(); ++i) {
        float best = -2.0f; size_t bi = 0;
        for (size_t c = 0; c < centroids.size(); ++c) {
            float s = cosine_or_floor(data[i], centroids[c]);   // 越大越近
            if (s > best) { best = s; bi = c; }
        }
        assign[i] = (uint32_t)bi;
    }
}

} // namespace

KMeansResult kmeans(const std::vector<std::vector<float>>& data,
                    size_t k, size_t iters) {
    const size_t n = data.size(), dim = data.empty() ? 0 : data[0].size();
    KMeansResult r;
    r.assign.resize(n, 0);
    if (n == 0) return r;

    std::mt19937 rng(42);                       // 固定种子 → 可复现
    std::uniform_int_distribution<size_t> pick(0, n - 1);
    for (size_t i = 0; i < k; ++i) r.centroids.push_back(data[pick(rng)]);

    for (size_t it = 0; it < iters; ++it) {
        // 1. 分配：每点归余弦相似度最大的中心
        assign_points(data, r.centroids, r.assign);
        // 2. 更新：中心 = 成员均值
        std::vector<std::vector<float>> sum(r.centroids.size(),
                                            std::vector<float>(dim, 0.0f));
        std::vector<size_t> cnt(r.centroids.size(), 0);
        for (size_t i = 0; i < n; ++i) {
            auto& s = sum[r.assign[i]]; ++cnt[r.assign[i]];
            for (size_t d = 0; d < dim; ++d) s[d] += data[i][d];
        }
        for (size_t c = 0; c < r.centroids.size(); ++c) {
            if (cnt[c] == 0) continue;          // 空簇保留旧中心
            for (size_t d = 0; d < dim; ++d)
                r.centroids[c][d] = sum[c][d] / (float)cnt[c];
        }
    }
    // 最后一次分配与最终中心对齐，Task 6 IVF 用 assign 建桶
    assign_points(data, r.centroids, r.assign);
    return r;
}

} // namespace core
