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
        // 1. 分配：每点归最近中心（余弦距离 = 1 - 相似度，或欧氏）
        for (size_t i = 0; i < n; ++i) {
            float best = -2.0f; size_t bi = 0;
            for (size_t c = 0; c < r.centroids.size(); ++c) {
                float s = cosine(data[i], r.centroids[c]);   // 越大越近
                if (s > best) { best = s; bi = c; }
            }
            r.assign[i] = (uint32_t)bi;
        }
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
    return r;
}

} // namespace core
