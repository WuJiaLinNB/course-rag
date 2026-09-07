#pragma once
#include <vector>
#include <cstdint>

namespace core {

struct KMeansResult {
    std::vector<std::vector<float>> centroids;  // k 个中心
    std::vector<uint32_t> assign;               // 每个点所属中心
};

// dim 维数据，k 个中心，iter 轮。内部固定随机种子（可复现）
KMeansResult kmeans(const std::vector<std::vector<float>>& data,
                    size_t k, size_t iters);

} // namespace core
