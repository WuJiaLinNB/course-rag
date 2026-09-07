#include <gtest/gtest.h>
#include <core/ivf_index.hpp>
#include <core/brute_index.hpp>
#include <vector>

using namespace core;

namespace {
std::vector<float> mk(float a, float b) { return {a, b}; }
}

TEST(Ivf, TrainThreshold) {
    // 设计规格：数据 < 1 万条时拒绝初始化，需要 Engine 降级——这里直接测异常
    IvfIndex idx(2, /*min_train=*/100);
    for (int i = 0; i < 50; ++i)
        idx.add(mk(1, 0), {"", "", "", ""});
    EXPECT_FALSE(idx.ready());          // 不足阈值：未训练
}

TEST(Ivf, ReadyAfterTrain) {
    IvfIndex idx(2, /*min_train=*/100);
    for (int i = 0; i < 150; ++i) {
        float s = (i % 2) ? 1.0f : -1.0f;
        idx.add(mk(s, 0.1f * (i % 7)), {"", "", "", ""});
    }
    EXPECT_TRUE(idx.ready());
    auto r = idx.search(mk(1, 0), 5, {});
    EXPECT_LE(r.size(), 5u);
}

TEST(Ivf, ConsistentWithBrute) {
    // IVF 与暴力在无过滤、数据成簇时应高度一致
    BruteIndex brute(2);
    IvfIndex ivf(2, /*min_train=*/100);
    for (int i = 0; i < 200; ++i) {
        float s = (i % 2) ? 1.0f : -1.0f;
        auto v = mk(s, 0.01f * i);
        brute.add(v, {"", "", "", ""});
        ivf.add(v, {"", "", "", ""});
    }
    auto rb = brute.search(mk(1, 0), 10, {});
    auto ri = ivf.search(mk(1, 0), 10, {});
    size_t overlap = 0;
    for (auto& b : rb)
        for (auto& x : ri) if (b.id == x.id) ++overlap;
    EXPECT_GE(overlap, 8u);   // ≥ 80% 重叠（IVF 本身允许少量损失）
}
