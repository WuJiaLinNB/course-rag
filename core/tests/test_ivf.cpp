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

TEST(Ivf, DegradedPathMatchesBruteBitwise) {  // 未训练时降级路径必须与暴力逐位一致（Engine 无感降级的依据）
    BruteIndex brute(2);
    IvfIndex ivf(2, /*min_train=*/100);
    for (int i = 0; i < 50; ++i) {
        auto v = mk(1.0f + 0.01f * i, 0.5f - 0.02f * i);
        brute.add(v, {"", "", "", ""});
        ivf.add(v, {"", "", "", ""});
    }
    ASSERT_FALSE(ivf.ready());
    auto rb = brute.search(mk(1, 0), 7, {});
    auto ri = ivf.search(mk(1, 0), 7, {});
    ASSERT_EQ(rb.size(), ri.size());
    for (size_t i = 0; i < rb.size(); ++i) {
        EXPECT_EQ(rb[i].id, ri[i].id);
        EXPECT_EQ(rb[i].similarity, ri[i].similarity);   // 逐位相等，不是 NEAR
    }
}

TEST(Ivf, PostTrainPointsVisible) {  // 训练后新增的点必须可被检索（增量补分配语义）
    IvfIndex idx(2, /*min_train=*/100);
    for (int i = 0; i < 100; ++i) idx.add(mk(1, 0), {"", "", "", ""});
    ASSERT_TRUE(idx.ready());
    idx.add(mk(-1, 0), {"", "", "", ""});        // 训练后新增
    auto r = idx.search(mk(-1, 0), 1, {});
    ASSERT_EQ(r.size(), 1u);
    EXPECT_EQ(r[0].id, 100u);
}

TEST(Ivf, FullProbeWithFilterMatchesBrute) {  // nprobe=全部桶 + 过滤：与暴力结果完全一致
    BruteIndex brute(2);
    IvfIndex ivf(2, /*min_train=*/100, /*k_centroids=*/8, /*nprobe=*/2);
    for (int i = 0; i < 120; ++i) {
        float s = (i % 2) ? 1.0f : -1.0f;
        auto v = mk(s, 0.1f * (i % 5));
        const char* c = (i % 2) ? "os" : "db";
        brute.add(v, {c, "", "", ""});
        ivf.add(v, {c, "", "", ""});
    }
    ASSERT_TRUE(ivf.ready());
    ivf.set_nprobe(8);                            // 探全部桶 → 等价暴力
    MetaFilter f; f.course = "os";
    auto rb = brute.search(mk(1, 0), 5, f);
    auto ri = ivf.search(mk(1, 0), 5, f);
    ASSERT_EQ(rb.size(), ri.size());
    for (size_t i = 0; i < rb.size(); ++i) {
        EXPECT_EQ(rb[i].id, ri[i].id);
        EXPECT_EQ(rb[i].similarity, ri[i].similarity);
    }
}
