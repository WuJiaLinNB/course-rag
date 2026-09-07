#include <gtest/gtest.h>
#include <core/kmeans.hpp>
#include <cmath>

using namespace core;

TEST(KMeans, TwoClusters) {
    // 构造两个明显分离的簇
    std::vector<std::vector<float>> data;
    for (int i = 0; i < 50; ++i) data.push_back({1.0f + i*0.01f, 0.0f});
    for (int i = 0; i < 50; ++i) data.push_back({-1.0f - i*0.01f, 0.0f});
    auto km = kmeans(data, 2, 20);
    ASSERT_EQ(km.centroids.size(), 2u);
    // 两个中心应分居原点两侧
    float c0 = km.centroids[0][0], c1 = km.centroids[1][0];
    EXPECT_TRUE((c0 < 0 && c1 > 0) || (c0 > 0 && c1 < 0));
    // 每个 assigned 中心都存在且 ∈ [0,2)
    for (auto a : km.assign) EXPECT_TRUE(a == 0u || a == 1u);
    EXPECT_NE(km.assign[0], km.assign[99]);   // 正侧点与负侧点必不同簇
}

TEST(KMeans, FixedSeed) {
    std::vector<std::vector<float>> data;
    for (int i = 0; i < 100; ++i)
        data.push_back({std::sin((float)i), std::cos((float)i)});
    auto a = kmeans(data, 4, 10);
    auto b = kmeans(data, 4, 10);
    EXPECT_EQ(a.assign, b.assign);   // 固定种子 → 可复现
    EXPECT_EQ(a.centroids, b.centroids);
}

TEST(KMeans, ZeroVectorNoThrow) {  // 零向量不抛异常，assign 仍有效
    std::vector<std::vector<float>> data = {{1,0},{0,0},{-1,0}};
    auto km = kmeans(data, 2, 5);
    ASSERT_EQ(km.assign.size(), 3u);
    for (auto a : km.assign) EXPECT_TRUE(a == 0u || a == 1u);
}

TEST(KMeans, EmptyInput) {  // 空输入不崩，返回空结果
    auto km = kmeans({}, 3, 5);
    EXPECT_TRUE(km.centroids.empty());
    EXPECT_TRUE(km.assign.empty());
}

TEST(KMeans, KGreaterThanN) {  // k > n：不崩，assign 编号仍 ∈ [0,k)
    std::vector<std::vector<float>> data = {{1,0},{0,1}};
    auto km = kmeans(data, 5, 3);
    ASSERT_EQ(km.centroids.size(), 5u);
    ASSERT_EQ(km.assign.size(), 2u);
    for (auto a : km.assign) EXPECT_LT(a, 5u);
}
