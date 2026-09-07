#include <gtest/gtest.h>
#include <core/brute_index.hpp>
#include <cmath>
#include <random>

using namespace core;

TEST(Brute, TopKOrdering) {
    BruteIndex idx(2);
    idx.add({1,0},  { "", "", "", "a"});     // 与 query 相似度 1.0
    idx.add({0,1},  { "", "", "", "b"});     // 0.0
    idx.add({-1,0}, { "", "", "", "c"});     // -1.0
    auto r = idx.search({1,0}, 3, {});
    ASSERT_EQ(r.size(), 3u);
    EXPECT_EQ(r[0].id, 0u);   // 最相似在前，同分按 id 升序
    EXPECT_EQ(r[1].id, 1u);
    EXPECT_EQ(r[2].id, 2u);
    EXPECT_NEAR(r[0].similarity, 1.0f, 1e-6);
}

TEST(Brute, PreFilter) {
    // pre-filter 正确性：只搜 course=="os" 的，b 虽然更相似也被排除
    BruteIndex idx(2);
    idx.add({0,1},  { "os", "", "", "a"});
    idx.add({1,0},  { "db", "", "", "b"});
    MetaFilter f; f.course = "os";
    auto r = idx.search({1,0}, 1, f);
    ASSERT_EQ(r.size(), 1u);
    EXPECT_EQ(r[0].id, 0u);
}

TEST(Brute, TopKClamp) {
    BruteIndex idx(2);
    idx.add({1,0}, { "", "", "", "a"});
    auto r = idx.search({1,0}, 100, {});     // k 超容量 → 返回全部
    EXPECT_EQ(r.size(), 1u);
}

TEST(Brute, DeterministicTie) {
    BruteIndex idx(2);
    idx.add({1,0}, {"", "", "", "x"});
    idx.add({1,0}, {"", "", "", "y"});       // 相似度并列 → id 小者在前
    auto r = idx.search({1,0}, 2, {});
    EXPECT_EQ(r[0].id, 0u);
    EXPECT_EQ(r[1].id, 1u);
}

TEST(Brute, DeterministicScores) {
    // ground truth 必须逐位可复现：同一查询反复搜，分数完全相等（double 累加器）
    BruteIndex idx(8);
    std::mt19937 rng(21);
    std::normal_distribution<float> g(0, 1);
    for (int i = 0; i < 200; ++i) {
        std::vector<float> v(8);
        for (auto& x : v) x = g(rng);
        idx.add(v, {"", "", "", ""});
    }
    std::vector<float> q(8);
    for (auto& x : q) x = g(rng);
    auto r1 = idx.search(q, 5, {});
    auto r2 = idx.search(q, 5, {});
    ASSERT_EQ(r1.size(), r2.size());
    for (size_t i = 0; i < r1.size(); ++i) {
        EXPECT_EQ(r1[i].id, r2[i].id);
        EXPECT_EQ(r1[i].similarity, r2[i].similarity);   // 逐位相等，不是 NEAR
    }
}
