#include <gtest/gtest.h>
#include <core/hnsw_index.hpp>
#include <core/brute_index.hpp>
#include <random>
#include <vector>

using namespace core;

namespace {
std::vector<float> rand_vec(size_t dim, std::mt19937& rng) {
    std::normal_distribution<float> g(0, 1);
    std::vector<float> v(dim);
    for (auto& x : v) x = g(rng);
    return v;
}
}

TEST(Hnsw, BasicSearch) {
    HnswIndex idx(8, /*M=*/8, /*ef_construction=*/100);
    std::mt19937 rng(1);
    for (int i = 0; i < 500; ++i)
        idx.add(rand_vec(8, rng), {"", "", "", ""});
    auto r = idx.search(rand_vec(8, rng), 10, {});
    EXPECT_EQ(r.size(), 10u);
}

TEST(Hnsw, RecallVsBrute) {
    HnswIndex hnsw(8, 8, 100);
    BruteIndex brute(8);
    std::mt19937 rng(7);
    std::vector<std::vector<float>> qs;
    for (int i = 0; i < 1000; ++i) {
        auto v = rand_vec(8, rng);
        hnsw.add(v, {"", "", "", ""});
        brute.add(v, {"", "", "", ""});
        if (i % 25 == 0) qs.push_back(v);      // 抽 40 个查询
    }
    size_t hits = 0, total = 0;
    for (auto& q : qs) {
        auto rh = hnsw.search(q, 10, {});
        auto rb = brute.search(q, 10, {});
        for (auto& b : rb)
            for (auto& h : rh) if (b.id == h.id) { ++hits; break; }
        total += 10;
    }
    float recall = (float)hits / total;
    EXPECT_GE(recall, 0.95f);      // 设计规格：一致性 ≥ 95%
}

TEST(Hnsw, PostFilter) {
    // HNSW 第一版：post-filter（声明限制）。查 k=10 但过滤后可能少于 10
    HnswIndex idx(8, 8, 100);
    std::mt19937 rng(3);
    for (int i = 0; i < 300; ++i) {
        auto v = rand_vec(8, rng);
        ChunkMeta m; m.course = (i % 2) ? "os" : "db";
        idx.add(v, m);
    }
    MetaFilter f; f.course = "os";
    auto r = idx.search(rand_vec(8, rng), 10, f);
    EXPECT_LE(r.size(), 10u);      // 可能不足 10（post-filter 特性）
}
