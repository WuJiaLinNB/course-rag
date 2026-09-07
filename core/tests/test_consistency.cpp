#include <gtest/gtest.h>
#include <core/engine.hpp>
#include <core/brute_index.hpp>
#include <core/hnsw_index.hpp>
#include <random>

using namespace core;

namespace {
std::vector<float> rv(size_t dim, std::mt19937& rng) {
    std::normal_distribution<float> g(0, 1);
    std::vector<float> v(dim);
    for (auto& x : v) x = g(rng);
    return v;
}
}

TEST(Consistency, HnswVsBrute95) {
    const size_t N = 5000, Q = 100, DIM = 32;
    BruteIndex brute(DIM);
    HnswIndex hnsw(DIM, 16, 200);
    hnsw.set_ef_search(64);   // 默认 efS=16 在 32 维随机数据上召回不足（实测 0.768）；
                              // ef 是公开的检索精度/速度旋钮：ef↑ → 召回↑ → QPS↓。
                              // 64 时实测 0.984 ≥ 0.95；默认值调优归 Task 13 benchmark。
    std::mt19937 rng(1234);
    std::vector<std::vector<float>> data, queries;
    for (size_t i = 0; i < N; ++i) {
        auto v = rv(DIM, rng);
        brute.add(v, {"", "", "", ""});
        hnsw.add(v, {"", "", "", ""});
        if (i < Q) queries.push_back(rv(DIM, rng));
    }
    size_t hits = 0;
    for (auto& q : queries) {
        auto rb = brute.search(q, 10, {});
        auto rh = hnsw.search(q, 10, {});
        for (auto& b : rb)
            for (auto& h : rh) if (b.id == h.id) { ++hits; break; }
    }
    EXPECT_GE((float)hits / (Q * 10), 0.95f);
}
