#include <gtest/gtest.h>
#include <core/engine.hpp>
#include <core/vector_store.hpp>
#include <thread>
#include <atomic>
#include <chrono>
#include <random>
#include <cstdio>

using namespace core;

TEST(Engine, DegradesToBrute) {
    Engine eng(8, Engine::Mode::Auto);      // 数据不足 → 用 brute
    std::mt19937 rng(5);
    std::normal_distribution<float> g(0, 1);
    for (int i = 0; i < 100; ++i) {         // 100 < 1 万
        std::vector<float> v(8);
        for (auto& x : v) x = g(rng);
        eng.add(v, {"", "", "", ""});
    }
    std::vector<float> q(8);
    for (auto& x : q) x = g(rng);
    auto r = eng.search(q, 5, {});
    EXPECT_EQ(r.size(), 5u);
    EXPECT_EQ(eng.active_index_name(), "brute");   // 降级标注
}

TEST(Engine, ConcurrentReadWrite) {
    // 并发压测：1 线程 insert + 4 线程 search，通过标准 = 不崩 + 结果合法
    Engine eng(8, Engine::Mode::Auto);
    std::mt19937 rng(9);
    std::normal_distribution<float> g(0, 1);
    for (int i = 0; i < 1000; ++i) {
        std::vector<float> v(8);
        for (auto& x : v) x = g(rng);
        eng.add(v, {"", "", "", ""});
    }
    std::atomic<bool> stop{false};
    std::atomic<int> errors{0};

    std::thread writer([&] {
        std::mt19937 r(1);
        while (!stop) {
            std::vector<float> v(8);
            for (auto& x : v) x = g(r);
            try { eng.add(v, {"", "", "", ""}); } catch (...) { ++errors; }
        }
    });
    std::vector<std::thread> readers;
    for (int t = 0; t < 4; ++t)
        readers.emplace_back([&] {
            std::mt19937 r(2 + t);
            while (!stop) {
                std::vector<float> q(8);
                for (auto& x : q) x = g(r);
                try {
                    auto res = eng.search(q, 5, {});
                    if (res.size() > 5) ++errors;      // 合法性抽检
                    for (size_t i = 1; i < res.size(); ++i)
                        if (res[i-1].similarity < res[i].similarity) ++errors;
                } catch (...) { ++errors; }
            }
        });
    std::this_thread::sleep_for(std::chrono::seconds(30));
    stop = true;
    writer.join();
    for (auto& t : readers) t.join();
    EXPECT_EQ(errors.load(), 0);
}

TEST(Engine, LoadHotSwap) {
    // 启动重建状态机：load 后 index_ready()==false（降级 brute）→ 后台重建完成 → 热切换 → true
    const std::string path = "hotswap_vectors.bin";
    {
        VectorStore vs(4);
        std::mt19937 rng(11);
        std::normal_distribution<float> g(0, 1);
        for (int i = 0; i < 50; ++i) {
            std::vector<float> v(4);
            for (auto& x : v) x = g(rng);
            vs.add(v);
        }
        vs.save(path);
    }
    Engine eng(4, Engine::Mode::Auto);
    std::vector<ChunkMeta> metas(50, {"", "", "", ""});
    eng.load(path, metas);
    EXPECT_EQ(eng.active_index_name(), "brute");   // 重建中：暴力在服务
    eng.wait_rebuild();                            // 测试钩子：等后台重建线程
    EXPECT_TRUE(eng.index_ready());
    EXPECT_EQ(eng.active_index_name(), "hnsw");    // 热切换完成
    std::remove(path.c_str());
}

TEST(Engine, LoadRejectsCountMismatch) {
    // 向量条数 ≠ chunks 条数 = id 错位 → 拒绝载入（张冠李戴比崩溃更恶劣）
    const std::string path = "mismatch_vectors.bin";
    { VectorStore vs(4); vs.add({1,0,0,0}); vs.save(path); }
    Engine eng(4, Engine::Mode::Auto);
    std::vector<ChunkMeta> metas(3, {"", "", "", ""});   // 1 个向量 vs 3 条元数据
    EXPECT_THROW(eng.load(path, metas), std::runtime_error);
    std::remove(path.c_str());
}

TEST(Engine, PersistRoundtripBruteOnly) {  // persist 落盘 → 新 Engine 载入 → 检索一致（优雅停机的数据链路）
    const std::string path = "persist_roundtrip.bin";
    const size_t dim = 4;
    std::vector<float> q;
    {
        Engine eng(dim, Engine::Mode::BruteOnly);
        std::mt19937 rng(13);
        std::normal_distribution<float> g(0, 1);
        for (int i = 0; i < 20; ++i) {
            std::vector<float> v(dim);
            for (auto& x : v) x = g(rng);
            eng.add(v, {"", "", "", ""});
        }
        q.resize(dim);
        for (auto& x : q) x = g(rng);
        EXPECT_TRUE(eng.dirty());                 // 有未落盘数据
        eng.persist(path);                        // 原子落盘（Task 3 的 .tmp+rename）
        EXPECT_FALSE(eng.dirty());                // 落盘成功后清除
        auto r1 = eng.search(q, 1, {});
        ASSERT_EQ(r1.size(), 1u);
        // r1 与 r2 在同一作用域比对：直接在这里建第二个 Engine，落盘-载入后逐位一致
        Engine eng2(dim, Engine::Mode::BruteOnly);
        std::vector<ChunkMeta> metas(20, {"", "", "", ""});
        eng2.load(path, metas);                   // BruteOnly：不触发后台重建
        EXPECT_EQ(eng2.size(), 20u);
        EXPECT_FALSE(eng2.index_ready());
        EXPECT_EQ(eng2.active_index_name(), "brute");
        auto r2 = eng2.search(q, 1, {});
        ASSERT_EQ(r2.size(), 1u);
        EXPECT_EQ(r1[0].id, r2[0].id);            // brute 是确定性 ground truth → 逐位一致
        EXPECT_EQ(r1[0].similarity, r2[0].similarity);
    }
    std::remove(path.c_str());
}
