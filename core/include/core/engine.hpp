#pragma once
#include <core/index.hpp>
#include <core/brute_index.hpp>
#include <core/hnsw_index.hpp>
#include <shared_mutex>
#include <atomic>
#include <thread>
#include <memory>
#include <string>
#include <vector>

namespace core {

// 门面：持有索引，负责模式选择/降级/读写锁/后台重建/热切换（设计文档第 3、6 节）
//
// 过关自测：
// 1) 读锁为什么可以多个并发、写锁必须独占？——search 是纯只读操作（brute/hnsw 的
//    search 都不改索引内部结构），多读并发只提升吞吐、无正确性风险；add 会修改
//    brute_/hnsw_ 的内部存储与图结构，写的同时有人读会看到撕裂/半成品状态，
//    所以写锁必须独占（读读共享、读写/写写互斥）。
// 2) 热切换为什么必须"独立实例建好后再指针交换"，而不是边服务边往 hnsw_ 里 add？
//    ——边服务边改有双重问题：其一，读者遍历图的同时写者改图 = 数据竞争；
//    其二，即便加了锁，读者也会查到边不全的半成品图，召回率不可控。
//    独立新实例建好 + 锁内指针整体替换，读者要么看到完整旧态（brute 服务），
//    要么看到完整新态（hnsw 服务），永远不存在中间态。
class Engine {
public:
    enum class Mode { BruteOnly, IvfAuto, HnswAuto, Auto };   // Auto = HnswAuto
    // v1 模式语义：hnsw 只由 load() 的后台重建产生（启动路径）；
    // BruteOnly 从不构建 hnsw；Auto/HnswAuto/IvfAuto 在重建完成后由 hnsw 服务，
    // 重建完成前降级 brute（1 万以下暴力延迟可接受）。IVF 并入 benchmark 对比，
    // 不进 Engine 默认链路（设计文档 4 节：Engine 用 brute/hnsw 二态）。

    Engine(size_t dim, Mode mode = Mode::Auto);
    ~Engine();                                // 析构必须 join 后台重建线程

    void add(std::vector<float> v, const ChunkMeta& meta);   // 写锁
    std::vector<SearchItem> search(const std::vector<float>& q,
                                   size_t k,
                                   const MetaFilter& f) const;  // 读锁
    size_t size() const;
    std::string active_index_name() const;   // 当前实际服务的索引名
    bool index_ready() const;                // hnsw 是否已在服务（API 的 index_ready 字段）

    // 读 vectors.bin（文件级校验在 VectorStore 内）+ 跨文件条数校验 + 后台重建 hnsw
    // metas 由 server 层从 chunks.json 解析而来——core 不碰 JSON
    void load(const std::string& vectors_path, const std::vector<ChunkMeta>& metas);
    void persist(const std::string& vectors_path) const;  // 原子全量重写 vectors.bin
    bool dirty() const;                      // 有未落盘数据（优雅停机时决定是否 flush）
    void wait_rebuild();                     // 等待后台重建结束（测试钩子/析构共用）

private:
    size_t dim_;
    Mode mode_;
    mutable std::shared_mutex mtx_;
    std::unique_ptr<BruteIndex> brute_;       // 永远保留（ground truth + 降级）
    std::unique_ptr<HnswIndex> hnsw_;         // 热切换对象：整体替换，绝不边服务边改
    std::thread rebuild_thread_;
    std::atomic<bool> rebuilding_{false};
    mutable std::atomic<bool> dirty_{false};
    mutable std::atomic<bool> warned_degraded_{false};   // 降级 WARN 只发一次，避免刷屏
};

} // namespace core
