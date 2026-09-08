#include <core/engine.hpp>
#include <core/log.hpp>
#include <core/vector_store.hpp>
#include <mutex>
#include <stdexcept>
#include <utility>

namespace core {

namespace {
const char* mode_name(Engine::Mode m) {
    switch (m) {
        case Engine::Mode::BruteOnly: return "brute_only";
        case Engine::Mode::IvfAuto:   return "ivf_auto";   // v1：IVF 仅 benchmark，不进默认链路
        case Engine::Mode::HnswAuto:  return "hnsw_auto";
        case Engine::Mode::Auto:      return "auto";
    }
    return "unknown";
}
} // namespace

Engine::Engine(size_t dim, Mode mode) : dim_(dim), mode_(mode) {
    brute_ = std::make_unique<BruteIndex>(dim_);   // ground truth 常驻：brute_ 永不为空
    LOG_INFO("engine up: dim=%zu mode=%s", dim_, mode_name(mode_));
}

Engine::~Engine() {
    wait_rebuild();   // 析构前必须 join：悬空线程对象析构 = std::terminate
    LOG_INFO("engine down");
}

void Engine::add(std::vector<float> v, const ChunkMeta& meta) {
    std::unique_lock lk(mtx_);                       // 写锁：独占
    if (rebuilding_)                                 // v1 契约：重建窗口禁止写（server 503 挡住）
        LOG_WARN("add during hnsw rebuild window: vector absent from hnsw until reload");
    if (hnsw_) {                                     // 热切换完成后双写，两索引保持同步
        hnsw_->add(v, meta);                         // 拷贝一份给 hnsw
        brute_->add(std::move(v), meta);             // 原向量给 brute（ground truth 优先）
    } else {
        brute_->add(std::move(v), meta);
    }
    dirty_ = true;
}

std::vector<SearchItem> Engine::search(const std::vector<float>& q,
                                       size_t k,
                                       const MetaFilter& f) const {
    std::shared_lock lk(mtx_);                       // 读锁：允许多读并发
    if (hnsw_)                                       // 热切换完成：hnsw 在服务
        return hnsw_->search(q, k, f);
    if (mode_ != Mode::BruteOnly && !warned_degraded_.exchange(true))
        LOG_WARN("degraded to brute index (hnsw not ready)");   // 仅一次，避免刷屏
    return brute_->search(q, k, f);                  // 降级路径：brute 永远在服务
}

size_t Engine::size() const {
    std::shared_lock lk(mtx_);
    return brute_->size();
}

std::string Engine::active_index_name() const {
    std::shared_lock lk(mtx_);
    return hnsw_ ? "hnsw" : "brute";
}

bool Engine::index_ready() const {
    std::shared_lock lk(mtx_);
    return hnsw_ != nullptr;
}

void Engine::load(const std::string& vectors_path,
                  const std::vector<ChunkMeta>& metas) {
    try {
        VectorStore loaded(dim_);
        loaded.load(vectors_path);              // 内部已校验 magic/版本/维度/count/字节数
        if (loaded.size() != metas.size())      // 跨文件条数校验：id 错位 = 张冠李戴
            throw std::runtime_error("vector/chunk count mismatch: "
                + std::to_string(loaded.size()) + " vs " + std::to_string(metas.size()));
        {
            std::unique_lock lk(mtx_);
            brute_ = std::make_unique<BruteIndex>(dim_);
            for (size_t i = 0; i < loaded.size(); ++i)
                brute_->add(loaded.get(i), metas[i]);   // 暴力先就位，立即可以降级服务
        }
        if (mode_ == Mode::BruteOnly) {         // BruteOnly：从不构建 hnsw
            LOG_INFO("loaded %zu vectors (brute-only, hnsw disabled)", loaded.size());
            return;
        }
        LOG_INFO("loaded %zu vectors, rebuilding hnsw in background", loaded.size());
        // v1 设计简化（明确写进设计，不是疏忽）：load() 只发生在启动时刻，
        // /documents（Task 12）在 index_ready()==false 期间返回 503——
        // 保证重建窗口内无写冲突，重建与热切换因此不需要处理并发写。
        if (rebuild_thread_.joinable())         // 防御：先 join 旧线程再赋值，否则 std::terminate
            rebuild_thread_.join();
        rebuilding_ = true;
        rebuild_thread_ = std::thread([this, snap = std::move(loaded), metas] {
            try {
                auto ni = std::make_unique<HnswIndex>(dim_, /*M=*/16, /*efC=*/200);   // 独立新实例，不碰在线索引
                // 默认 efS=16 在 10 万×1024 合成数据上召回仅 0.568（benchmark 首跑实测），
                // 定档 efS=128：召回 0.978（P99 3.8ms / QPS 397），满足 95% 关卡的最低档；
                // 完整曲线见 docs/benchmark.md（16→0.568 / 64→0.908 / 128→0.978 / 256→0.997）
                ni->set_ef_search(128);
                for (size_t i = 0; i < snap.size(); ++i) ni->add(snap.get(i), metas[i]);
                {
                    std::unique_lock lk(mtx_);
                    hnsw_ = std::move(ni);          // 锁内原子指针交换 = 热切换
                }
                LOG_INFO("hnsw rebuild done, hot-swapped");
            } catch (const std::exception& e) {
                // 重建失败不致命：hnsw_ 保持空，brute 继续服务（降级路径兜底）
                LOG_ERROR("hnsw rebuild failed: %s", e.what());
            }
            rebuilding_ = false;
        });
    } catch (const std::exception& e) {
        LOG_ERROR("engine load failed: %s", e.what());   // 致命错误：记日志后原样上抛
        throw;
    }
}

void Engine::persist(const std::string& vectors_path) const {
    std::shared_lock lk(mtx_);                  // save 只读数据，读锁即可
    brute_->save_vectors(vectors_path);         // VectorStore::save 失败抛异常 → dirty_ 保持 true
    dirty_ = false;                             // 成功落盘后才清脏标记
    // v1 persist 只落 vectors.bin；chunks.json 归 server 层（Task 12）负责
}

bool Engine::dirty() const { return dirty_; }

void Engine::wait_rebuild() {
    // 过关自测：为什么析构函数也要调用？——rebuild_thread_ 若在 join 前析构，
    // std::thread 析构函数对 joinable 线程直接 std::terminate，进程当场崩掉。
    // join 仅在 joinable 时做；join 后 joinable()==false，重复调用是空操作（幂等安全）。
    if (rebuild_thread_.joinable())
        rebuild_thread_.join();
}

} // namespace core
