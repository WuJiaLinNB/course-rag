#pragma once
#include <core/index.hpp>
#include <core/vector_store.hpp>
#include <string>
#include <vector>

namespace core {

// 暴力检索：全量扫描 + K 容量小顶堆守门员。
// 它是全项目的 ground truth——实现必须绝对正确且逐位可复现（见 search 内注释）
class BruteIndex : public Index {
public:
    explicit BruteIndex(size_t dim);
    void add(std::vector<float> v, const ChunkMeta& meta) override;
    size_t size() const override;
    std::vector<SearchItem> search(const std::vector<float>& q,
                                   size_t k,
                                   const MetaFilter& filter) const override;
    std::string name() const override { return "brute"; }
    // v1 persist 转发：VectorStore 是唯一权威数据源，原子写直接复用（Engine::persist 转调）
    void save_vectors(const std::string& path) const { store_.save(path); }
private:
    VectorStore store_;
    std::vector<ChunkMeta> metas_;
};

} // namespace core
