#pragma once
#include <core/types.hpp>
#include <cstddef>
#include <string>
#include <vector>

namespace core {

// 检索索引的抽象接口：所有索引（brute/ivf/hnsw）都实现它，
// server 层只面向此接口编程，可无缝换索引实现
class Index {
public:
    virtual ~Index() = default;
    // 追加一条已归一化向量（id = 当前容量）
    virtual void add(std::vector<float> v, const ChunkMeta& meta) = 0;
    virtual size_t size() const = 0;
    // 返回按相似度降序的 Top-K（同分按 id 升序，保证确定性）
    virtual std::vector<SearchItem> search(const std::vector<float>& q,
                                           size_t k,
                                           const MetaFilter& filter) const = 0;
    virtual std::string name() const = 0;   // "brute" / "ivf" / "hnsw"
};

} // namespace core
