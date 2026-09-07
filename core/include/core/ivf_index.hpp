#pragma once
#include <core/index.hpp>
#include <core/vector_store.hpp>
#include <vector>

namespace core {

// IVF 倒排索引：kmeans 聚成 k_ 个桶，查询只扫最相似的 nprobe 个桶。
// 数据量未达 min_train_ 时不训练（ready()==false），search 自动降级为
// 全量暴力扫描（与 BruteIndex 逐位一致的算分/排序），Engine 层可无感降级
class IvfIndex : public Index {
public:
    IvfIndex(size_t dim, size_t min_train, size_t k_centroids = 32,
             size_t nprobe = 8);
    void add(std::vector<float> v, const ChunkMeta& meta) override;
    size_t size() const override;
    std::vector<SearchItem> search(const std::vector<float>& q,
                                   size_t k,
                                   const MetaFilter& filter) const override;
    std::string name() const override { return "ivf"; }

    bool ready() const;                       // 是否已训练
    void maybe_train();                       // 数据量达阈值时训练（数据增长50%重训）
    void set_nprobe(size_t n);

private:
    VectorStore store_;
    std::vector<ChunkMeta> metas_;
    size_t min_train_, k_, nprobe_, trained_at_ = 0;
    std::vector<std::vector<float>> centroids_;
    std::vector<uint32_t> bucket_of_;        // 每个点属于哪个桶
    std::vector<std::vector<uint32_t>> buckets_;
    // 训练后新增的点按最近中心增量补分配，保证重训前新数据也可检索
    //（bucket_of_ 始终与 store_.size() 等长）
    void assign_new_points();
};

} // namespace core
